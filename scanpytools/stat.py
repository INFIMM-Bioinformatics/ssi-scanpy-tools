import scanpy as sc
import pandas as pd
from anndata import AnnData

def cluster_distribution(
    adata: AnnData,
    group: str,
    cluster: str,
    normalize: bool = True
) -> pd.DataFrame:
    """
    Calculate the distribution of clusters across groups in AnnData object.

    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix.
    group : str
        Column name in adata.obs for grouping.
    cluster : str
        Column name in adata.obs for clusters.
    normalize : bool, optional (default=True)
        If True, return percentages. If False, return counts.

    Returns:
    --------
    pd.DataFrame
        Matrix of cluster distributions across groups.

    Raises:
    -------
    ValueError
        If group or cluster columns are not found in adata.obs
    """
    # Input validation
    if group not in adata.obs.columns:
        raise ValueError(f"Group column '{group}' not found in adata.obs")
    if cluster not in adata.obs.columns:
        raise ValueError(f"Cluster column '{cluster}' not found in adata.obs")

    try:
        grouped = adata.obs[[group, cluster]].groupby(group, observed=True)
        value_counts = grouped.value_counts(normalize=normalize)
        
        if normalize:
            value_counts = value_counts * 100
            rounded_counts = value_counts.round(1)
        else:
            rounded_counts = value_counts

        result = rounded_counts.unstack().fillna(0).transpose()
        return result
    
    except Exception as e:
        raise RuntimeError(f"Error calculating cluster distribution: {str(e)}")
    
# def ensemble_rank_genes_groups(adata, groupby)
def ensemble_identify_cluster_markers(adata, methods=["wilcoxon", "t-test_overestim_var", "t-test", "logreg"]):
    """
    Perform differential expression analysis using multiple methods and combine results.
    This function runs multiple differential expression analyses on highly variable genes using
    different statistical methods, merges the results, and calculates a consensus ranking based
    on median ranks across methods.
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with the following requirements:
        - Must have 'highly_variable' column in .var
        - Must have 'cluster' column in .obs
    methods : list, optional
        List of methods to use for differential expression analysis.
        Default is ["wilcoxon", "t-test_overestim_var", "t-test", "logreg"]
        Must be valid methods accepted by scanpy.tl.rank_genes_groups
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing merged results from all methods with the following columns:
        - 'names': Gene names
        - 'cluster': Cluster identifiers
        - '{method}_rank': Rank of genes for each method
        - '{method}_scores': Scores for each method
        - '{method}_pvals': P-values if available
        - '{method}_pvals_adj': Adjusted p-values if available
        - '{method}_logfoldchanges': Log fold changes if available
        - 'median_rank': Consensus ranking based on median of method-specific ranks
    Notes
    -----
    - Only genes with adjusted p-value < 0.05 (when available) are included in results
    - Final results are sorted by cluster and median rank
    - Ranks are recalculated to be 1-based within each cluster
    """
    # Make a copy and subset to HVG
    adata_copy = adata[:,adata.var["highly_variable"]].copy()
    print(f"Running analysis on {adata_copy.n_vars} highly variable genes")
    
    # Initialize results dictionary
    results_dict = {}
    
    # For each method
    for method in methods:
        print(f"Processing method: {method}")
        
        # Run rank_genes_groups
        sc.tl.rank_genes_groups(adata_copy, 
                              groupby='cluster',
                              method=method,
                              rankby_abs=True,
                              pts=True,
                              key_added=method)
        
        # Process each cluster
        method_results = []
        clusters = sorted(adata_copy.obs['cluster'].unique())
        
        for cluster in clusters:
            # Get results for this cluster
            df = sc.get.rank_genes_groups_df(adata_copy, group=cluster, key=method)
            
            # Add cluster info and rank
            df.insert(0, 'cluster', cluster)
            df['rank'] = range(1, len(df) + 1)
            
            # Filter by adjusted p-value if available
            if 'pvals_adj' in df.columns:
                df = df[df['pvals_adj'] < 0.05]
            
            # Prefix column names with method except names and cluster
            df.columns = [f'{method}_{col}' if col not in ['names', 'cluster'] 
                         else col for col in df.columns]
            
            method_results.append(df)
        
        # Combine results for all clusters for this method
        results_dict[method] = pd.concat(method_results, axis=0)
    
    # Merge results from all methods
    final_df = None
    for method, df in results_dict.items():
        if final_df is None:
            final_df = df
        else:
            final_df = final_df.merge(df, on=['names', 'cluster'], how='outer')
    
    # After merging all methods, calculate cluster-wise median rank
    rank_columns = [f'{method}_rank' for method in methods]
    
    # Group by cluster and calculate median rank within each cluster
    for cluster in final_df['cluster'].unique():
        mask = final_df['cluster'] == cluster
        final_df.loc[mask, 'median_rank'] = final_df.loc[mask, rank_columns].median(axis=1)
        
        # Reset rank to be 1-based within each cluster
        final_df.loc[mask, 'median_rank'] = final_df.loc[mask, 'median_rank'].rank(method='min')
    
    # Sort by cluster and median rank
    final_df = final_df.sort_values(['cluster', 'median_rank'])
    
    return final_df

def ensemble_identify_cluster_markers_optimized(adata, methods=["wilcoxon", "t-test_overestim_var", "t-test", "logreg"]):
    """
    Optimized version of ensemble_identify_cluster_markers with improved performance.
    Uses numpy operations instead of pandas where possible and reduces memory allocations.
    
    Parameters and returns are the same as ensemble_identify_cluster_markers.
    """
    import numpy as np
    from scipy import stats
    
    # Make a copy and subset to HVG
    adata_copy = adata[:,adata.var["highly_variable"]].copy()
    print(f"Running analysis on {adata_copy.n_vars} highly variable genes")
    
    # Pre-allocate dictionaries for results
    clusters = sorted(adata_copy.obs['cluster'].unique())
    method_results = {method: {} for method in methods}
    
    # Process each method
    for method in methods:
        print(f"Processing method: {method}")
        
        # Run rank_genes_groups
        sc.tl.rank_genes_groups(adata_copy, 
                              groupby='cluster',
                              method=method,
                              rankby_abs=True,
                              pts=True,
                              key_added=method)
        
        # Pre-allocate lists for each cluster
        for cluster in clusters:
            df = sc.get.rank_genes_groups_df(adata_copy, group=cluster, key=method)
            
            # Filter by adjusted p-value if available
            if 'pvals_adj' in df.columns:
                mask = df['pvals_adj'] < 0.05
                df = df[mask]
            
            # Store as dict for faster access
            method_results[method][cluster] = {
                'names': df['names'].values,
                'scores': df['scores'].values if 'scores' in df else None,
                'pvals': df['pvals'].values if 'pvals' in df else None,
                'pvals_adj': df['pvals_adj'].values if 'pvals_adj' in df else None,
                'logfoldchanges': df['logfoldchanges'].values if 'logfoldchanges' in df else None
            }
    
    # Build final results using sets for faster lookups
    all_genes = set()
    for method_dict in method_results.values():
        for cluster_dict in method_dict.values():
            all_genes.update(cluster_dict['names'])
    
    # Create final dataframe more efficiently
    final_data = []
    
    for cluster in clusters:
        cluster_genes = set()
        for method in methods:
            cluster_genes.update(method_results[method][cluster]['names'])
        
        for gene in cluster_genes:
            row = {'cluster': cluster, 'names': gene}
            
            # Get ranks and scores for each method
            method_ranks = []
            for method in methods:
                cluster_data = method_results[method][cluster]
                try:
                    idx = np.where(cluster_data['names'] == gene)[0][0]
                    row[f'{method}_rank'] = idx + 1
                    if cluster_data['scores'] is not None:
                        row[f'{method}_scores'] = cluster_data['scores'][idx]
                    if cluster_data['pvals'] is not None:
                        row[f'{method}_pvals'] = cluster_data['pvals'][idx]
                    if cluster_data['pvals_adj'] is not None:
                        row[f'{method}_pvals_adj'] = cluster_data['pvals_adj'][idx]
                    if cluster_data['logfoldchanges'] is not None:
                        row[f'{method}_logfoldchanges'] = cluster_data['logfoldchanges'][idx]
                    method_ranks.append(idx + 1)
                except (IndexError, KeyError):
                    row[f'{method}_rank'] = np.nan
            
            # Calculate median rank
            if method_ranks:
                row['median_rank'] = np.nanmedian(method_ranks)
                final_data.append(row)
    
    final_df = pd.DataFrame(final_data)
    
    # Rank within clusters more efficiently using numpy
    for cluster in clusters:
        mask = final_df['cluster'] == cluster
        ranks = final_df.loc[mask, 'median_rank'].values
        final_df.loc[mask, 'median_rank'] = stats.rankdata(ranks, method='min')
    
    # Sort by cluster and median rank
    final_df = final_df.sort_values(['cluster', 'median_rank'])
    
    return final_df
