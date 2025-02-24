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

    This function computes either the counts or percentages of cells from each cluster
    that are present in each group. The result is a matrix where rows represent clusters
    and columns represent groups.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing cell annotations.
    group : str
        Column name in adata.obs containing group labels (e.g., 'condition', 'treatment').
    cluster : str
        Column name in adata.obs containing cluster labels (e.g., 'leiden', 'louvain').
    normalize : bool, optional (default=True)
        If True, return percentages (0-100).
        If False, return absolute counts.

    Returns
    -------
    pd.DataFrame
        Matrix of cluster distributions across groups.
        Rows: clusters
        Columns: groups
        Values: percentages (if normalize=True) or counts (if normalize=False)

    Raises
    ------
    ValueError
        If group or cluster columns are not found in adata.obs
    RuntimeError
        If an error occurs during calculation

    Examples
    --------
    >>> import scanpy as sc
    >>> adata = sc.datasets.pbmc3k()
    >>> sc.tl.leiden(adata)
    >>> result = cluster_distribution(adata, group='bulk_labels', cluster='leiden')
    >>> print(result)
    """
    # Validate input columns exist in adata.obs
    if group not in adata.obs.columns:
        raise ValueError(f"Group column '{group}' not found in adata.obs")
    if cluster not in adata.obs.columns:
        raise ValueError(f"Cluster column '{cluster}' not found in adata.obs")

    try:
        # Calculate distribution using pandas operations
        grouped = adata.obs[[group, cluster]].groupby(group, observed=True)
        value_counts = grouped.value_counts(normalize=normalize)
        
        # Convert to percentages if normalized
        if normalize:
            value_counts = value_counts * 100
            rounded_counts = value_counts.round(1)
        else:
            rounded_counts = value_counts

        # Reshape result to matrix form and fill missing values
        result = rounded_counts.unstack().fillna(0).transpose()
        return result
    
    except Exception as e:
        raise RuntimeError(f"Error calculating cluster distribution: {str(e)}")

def ensemble_identify_cluster_markers(
    adata: AnnData,
    methods: list = ["wilcoxon", "t-test_overestim_var", "t-test", "logreg"],
    n_cores: int = -1
) -> pd.DataFrame:
    """
    Perform ensemble differential expression analysis using multiple statistical methods.

    This function combines results from multiple differential expression methods to identify
    robust marker genes for each cluster. It calculates a consensus ranking based on the
    median ranks across all methods used.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix that must contain:
        - 'highly_variable' column in .var identifying variable genes
        - 'cluster' column in .obs containing cluster labels
    methods : list, optional
        Statistical methods to use for differential expression analysis.
        Must be valid methods for scanpy.tl.rank_genes_groups.
        Default: ["wilcoxon", "t-test_overestim_var", "t-test", "logreg"]
    n_cores : int, optional
        Number of CPU cores for parallel processing.
        Default: -1 (use all available cores)

    Returns
    -------
    pd.DataFrame
        Combined results with the following columns:
        - names: Gene names
        - cluster: Cluster identifiers
        - {method}_rank: Rankings from each method
        - {method}_scores: Scores from each method
        - {method}_pvals: P-values (if available)
        - {method}_pvals_adj: Adjusted p-values (if available)
        - logfoldchanges: Mean log fold changes across methods
        - pct_group: Mean percentage of cells expressing gene in cluster
        - pct_rest: Mean percentage of cells expressing gene in other clusters
        - pct_diff: Difference between pct_group and pct_rest
        - median_rank: Consensus ranking across all methods

    Notes
    -----
    - Analysis is performed only on highly variable genes
    - Results are filtered by adjusted p-value < 0.05 when available
    - Final rankings are 1-based within each cluster
    - Results are sorted by cluster and median rank

    Examples
    --------
    >>> import scanpy as sc
    >>> adata = sc.datasets.pbmc3k()
    >>> sc.pp.highly_variable_genes(adata)
    >>> sc.tl.leiden(adata, key_added='cluster')
    >>> markers = ensemble_identify_cluster_markers(adata)
    >>> print(markers.head())
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
                              key_added=method,
                              n_jobs=n_cores)
        
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
    
    # Keep only one logfoldchange column from methods that have it
    logfc_methods = [m for m in methods if m != 'logreg']
    if logfc_methods:
        logfc_columns = [f'{method}_logfoldchanges' for method in logfc_methods]
        final_df['logfoldchanges'] = final_df[logfc_columns].mean(axis=1).round(2)
        final_df = final_df.drop(columns=logfc_columns)
    
    # Add rank for logfoldchanges within each cluster
    if 'logfoldchanges' in final_df.columns:
        for cluster in final_df['cluster'].unique():
            mask = final_df['cluster'] == cluster
            final_df.loc[mask, 'logfoldchanges_rank'] = final_df.loc[mask, 'logfoldchanges'].abs().rank(method='min', ascending=False)

    # Keep only one *_pct_nz and *_pct_nz_reference column from methods that have it
    pct_nz_methods = [m for m in methods if m != 'logreg']
    if pct_nz_methods:
        pct_nz_group_columns = [f'{method}_pct_nz_group' for method in pct_nz_methods]
        pct_nz_reference_columns = [f'{method}_pct_nz_reference' for method in pct_nz_methods]
        final_df['pct_group'] = final_df[pct_nz_group_columns].mean(axis=1).round(2)
        final_df['pct_rest'] = final_df[pct_nz_reference_columns].mean(axis=1).round(2)
        final_df['pct_diff'] = final_df['pct_group'] - final_df['pct_rest']
        final_df['pct_diff'] = final_df['pct_diff'].round(2)
        final_df = final_df.drop(columns=pct_nz_group_columns + pct_nz_reference_columns)
    
    # Add rank for pct_diff within each cluster
    if 'pct_diff' in final_df.columns:
        for cluster in final_df['cluster'].unique():
            mask = final_df['cluster'] == cluster
            final_df.loc[mask, 'pct_diff_rank'] = final_df.loc[mask, 'pct_diff'].abs().rank(method='min', ascending=False)

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