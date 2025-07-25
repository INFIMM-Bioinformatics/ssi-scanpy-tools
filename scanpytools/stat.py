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
    var_cluster: str = "cluster",
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
        - var_cluster column in .obs containing cluster labels
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
    >>> sc.tl.leiden(adata, key_added=var_cluster)
    >>> markers = ensemble_identify_cluster_markers(adata)
    >>> print(markers.head())
    """
    # Make a copy and subset to HVG
    # Check if 'highly_variable' exists, if not, compute it
    if "highly_variable" not in adata.var.columns:
        print("Column 'highly_variable' not found in adata.var. Running sc.pp.highly_variable_genes...")
        sc.pp.highly_variable_genes(adata)
    adata_copy = adata[:, adata.var["highly_variable"]].copy()
    print(f"Running analysis on {adata_copy.n_vars} highly variable genes")
    
    # Initialize results dictionary
    results_dict = {}
    
    # For each method
    for method in methods:
        print(f"Processing method: {method}")
        
        # Run rank_genes_groups
        sc.tl.rank_genes_groups(adata_copy, 
                              groupby=var_cluster,
                              method=method,
                              rankby_abs=False,
                              pts=True,
                              key_added=method,
                              n_jobs=n_cores)
        
        # Process each cluster
        method_results = []
        clusters = sorted(adata_copy.obs[var_cluster].unique())
        
        for cluster in clusters:
            # Get results for this cluster - convert cluster to string to avoid KeyError
            df = sc.get.rank_genes_groups_df(adata_copy, group=str(cluster), key=method)
            
            # Add cluster info and rank
            df.insert(0, var_cluster, cluster)
            df['rank'] = range(1, len(df) + 1)
            
            # Filter by adjusted p-value if available
            if 'pvals_adj' in df.columns:
                df = df[df['pvals_adj'] < 0.05]
            
            # Prefix column names with method except names and cluster
            df.columns = [f'{method}_{col}' if col not in ['names', var_cluster] 
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
            final_df = final_df.merge(df, on=['names', var_cluster], how='outer')
    
    # Keep only one logfoldchange column from methods that have it
    logfc_methods = [m for m in methods if m != 'logreg']
    if logfc_methods:
        logfc_columns = [f'{method}_logfoldchanges' for method in logfc_methods]
        final_df['logfoldchanges'] = final_df[logfc_columns].mean(axis=1).round(2)
        final_df = final_df.drop(columns=logfc_columns)
    
    # Add rank for logfoldchanges within each cluster
    if 'logfoldchanges' in final_df.columns:
        for cluster in final_df[var_cluster].unique():
            mask = final_df[var_cluster] == cluster
            final_df.loc[mask, 'logfoldchanges_rank'] = final_df.loc[mask, 'logfoldchanges'].rank(method='min', ascending=False)

    # Keep only one *_pct_nz and *_pct_nz_reference column from methods that have it
    pct_nz_methods = [m for m in methods if m != 'logreg']
    if pct_nz_methods:
        pct_nz_group_columns = [f'{method}_pct_nz_group' for method in pct_nz_methods]
        pct_nz_reference_columns = [f'{method}_pct_nz_reference' for method in pct_nz_methods]
        final_df['pct_group'] = (final_df[pct_nz_group_columns].mean(axis=1).round(2))*100
        final_df['pct_rest'] = (final_df[pct_nz_reference_columns].mean(axis=1).round(2))*100
        final_df['pct_diff'] = final_df['pct_group'] - final_df['pct_rest']
        final_df['pct_diff'] = final_df['pct_diff'].round(2)
        final_df = final_df.drop(columns=pct_nz_group_columns + pct_nz_reference_columns)
    
    # Add rank for pct_diff within each cluster
    if 'pct_diff' in final_df.columns:
        for cluster in final_df[var_cluster].unique():
            mask = final_df[var_cluster] == cluster
            final_df.loc[mask, 'pct_diff_rank'] = final_df.loc[mask, 'pct_diff'].rank(method='min', ascending=False)

    # After merging all methods, calculate cluster-wise median rank
    rank_columns = [f'{method}_rank' for method in methods]
    
    # Group by cluster and calculate median rank within each cluster
    for cluster in final_df[var_cluster].unique():
        mask = final_df[var_cluster] == cluster
        final_df.loc[mask, 'median_rank'] = final_df.loc[mask, rank_columns].median(axis=1)
        final_df.loc[mask, 'mean_rank'] = final_df.loc[mask, rank_columns].mean(axis=1)

        # Reset rank to be 1-based within each cluster
        final_df.loc[mask, 'median_rank'] = final_df.loc[mask, 'median_rank'].rank(method='min')
        final_df.loc[mask, 'mean_rank'] = final_df.loc[mask, 'mean_rank'].rank(method='min')

    # Sort by cluster and median rank
    final_df = final_df.sort_values([var_cluster, 'median_rank', "mean_rank"])
    
    return final_df

def perform_subclustering_analysis(adata, project_name, pattern="leiden_", 
                                   abundance_cutoff=0.05, n_markers=50,
                                   min_resolution=None, max_resolution=None,
                                   group_column=None, group_filter=None,
                                   gemini_token=None, results_base_path=None):
    """
    Perform subclustering analysis with optional group filtering and AI-powered marker interpretation.
    
    Parameters:
    -----------
    adata : AnnData
        The input AnnData object
    project_name : str
        The project name for organizing results (e.g., "VITvacc")
    pattern : str, default "leiden_"
        Pattern to match clustering resolution columns
    abundance_cutoff : float, default 0.05
        Minimum fraction of cells required for a cluster to be analyzed
    n_markers : int, default 50
        Number of top markers to extract per cluster
    min_resolution : float, optional
        Minimum resolution to include (e.g., 0.1)
    max_resolution : float, optional
        Maximum resolution to include (e.g., 0.4)
    group_column : str, optional
        Column name in adata.obs to use for filtering (e.g., "group", "condition")
    group_filter : str or list, optional
        Value(s) to filter by in the group_column. If None, no filtering is applied.
        Can be a single value (str) or list of values to include.
    gemini_token : str, optional
        Gemini API token. If None, will try to load from environment
    results_base_path : str, optional
        Base path for results. If None, will use config
    
    Returns:
    --------
    dict : Dictionary containing analysis results and metadata
    
    Examples
    --------
    Basic usage with no filtering (analyze all data):
    
    >>> import scanpy as sc
    >>> import scanpytools as sctl
    >>> adata = sc.datasets.pbmc3k()
    >>> # Perform clustering at multiple resolutions
    >>> adata = sctl.tl.leiden_multi_resolution(adata)
    >>> results = sctl.stat.perform_subclustering_analysis(
    ...     adata, 
    ...     project_name="PBMC_analysis",
    ...     results_base_path="/path/to/results"
    ... )
    
    Filter by a specific group:
    
    >>> # Analyze only vaccinated samples
    >>> results = sctl.stat.perform_subclustering_analysis(
    ...     adata, 
    ...     project_name="VIT_vacc",
    ...     group_column="treatment", 
    ...     group_filter="vaccinated",
    ...     results_base_path="/path/to/results"
    ... )
    
    Filter by multiple groups:
    
    >>> # Analyze both treated and control samples
    >>> results = sctl.stat.perform_subclustering_analysis(
    ...     adata, 
    ...     project_name="VIT_comparison",
    ...     group_column="condition", 
    ...     group_filter=["treated", "control"],
    ...     results_base_path="/path/to/results"
    ... )
    
    Advanced usage with resolution filtering:
    
    >>> # Focus on specific resolution range and cluster pattern
    >>> results = sctl.stat.perform_subclustering_analysis(
    ...     adata, 
    ...     project_name="VIT_detailed",
    ...     pattern="leiden_",
    ...     min_resolution=0.1,
    ...     max_resolution=0.5,
    ...     abundance_cutoff=0.03,  # Include smaller clusters (3%)
    ...     n_markers=100,          # Extract more markers per cluster
    ...     group_column="group", 
    ...     group_filter="experimental",
    ...     results_base_path="/path/to/results"
    ... )
    
    Using custom clustering pattern:
    
    >>> # Analyze Louvain clustering instead of Leiden
    >>> results = sctl.stat.perform_subclustering_analysis(
    ...     adata, 
    ...     project_name="VIT_louvain",
    ...     pattern="louvain_",
    ...     results_base_path="/path/to/results"
    ... )
    
    Notes
    -----
    - The function requires 'highly_variable' genes to be computed in adata.var
    - AI analysis requires a valid Gemini API token (set GEMINI_API_KEY environment variable)
    - Results are organized in subdirectories by resolution (e.g., leiden_0.3/, leiden_0.5/)
    - Each cluster gets: DEG analysis, AI interpretation, gene info CSV, and dotplot visualization
    """
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import scanpytools as sctl
    import os
    import shutil
    from dotenv import load_dotenv
    from google.api_core import retry
    from google import genai
    from datetime import datetime
    
    # Setup results path
    # if results_base_path is None:
    #     results_path = config.get_results_path(project_name, 'subclustering')
    # else:
    #     results_path = results_base_path
    
    # Setup logging
    log_file_path = os.path.join(results_base_path, f"subclustering_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    def log_print(message):
        """Print to console and write to log file"""
        print(message)
        with open(log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    
    # Create results directory if it doesn't exist
    os.makedirs(results_base_path, exist_ok=True)
    
    # Initialize log file
    log_print(f"=== Starting Subclustering Analysis ===")
    log_print(f"Project: {project_name}")
    log_print(f"Group column: {group_column}")
    log_print(f"Group filter: {group_filter}")
    log_print(f"Pattern: {pattern}")
    log_print(f"Abundance cutoff: {abundance_cutoff}")
    log_print(f"Number of markers: {n_markers}")
    log_print(f"Resolution range: [{min_resolution}, {max_resolution}]")
    log_print(f"Results path: {results_base_path}")
    log_print(f"Log file: {log_file_path}")
    
    # Clean existing results (except the log file we just created)
    if os.path.exists(results_base_path):
        for item in os.listdir(results_base_path):
            item_path = os.path.join(results_base_path, item)
            if item_path != log_file_path:  # Don't delete the log file
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
        log_print(f"Cleaned existing results directory (preserved log file)")
    else:
        log_print(f"Created new results directory: {results_base_path}")
    
    # Setup AI configuration
    if gemini_token is None:
        load_dotenv()
        gemini_token = os.getenv("GEMINI_API_KEY")
    
    platform = "google"
    model = "gemini-2.5-flash-preview-05-20"
    max_tokens = 4000
    
    log_print(f"AI Configuration: {platform}, {model}, max_tokens={max_tokens}")
    
    # Setup retry for Gemini API
    is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})
    genai.models.Models.generate_content = retry.Retry(
        predicate=is_retriable)(genai.models.Models.generate_content)
    
    # Filter data by group if specified
    if group_column is not None and group_filter is not None:
        if isinstance(group_filter, str):
            adata_group = adata[adata.obs[group_column] == group_filter].copy()
            filter_description = f"{group_column} == '{group_filter}'"
        elif isinstance(group_filter, list):
            adata_group = adata[adata.obs[group_column].isin(group_filter)].copy()
            filter_description = f"{group_column} in {group_filter}"
        else:
            raise ValueError("group_filter must be a string or list of strings")
        
        log_print(f"1. Filtered data by {filter_description}: \n\n" + str(adata_group) + "\n")
    else:
        adata_group = adata.copy()
        log_print(f"1. Using all data (no group filtering): \n\n" + str(adata_group) + "\n")
    
    log_print("Number of highly variable genes: " + str(np.sum(adata_group.var["highly_variable"] == True)))
    
    # Find pattern clusters
    log_print(f"2. Analyze resolution with pattern: {pattern}\n")
    pattern_clusters = [col for col in adata_group.obs.columns if pattern in col]
    
    # Filter by resolution range if specified
    if min_resolution is not None or max_resolution is not None:
        filtered_clusters = []
        for col in pattern_clusters:
            try:
                # Extract resolution value from column name (e.g., "leiden_0.1" -> 0.1)
                resolution_str = col.replace(pattern, "")
                resolution_val = float(resolution_str)
                
                # Check if within range
                if min_resolution is not None and resolution_val < min_resolution:
                    continue
                if max_resolution is not None and resolution_val > max_resolution:
                    continue
                    
                filtered_clusters.append(col)
            except ValueError:
                # Skip columns that don't have numeric resolution values
                log_print(f"Warning: Could not parse resolution from {col}, skipping...")
                continue
        
        pattern_clusters = filtered_clusters
        log_print(f"Filtered to resolution range [{min_resolution}, {max_resolution}]: {pattern_clusters}")
    
    log_print("Columns matching pattern: " + str(pattern_clusters) + "\n")
    
    results_summary = {
        'project': project_name,
        'group_column': group_column,
        'group_filter': group_filter,
        'total_cells': len(adata_group),
        'resolutions_analyzed': [],
        'clusters_analyzed': {},
        'files_generated': [log_file_path]  # Include log file in generated files
    }
    
    # Loop over all filtered pattern_clusters
    for one_resolution in pattern_clusters:
        log_print(f"\n=== Processing resolution: {one_resolution} ===")
        
        # Find abundant clusters
        composition = round(adata_group.obs[one_resolution].value_counts(normalize=True), 2)
        log_print(f"Cluster abundance for {one_resolution}: (composition: {composition})")

        one_resolution_clusters = composition[composition > abundance_cutoff].index.tolist()
        log_print(f"Abundant clusters found: {one_resolution_clusters}")
        
        if len(one_resolution_clusters) <= 1:
            log_print(f"Not enough abundant clusters found for {one_resolution} (need >1), skipping...")
            continue
        
        results_summary['resolutions_analyzed'].append(one_resolution)
        results_summary['clusters_analyzed'][one_resolution] = one_resolution_clusters
        
        # Subset data to abundant clusters
        adata_group_oneres_abundant = adata_group[adata_group.obs[one_resolution].isin(one_resolution_clusters)].copy()
        
        # Perform highly variable gene selection
        sc.pp.highly_variable_genes(adata_group_oneres_abundant)
        log_print("Number of highly variable genes: " + str(np.sum(adata_group_oneres_abundant.var["highly_variable"] == True)) + "\n")
        
        log_print(f"3. Analyze resolution {one_resolution}: " + ", ".join(map(str, one_resolution_clusters)) + "\n")
        
        # Identify cluster markers
        DEG_df = sctl.stat.ensemble_identify_cluster_markers(
            adata_group_oneres_abundant, 
            var_cluster=one_resolution,
            methods=["wilcoxon", "t-test_overestim_var", "t-test"]
        )
        
        # Save DEG results
        resolution_results_path = os.path.join(results_base_path, one_resolution)
        os.makedirs(resolution_results_path, exist_ok=True)        
        deg_file = os.path.join(resolution_results_path, "DEG.csv")
        DEG_df.to_csv(deg_file, index=False)
        results_summary['files_generated'].append(deg_file)
        
        # Analyze each cluster
        for one_resolution_one_cluster in one_resolution_clusters:
            log_print(f"Processing cluster {one_resolution_one_cluster} in resolution {one_resolution}")
            
            # Get top markers for this cluster
            markers_one_resolution_one_cluster = \
                DEG_df[DEG_df[one_resolution] == one_resolution_one_cluster].\
                sort_values("median_rank").\
                head(n_markers)["names"].tolist()
            
            if not markers_one_resolution_one_cluster:
                log_print(f"No markers found for cluster {one_resolution_one_cluster}, skipping...")
                continue
            
            # AI analysis
            try:
                ai_result = sctl.ai.prioritize_genes(
                    markers_one_resolution_one_cluster, 
                    context="Vaccine induced T cells", 
                    api_token=gemini_token, 
                    n=20,
                    platform=platform,
                    llm_model=model, 
                    max_tokens=max_tokens
                )
                
                # Unpack AI result
                ai_result_full, ai_result_top_g, ai_result_summary_ai, ai_result_celltype = ai_result
                
                # Save AI results
                ai_file = os.path.join(resolution_results_path, f"cluster_{one_resolution_one_cluster}_ai_inference.txt")
                with open(ai_file, "w") as file:
                    file.write(ai_result_full)
                results_summary['files_generated'].append(ai_file)
                
                # Save gene info
                ai_result_top_g_df = pd.DataFrame(
                    list(ai_result_top_g.items()), 
                    columns=['Gene', 'Description']
                )
                gene_info_file = os.path.join(resolution_results_path, f"cluster_{one_resolution_one_cluster}_g_info.csv")
                ai_result_top_g_df.to_csv(gene_info_file, index=False)
                results_summary['files_generated'].append(gene_info_file)
                
                log_print(f"AI analysis completed for cluster {one_resolution_one_cluster}")
                
            except Exception as e:
                log_print(f"AI analysis failed for cluster {one_resolution_one_cluster}: {str(e)}")
                continue
            
            # Generate dotplot
            try:
                markers_ordered = [gene for gene in adata_group_oneres_abundant.var.sort_values(by='means', ascending=False).index if gene in markers_one_resolution_one_cluster]
                
                sc.settings.figdir = resolution_results_path
                sc.set_figure_params(dpi=300)
                
                sc.pl.dotplot(
                    adata_group_oneres_abundant,
                    markers_ordered,
                    groupby=one_resolution,
                    swap_axes=True,
                    title=f"Top {n_markers} markers for cluster {one_resolution_one_cluster} \n (resolution {one_resolution})",
                    save=f"cluster_{one_resolution_one_cluster}.png",
                    show=False
                )
                
                # Rename dotplot file
                old_name = os.path.join(resolution_results_path, f"dotplot_cluster_{one_resolution_one_cluster}.png")
                new_name = os.path.join(resolution_results_path, f"cluster_{one_resolution_one_cluster}_dotplot.png")
                if os.path.exists(old_name):
                    os.rename(old_name, new_name)
                    results_summary['files_generated'].append(new_name)
                
                log_print(f"Dotplot generated for cluster {one_resolution_one_cluster}")
                
            except Exception as e:
                log_print(f"Dotplot generation failed for cluster {one_resolution_one_cluster}: {str(e)}")
                continue
    
    log_print(f"\n=== Analysis Complete ===")
    log_print(f"Project: {project_name}")
    log_print(f"Resolutions analyzed: {len(results_summary['resolutions_analyzed'])}")
    log_print(f"Total files generated: {len(results_summary['files_generated'])}")
    
    return results_summary

def perform_celltype_specific_deg_across_resolutions(adata, project_name, 
                                                    analysis_configs=None,
                                                    resolutions=[0.1, 0.2, 0.3],
                                                    pattern="leiden_",
                                                    n_top_deg=50,
                                                    min_cells_per_group=10,
                                                    methods=["wilcoxon", "t-test_overestim_var", "t-test"],
                                                    results_base_path=None):
    """
    Perform cell type-specific differential expression analysis across multiple clustering resolutions and group configurations.
    
    This function systematically analyzes differential gene expression across different clustering
    resolutions and various group/filter combinations. It's designed for comprehensive comparative
    analysis of cell type-specific responses across experimental conditions.
    
    Parameters
    ----------
    adata : AnnData
        The input AnnData object containing processed single-cell data
    project_name : str
        The project name for organizing results (e.g., "VITINFECT")
    analysis_configs : list of dict, optional
        List of analysis configurations. Each dict should contain:
        - 'group_var': Main grouping variable for DEG analysis
        - 'filter_var': Variable to filter on (optional, can be None)
        - 'filter_value': Value to filter for (required if filter_var is specified)
        If None, will use default configurations.
    resolutions : list of float, default [0.1, 0.2, 0.3]
        List of clustering resolutions to analyze
    pattern : str, default "leiden_"
        Pattern prefix for clustering resolution columns
    n_top_deg : int, default 50
        Number of top differential genes to retain per group
    min_cells_per_group : int, default 10
        Minimum number of cells required per group for analysis
    methods : list, default ["wilcoxon", "t-test_overestim_var", "t-test"]
        Statistical methods for differential expression analysis
    results_base_path : str, optional
        Base path for results output
    
    Returns
    -------
    dict
        Dictionary containing analysis summary with:
        - 'project': Project name
        - 'total_configs': Number of analysis configurations
        - 'total_resolutions': Number of resolutions analyzed
        - 'files_generated': List of output files
        - 'analysis_summary': Detailed breakdown by configuration
    
    Examples
    --------
    Basic usage with default configurations:
    
    >>> import scanpy as sc
    >>> import scanpytools as sctl
    >>> adata = sc.datasets.pbmc3k()
    >>> # Assume clustering and metadata are already added
    >>> results = sctl.stat.perform_celltype_specific_deg_across_resolutions(
    ...     adata,
    ...     project_name="PBMC_DEG",
    ...     results_base_path="/path/to/results"
    ... )
    
    Custom analysis configurations:
    
    >>> analysis_configs = [
    ...     # No filtering - analyze all samples
    ...     {"group_var": "treatment", "filter_var": None, "filter_value": None},
    ...     {"group_var": "condition", "filter_var": None, "filter_value": None},
    ...     # With filtering - analyze treatment within specific subgroups
    ...     {"group_var": "treatment", "filter_var": "cell_type", "filter_value": "T_cell"},
    ...     {"group_var": "condition", "filter_var": "timepoint", "filter_value": "day7"},
    ... ]
    >>> results = sctl.stat.perform_celltype_specific_deg_across_resolutions(
    ...     adata,
    ...     project_name="Custom_DEG",
    ...     analysis_configs=analysis_configs,
    ...     resolutions=[0.2, 0.4, 0.6],
    ...     results_base_path="/path/to/results"
    ... )
    
    Focus on specific resolution range:
    
    >>> results = sctl.stat.perform_celltype_specific_deg_across_resolutions(
    ...     adata,
    ...     project_name="High_res_DEG",
    ...     resolutions=[0.5, 0.7, 0.9, 1.1],
    ...     n_top_deg=100,  # More genes per group
    ...     min_cells_per_group=5,  # Allow smaller groups
    ...     results_base_path="/path/to/results"
    ... )
    
    Notes
    -----
    - Requires clustering columns matching the pattern (e.g., 'leiden_0.1', 'leiden_0.2')
    - Each cluster within each resolution is analyzed separately for cell type-specific responses
    - Results are organized hierarchically: analysis_config/resolution/cluster
    - Skips configurations with insufficient group representation
    - Automatically performs highly variable gene selection per cluster
    """
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import scanpytools as sctl
    import os
    import shutil
    from datetime import datetime
    
    # Default analysis configurations if not provided
    if analysis_configs is None:
        analysis_configs = [
            {"group_var": "treatment", "filter_var": None, "filter_value": None},
            {"group_var": "condition", "filter_var": None, "filter_value": None},
        ]
    
    # Setup logging
    log_file_path = os.path.join(results_base_path, f"celltype_deg_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    def log_print(message):
        """Print to console and write to log file"""
        print(message)
        with open(log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    
    # Create results directory if it doesn't exist
    os.makedirs(results_base_path, exist_ok=True)
    
    # Initialize log file
    log_print(f"=== Starting Cell Type-Specific DEG Analysis Across Resolutions ===")
    log_print(f"Project: {project_name}")
    log_print(f"Number of configurations: {len(analysis_configs)}")
    log_print(f"Resolutions: {resolutions}")
    log_print(f"Pattern: {pattern}")
    log_print(f"Top DEGs per group: {n_top_deg}")
    log_print(f"Min cells per group: {min_cells_per_group}")
    log_print(f"Methods: {methods}")
    log_print(f"Results path: {results_base_path}")
    log_print(f"Log file: {log_file_path}")
    
    # Clean existing results (except the log file we just created)
    if os.path.exists(results_base_path):
        for item in os.listdir(results_base_path):
            item_path = os.path.join(results_base_path, item)
            if item_path != log_file_path:  # Don't delete the log file
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
        log_print(f"Cleaned existing results directory (preserved log file)")
    else:
        log_print(f"Created new results directory: {results_base_path}")
    
    # Initialize results summary
    results_summary = {
        'project': project_name,
        'total_configs': len(analysis_configs),
        'total_resolutions': len(resolutions),
        'files_generated': [log_file_path],
        'analysis_summary': {}
    }
    
    # Loop over analysis configurations
    for config_idx, analysis_config in enumerate(analysis_configs):
        group_var = analysis_config["group_var"]
        filter_var = analysis_config.get("filter_var")
        filter_value = analysis_config.get("filter_value")
        
        # Create descriptive name for this analysis
        if filter_var is None:
            analysis_name = f"group_{group_var}"
            filter_description = "all samples"
        else:
            analysis_name = f"group_{group_var}_{filter_var}{filter_value}"
            filter_description = f"{filter_var}={filter_value} samples only"
        
        log_print(f"\n=== Configuration {config_idx + 1}/{len(analysis_configs)}: Analyzing {group_var} in {filter_description} ===")
        
        # Apply filtering if specified
        if filter_var is not None:
            adata_filtered = adata[adata.obs[filter_var] == filter_value, :].copy()
            log_print(f"Filtered from {adata.n_obs} to {adata_filtered.n_obs} cells based on {filter_var}={filter_value}")
            
            # Check if we have enough samples for the main grouping variable after filtering
            group_counts_after_filter = adata_filtered.obs[group_var].value_counts()
            log_print(f"Group counts after filtering: {group_counts_after_filter.to_dict()}")
            
            if len(group_counts_after_filter) < 2:
                log_print(f"Skipping this configuration - only {len(group_counts_after_filter)} group(s) available after filtering")
                continue
            
            if any(group_counts_after_filter < 50):
                log_print(f"Warning: Some groups have fewer than 50 cells after filtering: {group_counts_after_filter.to_dict()}")
        else:
            adata_filtered = adata.copy()
            log_print(f"Using all {adata_filtered.n_obs} cells (no filtering applied)")
        
        # Create directory for this analysis configuration
        group_results_dir = os.path.join(results_base_path, analysis_name)
        os.makedirs(group_results_dir, exist_ok=True)
        
        # Initialize config summary
        config_summary = {
            'group_var': group_var,
            'filter_var': filter_var,
            'filter_value': filter_value,
            'total_cells': len(adata_filtered),
            'resolutions_processed': [],
            'files_generated': []
        }
        
        # Loop over resolutions
        for resolution in resolutions:
            log_print(f"  Processing resolution {resolution}")
            
            # Create directory for results at this resolution
            deg_results_resolution_dir = os.path.join(group_results_dir, f"res_{resolution}")
            os.makedirs(deg_results_resolution_dir, exist_ok=True)
            
            resolution_column = f"{pattern}{resolution}"
            
            if resolution_column not in adata_filtered.obs.columns:
                log_print(f"    Warning: Resolution column '{resolution_column}' not found, skipping...")
                continue
                
            log_print(f"    Analyzing resolution: {resolution_column}, group: {group_var}, filter: {filter_description}")
            
            # Print cluster composition
            log_print("    Composition (%)")
            composition_pct = sctl.stat.cluster_distribution(adata_filtered, group=group_var, cluster=resolution_column)
            log_print(str(composition_pct))
            
            log_print("    Cell number (N)")
            composition_n = sctl.stat.cluster_distribution(adata_filtered, group=group_var, cluster=resolution_column, normalize=False)
            log_print(str(composition_n))
            
            # Get clusters for this resolution
            resolution_clusters = sorted(adata_filtered.obs[resolution_column].unique().tolist())
            log_print(f"    Clusters at resolution {resolution_column}: {resolution_clusters}")
            
            config_summary['resolutions_processed'].append(resolution)
            
            # Loop over clusters within this resolution
            for cluster in resolution_clusters:
                log_print(f"    Processing cluster {cluster}")
                adata_cluster = adata_filtered[adata_filtered.obs[resolution_column] == cluster, :].copy()
                
                # Check if there are enough cells in different groups for comparison
                group_counts = adata_cluster.obs[group_var].value_counts()
                log_print(f"      Group counts in cluster {cluster}: {group_counts.to_dict()}")
                
                # Skip if any group has fewer than minimum required cells
                if any(group_counts < min_cells_per_group):
                    log_print(f"      Skipping cluster {cluster} - insufficient cells in some groups (min {min_cells_per_group} required)")
                    continue
                
                # Perform highly variable gene selection
                sc.pp.highly_variable_genes(adata_cluster)
                
                # Subset the data to the highly variable genes
                adata_cluster = adata_cluster[:, adata_cluster.var['highly_variable']].copy()
                
                # Identify differential genes
                DEG_df = sctl.stat.ensemble_identify_cluster_markers(
                    adata_cluster,
                    methods=methods,
                    var_cluster=group_var
                )
                
                # Filter top genes
                DEG_df = DEG_df[DEG_df["median_rank"] <= n_top_deg]
                
                # Add metadata columns
                DEG_df['group_variable'] = group_var
                DEG_df['filter_variable'] = filter_var if filter_var is not None else 'none'
                DEG_df['filter_value'] = filter_value if filter_value is not None else 'none'
                DEG_df['resolution'] = resolution
                DEG_df['cluster'] = cluster
                DEG_df['resolution_column'] = resolution_column
                DEG_df['analysis_config'] = analysis_name
                
                # Save results
                out_file = os.path.join(deg_results_resolution_dir, f"cluster_{cluster}.csv")
                DEG_df.to_csv(out_file, index=False)
                log_print(f"      Saved DEG results to {out_file}")
                
                config_summary['files_generated'].append(out_file)
                results_summary['files_generated'].append(out_file)
                
                # Generate condition-specific dotplots for DEG markers
                try:
                    # Get the unique conditions/groups being compared
                    conditions = sorted(adata_cluster.obs[group_var].unique())
                    log_print(f"      Conditions being compared in cluster {cluster}: {conditions}")
                    
                    # Generate dotplot for each condition showing their specific markers
                    for condition in conditions:
                        # Get top markers specific to this condition
                        condition_markers = DEG_df[DEG_df[group_var] == condition]["names"].tolist()
                        
                        if condition_markers:
                            # Order markers by mean expression within this condition
                            condition_cells = adata_cluster[adata_cluster.obs[group_var] == condition, :].copy()
                            if condition_cells.n_obs > 0:
                                # Calculate mean expression for ordering
                                # mean_expr = pd.Series(
                                #     condition_cells.X.mean(axis=0).A1 if hasattr(condition_cells.X, 'A1') else condition_cells.X.mean(axis=0),
                                #     index=condition_cells.var.index
                                # )
                                sc.pp.calculate_qc_metrics(condition_cells, inplace=True)
                                markers_ordered = [gene for gene in condition_cells.var.sort_values("means", ascending=False).index if gene in condition_markers]
                                
                                if markers_ordered:
                                    sc.settings.figdir = deg_results_resolution_dir
                                    sc.set_figure_params(dpi=300)
                                    
                                    sc.pl.dotplot(
                                        adata_cluster,
                                        markers_ordered,
                                        groupby=group_var,
                                        swap_axes=True,
                                        title=f"Top markers for {condition} in cluster {cluster} \n (resolution {resolution_column}, {filter_description})",
                                        save=f"cluster_{cluster}_{condition}_markers_dotplot.png",
                                        show=False
                                    )
                                    
                                    # Rename dotplot file to ensure consistent naming
                                    old_name = os.path.join(deg_results_resolution_dir, f"dotplot_cluster_{cluster}_{condition}_markers_dotplot.png")
                                    new_name = os.path.join(deg_results_resolution_dir, f"cluster_{cluster}_{condition}_markers_dotplot.png")
                                    if os.path.exists(old_name):
                                        os.rename(old_name, new_name)
                                        config_summary['files_generated'].append(new_name)
                                        results_summary['files_generated'].append(new_name)
                                        log_print(f"      Generated dotplot for {condition}-specific markers in cluster {cluster}")
                                    else:
                                        log_print(f"      Warning: Expected dotplot file not found: {old_name}")
                                else:
                                    log_print(f"      No valid markers found for {condition} dotplot in cluster {cluster}")
                            else:
                                log_print(f"      No cells found for condition {condition} in cluster {cluster}")
                        else:
                            log_print(f"      No {condition}-specific markers available for dotplot in cluster {cluster}")
                                            
                except Exception as e:
                    log_print(f"      Dotplot generation failed for cluster {cluster}: {str(e)}")
        
        results_summary['analysis_summary'][analysis_name] = config_summary
        log_print(f"=== Completed configuration {config_idx + 1}: {analysis_name} ===")
    
    log_print(f"\n=== Cell Type-Specific DEG Analysis Complete ===")
    log_print(f"Project: {project_name}")
    log_print(f"Configurations processed: {len(results_summary['analysis_summary'])}")
    log_print(f"Total files generated: {len(results_summary['files_generated'])}")
    
    return results_summary