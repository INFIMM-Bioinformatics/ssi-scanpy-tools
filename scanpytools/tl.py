import anndata as ad
import pandas as pd
import numpy as np
import scanpy as sc

def reorder_clusters(adata, old_cluster, new_cluster="new_cluster", inplace=False):
    """
    Reorder clusters in an AnnData object by descending cell count.

    Parameters:
    ----------
    adata : anndata.AnnData
        The annotated data matrix.
    old_cluster : str
        The name of the column in adata.obs containing the original cluster labels.
    new_cluster : str, optional (default="new_cluster")
        The name of the new column to store the reordered cluster labels.
    inplace : bool, optional (default=False)
        If True, modify the adata object in place.

    Returns:
    -------
    anndata.AnnData or None
        A new AnnData object with reordered clusters if inplace=False, otherwise None.

    Raises:
    ------
    ValueError
        If old_cluster is not in adata.obs or cluster values cannot be coerced to numeric.
    """
    if not isinstance(adata, ad.AnnData):
        raise TypeError("adata must be an AnnData object")
    
    if old_cluster not in adata.obs:
        raise ValueError(f"Column '{old_cluster}' not found in adata.obs")
    
    # Create a copy of the adata object if not inplace
    adata_reorder = adata if inplace else adata.copy()
    
    # Get cluster counts and create mapping
    cluster_counts = adata.obs[old_cluster].value_counts()
    mapping = {str(old): str(new) for new, old in enumerate(cluster_counts.index)}
    
    # Apply mapping and set categories
    new_clusters = adata.obs[old_cluster].astype(str).map(mapping)
    categories = [str(i) for i in range(len(mapping))]
    adata_reorder.obs[new_cluster] = pd.Categorical(new_clusters, categories=categories)
    
    return None if inplace else adata_reorder

def leiden_multi_resolution(adata, min_resolution=0, max_resolution=1, step=0.1):
    """
    Performs Leiden clustering at multiple resolutions.

    This function applies the Leiden community detection algorithm across a range of
    resolution parameters, storing the results in new columns of the AnnData object.
    Each clustering result is named 'leiden_X' where X is the resolution value.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix with connectivity information stored in
        `adata.obsp['connectivities']`.
    min_resolution : float, optional (default: 0)
        Minimum resolution parameter value to use for clustering.
    max_resolution : float, optional (default: 1)
        Maximum resolution parameter value to use for clustering.
    step : float, optional (default: 0.1)
        Step size for the resolution parameter range.

    Returns
    -------
    anndata.AnnData
        Updated AnnData object with new leiden clustering results stored in
        `.obs['leiden_X']` for each resolution X.

    Notes
    -----
    - Uses igraph implementation of the Leiden algorithm
    - Performs 2 iterations for each resolution
    - Clusters are reordered after each clustering
    """
    for resolution in np.arange(min_resolution, max_resolution + .01, step):
        resolution = round(resolution, 1)
        sc.tl.leiden(adata, resolution=resolution, flavor='igraph', n_iterations=2, key_added=f'leiden_{resolution}')
        adata = reorder_clusters(adata, f'leiden_{resolution}', f'leiden_{resolution}')
    return adata
