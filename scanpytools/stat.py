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