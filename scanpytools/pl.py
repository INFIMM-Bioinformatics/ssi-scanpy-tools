import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt

def cluster_sankey_diagram(adata, prefix="leiden", return_fig=False):
    """
    Creates a Sankey diagram visualizing the transitions of cluster identities across different resolutions.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing clustering results at different resolutions.
        Must have cluster annotations in .obs with names formatted as '{prefix}_{resolution}'.
    prefix : str, optional (default: 'leiden')
        The prefix used in the column names for clustering results.
    return_fig : bool, optional (default: False)
        If True, returns the figure object instead of displaying it.

    Returns
    -------
    plotly.graph_objects.Figure or None
        Returns the figure object if return_fig=True, otherwise displays the plot and returns None.
    """
    # Extract the cluster identities for each resolution
    resolutions = [round(x * 0.1, 1) for x in range(11)]
    cluster_data = {f'{prefix}_{res}': adata.obs[f'{prefix}_{res}'] for res in resolutions}

    # Create a DataFrame to store the transitions
    transitions = []
    for i in range(len(resolutions) - 1):
        res1 = f'{prefix}_{resolutions[i]}'
        res2 = f'{prefix}_{resolutions[i + 1]}'
        for cluster1 in cluster_data[res1].unique():
            for cluster2 in cluster_data[res2].unique():
                count = ((cluster_data[res1] == cluster1) & (cluster_data[res2] == cluster2)).sum()
                if count > 0:
                    transitions.append([res1, cluster1, res2, cluster2, count])

    # Convert transitions to a DataFrame
    transitions_df = pd.DataFrame(transitions, columns=['source_res', 'source_cluster', 'target_res', 'target_cluster', 'count'])

    # Create unique labels for nodes
    unique_labels = sorted(set([f'{res}_{cluster}' for res, cluster, _, _, _ in transitions] + 
                               [f'{res}_{cluster}' for _, _, res, cluster, _ in transitions]))

    unique_labels_show = [label.split('_')[2] for label in unique_labels]

    # Create a mapping from label to index
    label_to_index = {label: i for i, label in enumerate(unique_labels)}

    # Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=unique_labels_show,
            color="blue"
        ),
        link=dict(
            source=[label_to_index[f'{src}_{src_cluster}'] for src, src_cluster, _, _, _ in transitions],
            target=[label_to_index[f'{tgt}_{tgt_cluster}'] for _, _, tgt, tgt_cluster, _ in transitions],
            value=[count for _, _, _, _, count in transitions]
        )
    )])

    # Add annotations for resolutions
    annotations = []
    for i, res in enumerate(resolutions):
        annotations.append(dict(
            x=i / (len(resolutions) - 1),
            y=1.1,
            text=f'{res}',
            showarrow=False,
            xref='paper',
            yref='paper',
            font=dict(size=12)
        ))

    fig.update_layout(
        title_text="Sankey Diagram of Cluster Identity Transitions in Different Resolutions",
        title_x=0.5,  # Center the title
        font_size=10,
        annotations=annotations,    
    )

    if return_fig:
        return fig
    else:
        fig.show()


def plot_multi_resolution_umap(adata, group, resolution_prefix, ncols=2, n_resolutions = None):
    """
    Plot UMAPS showing clustering results for multiple resolutions.
    This function creates a multi-panel UMAP visualization showing a given grouping variable
    and multiple Leiden clustering results at different resolutions.
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix that contains UMAP coordinates and clustering results.
    group : str
        Column name in adata.obs for the reference grouping variable to show in the first panel.
    resolution_prefix : str
        Prefix used to identify columns in adata.obs that contain clustering results at different resolutions.
        The function will select all columns that start with this prefix.
    ncols : int, optional (default: 2)
        Number of panels per row in the output figure.
    n_resolutions : int, optional (default: None)
        Number of resolutions to plot. If None, all found resolutions will be plotted.
    Returns
    -------
    matplotlib.figure.Figure
        A figure containing the UMAP plots.
    Examples
    --------
    >>> plot_multi_resolution_umap(adata, 'cell_type', 'leiden_res_', ncols=3, n_resolutions=5)
    """
    resolutions = [col for col in adata.obs.columns if col.startswith(resolution_prefix)]

    if n_resolutions is not None:
        resolutions = resolutions[:n_resolutions]
    

    titles = [f'UMAP ({group})'] + [f'UMAP (res={res.split("_")[1]})' for res in resolutions]

    with plt.rc_context({'figure.figsize': (3, 3)}):
        umap = sc.pl.umap(adata, color=[group] + resolutions, legend_loc='on data', ncols=ncols, title=titles, return_fig=True)
    return umap