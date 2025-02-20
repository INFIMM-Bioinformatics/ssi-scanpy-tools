import plotly.graph_objects as go
import pandas as pd

def cluster_sankey_diagram(adata, prefix="leiden"):
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

    fig.show()