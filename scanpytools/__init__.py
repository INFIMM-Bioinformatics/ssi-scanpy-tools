"""
scanpytools

A collection of utility functions extending Scanpy functionality for single-cell RNA sequencing analysis.

Main Components:
---------------
- tl: Analysis tools including cluster reordering and multi-resolution Leiden clustering
- pl: Visualization tools including Sankey diagrams for cluster comparison
- stat: Statistical analysis tools including cluster distribution analysis

Functions:
---------
- reorder_clusters: Reorder cluster labels based on specific criteria
- leiden_multi_resolution: Perform Leiden clustering at multiple resolutions
- cluster_sankey_diagram: Create Sankey diagram to visualize cluster relationships
- cluster_distribution: Analyze and compute cluster distribution statistics
"""

from .tl import reorder_clusters, leiden_multi_resolution
from .pl import cluster_sankey_diagram
from .stat import cluster_distribution, ensemble_identify_cluster_markers

__version__ = '0.1.0'

