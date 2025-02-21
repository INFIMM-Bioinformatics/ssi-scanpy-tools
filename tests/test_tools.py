import unittest
import anndata as ad
import pandas as pd
from scanpytools.tl import tools

class TestTools(unittest.TestCase):
    def setUp(self):
        # Create a sample AnnData.obs object for testing
        data = pd.DataFrame({
            'gene1': [1, 2, 3, 4],
            'gene2': [4, 3, 2, 1]
        })
        obs = pd.DataFrame({
            'clusters': ['A', 'B', 'A', 'B']
        })
        self.adata = ad.AnnData(X=data, obs=obs)

    def test_reorder_clusters_inplace_false(self):
        result = tools.reorder_clusters(self.adata, 'clusters', inplace=False)
        self.assertIsInstance(result, ad.AnnData)
        self.assertIn('new_cluster', result.obs.columns)
        self.assertEqual(result.obs['new_cluster'].tolist(), ['0', '1', '0', '1'])

    def test_reorder_clusters_inplace_true(self):
        tools.reorder_clusters(self.adata, 'clusters', inplace=True)
        self.assertIn('new_cluster', self.adata.obs.columns)
        self.assertEqual(self.adata.obs['new_cluster'].tolist(), ['0', '1', '0', '1'])

    def test_invalid_adata_type(self):
        with self.assertRaises(TypeError):
            tools.reorder_clusters("invalid_adata", 'clusters')

    def test_missing_old_cluster_column(self):
        with self.assertRaises(ValueError):
            tools.reorder_clusters(self.adata, 'missing_column')

    def test_empty_adata(self):
        empty_adata = ad.AnnData()
        with self.assertRaises(ValueError):
            tools.reorder_clusters(empty_adata, 'clusters')

if __name__ == '__main__':
    unittest.main()