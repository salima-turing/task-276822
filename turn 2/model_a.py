import unittest
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.datasets import make_blobs


class TestProductSimilarityScaling(unittest.TestCase):

    def test_cosine_similarity_scaling(self):
        # Generate a large sparse dataset
        n_samples = 10000
        n_features = 1000
        data, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=10, random_state=0, cluster_std=0.1)
        data = data.astype(np.float32)  # Use float32 to save memory

        # Compute cosine similarity using dense method
        dense_sim = cosine_similarity(data)

        # Take a sample for testing
        sample_size = 100
        idx = np.random.choice(n_samples, size=sample_size, replace=False)
        sample_data = data[idx]
        sample_dense_sim = dense_sim[idx][:, idx]

        # Compute cosine similarity using sparse method (approximate for large datasets)
        from sklearn.metrics.pairwise import cosine_distances
        sparse_sim = 1 - cosine_distances(sample_data, sample_data)

        # Calculate maximum absolute error
        max_error = np.abs(sample_dense_sim - sparse_sim).max()
        self.assertLess(max_error, 0.01, "Maximum absolute error for cosine similarity is too high")


if __name__ == '__main__':
    unittest.main()
