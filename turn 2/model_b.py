import unittest
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class TestProductSimilarity(unittest.TestCase):

    def test_cosine_similarity_with_large_sample(self):
        # Generate a large sample of random vectors
        np.random.seed(0)
        num_products = 10000
        feature_dim = 100
        vec1 = np.random.rand(num_products, feature_dim)
        vec2 = np.random.rand(num_products, feature_dim)

        # Calculate cosine similarity using scikit-learn
        expected_scores = cosine_similarity(vec1, vec2)

        # Your custom implementation
        your_scores = np.array([cosine_similarity(vec1[i], vec2[i]) for i in range(num_products)])

        # Assert that the calculated scores are close to the expected ones
        self.assertTrue(np.allclose(your_scores, expected_scores, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
