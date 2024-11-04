import unittest
from scipy.spatial.distance import cosine, cityblock, jaccard
import numpy as np


class TestProductSimilarity(unittest.TestCase):
    def setUp(self):
        # Sample product data for testing
        self.product_1 = np.array([1, 0, 1, 0, 1])
        self.product_2 = np.array([0, 1, 1, 1, 0])
        self.product_3 = np.array([1, 1, 0, 0, 0])
        # Expected similarity values based on some established standards
        self.expected_cosine_similarity = 0.6
        self.expected_jaccard_similarity = 0.4
        # Your implementation of similarity metrics functions
        self.cosine_similarity_func = lambda x, y: 1 - cosine(x, y)
        self.jaccard_similarity_func = lambda x, y: 1 - jaccard(x, y)

    def test_cosine_similarity(self):
        """Test cosine similarity calculation against expected value."""
        similarity = self.cosine_similarity_func(self.product_1, self.product_2)
        self.assertAlmostEqual(similarity, self.expected_cosine_similarity, places=2)

    def test_jaccard_similarity(self):
        """Test jaccard similarity calculation against expected value."""
        similarity = self.jaccard_similarity_func(self.product_1, self.product_2)
        self.assertAlmostEqual(similarity, self.expected_jaccard_similarity, places=2)


if __name__ == '__main__':
    unittest.main()
