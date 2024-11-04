import unittest
from collections import defaultdict
from math import sqrt


def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)


def cosine_similarity(vec1, vec2):
    dot_product = sum(x * y for x, y in zip(vec1, vec2))
    norm1 = sqrt(sum(x ** 2 for x in vec1))
    norm2 = sqrt(sum(x ** 2 for x in vec2))
    return dot_product / (norm1 * norm2)


class TestProductSimilarity(unittest.TestCase):

    def test_jaccard_similarity(self):
        test_cases = (
            ({"apple", "banana", "orange"}, {"apple", "banana"}, 0.6),
            ({"apple", "banana"}, {"orange", "mango"}, 0.0),
            (set(), {"apple", "banana"}, 0.0),
            ({"apple", "banana"}, set(), 0.0),
            ({"apple", "banana"}, {"apple", "banana"}, 1.0),
        )
        for set1, set2, expected_score in test_cases:
            with self.subTest(set1=set1, set2=set2):
                self.assertAlmostEqual(jaccard_similarity(set1, set2), expected_score)

    def test_cosine_similarity(self):
        test_cases = (
            ([1, 2, 3], [4, 5, 6], 0.26726),
            ([1, 0], [0, 1], 0.0),
            ([1, 2], [3, 4], -0.28284),
            ([3, 4], [1, 2], -0.28284),
        )
        for vec1, vec2, expected_score in test_cases:
            with self.subTest(vec1=vec1, vec2=vec2):
                self.assertAlmostEqual(cosine_similarity(vec1, vec2), expected_score)


if __name__ == '__main__':
    unittest.main()
