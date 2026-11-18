import unittest
from pathlib import Path

from med_retrieval import RetrievalIndex, cosine_similarity, extract_features, load_pgm
from med_retrieval.features import gradient_features, histogram_features, statistical_features

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _uniform(value, size=8):
    return [[float(value)] * size for _ in range(size)]


def _bright_center(size=8):
    image = [[40.0] * size for _ in range(size)]
    for r in range(2, 6):
        for c in range(2, 6):
            image[r][c] = 220.0
    return image


class TestFeatures(unittest.TestCase):
    def test_feature_vector_length(self):
        features = extract_features(_bright_center())
        self.assertEqual(len(features), 16)

    def test_histogram_sums_to_one(self):
        pixels = [v for row in _bright_center() for v in row]
        self.assertAlmostEqual(sum(histogram_features(pixels)), 1.0, places=6)

    def test_uniform_image_statistics(self):
        pixels = [v for row in _uniform(100) for v in row]
        mean, std, *_ = statistical_features(pixels)
        self.assertAlmostEqual(mean, 100 / 255.0, places=6)
        self.assertAlmostEqual(std, 0.0, places=6)

    def test_uniform_image_no_gradient(self):
        self.assertEqual(gradient_features(_uniform(100)), [0.0, 0.0])

    def test_empty_image_rejected(self):
        with self.assertRaises(ValueError):
            extract_features([])


class TestSimilarity(unittest.TestCase):
    def test_identical_vectors_score_one(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 2.0], [1.0, 2.0]), 1.0, places=6)

    def test_mismatched_lengths_rejected(self):
        with self.assertRaises(ValueError):
            cosine_similarity([1.0], [1.0, 2.0])


class TestIndex(unittest.TestCase):
    def test_search_returns_query_itself_first(self):
        index = RetrievalIndex()
        bright = extract_features(_bright_center())
        dark = extract_features(_uniform(20))
        index.add("bright", "bright-center", bright)
        index.add("dark", "dark-uniform", dark)
        results = index.search(bright, top_k=2)
        self.assertEqual(results[0][0], "bright")
        self.assertAlmostEqual(results[0][2], 1.0, places=6)

    def test_similar_pattern_ranks_above_dissimilar(self):
        index = RetrievalIndex()
        query = extract_features(_bright_center())
        near = extract_features([[40.0] * 8 for _ in range(8)])
        # Near image: same background, slightly brighter centre.
        near_image = [[40.0] * 8 for _ in range(8)]
        for r in range(2, 6):
            for c in range(2, 6):
                near_image[r][c] = 200.0
        index.add("near", "bright-center", extract_features(near_image))
        index.add("far", "dark-uniform", extract_features(_uniform(20)))
        results = index.search(query, top_k=2)
        self.assertEqual(results[0][0], "near")

    def test_duplicate_id_rejected(self):
        index = RetrievalIndex()
        index.add("a", "x", [1.0, 0.0])
        with self.assertRaises(ValueError):
            index.add("a", "y", [0.0, 1.0])

    def test_json_round_trip(self):
        index = RetrievalIndex()
        index.add("bright", "bright-center", extract_features(_bright_center()))
        restored = RetrievalIndex.from_json(index.to_json())
        self.assertEqual([e.id for e in restored.entries], ["bright"])
        self.assertEqual(restored.search(extract_features(_bright_center()))[0][0], "bright")


class TestIO(unittest.TestCase):
    def test_load_pgm_example(self):
        image = load_pgm(EXAMPLES / "images" / "img01.pgm")
        self.assertEqual(len(image), 8)
        self.assertEqual(len(image[0]), 8)

    def test_load_pgm_rejects_bad_file(self):
        with self.assertRaises(ValueError):
            load_pgm(EXAMPLES / "index.csv")


if __name__ == "__main__":
    unittest.main()
