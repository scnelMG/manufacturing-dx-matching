import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.matching import retrieve_technology_candidates, rank_provider_candidates


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.vectors = np.array([[20., 0.], [1., 1.], [-3., 0.]])
        self.demand = np.array([2., 0.])
        self.metadata = pd.DataFrame({"provider_company": ["A", "B", "C"],
                                      "technology_name": ["a", "b", "c"]},
                                     index=[70, 10, 30])

    def test_cosine_search_preserves_metadata_alignment_and_inputs(self):
        vectors, demand = self.vectors.copy(), self.demand.copy()
        metadata = self.metadata.copy(deep=True)
        result = retrieve_technology_candidates(demand, vectors, metadata, top_k=2)
        self.assertEqual(result["provider_company"].tolist(), ["A", "B"])
        self.assertEqual(result["technology_name"].tolist(), ["a", "b"])
        np.testing.assert_allclose(result["similarity"], [1., 1 / np.sqrt(2)], atol=1e-6)
        np.testing.assert_array_equal(vectors, self.vectors)
        np.testing.assert_array_equal(demand, self.demand)
        assert_frame_equal(metadata, self.metadata)

    def test_invalid_top_k_does_not_return_faiss_padding_rows(self):
        for top_k in (0, 4):
            with self.subTest(top_k=top_k), self.assertRaises(ValueError):
                retrieve_technology_candidates(self.demand, self.vectors, self.metadata, top_k)

    def test_embedding_shape_and_metadata_length_must_match(self):
        for demand, vectors, metadata in (
            (np.ones(3), self.vectors, self.metadata),
            (self.demand.reshape(1, -1), self.vectors, self.metadata),
            (self.demand, self.vectors.ravel(), self.metadata),
            (self.demand, self.vectors, self.metadata.iloc[:2]),
        ):
            with self.subTest(shape=vectors.shape), self.assertRaises(ValueError):
                retrieve_technology_candidates(demand, vectors, metadata, top_k=1)


class RankingTests(unittest.TestCase):
    def setUp(self):
        self.candidates = pd.DataFrame({
            "provider_company": ["A", "A", "B"],
            "technology_name": ["a1", "a2", "b1"],
            "similarity": [0.8, 0.6, 0.9],
        })
        self.evidence = pd.DataFrame({"provider_company": ["A", "B"],
                                      "award_count": [0, 3], "certificate_count": [0, 1]})

    def test_aggregation_and_weights_change_ranking(self):
        original = self.candidates.copy(deep=True)
        result = rank_provider_candidates(self.candidates, self.evidence)
        self.assertEqual(result["provider_company"].tolist(), ["A", "B"])
        np.testing.assert_allclose(result["final_score"], [80., 20.])
        self.assertEqual(result["matched_technology_count"].tolist(), [2, 1])
        self.assertAlmostEqual(result.iloc[0]["avg_similarity"], 0.7)
        credibility_only = rank_provider_candidates(self.candidates, self.evidence, 0)
        self.assertEqual(credibility_only.iloc[0]["provider_company"], "B")
        assert_frame_equal(self.candidates, original)

    def test_missing_evidence_and_constant_scores_stay_finite(self):
        result = rank_provider_candidates(self.candidates.iloc[:2], self.evidence.iloc[:0])
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["final_score"], 0)
        self.assertEqual(result.iloc[0]["award_count"], 0)
        self.assertEqual(result.iloc[0]["certificate_count"], 0)

    def test_invalid_weights_and_evidence_are_rejected(self):
        for weight in (-0.1, 1.1):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                rank_provider_candidates(self.candidates, self.evidence, weight)
        duplicate = pd.concat([self.evidence, self.evidence.iloc[:1]])
        negative = self.evidence.assign(award_count=-1)
        for evidence in (duplicate, negative):
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                rank_provider_candidates(self.candidates, evidence)

    def test_missing_columns_and_similarity_are_rejected(self):
        with self.assertRaises(KeyError):
            rank_provider_candidates(self.candidates.drop(columns="similarity"), self.evidence)
        with self.assertRaises(KeyError):
            rank_provider_candidates(self.candidates, self.evidence.drop(columns="award_count"))
        with self.assertRaises(ValueError):
            rank_provider_candidates(self.candidates.assign(similarity=np.nan), self.evidence)


if __name__ == "__main__":
    unittest.main()
