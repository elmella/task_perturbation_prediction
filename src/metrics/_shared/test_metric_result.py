"""Contract tests for the shared metric result."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parent))

from metric_result import MetricResult, to_score_anndata


class MetricResultTests(unittest.TestCase):
    def test_serializes_required_and_auditing_fields(self) -> None:
        result = MetricResult(
            dataset_id="dataset",
            method_id="method",
            metric_values={"first": 1.0, "second": np.nan},
            per_unit_ids=["a", "b"],
            per_unit_metric_values=np.asarray([[1.0, 1.0], [np.nan, 2.0]]),
            provenance={"protocol": "test"},
        )
        score = to_score_anndata(result)
        self.assertEqual(score.uns["metric_ids"].tolist(), ["first", "second"])
        self.assertEqual(score.uns["metric_scored_unit_counts"].tolist(), [2, 1])
        self.assertEqual(score.uns["provenance_json"], '{"protocol": "test"}')

    def test_rejects_misaligned_per_unit_matrix(self) -> None:
        result = MetricResult(
            dataset_id="dataset",
            method_id="method",
            metric_values={"first": 1.0},
            per_unit_ids=["a", "b"],
            per_unit_metric_values=np.asarray([[1.0]]),
        )
        with self.assertRaisesRegex(ValueError, "must have shape"):
            result.validate()


if __name__ == "__main__":
    unittest.main()
