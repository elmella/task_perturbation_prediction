"""Unit tests for the historical DEG metric implementation."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).parents[2] / "_shared"))
sys.path.insert(0, str(Path(__file__).parent))

from rowwise_suite import evaluate_rowwise_metrics, rowwise_metric_values, zapsmall


class RowwiseSuiteTests(unittest.TestCase):
    def test_python_formulas_match_original_r_reference(self) -> None:
        truth = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [5.0, 4.0, 3.0, 2.0, 1.0],
                [2.0, 2.0, 2.0, 2.0, 2.0],
                [0.0, 1.0, 1.0, 4.0, 9.0],
            ]
        )
        prediction = np.array(
            [
                [1.0, 2.5, 2.0, 4.5, 4.0],
                [4.0, 4.0, 2.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 2.0, 5.0, 8.0],
            ]
        )
        expected = np.array(
            [0.8739902908, 0.8, 0.7032092002, 0.6739330165, 0.9833123797]
        )
        per_condition = rowwise_metric_values(truth, prediction)
        observed = zapsmall(
            np.asarray([np.mean(values) for values in per_condition.values()]),
            digits=10,
        )
        np.testing.assert_array_equal(observed, expected)

    def test_constant_rows_follow_legacy_zero_policy(self) -> None:
        truth = np.ones((1, 3))
        prediction = np.ones((1, 3))
        values = rowwise_metric_values(truth, prediction)
        self.assertEqual(values["mean_rowwise_pearson"][0], 0.0)
        self.assertEqual(values["mean_rowwise_spearman"][0], 0.0)

    def test_adapter_aligns_conditions_and_genes(self) -> None:
        truth_values = np.asarray([[1.0, 2.0, 4.0], [4.0, 2.0, 1.0]])
        conditions = ["a", "b"]
        genes = ["g1", "g2", "g3"]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            truth_path = temp_path / "truth.h5ad"
            prediction_path = temp_path / "prediction.h5ad"
            truth = ad.AnnData(
                X=np.zeros_like(truth_values),
                obs=pd.DataFrame(index=conditions),
                var=pd.DataFrame(index=genes),
                uns={"dataset_id": "dataset"},
            )
            truth.layers["truth"] = truth_values
            truth.write_h5ad(truth_path)
            prediction = ad.AnnData(
                X=np.zeros_like(truth_values),
                obs=pd.DataFrame(index=list(reversed(conditions))),
                var=pd.DataFrame(index=list(reversed(genes))),
                uns={"method_id": "method"},
            )
            prediction.layers["prediction"] = truth_values[::-1, ::-1]
            prediction.write_h5ad(prediction_path)

            result = evaluate_rowwise_metrics(
                truth_path,
                "truth",
                prediction_path,
                "prediction",
                "intersection",
            )

        self.assertEqual(result.per_unit_ids, conditions)
        self.assertEqual(result.metric_values["mean_rowwise_rmse"], 0.0)
        self.assertEqual(result.metric_values["mean_rowwise_pearson"], 1.0)


if __name__ == "__main__":
    unittest.main()
