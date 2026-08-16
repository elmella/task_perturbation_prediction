"""Unit tests for the scPertEval-to-Open-Problems adapter."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).parents[2] / "_shared"))
sys.path.insert(0, str(Path(__file__).parent))

from scperteval_distributional_runner import (
    PROTOCOLS,
    evaluate_scperteval_distributional,
)


class _FakeResult:
    def __init__(self, value: float):
        self.per_perturbation = pd.DataFrame(
            {
                "perturbation": ["drug_a", "drug_b"],
                "score": [value, value + 1.0],
            }
        )


class _FakeBackend:
    __version__ = "test"

    def __init__(self, values: list[float]):
        self.values = iter(values)
        self.prepare_calls: list[tuple[ad.AnnData, dict]] = []
        self.score_calls: list[tuple[str, ad.AnnData, str]] = []

    def prepare(self, observed: ad.AnnData, protocols: list[str], **kwargs):
        self.prepare_calls.append((observed, {"protocols": protocols, **kwargs}))
        return object()

    def score(
        self,
        prepared: object,
        protocol: str,
        prediction: ad.AnnData,
        de_method: str,
    ) -> _FakeResult:
        self.score_calls.append((protocol, prediction, de_method))
        return _FakeResult(next(self.values))


class ScPertEvalAdapterTests(unittest.TestCase):
    def _inputs(self) -> tuple[ad.AnnData, ad.AnnData]:
        rng = np.random.default_rng(11)
        genes = [f"gene_{i}" for i in range(50)]
        observed = ad.AnnData(
            X=rng.normal(size=(60, 50)),
            obs=pd.DataFrame(
                {"perturbation": ["control"] * 20 + ["drug_a"] * 20 + ["drug_b"] * 20},
                index=[f"observed_{i}" for i in range(60)],
            ),
        )
        observed.var_names = genes
        observed.uns.update(
            {"dataset_id": "synthetic", "expression_scale": "log1p_cp10k"}
        )
        prediction = ad.AnnData(
            X=rng.normal(size=(40, 50)),
            obs=pd.DataFrame(
                {"perturbation": ["drug_a"] * 20 + ["drug_b"] * 20},
                index=[f"predicted_{i}" for i in range(40)],
            ),
        )
        prediction.var_names = list(reversed(genes))
        prediction.uns.update(
            {"method_id": "fake_method", "expression_scale": "log1p_cp10k"}
        )
        return observed, prediction

    def test_interface_prepares_once_and_returns_shared_result(self) -> None:
        observed, prediction = self._inputs()
        backend = _FakeBackend([float(i) for i in range(len(PROTOCOLS))])
        result = evaluate_scperteval_distributional(
            observed,
            prediction,
            min_cells=10,
            subsample=128,
            seed=7,
            workers=1,
            backend=backend,
        )

        self.assertEqual(len(backend.prepare_calls), 1)
        self.assertEqual(len(backend.score_calls), len(PROTOCOLS))
        self.assertEqual(result.dataset_id, "synthetic")
        self.assertEqual(result.method_id, "fake_method")
        self.assertEqual(list(result.metric_values), list(PROTOCOLS))
        np.testing.assert_array_equal(
            list(result.metric_values.values()),
            np.arange(len(PROTOCOLS), dtype=float) + 0.5,
        )
        self.assertEqual(result.per_unit_metric_values.shape, (6, 2))
        self.assertEqual(
            backend.score_calls[0][1].var_names.tolist(),
            observed.var_names.tolist(),
        )

    def test_missing_control_is_rejected_before_scoring(self) -> None:
        observed, prediction = self._inputs()
        observed.obs["perturbation"] = "drug_a"
        with self.assertRaisesRegex(ValueError, "contains no 'control' cells"):
            evaluate_scperteval_distributional(
                observed,
                prediction,
                backend=_FakeBackend([]),
            )

    def test_expression_scale_must_match(self) -> None:
        observed, prediction = self._inputs()
        prediction.uns["expression_scale"] = "counts"
        with self.assertRaisesRegex(ValueError, "expression_scale values must match"):
            evaluate_scperteval_distributional(
                observed,
                prediction,
                backend=_FakeBackend([]),
            )

    def test_group_keys_prepare_each_covariate_group_separately(self) -> None:
        observed, prediction = self._inputs()
        observed_a, observed_b = observed.copy(), observed.copy()
        prediction_a, prediction_b = prediction.copy(), prediction.copy()
        observed_a.obs["cell_type"] = "A"
        observed_b.obs["cell_type"] = "B"
        prediction_a.obs["cell_type"] = "A"
        prediction_b.obs["cell_type"] = "B"
        observed = ad.concat([observed_a, observed_b], index_unique="-")
        prediction = ad.concat([prediction_a, prediction_b], index_unique="-")
        observed.uns.update(
            {"dataset_id": "synthetic", "expression_scale": "log1p_cp10k"}
        )
        prediction.uns.update(
            {"method_id": "fake_method", "expression_scale": "log1p_cp10k"}
        )
        backend = _FakeBackend(
            [
                float(group_offset + metric_index)
                for group_offset in (0, 10)
                for metric_index in range(len(PROTOCOLS))
            ]
        )
        result = evaluate_scperteval_distributional(
            observed,
            prediction,
            group_keys=["cell_type"],
            min_cells=10,
            workers=1,
            backend=backend,
        )

        self.assertEqual(len(backend.prepare_calls), 2)
        self.assertEqual(len(backend.score_calls), 2 * len(PROTOCOLS))
        self.assertEqual(result.per_unit_metric_values.shape, (6, 4))
        np.testing.assert_allclose(
            list(result.metric_values.values()),
            np.arange(len(PROTOCOLS), dtype=float) + 5.5,
        )


if __name__ == "__main__":
    unittest.main()
