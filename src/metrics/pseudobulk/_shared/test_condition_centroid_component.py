"""Shared Viash integration test for every condition-centroid component."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np


## VIASH START
meta = {"executable": ""}
## VIASH END


def main() -> None:
    truth = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 0.0],
            [0.0, 2.0, 4.0, 6.0],
            [6.0, 4.0, 2.0, 0.0],
        ]
    )
    zeros = np.zeros_like(truth)
    with tempfile.TemporaryDirectory(prefix="condition-centroid-") as temp_dir:
        temp_path = Path(temp_dir)
        prepared_path = temp_path / "prepared.npz"
        output_path = temp_path / "score.h5ad"
        np.savez_compressed(
            prepared_path,
            truth=truth,
            prediction=truth,
            control_reference=zeros,
            perturbed_mean_reference=zeros,
            deg_mask=np.ones_like(truth, dtype=bool),
            deg_weights=np.full_like(truth, 0.25),
            candidate_groups=np.asarray(["a", "a", "b", "b"]),
            condition_keys=np.asarray(["a1", "a2", "b1", "b2"]),
            metadata_json=np.asarray(
                json.dumps(
                    {"dataset_id": "synthetic", "method_id": "perfect_prediction"}
                )
            ),
        )
        subprocess.run(
            [
                meta["executable"],
                "--prepared",
                str(prepared_path),
                "--output",
                str(output_path),
            ],
            check=True,
        )

        score = ad.read_h5ad(output_path)
        assert score.shape == (0, 0)
        assert score.uns["dataset_id"] == "synthetic"
        assert score.uns["method_id"] == "perfect_prediction"
        metric_ids = score.uns["metric_ids"].tolist()
        values = dict(zip(metric_ids, score.uns["metric_values"]))
        assert len(values) == 14
        assert values["mse"] == 0.0
        assert values["weighted_mse"] == 0.0
        for metric_id, value in values.items():
            if metric_id not in {"mse", "weighted_mse"}:
                assert np.isclose(value, 1.0), metric_id
        assert score.uns["per_unit_metric_values"].shape == (14, 4)


if __name__ == "__main__":
    main()
