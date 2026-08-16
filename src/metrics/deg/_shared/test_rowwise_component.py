"""Viash regression test for the consolidated rowwise metric suite."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


## VIASH START
meta = {"executable": ""}
## VIASH END


R_REFERENCE_VALUES = {
    "mean_rowwise_rmse": 0.8739902908,
    "mean_rowwise_mae": 0.8,
    "mean_rowwise_pearson": 0.7032092002,
    "mean_rowwise_spearman": 0.6739330165,
    "mean_rowwise_cosine": 0.9833123797,
}


def main() -> None:
    truth = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 2.0],
            [0.0, 1.0, 1.0, 4.0, 9.0],
        ]
    )
    predicted = np.array(
        [
            [1.0, 2.5, 2.0, 4.5, 4.0],
            [4.0, 4.0, 2.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, np.nan, 2.0, 5.0, 8.0],
        ]
    )
    genes = [f"g{index}" for index in range(truth.shape[1])]
    conditions = [f"c{index}" for index in range(truth.shape[0])]

    with tempfile.TemporaryDirectory(prefix="rowwise-") as temp_dir:
        temp_path = Path(temp_dir)
        truth_path = temp_path / "truth.h5ad"
        prediction_path = temp_path / "prediction.h5ad"
        output_path = temp_path / "score.h5ad"

        truth_data = ad.AnnData(
            X=np.zeros_like(truth),
            obs=pd.DataFrame(index=conditions),
            var=pd.DataFrame(index=genes),
            uns={"dataset_id": "synthetic"},
        )
        truth_data.layers["truth"] = truth
        truth_data.write_h5ad(truth_path)

        prediction = ad.AnnData(
            X=np.zeros_like(predicted),
            obs=pd.DataFrame(index=list(reversed(conditions))),
            var=pd.DataFrame(index=list(reversed(genes))),
            uns={"method_id": "candidate"},
        )
        prediction.layers["prediction"] = predicted[::-1, ::-1]
        prediction.write_h5ad(prediction_path)

        subprocess.run(
            [
                meta["executable"],
                "--de_test",
                str(truth_path),
                "--de_test_layer",
                "truth",
                "--prediction",
                str(prediction_path),
                "--prediction_layer",
                "prediction",
                "--resolve_genes",
                "intersection",
                "--output",
                str(output_path),
            ],
            check=True,
        )

        score = ad.read_h5ad(output_path)
        assert score.uns["dataset_id"] == "synthetic"
        assert score.uns["method_id"] == "candidate"
        assert score.uns["metric_ids"].tolist() == list(R_REFERENCE_VALUES)
        np.testing.assert_array_equal(
            score.uns["metric_values"],
            np.asarray(list(R_REFERENCE_VALUES.values())),
        )
        assert score.uns["per_unit_metric_values"].shape == (5, 4)


if __name__ == "__main__":
    main()
