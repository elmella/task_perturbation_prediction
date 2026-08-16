"""Viash integration test for the packaged scPertEval distributional component."""

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


def main() -> None:
    rng = np.random.default_rng(19)
    genes = [f"gene_{index}" for index in range(50)]
    control = rng.normal(0.0, 1.0, size=(20, 50))
    drug_a = rng.normal(0.3, 1.0, size=(20, 50))
    drug_b = rng.normal(-0.3, 1.0, size=(20, 50))
    observed = ad.AnnData(
        X=np.vstack([control, drug_a, drug_b]),
        obs=pd.DataFrame(
            {"perturbation": ["control"] * 20 + ["drug_a"] * 20 + ["drug_b"] * 20},
            index=[f"observed_{i}" for i in range(60)],
        ),
    )
    observed.var_names = genes
    observed.uns["dataset_id"] = "synthetic"
    observed.uns["expression_scale"] = "log1p_cp10k"
    prediction = ad.AnnData(
        X=np.vstack([drug_a, drug_b]),
        obs=pd.DataFrame(
            {"perturbation": ["drug_a"] * 20 + ["drug_b"] * 20},
            index=[f"predicted_{i}" for i in range(40)],
        ),
    )
    prediction.var_names = genes
    prediction.uns["method_id"] = "perfect_cells"
    prediction.uns["expression_scale"] = "log1p_cp10k"

    with tempfile.TemporaryDirectory(prefix="scperteval-distributional-") as temp_dir:
        temp_path = Path(temp_dir)
        observed_path = temp_path / "observed.h5ad"
        prediction_path = temp_path / "prediction.h5ad"
        output_path = temp_path / "score.h5ad"
        observed.write_h5ad(observed_path)
        prediction.write_h5ad(prediction_path)
        subprocess.run(
            [
                meta["executable"],
                "--observed",
                str(observed_path),
                "--prediction",
                str(prediction_path),
                "--output",
                str(output_path),
                "--min_cells",
                "10",
                "--workers",
                "1",
                "--subsample",
                "40",
            ],
            check=True,
        )
        output = ad.read_h5ad(output_path)
        assert output.uns["dataset_id"] == "synthetic"
        assert output.uns["method_id"] == "perfect_cells"
        assert len(output.uns["metric_ids"]) == 6
        assert np.isfinite(output.uns["metric_values"]).all()
        assert output.uns["per_unit_metric_values"].shape == (6, 2)
        assert output.uns["metric_scored_unit_counts"].tolist() == [2] * 6


if __name__ == "__main__":
    main()
