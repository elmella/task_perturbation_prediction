import sys
from pathlib import Path


## VIASH START
par = {
    "observed": "observed.h5ad",
    "prediction": "prediction.h5ad",
    "output": "output.h5ad",
    "perturbation_key": "perturbation",
    "group_keys": [],
    "control_label": "control",
    "min_cells": 30,
    "subsample": 8192,
    "seed": 42,
    "workers": 0,
    "de_method": "t-test",
}
meta = {
    "resources_dir": str(Path(__file__).resolve().parents[1] / "_shared")
}
## VIASH END


sys.path.extend(
    [meta["resources_dir"], str(Path(__file__).resolve().parents[2] / "_shared")]
)
from scperteval_distributional_runner import run_scperteval_distributional  # noqa: E402


run_scperteval_distributional(
    observed_path=par["observed"],
    prediction_path=par["prediction"],
    output_path=par["output"],
    perturbation_key=par["perturbation_key"],
    group_keys=par["group_keys"] or [],
    control_label=par["control_label"],
    min_cells=par["min_cells"],
    subsample=par["subsample"],
    seed=par["seed"],
    workers=par["workers"],
    de_method=par["de_method"],
)
