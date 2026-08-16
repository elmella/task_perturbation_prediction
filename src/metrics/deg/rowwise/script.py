import sys
from pathlib import Path


## VIASH START
par = {
    "de_test": "resources/datasets/neurips-2023-data/de_test.h5ad",
    "de_test_layer": "clipped_sign_log10_pval",
    "prediction": "resources/datasets/neurips-2023-data/prediction.h5ad",
    "prediction_layer": "prediction",
    "resolve_genes": "de_test",
    "output": "output.h5ad",
}
meta = {"resources_dir": str(Path(__file__).resolve().parents[1] / "_shared")}
## VIASH END


sys.path.extend(
    [meta["resources_dir"], str(Path(__file__).resolve().parents[2] / "_shared")]
)
from rowwise_suite import run_rowwise_metrics  # noqa: E402


run_rowwise_metrics(
    de_test_path=par["de_test"],
    de_test_layer=par["de_test_layer"],
    prediction_path=par["prediction"],
    prediction_layer=par["prediction_layer"],
    resolve_genes=par["resolve_genes"],
    output_path=par["output"],
)
