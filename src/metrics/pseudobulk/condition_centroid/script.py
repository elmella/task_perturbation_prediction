import sys
from pathlib import Path


## VIASH START
par = {"prepared": "prepared.npz", "output": "output.h5ad"}
meta = {"resources_dir": str(Path(__file__).resolve().parents[1] / "_shared")}
## VIASH END


sys.path.extend(
    [meta["resources_dir"], str(Path(__file__).resolve().parents[2] / "_shared")]
)
from condition_centroid_suite import run_condition_centroid_suite  # noqa: E402


run_condition_centroid_suite(par["prepared"], par["output"])
