# Single-cell metrics

Components in this family require predicted and observed cell populations with
genes and perturbation-condition labels aligned on a declared expression
scale. They may use distributional information that is absent from a condition
centroid, including variance, covariance, multimodality, and tail behavior.

`scperteval_distributional/` wraps the public API of the authors' pinned
`scperteval[sinkhorn]==0.1.0` package. One preparation pass is reused across
six paper-defined protocols:

- unbiased median-bandwidth RBF MMD squared in top-50 and PCA-50 spaces;
- bias-corrected energy distance in top-50 and PCA-50 spaces; and
- debiased Sinkhorn W2 in top-50 and PCA-50 spaces.

The component consumes normalized expression in `X`. Both H5AD inputs must
contain the same genes and the configured perturbation label column. The
observed file must include control cells and `uns["dataset_id"]`; predictions
must contain `uns["method_id"]`. Both files must declare the same normalization
and transformation identifier in `uns["expression_scale"]`. scPertEval constructs top-effect-size gene
sets and fits PCA evaluator-side from observed data, never from predictions.
For multi-covariate datasets, `--group_keys` runs an independent preparation
within each group (for example, each cell type) before macro-aggregating the
condition-level scores. This avoids mixing incompatible covariate populations
in the control reference and PCA fit.

The shared score contract contains the macro mean for each protocol and the
aligned metric-by-perturbation matrix, finite perturbation counts, and exact
protocol provenance. The unbiased MMD and energy estimates can be negative at
finite sample sizes; the adapter preserves those raw package values.
