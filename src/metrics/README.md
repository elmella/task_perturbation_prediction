# Metric families

Metric components are grouped by the least-derived prediction representation
they require:

- `single_cell/` consumes predicted cells by genes with condition labels;
- `pseudobulk/` consumes one gene-expression centroid per condition; and
- `deg/` consumes one differential-expression result per condition.

An evaluator may derive pseudobulk centroids and DE results from a valid
single-cell prediction. It may also derive DE results from pseudobulk output
when the output preserves the replicate and control structure required by the
declared DE procedure. The reverse transformations are not valid.

Each family exposes one suite component: `deg/rowwise`,
`pseudobulk/condition_centroid`, and
`single_cell/scperteval_distributional`. Their scientific implementations stay
family-local, while all three return the score contract in
`_shared/metric_result.py`. `metric_suite.yaml` records their scoring
directions, sources, aliases, and protocol-specific requirements.

This directory contains direct prediction metrics only. Metric-evaluation
protocols belong outside these representation families.

Run the local contract, manifest, and formula tests with:

```bash
python3 src/metrics/_shared/test_metric_result.py
python3 src/metrics/_shared/test_metric_suite_manifest.py
python3 src/metrics/deg/_shared/test_rowwise_suite.py
python3 src/metrics/pseudobulk/_shared/test_condition_centroid_metrics.py
python3 src/metrics/single_cell/_shared/test_scperteval_distributional_runner.py
```
