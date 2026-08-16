"""Python implementation of the historical Open Problems DEG metrics."""

from __future__ import annotations

import warnings
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse
from scipy.stats import rankdata

from metric_result import MetricResult, write_metric_result


METRIC_DIRECTIONS = {
    "mean_rowwise_rmse": "minimize",
    "mean_rowwise_mae": "minimize",
    "mean_rowwise_pearson": "maximize",
    "mean_rowwise_spearman": "maximize",
    "mean_rowwise_cosine": "maximize",
}


def _dense(values: object, name: str) -> np.ndarray:
    if sparse.issparse(values):
        values = values.toarray()
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    return array


def _aligned_layers(
    de_test_path: str | Path,
    de_test_layer: str,
    prediction_path: str | Path,
    prediction_layer: str,
    resolve_genes: str,
) -> tuple[np.ndarray, np.ndarray, list[str], str, str]:
    de_test = ad.read_h5ad(de_test_path)
    prediction = ad.read_h5ad(prediction_path)
    if not de_test.obs_names.is_unique or not prediction.obs_names.is_unique:
        raise ValueError("DE truth and prediction condition identifiers must be unique")
    missing_conditions = de_test.obs_names.difference(prediction.obs_names).tolist()
    extra_conditions = prediction.obs_names.difference(de_test.obs_names).tolist()
    if missing_conditions or extra_conditions:
        raise ValueError(
            "DE truth/prediction condition mismatch; "
            f"missing={missing_conditions or 'none'}, extra={extra_conditions or 'none'}"
        )
    prediction = prediction[de_test.obs_names]

    if not de_test.var_names.is_unique or not prediction.var_names.is_unique:
        raise ValueError("DE truth and prediction gene identifiers must be unique")
    if resolve_genes == "de_test":
        missing_genes = de_test.var_names.difference(prediction.var_names).tolist()
        if missing_genes:
            raise ValueError(
                "Prediction is missing genes required by de_test: "
                + ", ".join(missing_genes[:10])
            )
        genes = de_test.var_names.tolist()
    elif resolve_genes == "intersection":
        prediction_genes = set(prediction.var_names)
        genes = [gene for gene in de_test.var_names if gene in prediction_genes]
        if not genes:
            raise ValueError("DE truth and prediction have no genes in common")
    else:
        raise ValueError(f"Unknown resolve_genes value: {resolve_genes}")

    if de_test_layer not in de_test.layers:
        raise ValueError(f"de_test is missing layer {de_test_layer!r}")
    if prediction_layer not in prediction.layers:
        raise ValueError(f"prediction is missing layer {prediction_layer!r}")
    truth = _dense(de_test[:, genes].layers[de_test_layer], "de_test layer")
    predicted = _dense(
        prediction[:, genes].layers[prediction_layer], "prediction layer"
    )
    if not np.isfinite(truth).all():
        raise ValueError("de_test layer must contain only finite values")
    if np.isnan(predicted).any():
        warnings.warn(
            "NA values in prediction layer were replaced with zero to preserve "
            "the historical metric protocol",
            RuntimeWarning,
            stacklevel=2,
        )
        predicted = np.nan_to_num(predicted, nan=0.0)
    if not np.isfinite(predicted).all():
        raise ValueError("prediction layer must not contain infinite values")
    return (
        truth,
        predicted,
        de_test.obs_names.astype(str).tolist(),
        str(de_test.uns["dataset_id"]),
        str(prediction.uns["method_id"]),
    )


def _rowwise_pearson(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    numerator = np.sum(left_centered * right_centered, axis=1)
    denominator = np.sqrt(
        np.sum(left_centered**2, axis=1) * np.sum(right_centered**2, axis=1)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        values = numerator / denominator
    return np.where(np.isfinite(values), values, 0.0)


def _rowwise_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1)
    denominator = np.sqrt(np.sum(left**2, axis=1) * np.sum(right**2, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        values = numerator / denominator
    return np.where(np.isfinite(values), values, 0.0)


def rowwise_metric_values(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return condition-level values for all five historical formulas."""

    difference = np.asarray(truth, dtype=float) - np.asarray(prediction, dtype=float)
    truth_ranks = rankdata(truth, method="average", axis=1)
    prediction_ranks = rankdata(prediction, method="average", axis=1)
    return {
        "mean_rowwise_rmse": np.sqrt(np.mean(difference**2, axis=1)),
        "mean_rowwise_mae": np.mean(np.abs(difference), axis=1),
        "mean_rowwise_pearson": _rowwise_pearson(truth, prediction),
        "mean_rowwise_spearman": _rowwise_pearson(truth_ranks, prediction_ranks),
        "mean_rowwise_cosine": _rowwise_cosine(truth, prediction),
    }


def zapsmall(values: np.ndarray, digits: int = 10) -> np.ndarray:
    """Match base R's ``zapsmall(values, digits)`` output rounding."""

    values = np.asarray(values, dtype=float)
    finite = values[~np.isnan(values)]
    if finite.size == 0:
        return values
    maximum = np.max(np.abs(finite))
    if maximum == 0 or (np.isinf(maximum) and np.isinf(digits)):
        return values
    decimals = int(np.rint(max(0.0, digits - np.log10(maximum))))
    return np.round(values, decimals=decimals)


def evaluate_rowwise_metrics(
    de_test_path: str | Path,
    de_test_layer: str,
    prediction_path: str | Path,
    prediction_layer: str,
    resolve_genes: str,
) -> MetricResult:
    """Evaluate all historical DEG metrics from aligned H5AD layers."""

    truth, prediction, condition_ids, dataset_id, method_id = _aligned_layers(
        de_test_path,
        de_test_layer,
        prediction_path,
        prediction_layer,
        resolve_genes,
    )
    per_condition = rowwise_metric_values(truth, prediction)
    raw_aggregates = np.asarray(
        [np.mean(per_condition[metric_id]) for metric_id in METRIC_DIRECTIONS]
    )
    aggregates = zapsmall(raw_aggregates, digits=10)
    return MetricResult(
        dataset_id=dataset_id,
        method_id=method_id,
        metric_values=dict(zip(METRIC_DIRECTIONS, aggregates)),
        per_unit_ids=condition_ids,
        per_unit_metric_values=np.vstack(
            [per_condition[metric_id] for metric_id in METRIC_DIRECTIONS]
        ),
        provenance={
            "de_test_layer": de_test_layer,
            "prediction_layer": prediction_layer,
            "resolve_genes": resolve_genes,
            "aggregation": "mean_over_conditions",
            "implementation": "python_port_parity_checked_against_original_r",
        },
        details={
            "metric_directions": np.asarray(
                list(METRIC_DIRECTIONS.values()), dtype=str
            )
        },
    )


def run_rowwise_metrics(
    de_test_path: str | Path,
    de_test_layer: str,
    prediction_path: str | Path,
    prediction_layer: str,
    resolve_genes: str,
    output_path: str | Path,
) -> None:
    result = evaluate_rowwise_metrics(
        de_test_path,
        de_test_layer,
        prediction_path,
        prediction_layer,
        resolve_genes,
    )
    write_metric_result(result, output_path)


__all__ = [
    "METRIC_DIRECTIONS",
    "evaluate_rowwise_metrics",
    "rowwise_metric_values",
    "run_rowwise_metrics",
    "zapsmall",
]
