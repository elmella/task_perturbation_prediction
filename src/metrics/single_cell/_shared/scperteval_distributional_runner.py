"""Open Problems adapter for scPertEval's packaged distributional protocols."""

from __future__ import annotations

import importlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np

from metric_result import MetricResult, write_metric_result


PROTOCOLS = OrderedDict(
    (
        ("unbiased_mmd_median_top_k", "unbiased_mmd_median_top_k=50"),
        ("unbiased_mmd_median_pca_k", "unbiased_mmd_median_pca_k=50"),
        ("energy_distance_top_k", "energy_distance_top_k=50"),
        ("energy_distance_pca_k", "energy_distance_pca_k=50"),
        ("sinkhorn_w2_top_k", "sinkhorn_w2_top_k=50"),
        ("sinkhorn_w2_pca_k", "sinkhorn_w2_pca_k=50"),
    )
)


def _required_uns(adata: ad.AnnData, key: str, input_name: str) -> str:
    if key not in adata.uns or not str(adata.uns[key]):
        raise ValueError(f"{input_name}.uns must contain non-empty {key!r}")
    return str(adata.uns[key])


def _validate_inputs(
    observed: ad.AnnData,
    prediction: ad.AnnData,
    perturbation_key: str,
    control_label: str,
    group_keys: list[str],
) -> tuple[str, str, str]:
    dataset_id = _required_uns(observed, "dataset_id", "observed")
    method_id = _required_uns(prediction, "method_id", "prediction")
    observed_scale = _required_uns(observed, "expression_scale", "observed")
    prediction_scale = _required_uns(prediction, "expression_scale", "prediction")
    if observed_scale != prediction_scale:
        raise ValueError(
            "observed and prediction expression_scale values must match; "
            f"got {observed_scale!r} and {prediction_scale!r}"
        )
    if "dataset_id" in prediction.uns and str(prediction.uns["dataset_id"]) != dataset_id:
        raise ValueError("prediction dataset_id does not match observed dataset_id")

    for input_name, adata in (("observed", observed), ("prediction", prediction)):
        if perturbation_key not in adata.obs:
            raise ValueError(
                f"{input_name}.obs must contain perturbation key {perturbation_key!r}"
            )
        if not adata.var_names.is_unique:
            raise ValueError(f"{input_name}.var_names must be unique")
        missing_groups = [key for key in group_keys if key not in adata.obs]
        if missing_groups:
            raise ValueError(
                f"{input_name}.obs is missing group keys: {', '.join(missing_groups)}"
            )
    missing_genes = observed.var_names.difference(prediction.var_names).tolist()
    extra_genes = prediction.var_names.difference(observed.var_names).tolist()
    if missing_genes or extra_genes:
        raise ValueError(
            "prediction/observed gene mismatch; "
            f"missing={missing_genes[:10] or 'none'}, extra={extra_genes[:10] or 'none'}"
        )
    if observed.n_vars < 50:
        raise ValueError(
            "The fixed scPertEval protocols require at least 50 genes for their "
            "top-50 and PCA-50 feature spaces"
        )
    if observed.n_obs < 50:
        raise ValueError("PCA-50 requires at least 50 observed cells")
    if control_label not in set(observed.obs[perturbation_key].astype(str)):
        raise ValueError(
            f"observed contains no {control_label!r} cells in {perturbation_key!r}"
        )
    return dataset_id, method_id, observed_scale


def _group_indices(
    adata: ad.AnnData, group_keys: list[str]
) -> dict[tuple[str, ...], np.ndarray]:
    if not group_keys:
        return {(): np.arange(adata.n_obs)}
    values = adata.obs[group_keys].astype(str).itertuples(index=False, name=None)
    grouped: dict[tuple[str, ...], list[int]] = {}
    for index, value in enumerate(values):
        grouped.setdefault(tuple(value), []).append(index)
    return {key: np.asarray(indices, dtype=int) for key, indices in grouped.items()}


def _evaluation_groups(
    observed: ad.AnnData,
    prediction: ad.AnnData,
    group_keys: list[str],
    perturbation_key: str,
    control_label: str,
) -> list[tuple[tuple[str, ...], ad.AnnData, ad.AnnData]]:
    observed_groups = _group_indices(observed, group_keys)
    prediction_groups = _group_indices(prediction, group_keys)
    missing = sorted(set(observed_groups) - set(prediction_groups))
    extra = sorted(set(prediction_groups) - set(observed_groups))
    if missing or extra:
        raise ValueError(
            "prediction/observed evaluation-group mismatch; "
            f"missing={missing or 'none'}, extra={extra or 'none'}"
        )
    result = []
    for group in sorted(observed_groups):
        group_observed = observed[observed_groups[group]].copy()
        group_prediction = prediction[prediction_groups[group]].copy()
        labels = set(group_observed.obs[perturbation_key].astype(str))
        if control_label not in labels:
            details = dict(zip(group_keys, group))
            raise ValueError(
                f"observed evaluation group {details} contains no {control_label!r} "
                f"cells in {perturbation_key!r}"
            )
        if min(group_observed.n_obs, group_observed.n_vars) < 50:
            details = dict(zip(group_keys, group))
            raise ValueError(
                "PCA-50 requires at least 50 observed cells and genes per "
                f"evaluation group; group {details} has shape {group_observed.shape}"
            )
        result.append((group, group_observed, group_prediction))
    return result


def _aggregate_values(
    scores: OrderedDict[str, list[tuple[str, float]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not scores or not next(iter(scores.values())):
        raise ValueError("scPertEval returned no perturbation scores")
    condition_ids = np.asarray([condition for condition, _ in next(iter(scores.values()))])
    per_perturbation = []
    for metric_id, metric_scores in scores.items():
        current_ids = np.asarray([condition for condition, _ in metric_scores])
        if not np.array_equal(current_ids, condition_ids):
            raise RuntimeError(
                f"scPertEval returned inconsistent conditions for {metric_id}"
            )
        per_perturbation.append([value for _, value in metric_scores])
    values = np.asarray(per_perturbation, dtype=float)
    means = np.asarray(
        [np.mean(row[np.isfinite(row)]) if np.isfinite(row).any() else np.nan for row in values]
    )
    medians = np.asarray(
        [np.median(row[np.isfinite(row)]) if np.isfinite(row).any() else np.nan for row in values]
    )
    return means, medians, condition_ids, values


def evaluate_scperteval_distributional(
    observed: ad.AnnData,
    prediction: ad.AnnData,
    perturbation_key: str = "perturbation",
    group_keys: list[str] | None = None,
    control_label: str = "control",
    min_cells: int = 30,
    subsample: int = 8192,
    seed: int = 42,
    workers: int = 0,
    de_method: str = "t-test",
    backend: Any | None = None,
) -> MetricResult:
    """Evaluate all six packaged protocols and return a shared metric result."""

    group_keys = list(group_keys or [])
    if len(group_keys) != len(set(group_keys)):
        raise ValueError("group_keys must be unique")
    if perturbation_key in group_keys:
        raise ValueError("perturbation_key cannot also be a group key")
    if min_cells < 2 or subsample < 2 or workers < 0:
        raise ValueError(
            "min_cells and subsample must be at least 2; workers cannot be negative"
        )
    dataset_id, method_id, expression_scale = _validate_inputs(
        observed, prediction, perturbation_key, control_label, group_keys
    )
    prediction = prediction[:, observed.var_names].copy()
    backend = backend or importlib.import_module("scperteval")

    concrete_protocols = list(PROTOCOLS.values())
    scores: OrderedDict[str, list[tuple[str, float]]] = OrderedDict(
        (metric_id, []) for metric_id in PROTOCOLS
    )
    for group, group_observed, group_prediction in _evaluation_groups(
        observed, prediction, group_keys, perturbation_key, control_label
    ):
        group_dict = dict(zip(group_keys, group))
        prepared = backend.prepare(
            group_observed,
            concrete_protocols,
            subsample=int(subsample),
            seed=int(seed),
            min_cells=int(min_cells),
            perturbation_key=perturbation_key,
            control_label=control_label,
            workers=int(workers),
            name=json.dumps(group_dict, sort_keys=True) if group_dict else dataset_id,
        )
        for metric_id, protocol in PROTOCOLS.items():
            result = backend.score(
                prepared,
                protocol,
                group_prediction,
                de_method=de_method,
            )
            for row in result.per_perturbation.itertuples(index=False):
                condition = {**group_dict, perturbation_key: str(row.perturbation)}
                condition_id = json.dumps(
                    condition, sort_keys=True, separators=(",", ":")
                )
                scores[metric_id].append((condition_id, float(row.score)))

    means, medians, condition_ids, per_perturbation = _aggregate_values(scores)
    package_version = str(getattr(backend, "__version__", "unknown"))
    return MetricResult(
        dataset_id=dataset_id,
        method_id=method_id,
        metric_values=dict(zip(PROTOCOLS, means)),
        per_unit_ids=condition_ids.tolist(),
        per_unit_metric_values=per_perturbation,
        provenance={
            "package": "scperteval",
            "package_version": package_version,
            "protocols": concrete_protocols,
            "de_method": de_method,
            "perturbation_key": perturbation_key,
            "group_keys": group_keys,
            "control_label": control_label,
            "min_cells": int(min_cells),
            "subsample": int(subsample),
            "seed": int(seed),
            "workers": int(workers),
            "expression_scale": expression_scale,
            "aggregation": "macro_mean_over_finite_perturbations",
        },
        details={
            "metric_medians": medians,
            "metric_protocols": np.asarray(concrete_protocols, dtype=str),
            "scperteval_version": package_version,
        },
    )


def run_scperteval_distributional(
    observed_path: str | Path,
    prediction_path: str | Path,
    output_path: str | Path,
    **kwargs: Any,
) -> None:
    """Read inputs, evaluate the suite, and write one score artifact."""

    observed = ad.read_h5ad(observed_path)
    prediction = ad.read_h5ad(prediction_path)
    result = evaluate_scperteval_distributional(observed, prediction, **kwargs)
    write_metric_result(result, output_path)


__all__ = [
    "PROTOCOLS",
    "evaluate_scperteval_distributional",
    "run_scperteval_distributional",
]
