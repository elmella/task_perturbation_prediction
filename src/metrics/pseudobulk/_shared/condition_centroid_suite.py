"""Evaluation adapter for the complete condition-centroid metric family."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from condition_centroid_metrics import (
    METRIC_DIRECTIONS,
    aggregate_metric_scores,
    score_prediction_metrics,
)
from metric_result import MetricResult, write_metric_result


_REQUIRED_ARRAYS = {
    "truth",
    "prediction",
    "control_reference",
    "perturbed_mean_reference",
    "deg_mask",
    "deg_weights",
    "candidate_groups",
    "condition_keys",
    "metadata_json",
}


def evaluate_condition_centroids(prepared_path: str | Path) -> MetricResult:
    """Evaluate every paper-derived centroid metric from one aligned bundle."""

    prepared_path = Path(prepared_path)
    with np.load(prepared_path, allow_pickle=False) as prepared:
        missing = sorted(_REQUIRED_ARRAYS.difference(prepared.files))
        if missing:
            raise ValueError(
                "Condition-centroid bundle is missing arrays: " + ", ".join(missing)
            )
        metadata = json.loads(str(prepared["metadata_json"].item()))
        if not isinstance(metadata, dict):
            raise ValueError("metadata_json must encode a JSON object")
        for key in ("dataset_id", "method_id"):
            if not metadata.get(key):
                raise ValueError(f"metadata_json must contain non-empty {key!r}")

        if prepared["condition_keys"].ndim != 1:
            raise ValueError("condition_keys must be one-dimensional")
        if prepared["candidate_groups"].ndim != 1:
            raise ValueError("candidate_groups must be one-dimensional")
        condition_keys = [str(value) for value in prepared["condition_keys"].tolist()]
        if len(condition_keys) != len(set(condition_keys)):
            raise ValueError("condition_keys must be unique")
        scores = score_prediction_metrics(
            truth=prepared["truth"],
            prediction=prepared["prediction"],
            control_reference=prepared["control_reference"],
            perturbed_mean_reference=prepared["perturbed_mean_reference"],
            deg_mask=prepared["deg_mask"].astype(bool),
            deg_weights=prepared["deg_weights"],
            candidate_groups=prepared["candidate_groups"].tolist(),
        )

    metric_ids = list(METRIC_DIRECTIONS)
    if any(len(scores[metric_id]) != len(condition_keys) for metric_id in metric_ids):
        raise ValueError("condition_keys must contain one identifier per score row")
    aggregates = aggregate_metric_scores(scores)
    return MetricResult(
        dataset_id=str(metadata["dataset_id"]),
        method_id=str(metadata["method_id"]),
        metric_values={metric_id: aggregates[metric_id] for metric_id in metric_ids},
        per_unit_ids=condition_keys,
        per_unit_metric_values=np.vstack([scores[metric_id] for metric_id in metric_ids]),
        provenance={
            **metadata,
            "input_format": "condition_centroid_npz_v1",
            "aggregation": "macro_mean_over_finite_conditions",
        },
        details={
            "metric_directions": np.asarray(
                [METRIC_DIRECTIONS[metric_id] for metric_id in metric_ids], dtype=str
            )
        },
    )


def run_condition_centroid_suite(
    prepared_path: str | Path,
    output_path: str | Path,
) -> None:
    """Evaluate the suite and write one multi-metric score artifact."""

    result = evaluate_condition_centroids(prepared_path)
    write_metric_result(result, output_path)
    print(f"Wrote {len(result.metric_values)} scores to {output_path}")


__all__ = ["evaluate_condition_centroids", "run_condition_centroid_suite"]
