"""Validated score-file result shared by all metric families."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import anndata as ad
import numpy as np


_RESERVED_UNS_KEYS = {
    "dataset_id",
    "method_id",
    "metric_ids",
    "metric_values",
    "per_unit_ids",
    "per_unit_metric_values",
    "metric_scored_unit_counts",
    "provenance_json",
}


@dataclass(frozen=True)
class MetricResult:
    """In-memory result returned by a metric-family evaluator."""

    dataset_id: str
    method_id: str
    metric_values: Mapping[str, float]
    per_unit_ids: Sequence[str] = ()
    per_unit_metric_values: np.ndarray | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        metric_ids = list(self.metric_values)
        if not self.dataset_id:
            raise ValueError("dataset_id must be non-empty")
        if not self.method_id:
            raise ValueError("method_id must be non-empty")
        if not metric_ids:
            raise ValueError("metric_values must contain at least one metric")
        if len(metric_ids) != len(set(metric_ids)):
            raise ValueError("metric identifiers must be unique")
        if any(not metric_id for metric_id in metric_ids):
            raise ValueError("metric identifiers must be non-empty")

        unit_ids = [str(value) for value in self.per_unit_ids]
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("per-unit identifiers must be unique")
        if self.per_unit_metric_values is None:
            if unit_ids:
                raise ValueError("per-unit identifiers require per-unit values")
        else:
            values = np.asarray(self.per_unit_metric_values, dtype=float)
            expected_shape = (len(metric_ids), len(unit_ids))
            if values.shape != expected_shape:
                raise ValueError(
                    "per_unit_metric_values must have shape "
                    f"{expected_shape}, got {values.shape}"
                )

        conflicts = sorted(_RESERVED_UNS_KEYS.intersection(self.details))
        if conflicts:
            raise ValueError(
                "details cannot replace required score fields: " + ", ".join(conflicts)
            )


def to_score_anndata(result: MetricResult) -> ad.AnnData:
    """Convert a validated metric result to the repository score contract."""

    result.validate()
    metric_ids = list(result.metric_values)
    uns: dict[str, Any] = {
        "dataset_id": result.dataset_id,
        "method_id": result.method_id,
        "metric_ids": np.asarray(metric_ids, dtype=str),
        "metric_values": np.asarray(
            [result.metric_values[metric_id] for metric_id in metric_ids], dtype=float
        ),
        "provenance_json": json.dumps(result.provenance, sort_keys=True),
    }
    if result.per_unit_metric_values is not None:
        per_unit_values = np.asarray(result.per_unit_metric_values, dtype=float)
        uns.update(
            {
                "per_unit_ids": np.asarray(result.per_unit_ids, dtype=str),
                "per_unit_metric_values": per_unit_values,
                "metric_scored_unit_counts": np.isfinite(per_unit_values).sum(axis=1),
            }
        )
    uns.update(result.details)
    return ad.AnnData(shape=(0, 0), uns=uns)


def write_metric_result(result: MetricResult, output_path: str | Path) -> None:
    """Validate and write one score artifact."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    to_score_anndata(result).write_h5ad(output_path, compression="gzip")


__all__ = ["MetricResult", "to_score_anndata", "write_metric_result"]
