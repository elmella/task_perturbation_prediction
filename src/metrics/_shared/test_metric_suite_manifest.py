"""Check that the scientific metric registry matches Viash suite configs."""

from __future__ import annotations

import unittest
from collections import defaultdict
from pathlib import Path

import yaml


METRICS_ROOT = Path(__file__).parents[1]


def _registered_metrics(manifest: dict) -> list[dict]:
    return [
        *manifest["active_historical_metrics"],
        *manifest["paper_model_metrics"]["metrics"],
        *manifest["single_cell_model_metrics"]["metrics"],
    ]


class MetricSuiteManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = yaml.safe_load((METRICS_ROOT / "metric_suite.yaml").read_text())
        cls.metrics = _registered_metrics(cls.manifest)

    def test_metric_ids_are_unique(self) -> None:
        metric_ids = [metric["id"] for metric in self.metrics]
        self.assertEqual(len(metric_ids), len(set(metric_ids)))

    def test_component_metric_ids_and_directions_match(self) -> None:
        expected: defaultdict[str, dict[str, str]] = defaultdict(dict)
        default_single_cell_path = self.manifest["single_cell_model_metrics"][
            "component_path"
        ]
        for metric in self.metrics:
            component_path = metric.get("component_path", default_single_cell_path)
            expected[component_path][metric["id"]] = metric["direction"]

        for component_path, registered in expected.items():
            config_path = METRICS_ROOT / component_path / "config.vsh.yaml"
            self.assertTrue(config_path.is_file(), component_path)
            config = yaml.safe_load(config_path.read_text())
            configured = {
                metric["name"]: "maximize" if metric["maximize"] else "minimize"
                for metric in config["info"]["metrics"]
            }
            self.assertEqual(configured, registered, component_path)

    def test_aliases_resolve_to_registered_metrics(self) -> None:
        metric_ids = {metric["id"] for metric in self.metrics}
        for metric in self.metrics:
            if "alias_of" in metric:
                self.assertIn(metric["alias_of"], metric_ids)
                self.assertEqual(metric.get("role"), "diagnostic_alias")


if __name__ == "__main__":
    unittest.main()
