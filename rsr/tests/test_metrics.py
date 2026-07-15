from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from rsr.metrics import compute_iou, score_prediction_mask
from rsr.common import write_json
from rsr.run import write_reports


class MetricTests(unittest.TestCase):
    def test_iou_exactly_half_is_rsr_zero(self) -> None:
        instances = np.array([[1, 1], [0, 0]], dtype=np.int32)
        predicted = np.array([[1, 0], [0, 0]], dtype=bool)
        score = score_prediction_mask(predicted, instances, [0], threshold=0.5)
        self.assertEqual(score["ssr"], 0.5)
        self.assertEqual(score["rsr"], 0)

    def test_greater_than_half_is_rsr_one(self) -> None:
        instances = np.array([[1, 1], [1, 0]], dtype=np.int32)
        predicted = np.array([[1, 1], [0, 0]], dtype=bool)
        score = score_prediction_mask(predicted, instances, "0", threshold=0.5)
        self.assertAlmostEqual(score["ssr"], 2 / 3)
        self.assertEqual(score["rsr"], 1)

    def test_uses_best_of_multiple_ground_truth_ids(self) -> None:
        instances = np.array([[1, 1], [2, 2]], dtype=np.int32)
        predicted = np.array([[0, 0], [1, 1]], dtype=bool)
        score = score_prediction_mask(predicted, instances, "0,1")
        self.assertEqual(score["ssr"], 1.0)
        self.assertEqual(score["rsr"], 1)
        self.assertEqual(score["best_ground_truth_object_id"], 1)

    def test_algorithm_failure_mask_is_zero_and_includable(self) -> None:
        instances = np.array([[1, 1], [0, 0]], dtype=np.int32)
        score = score_prediction_mask(None, instances, [0])
        self.assertEqual(score["ssr"], 0.0)
        self.assertEqual(score["rsr"], 0)

    def test_compute_iou_empty_union_is_zero(self) -> None:
        empty = np.zeros((2, 2), dtype=bool)
        self.assertEqual(compute_iou(empty, empty), 0.0)

    def test_report_excludes_api_failure_from_denominator(self) -> None:
        with TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            record_root = output_root / "reason" / "gt" / "testcase_easy" / "scene_1"
            common = {
                "testcase": "testcase_easy",
                "scene_id": 1,
                "localization_mode": "gt",
                "status": "ok",
            }
            write_json(record_root / "split_0.json", {
                **common, "split": 0, "ssr": 0.75, "rsr": 1,
            })
            write_json(record_root / "split_1.json", {
                **common,
                "split": 1,
                "status": "segmentation_failure",
                "ssr": 0.0,
                "rsr": 0,
            })
            write_json(output_root / "run_failures.json", {"failures": [{
                "stage": "reason",
                "failure_type": "api_or_transport_failure",
                "excluded_from_statistics": True,
                "ssr": None,
                "rsr": None,
            }, {
                "stage": "reason",
                "failure_type": "segmentation_infrastructure_failure",
                "excluded_from_statistics": True,
                "ssr": None,
                "rsr": None,
            }]})
            report = write_reports(output_root)
            self.assertEqual(report["num_evaluated"], 2)
            self.assertEqual(report["num_api_failures_excluded_current_run"], 1)
            self.assertEqual(report["num_infrastructure_failures_excluded_current_run"], 1)
            self.assertEqual(report["overall"]["count"], 2)
            self.assertEqual(report["overall"]["mean_ssr"], 0.375)
            self.assertEqual(report["overall"]["mean_rsr"], 0.5)

    def test_report_can_limit_results_to_molmo_mode(self) -> None:
        with TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            for mode in ("gt", "molmo"):
                write_json(
                    output_root
                    / "reason"
                    / mode
                    / "01_hard_ambiguous"
                    / "scene_815"
                    / "split_0.json",
                    {
                        "testcase": "01_hard_ambiguous",
                        "scene_id": 815,
                        "split": 0,
                        "localization_mode": mode,
                        "status": "ok",
                        "ssr": 1.0,
                        "rsr": 1,
                    },
                )
            report = write_reports(output_root, localization_modes={"molmo"})
            self.assertEqual(report["num_predictions"], 1)
            self.assertEqual(report["localization_mode_counts"], {"molmo": 1})
            self.assertEqual(report["by_localization_mode"]["gt"]["count"], 0)


if __name__ == "__main__":
    unittest.main()
