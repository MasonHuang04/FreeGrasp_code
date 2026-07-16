from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from rsr.metrics import (
    compute_iou,
    save_ground_truth_masks,
    score_prediction_against_gt_files,
    score_prediction_mask,
)
from rsr.common import write_json
from rsr.evaluate import evaluate_saved_masks
from rsr.run import write_reports


class MetricTests(unittest.TestCase):
    def test_scores_extracted_gt_id_plus_one_file(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            predicted = np.array([[1, 0], [0, 0]], dtype=bool)
            gt = np.array([[1, 1], [0, 0]], dtype=np.uint8) * 255
            Image.fromarray(gt).save(root / "mask_001_gt.png")
            score = score_prediction_against_gt_files(
                predicted, [0], root, threshold=0.5
            )
            self.assertEqual(score["ssr"], 0.5)
            self.assertEqual(score["rsr"], 0)
            self.assertEqual(
                Path(score["compared_ground_truth_masks"][0]["path"]).name,
                "mask_001_gt.png",
            )

    def test_saved_gt_mask_uses_dataset_id_plus_one_npz_label(self) -> None:
        instances = np.array([[0, 1], [2, 2]], dtype=np.int32)
        with TemporaryDirectory() as temporary:
            records = save_ground_truth_masks(
                instances, [0], Path(temporary), "split_0"
            )
            saved = np.asarray(Image.open(records[0]["path"]).convert("L")) > 0
            np.testing.assert_array_equal(saved, instances == 1)
            self.assertEqual(records[0]["dataset_object_id"], 0)
            self.assertEqual(records[0]["npz_instance_label"], 1)

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

    def test_report_excludes_missing_mask_from_denominator(self) -> None:
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
                "ssr": None,
                "rsr": None,
                "excluded_from_statistics": True,
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
            self.assertEqual(report["num_evaluated"], 1)
            self.assertEqual(report["num_api_failures_excluded_current_run"], 1)
            self.assertEqual(report["num_infrastructure_failures_excluded_current_run"], 1)
            self.assertEqual(report["overall"]["count"], 1)
            self.assertEqual(report["overall"]["mean_ssr"], 0.75)
            self.assertEqual(report["overall"]["mean_rsr"], 1.0)

    def test_offline_evaluate_excludes_missing_selected_id_and_mask(self) -> None:
        with TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            input_root = data_root / "input"
            output_root = data_root / "output"
            scene_root = input_root / "01_hard_ambiguous" / "scene_815"
            gt_root = (
                data_root / "ground_truth_masks" / "01_hard_ambiguous"
                / "scene_815" / "gt"
            )
            write_json(scene_root / "metadata.json", {
                "testcase": "01_hard_ambiguous",
                "scene_id": 815,
                "ground_truth_object_ids": "0",
            })
            write_json(gt_root / "summary.json", {"masks": []})
            result_path = (
                output_root / "reason" / "molmo" / "01_hard_ambiguous"
                / "scene_815" / "split_0.json"
            )
            write_json(result_path, {
                "testcase": "01_hard_ambiguous",
                "scene_id": 815,
                "split": 0,
                "localization_mode": "molmo",
                "status": "unparsed_response",
                "predicted_localization_id": None,
                "predicted_object_id": None,
                "predicted_mask": None,
                "ssr": 0.0,
                "rsr": 0,
            })
            summary = evaluate_saved_masks(
                input_root,
                output_root,
                0.5,
                testcases={"01_hard_ambiguous"},
                localization_modes={"molmo"},
            )
            updated = __import__("json").loads(result_path.read_text())
            self.assertIsNone(updated["ssr"])
            self.assertIsNone(updated["rsr"])
            self.assertTrue(updated["excluded_from_statistics"])
            self.assertEqual(summary["evaluated"], 0)
            self.assertEqual(summary["excluded_missing_selected_object_or_mask"], 1)

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

    def test_report_marks_manual_mask_records_as_not_computed(self) -> None:
        with TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            write_json(
                output_root
                / "reason"
                / "molmo"
                / "01_hard_ambiguous"
                / "scene_815"
                / "split_0.json",
                {
                    "testcase": "01_hard_ambiguous",
                    "scene_id": 815,
                    "split": 0,
                    "localization_mode": "molmo",
                    "status": "ok",
                    "ssr": None,
                    "rsr": None,
                    "ground_truth_compared": False,
                },
            )
            report = write_reports(output_root, localization_modes={"molmo"})
            self.assertEqual(report["num_predictions"], 1)
            self.assertEqual(report["num_evaluated"], 0)
            self.assertFalse(report["rsr_is_computed"])
            self.assertFalse(report["ground_truth_compared"])


if __name__ == "__main__":
    unittest.main()
