from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .common import INPUT_ROOT, OUTPUT_ROOT, read_json, write_json
from .metrics import DEFAULT_IOU_THRESHOLD, score_prediction_against_gt_files


def evaluate_saved_masks(
    input_root: Path,
    output_root: Path,
    threshold: float,
    *,
    testcases: set[str] | None = None,
    scene_ids: set[int] | None = None,
    splits: set[int] | None = None,
    localization_modes: set[str] | None = None,
) -> dict[str, object]:
    """Compare saved masks only; never run localization, API, or segmentation."""
    result_paths = sorted((output_root / "reason").glob("*/*/scene_*/split_*.json"))
    counts = {
        "evaluated": 0,
        "missing_predicted_mask": 0,
        "excluded_missing_selected_object_or_mask": 0,
    }
    comparisons = []
    exclusions = []
    for result_path in result_paths:
        record = read_json(result_path)
        if testcases and record.get("testcase") not in testcases:
            continue
        if scene_ids and int(record.get("scene_id")) not in scene_ids:
            continue
        if splits and int(record.get("split")) not in splits:
            continue
        if localization_modes and record.get("localization_mode") not in localization_modes:
            continue

        scene_dir = input_root / record["testcase"] / f"scene_{record['scene_id']}"
        metadata = read_json(scene_dir / "metadata.json")
        mask_value = record.get("predicted_mask")
        predicted_path = Path(mask_value) if mask_value else None
        predicted_mask = None
        if predicted_path is not None and predicted_path.exists():
            predicted_mask = np.asarray(Image.open(predicted_path).convert("L")) > 127
        else:
            counts["missing_predicted_mask"] += 1

        gt_root = (
            input_root.parent
            / "ground_truth_masks"
            / record["testcase"]
            / f"scene_{record['scene_id']}"
            / "gt"
        )
        summary_path = gt_root / "summary.json"
        gt_summary = read_json(summary_path)
        has_selected_object = (
            record.get("predicted_localization_id") is not None
            and record.get("predicted_object_id") is not None
        )
        if not has_selected_object or predicted_mask is None:
            record.update({
                "iou": None,
                "ssr": None,
                "rsr": None,
                "iou_threshold": float(threshold),
                "threshold_operator": ">",
                "best_ground_truth_object_id": None,
                "per_ground_truth_iou": {},
                "compared_ground_truth_masks": [],
                "metric_definition": (
                    "missing selected object ID or corresponding predicted mask; excluded"
                ),
            })
            record["ground_truth_object_ids"] = metadata.get("ground_truth_object_ids")
            record["ground_truth_mask_manifest"] = str(summary_path.resolve())
            record["ground_truth_masks"] = gt_summary.get("masks", [])
            record["rsr_success"] = None
            record["ground_truth_compared"] = False
            record["excluded_from_statistics"] = True
            record["metric_status"] = "missing_selected_object_or_mask_excluded"
            write_json(result_path, record)
            counts["excluded_missing_selected_object_or_mask"] += 1
            exclusions.append({
                "testcase": record["testcase"],
                "scene_id": int(record["scene_id"]),
                "split": int(record["split"]),
                "localization_mode": record["localization_mode"],
                "status": record.get("status"),
                "reason": "missing_selected_object_or_mask",
                "result_json": str(result_path.resolve()),
            })
            continue

        metrics = score_prediction_against_gt_files(
            predicted_mask,
            metadata.get("ground_truth_object_ids"),
            gt_root / "mask",
            threshold=threshold,
        )
        record.update(metrics)
        record["ground_truth_object_ids"] = metadata.get("ground_truth_object_ids")
        record["ground_truth_mask_manifest"] = str(summary_path.resolve())
        record["ground_truth_masks"] = gt_summary.get("masks", [])
        record["rsr_success"] = metrics["rsr"]
        record["ground_truth_compared"] = True
        record["excluded_from_statistics"] = False
        record["metric_status"] = (
            "ok" if predicted_mask is not None else "missing_mask_counted_as_zero"
        )
        write_json(result_path, record)
        counts["evaluated"] += 1
        comparisons.append({
            "testcase": record["testcase"],
            "scene_id": int(record["scene_id"]),
            "split": int(record["split"]),
            "localization_mode": record["localization_mode"],
            "predicted_mask": str(predicted_path.resolve()),
            "compared_ground_truth_masks": metrics["compared_ground_truth_masks"],
            "ssr": metrics["ssr"],
            "rsr": metrics["rsr"],
            "result_json": str(result_path.resolve()),
        })

    from .run import write_reports

    report = write_reports(output_root, localization_modes=localization_modes)

    def aggregate(selected_comparisons, selected_exclusions):
        count = len(selected_comparisons)
        return {
            "count": count,
            "num_excluded_missing_selected_object_or_mask": len(selected_exclusions),
            "mean_ssr": (
                float(sum(item["ssr"] for item in selected_comparisons) / count)
                if count else None
            ),
            "mean_rsr": (
                float(sum(item["rsr"] for item in selected_comparisons) / count)
                if count else None
            ),
        }

    testcase_names = sorted({item["testcase"] for item in comparisons + exclusions})
    selection_report = {
        "overall": aggregate(comparisons, exclusions),
        "by_testcase": {
            testcase: aggregate(
                [item for item in comparisons if item["testcase"] == testcase],
                [item for item in exclusions if item["testcase"] == testcase],
            )
            for testcase in testcase_names
        },
    }
    summary = {
        **counts,
        "iou_threshold": threshold,
        "threshold_operator": ">",
        "definition": (
            "SSR=max IoU(predicted mask, extracted GT ID+1 mask); "
            "RSR=1 if SSR>threshold else 0"
        ),
        "localization_or_api_called": False,
        "comparisons": comparisons,
        "exclusions": exclusions,
        "selection_report": selection_report,
        "report": report,
    }
    write_json(output_root / "reports" / "metric_recompute_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Offline-only comparison of saved predicted masks with extracted GT PNGs; "
            "does not run Molmo, an API, or LangSAM."
        )
    )
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    parser.add_argument("--testcase", action="append", default=None)
    parser.add_argument("--scene-id", type=int, action="append", default=None)
    parser.add_argument("--split", type=int, action="append", choices=[0, 1, 2], default=None)
    parser.add_argument(
        "--localization-mode",
        action="append",
        choices=["gt", "molmo"],
        default=None,
    )
    args = parser.parse_args()
    print(json.dumps(
        evaluate_saved_masks(
            args.input_root.resolve(),
            args.output_root.resolve(),
            args.iou_threshold,
            testcases=set(args.testcase or []) or None,
            scene_ids=set(args.scene_id or []) or None,
            splits=set(args.split or []) or None,
            localization_modes=set(args.localization_mode or []) or None,
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
