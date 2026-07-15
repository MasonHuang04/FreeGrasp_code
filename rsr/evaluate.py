from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .common import INPUT_ROOT, OUTPUT_ROOT, read_json, write_json
from .metrics import DEFAULT_IOU_THRESHOLD, score_prediction_mask


def evaluate_saved_masks(input_root: Path, output_root: Path, threshold: float) -> dict[str, object]:
    result_paths = sorted((output_root / "reason").glob("*/*/scene_*/split_*.json"))
    counts = {"evaluated": 0, "missing_mask": 0}
    for result_path in result_paths:
        record = read_json(result_path)
        scene_dir = input_root / record["testcase"] / f"scene_{record['scene_id']}"
        metadata = read_json(scene_dir / "metadata.json")
        instances = np.load(scene_dir / "instances_objects.npy")
        mask_value = record.get("predicted_mask")
        mask_path = Path(mask_value) if mask_value else result_path.with_name("predicted_mask.png")
        predicted_mask = None
        if mask_path.exists():
            predicted_mask = np.asarray(Image.open(mask_path).convert("L")) > 127
        else:
            counts["missing_mask"] += 1
        metrics = score_prediction_mask(
            predicted_mask,
            instances,
            metadata.get("ground_truth_object_ids"),
            threshold=threshold,
        )
        record.update(metrics)
        record["rsr_success"] = metrics["rsr"]
        record["ground_truth_compared"] = True
        record["metric_status"] = "ok" if predicted_mask is not None else "missing_mask_counted_as_zero"
        write_json(result_path, record)
        counts["evaluated"] += 1

    from .run import write_reports

    report = write_reports(output_root)
    summary = {
        **counts,
        "iou_threshold": threshold,
        "threshold_operator": ">",
        "definition": "SSR=max IoU; RSR=1 if IoU>threshold else 0",
        "report": report,
    }
    write_json(output_root / "reports" / "metric_recompute_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute revised SSR/RSR from saved masks.")
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    args = parser.parse_args()
    print(json.dumps(
        evaluate_saved_masks(args.input_root.resolve(), args.output_root.resolve(), args.iou_threshold),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()

