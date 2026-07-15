from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from rsr.metrics import score_prediction_mask
from rsr.segmentation import predict_object_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate cached FreeGrasp segmentation without calling an API.")
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--point-x", required=True, type=int)
    parser.add_argument("--point-y", required=True, type=int)
    parser.add_argument("--iou-threshold", default=0.5, type=float)
    args = parser.parse_args()

    # utils.config constructs an OpenAI client while importing LangSAM, but this
    # segmentation-only utility never sends an API request.
    os.environ.setdefault("OPENAI_API_KEY", "unused-for-segmentation-only")

    scene_dir = Path(args.scene_dir).resolve()
    metadata = json.loads((scene_dir / "metadata.json").read_text())
    mask, segmentation = predict_object_mask(
        str(scene_dir / "image.png"),
        args.class_name,
        (args.point_x, args.point_y),
    )
    metrics = score_prediction_mask(
        mask,
        np.load(scene_dir / "instances_objects.npy"),
        metadata.get("ground_truth_object_ids"),
        threshold=args.iou_threshold,
    )
    print(
        json.dumps(
            {
                "scene_id": metadata.get("scene_id"),
                "class_name": args.class_name,
                "point": [args.point_x, args.point_y],
                "mask_shape": list(mask.shape),
                "mask_pixels": int(mask.sum()),
                "segmentation": segmentation,
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
