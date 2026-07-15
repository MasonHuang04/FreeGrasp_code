from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_IOU_THRESHOLD = 0.5


def parse_object_ids(value: object) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return sorted({int(item) for item in value})
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return sorted({int(item.strip()) for item in text.split(",") if item.strip()})


def _binary_mask(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    mask = np.squeeze(np.asarray(value))
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {mask.shape}")
    return mask.astype(bool)


def compute_iou(predicted_mask: Any, ground_truth_mask: Any) -> float:
    predicted = _binary_mask(predicted_mask)
    ground_truth = _binary_mask(ground_truth_mask)
    if predicted.shape != ground_truth.shape:
        raise ValueError(
            f"Mask shape mismatch: predicted={predicted.shape}, ground_truth={ground_truth.shape}"
        )
    intersection = int(np.logical_and(predicted, ground_truth).sum())
    union = int(np.logical_or(predicted, ground_truth).sum())
    return float(intersection / union) if union else 0.0


def score_prediction_mask(
    predicted_mask: Any | None,
    instances_objects: Any,
    ground_truth_object_ids: object,
    *,
    threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict[str, Any]:
    """Revised metrics: SSR=max IoU and RSR=1[SSR>threshold]."""
    instances = np.asarray(instances_objects)
    if instances.ndim != 2:
        raise ValueError(f"Expected a 2D instance map, got shape {instances.shape}")
    dataset_ids = parse_object_ids(ground_truth_object_ids)
    predicted = None if predicted_mask is None else _binary_mask(predicted_mask)
    if predicted is not None and predicted.shape != instances.shape:
        raise ValueError(
            f"Mask shape mismatch: predicted={predicted.shape}, instances={instances.shape}"
        )

    per_ground_truth_iou: dict[str, float] = {}
    best_iou = 0.0
    best_dataset_id = None
    for dataset_id in dataset_ids:
        ground_truth = instances == (int(dataset_id) + 1)
        iou = 0.0 if predicted is None else compute_iou(predicted, ground_truth)
        per_ground_truth_iou[str(dataset_id)] = iou
        if best_dataset_id is None or iou > best_iou:
            best_iou = iou
            best_dataset_id = int(dataset_id)

    return {
        "iou": best_iou,
        "ssr": best_iou,
        "rsr": int(best_iou > threshold),
        "iou_threshold": float(threshold),
        "threshold_operator": ">",
        "best_ground_truth_object_id": best_dataset_id,
        "per_ground_truth_iou": per_ground_truth_iou,
        "metric_definition": "ssr = max_gt IoU; rsr = 1 if ssr > threshold else 0",
    }

