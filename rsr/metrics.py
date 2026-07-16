from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_IOU_THRESHOLD = 0.5


class GroundTruthMaskError(RuntimeError):
    """A required extracted GT mask file is missing or invalid."""


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


def extract_ground_truth_masks(
    instances_objects: Any,
    ground_truth_object_ids: object,
) -> dict[int, np.ndarray]:
    """Extract exact GT masks using zero-based dataset ID + 1 NPZ label."""
    instances = np.asarray(instances_objects)
    if instances.ndim != 2:
        raise ValueError(f"Expected a 2D instance map, got shape {instances.shape}")
    return {
        dataset_id: instances == (dataset_id + 1)
        for dataset_id in parse_object_ids(ground_truth_object_ids)
    }


def save_ground_truth_masks(
    instances_objects: Any,
    ground_truth_object_ids: object,
    output_dir: Path,
    prefix: str,
) -> list[dict[str, Any]]:
    """Persist the exact binary GT masks used by the IoU calculation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for dataset_id, mask in extract_ground_truth_masks(
        instances_objects, ground_truth_object_ids
    ).items():
        path = output_dir / f"{prefix}_ground_truth_mask_object_{dataset_id}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(path, format="PNG")
        records.append({
            "dataset_object_id": dataset_id,
            "npz_instance_label": dataset_id + 1,
            "mask_pixels": int(mask.sum()),
            "path": str(path.resolve()),
        })
    return records


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
    predicted = None if predicted_mask is None else _binary_mask(predicted_mask)
    if predicted is not None and predicted.shape != instances.shape:
        raise ValueError(
            f"Mask shape mismatch: predicted={predicted.shape}, instances={instances.shape}"
        )

    per_ground_truth_iou: dict[str, float] = {}
    best_iou = 0.0
    best_dataset_id = None
    for dataset_id, ground_truth in extract_ground_truth_masks(
        instances, ground_truth_object_ids
    ).items():
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


def score_prediction_against_gt_files(
    predicted_mask: Any | None,
    ground_truth_object_ids: object,
    gt_mask_dir: Path,
    *,
    threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict[str, Any]:
    """Compare with extracted mask_(dataset_id+1)_gt.png files."""
    dataset_ids = parse_object_ids(ground_truth_object_ids)
    if not dataset_ids:
        raise GroundTruthMaskError("No ground-truth dataset object IDs")
    predicted = None if predicted_mask is None else _binary_mask(predicted_mask)
    per_ground_truth_iou: dict[str, float] = {}
    compared_masks = []
    best_iou = 0.0
    best_dataset_id = None
    for dataset_id in dataset_ids:
        npz_label = dataset_id + 1
        gt_path = gt_mask_dir / f"mask_{npz_label:03d}_gt.png"
        if not gt_path.exists():
            raise GroundTruthMaskError(
                f"Missing GT mask for dataset ID {dataset_id}: {gt_path}"
            )
        ground_truth = np.asarray(Image.open(gt_path).convert("L")) > 127
        try:
            iou = 0.0 if predicted is None else compute_iou(predicted, ground_truth)
        except ValueError as exc:
            raise GroundTruthMaskError(
                f"Cannot compare predicted mask with {gt_path}: {exc}"
            ) from exc
        per_ground_truth_iou[str(dataset_id)] = iou
        compared_masks.append({
            "dataset_object_id": dataset_id,
            "npz_instance_label": npz_label,
            "path": str(gt_path.resolve()),
            "iou": iou,
        })
        if best_dataset_id is None or iou > best_iou:
            best_iou = iou
            best_dataset_id = dataset_id

    return {
        "iou": best_iou,
        "ssr": best_iou,
        "rsr": int(best_iou > threshold),
        "iou_threshold": float(threshold),
        "threshold_operator": ">",
        "best_ground_truth_object_id": best_dataset_id,
        "per_ground_truth_iou": per_ground_truth_iou,
        "compared_ground_truth_masks": compared_masks,
        "metric_definition": (
            "ssr = max IoU(predicted_mask, mask_(groundTruthId+1)_gt.png); "
            "rsr = 1 if ssr > threshold else 0"
        ),
    }
