from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


class SegmentationInfrastructureError(RuntimeError):
    """The segmentation backend could not be loaded in this environment."""


def _mask_numpy(mask: Any) -> np.ndarray:
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    result = np.squeeze(np.asarray(mask)).astype(bool)
    if result.ndim != 2:
        raise ValueError(f"Expected a 2D predicted mask, got {result.shape}")
    return result


def _select_mask(masks: Any, logits: Any, point: tuple[int, int] | None) -> tuple[np.ndarray, int]:
    if len(masks) == 0:
        raise RuntimeError("LangSAM returned no masks")
    values = [_mask_numpy(mask) for mask in masks]
    if point is None:
        scores = logits.detach().cpu().numpy() if hasattr(logits, "detach") else np.asarray(logits)
        index = int(np.argmax(scores))
        return values[index], index

    x, y = point
    best_index = None
    best_distance = float("inf")
    for index, mask in enumerate(values):
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
            return mask, index
        coordinates = np.column_stack(np.where(mask))
        if not len(coordinates):
            continue
        distance = float(np.linalg.norm(coordinates - np.array([y, x]), axis=1).min())
        if distance < best_distance:
            best_distance = distance
            best_index = index
    if best_index is None:
        raise RuntimeError("LangSAM masks contain no foreground pixels")
    return values[best_index], best_index


def predict_object_mask(
    image_path: str,
    class_name: str,
    point: tuple[int, int] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use the unmodified FreeGrasp LangSAM actor and its selection policy."""
    try:
        from utils.config import langsam_actor
    except Exception as exc:
        raise SegmentationInfrastructureError(
            "FreeGrasp LangSAM is unavailable. Install GroundingDINO and its weights first."
        ) from exc

    image = Image.open(image_path).convert("RGB")
    masks, _boxes, phrases, logits = langsam_actor.predict(image, class_name)
    selected_mask, index = _select_mask(masks, logits, point)
    logit = None
    try:
        value = logits[index]
        logit = float(value.detach().cpu().item() if hasattr(value, "detach") else value)
    except Exception:
        pass
    return selected_mask, {
        "selected_mask_index": index,
        "phrase": str(phrases[index]) if index < len(phrases) else None,
        "logit": logit,
        "selection_method": "contains_or_nearest_point" if point is not None else "highest_logit",
    }
