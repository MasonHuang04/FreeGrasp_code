from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


RSR_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RSR_ROOT.parent
DATA_ROOT = RSR_ROOT / "data"
INPUT_ROOT = DATA_ROOT / "input"
OUTPUT_ROOT = DATA_ROOT / "output"
SOURCE_ROOT = DATA_ROOT / "source"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_bool(value: object) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def parse_freegrasp_response(output: str, fallback_text: str) -> dict[str, Any]:
    """Parse the two response formats accepted by FreeGrasp's source code."""
    first = re.search(r"\[(\d+),\s*(.+?)\]", output)
    second = re.search(r"\[pick object,\s*(\d+),\s*(.+?)\]", output, re.IGNORECASE)
    match = first or second
    if match is None:
        return {
            "selected_object_id": None,
            "class_name": fallback_text,
            "parsed": False,
        }
    return {
        "selected_object_id": int(match.group(1)),
        "class_name": match.group(2).strip().lower(),
        "parsed": True,
    }


def point_to_dataset_id(
    instances_objects: Any,
    x: int,
    y: int,
) -> dict[str, int | None]:
    """Map a Molmo point to FreeGraspData's zero-based object ID."""
    height, width = instances_objects.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        return {"npz_label": None, "dataset_object_id": None}
    npz_label = int(instances_objects[y, x])
    if npz_label <= 0:
        return {"npz_label": npz_label, "dataset_object_id": None}
    return {"npz_label": npz_label, "dataset_object_id": npz_label - 1}

