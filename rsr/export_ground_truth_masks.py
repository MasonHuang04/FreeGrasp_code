from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .common import DATA_ROOT, write_json
from .metrics import parse_object_ids
from .prepare_inputs import build_npz_index, read_cases, read_selected_parquet


def _instances_array(npz: np.lib.npyio.NpzFile) -> np.ndarray:
    for key in ("instances_objects", "instances_objects.npy"):
        if key in npz.files:
            value = np.asarray(npz[key])
            if value.ndim != 2:
                raise ValueError(f"Expected a 2D instance map, got {value.shape}")
            return value
    raise KeyError(f"instances_objects is missing from {npz.files}")


def export_instance_masks(
    instances: np.ndarray,
    mask_dir: Path,
) -> list[dict[str, Any]]:
    """Losslessly split every non-background NPZ label into a GT PNG."""
    if instances.ndim != 2:
        raise ValueError(f"Expected a 2D instance map, got {instances.shape}")
    mask_dir.mkdir(parents=True, exist_ok=True)
    records = []
    labels = sorted(int(value) for value in np.unique(instances) if int(value) > 0)
    for npz_label in labels:
        mask = instances == npz_label
        path = mask_dir / f"mask_{npz_label:03d}_gt.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(path, format="PNG")
        records.append({
            "object_id": npz_label,
            "npz_instance_label": npz_label,
            "dataset_object_id": npz_label - 1,
            "mask_pixels": int(mask.sum()),
            "path": str(path.resolve()),
        })
    return records


def export_from_archives(
    source_root: Path,
    output_root: Path,
    scene_ids: set[int] | None = None,
) -> dict[str, Any]:
    cases_path = source_root / "rsr_case.csv"
    parquet_paths = [
        source_root / f"train-{index:05d}-of-00002.parquet"
        for index in range(2)
    ]
    npz_zip_path = source_root / "npz_file.zip"
    for path in [cases_path, *parquet_paths, npz_zip_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    cases, case_stats = read_cases(cases_path)
    if scene_ids:
        cases = [case for case in cases if int(case["scene_id"]) in scene_ids]
    selected_rows = read_selected_parquet(parquet_paths, cases)
    npz_index = build_npz_index(npz_zip_path)

    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for case in cases:
        groups.setdefault(
            (case["testcase"], int(case["scene_id"]), int(case["query_obj_id"])),
            [],
        ).append(case)

    exported = []
    with zipfile.ZipFile(npz_zip_path) as archive:
        for (testcase, scene_id, query_obj_id), group in sorted(groups.items()):
            member = npz_index.get(scene_id)
            if member is None:
                raise FileNotFoundError(f"No NPZ member for scene {scene_id}")
            with np.load(io.BytesIO(archive.read(member)), allow_pickle=True) as npz:
                instances = _instances_array(npz)

            rows = [
                selected_rows[(scene_id, query_obj_id, int(case["split"]))]
                for case in group
            ]
            gt_values = {str(row["groundTruthObjIds"]) for row in rows}
            if len(gt_values) != 1:
                raise RuntimeError(
                    f"Inconsistent groundTruthObjIds for scene {scene_id}: {gt_values}"
                )
            dataset_gt_ids = parse_object_ids(next(iter(gt_values)))

            gt_root = output_root / testcase / f"scene_{scene_id}" / "gt"
            mask_records = export_instance_masks(instances, gt_root / "mask")
            summary = {
                "schema_version": 1,
                "scene_id": scene_id,
                "query_obj_id": query_obj_id,
                "testcase": testcase,
                "source_npz_archive": str(npz_zip_path.resolve()),
                "source_npz_member": member,
                "source_parquet_shards": [str(path.resolve()) for path in parquet_paths],
                "ground_truth_dataset_object_ids": dataset_gt_ids,
                "ground_truth_npz_labels": [value + 1 for value in dataset_gt_ids],
                "id_mapping": "dataset_object_id = npz_instance_label - 1",
                "export_policy": (
                    "all non-background masks extracted directly from copied NPZ; "
                    "no model and no metric"
                ),
                "metric_computed": False,
                "masks": mask_records,
            }
            write_json(gt_root / "summary.json", summary)
            exported.append(summary)

    manifest = {
        "schema_version": 1,
        "source_root": str(source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "source_files": [str(path.resolve()) for path in [*parquet_paths, npz_zip_path]],
        "num_scenes": len(exported),
        "num_masks": sum(len(scene["masks"]) for scene in exported),
        "metric_computed": False,
        "case_stats": case_stats,
        "scenes": [{
            "testcase": scene["testcase"],
            "scene_id": scene["scene_id"],
            "summary": str(
                (
                    output_root
                    / scene["testcase"]
                    / f"scene_{scene['scene_id']}"
                    / "gt"
                    / "summary.json"
                ).resolve()
            ),
            "num_masks": len(scene["masks"]),
        } for scene in exported],
    }
    write_json(output_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract standard mask_NNN_gt.png files directly from the copied "
            "FreeGraspData NPZ archive without running a model or metric."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DATA_ROOT / "source",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DATA_ROOT / "ground_truth_masks",
    )
    parser.add_argument("--scene-id", type=int, action="append", default=None)
    args = parser.parse_args()
    print(json.dumps(
        export_from_archives(
            args.source_root.resolve(),
            args.output_root.resolve(),
            set(args.scene_id or []) or None,
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
