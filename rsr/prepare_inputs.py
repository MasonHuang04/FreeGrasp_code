from __future__ import annotations

import argparse
import csv
import hashlib
import io
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .common import INPUT_ROOT, PROJECT_ROOT, SOURCE_ROOT, parse_bool, write_json


CASE_FIELDS = (
    "testcase",
    "scene_id",
    "query_obj_id",
    "difficulty",
    "ambiguous",
    "split",
)
PARQUET_FIELDS = (
    "sceneId",
    "queryObjId",
    "annotation",
    "groundTruthObjIds",
    "difficulty",
    "ambiguious",
    "split",
    "image",
)


def _image_bytes(value: Any) -> bytes:
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return bytes(value["bytes"])
        if value.get("path"):
            return Path(value["path"]).read_bytes()
    raise ValueError(f"Unsupported parquet image field: {type(value)!r}")


def read_cases(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        raw_rows = list(csv.DictReader(handle))

    normalized: list[dict[str, Any]] = []
    for raw in raw_rows:
        missing = [field for field in CASE_FIELDS if not str(raw.get(field, "")).strip()]
        if missing:
            raise ValueError(f"Missing {missing} in rsr_case.csv row: {raw}")
        normalized.append({
            "testcase": str(raw["testcase"]).strip(),
            "scene_id": int(raw["scene_id"]),
            "query_obj_id": int(raw["query_obj_id"]),
            "difficulty": str(raw["difficulty"]).strip(),
            "ambiguous": parse_bool(raw["ambiguous"]),
            "split": int(raw["split"]),
        })

    key = lambda row: tuple(row[field] for field in CASE_FIELDS)
    multiplicities = Counter(key(row) for row in normalized)
    unique = [dict(zip(CASE_FIELDS, values)) for values in sorted(multiplicities)]
    stats = {
        "source_rows": len(raw_rows),
        "unique_annotation_cases": len(unique),
        "duplicate_rows_removed": len(raw_rows) - len(unique),
        "multiplicity_histogram": {
            str(count): sum(1 for value in multiplicities.values() if value == count)
            for count in sorted(set(multiplicities.values()))
        },
    }
    return unique, stats


def read_selected_parquet(
    parquet_paths: list[Path],
    cases: list[dict[str, Any]],
) -> dict[tuple[int, int, int], dict[str, Any]]:
    wanted = {(row["scene_id"], row["query_obj_id"], row["split"]) for row in cases}
    scene_ids = sorted({row["scene_id"] for row in cases})
    selected: dict[tuple[int, int, int], dict[str, Any]] = {}
    for path in parquet_paths:
        frame = pd.read_parquet(
            path,
            columns=list(PARQUET_FIELDS),
            filters=[("sceneId", "in", scene_ids)],
        )
        for source_row, row in enumerate(frame.to_dict(orient="records")):
            row_key = (int(row["sceneId"]), int(row["queryObjId"]), int(row["split"]))
            if row_key not in wanted:
                continue
            if row_key in selected:
                raise RuntimeError(f"Duplicate parquet row for {row_key}")
            row["source_parquet"] = str(path.resolve())
            row["source_shard"] = path.name
            row["source_filtered_row"] = source_row
            selected[row_key] = row
    missing = sorted(wanted - set(selected))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} selected parquet rows; first entries: {missing[:10]}")
    return selected


def build_npz_index(path: Path) -> dict[int, str]:
    with zipfile.ZipFile(path) as archive:
        index = {
            int(Path(member).stem): member
            for member in archive.namelist()
            if member.endswith(".npz") and Path(member).stem.isdigit()
        }
    if not index:
        raise RuntimeError(f"No numeric NPZ members found in {path}")
    return index


def _npz_array(npz: np.lib.npyio.NpzFile, *names: str) -> np.ndarray:
    for name in names:
        if name in npz.files:
            return np.asarray(npz[name])
    raise KeyError(f"None of {names} found in {npz.files}")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    project_root = args.project_root.resolve()
    input_root = args.input_root.resolve()
    source_root = args.source_root.resolve()
    cases_path = args.cases.resolve()
    parquet_paths = [project_root / f"train-{index:05d}-of-00002.parquet" for index in range(2)]
    npz_zip = project_root / "npz_file.zip"
    for path in [cases_path, *parquet_paths, npz_zip]:
        if not path.exists():
            raise FileNotFoundError(path)

    cases, case_stats = read_cases(cases_path)
    selected_rows = read_selected_parquet(parquet_paths, cases)
    npz_index = build_npz_index(npz_zip)

    if input_root.exists():
        if not args.force:
            raise FileExistsError(f"{input_root} already exists; pass --force to replace it")
        shutil.rmtree(input_root)
    input_root.mkdir(parents=True)
    source_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cases_path, source_root / "rsr_case.csv")

    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for case in cases:
        groups.setdefault(
            (case["testcase"], case["scene_id"], case["query_obj_id"]),
            [],
        ).append(case)

    exported: list[dict[str, Any]] = []
    with zipfile.ZipFile(npz_zip) as archive:
        for (testcase, scene_id, query_obj_id), group in sorted(groups.items()):
            if sorted(item["split"] for item in group) != [0, 1, 2]:
                raise RuntimeError(
                    f"Expected splits 0,1,2 for {(testcase, scene_id, query_obj_id)}, got {group}"
                )
            rows = [selected_rows[(scene_id, query_obj_id, item["split"])] for item in group]
            case0 = group[0]
            for row in rows:
                if str(row["difficulty"]) != case0["difficulty"]:
                    raise RuntimeError(f"Difficulty mismatch for scene {scene_id}")
                if bool(row["ambiguious"]) != bool(case0["ambiguous"]):
                    raise RuntimeError(f"Ambiguity mismatch for scene {scene_id}")

            images = [_image_bytes(row["image"]) for row in rows]
            image_hashes = {hashlib.sha256(value).hexdigest() for value in images}
            if len(image_hashes) != 1:
                raise RuntimeError(f"The three splits have different images for scene {scene_id}")

            member = npz_index.get(scene_id)
            if member is None:
                raise FileNotFoundError(f"No NPZ member for scene {scene_id}")
            npz_bytes = archive.read(member)

            scene_dir = input_root / testcase / f"scene_{scene_id}"
            scene_dir.mkdir(parents=True, exist_ok=False)
            Image.open(io.BytesIO(images[0])).convert("RGB").save(scene_dir / "image.png")
            (scene_dir / "source.npz").write_bytes(npz_bytes)
            with np.load(io.BytesIO(npz_bytes), allow_pickle=True) as npz:
                instances = _npz_array(npz, "instances_objects", "instances_objects.npy")
            np.save(scene_dir / "instances_objects.npy", instances.astype(np.int32))

            annotations = []
            for case in sorted(group, key=lambda item: item["split"]):
                row = selected_rows[(scene_id, query_obj_id, case["split"])]
                annotations.append({
                    "split": case["split"],
                    "annotation": str(row["annotation"]).strip(),
                    "source_parquet": row["source_parquet"],
                    "source_shard": row["source_shard"],
                    "source_filtered_row": int(row["source_filtered_row"]),
                })
            metadata = {
                "schema_version": 1,
                "testcase": testcase,
                "scene_id": scene_id,
                "query_obj_id": query_obj_id,
                "difficulty": case0["difficulty"],
                "ambiguous": bool(case0["ambiguous"]),
                "ground_truth_object_ids": str(rows[0]["groundTruthObjIds"]),
                "id_convention": {
                    "parquet_object_ids": "zero_based",
                    "npz_instance_labels": "one_based_with_zero_as_background",
                    "predicted_object_id": "npz_label_minus_one",
                },
                "source_npz_member": member,
                "image": "image.png",
                "instances_objects": "instances_objects.npy",
                "annotations": annotations,
            }
            write_json(scene_dir / "metadata.json", metadata)
            exported.append(metadata)

    testcases = []
    for testcase in sorted({item["testcase"] for item in exported}):
        subset = [item for item in exported if item["testcase"] == testcase]
        testcases.append({
            "name": testcase,
            "difficulty": subset[0]["difficulty"],
            "ambiguous": subset[0]["ambiguous"],
            "num_scenes": len(subset),
            "scene_ids": [item["scene_id"] for item in subset],
        })
    manifest = {
        "schema_version": 1,
        "source_cases": str(cases_path),
        "source_case_snapshot": str((source_root / "rsr_case.csv").resolve()),
        **case_stats,
        "num_scenes": len(exported),
        "num_annotation_runs": sum(len(item["annotations"]) for item in exported),
        "testcases": testcases,
        "rsr_is_computed": False,
    }
    write_json(input_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare only the FreeGrasp scenes listed by rsr_case.csv."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--cases", type=Path, default=PROJECT_ROOT / "rsr_case.csv")
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(__import__("json").dumps(prepare(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

