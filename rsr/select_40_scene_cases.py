from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

from .common import DATA_ROOT, PROJECT_ROOT, parse_bool, write_json
from .prepare_inputs import CASE_FIELDS, read_cases


CATEGORY_TO_TESTCASE = {
    ("Hard", True): "01_hard_ambiguous",
    ("Medium", True): "02_medium_ambiguous",
    ("Easy", True): "03_easy_ambiguous",
    ("Easy", False): "04_easy_unambiguous",
    ("Medium", False): "05_medium_unambiguous",
    ("Hard", False): "06_hard_unambiguous",
}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return parse_bool(value)


def select_cases(
    base_cases_path: Path,
    parquet_paths: list[Path],
    output_path: Path,
    *,
    scenes_per_testcase: int,
    seed: int,
) -> dict[str, Any]:
    base_cases, base_stats = read_cases(base_cases_path)
    base_groups: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for case in base_cases:
        key = (case["testcase"], int(case["scene_id"]), int(case["query_obj_id"]))
        base_groups.setdefault(key, []).append(case)

    existing_by_testcase: dict[str, list[tuple[int, int]]] = {
        testcase: [] for testcase in CATEGORY_TO_TESTCASE.values()
    }
    for (testcase, scene_id, query_obj_id), cases in sorted(base_groups.items()):
        if sorted(int(item["split"]) for item in cases) != [0, 1, 2]:
            raise RuntimeError(
                f"Base scene does not contain splits 0,1,2: {testcase} scene={scene_id}"
            )
        expected = CATEGORY_TO_TESTCASE[(cases[0]["difficulty"], cases[0]["ambiguous"])]
        if testcase != expected:
            raise RuntimeError(f"Base testcase/category mismatch: {testcase} != {expected}")
        existing_by_testcase[testcase].append((scene_id, query_obj_id))

    frames = [
        pd.read_parquet(
            path,
            columns=["sceneId", "queryObjId", "difficulty", "ambiguious", "split"],
        )
        for path in parquet_paths
    ]
    frame = pd.concat(frames, ignore_index=True)
    available: dict[str, list[tuple[int, int]]] = {
        testcase: [] for testcase in CATEGORY_TO_TESTCASE.values()
    }
    for (scene_id, query_obj_id), rows in frame.groupby(["sceneId", "queryObjId"]):
        splits = sorted(int(value) for value in rows["split"].tolist())
        if splits != [0, 1, 2]:
            raise RuntimeError(
                f"Parquet scene does not contain splits 0,1,2: {(scene_id, query_obj_id)}"
            )
        difficulties = {str(value) for value in rows["difficulty"].tolist()}
        ambiguities = {_as_bool(value) for value in rows["ambiguious"].tolist()}
        if len(difficulties) != 1 or len(ambiguities) != 1:
            raise RuntimeError(f"Inconsistent category for scene {scene_id}")
        category = (next(iter(difficulties)), next(iter(ambiguities)))
        testcase = CATEGORY_TO_TESTCASE.get(category)
        if testcase is None:
            raise RuntimeError(f"Unknown category: {category}")
        available[testcase].append((int(scene_id), int(query_obj_id)))

    selected: dict[str, dict[str, list[tuple[int, int]]]] = {}
    output_rows = []
    for testcase in sorted(available):
        existing = sorted(set(existing_by_testcase[testcase]))
        if len(existing) > scenes_per_testcase:
            raise RuntimeError(
                f"{testcase} already has {len(existing)} scenes, over target {scenes_per_testcase}"
            )
        existing_scene_ids = {scene_id for scene_id, _ in existing}
        candidates = sorted(
            item for item in set(available[testcase]) if item[0] not in existing_scene_ids
        )
        rng = random.Random(f"{seed}:{testcase}")
        rng.shuffle(candidates)
        needed = scenes_per_testcase - len(existing)
        if len(candidates) < needed:
            raise RuntimeError(
                f"{testcase} needs {needed} new scenes but only {len(candidates)} are available"
            )
        new = sorted(candidates[:needed])
        combined = sorted(existing + new)
        category = next(key for key, value in CATEGORY_TO_TESTCASE.items() if value == testcase)
        difficulty, ambiguous = category
        for scene_id, query_obj_id in combined:
            for split in (0, 1, 2):
                output_rows.append({
                    "testcase": testcase,
                    "scene_id": scene_id,
                    "query_obj_id": query_obj_id,
                    "difficulty": difficulty,
                    "ambiguous": str(ambiguous).lower(),
                    "split": split,
                })
        selected[testcase] = {
            "existing": existing,
            "new": new,
            "combined": combined,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CASE_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    manifest = {
        "schema_version": 1,
        "base_cases": str(base_cases_path.resolve()),
        "parquet_files": [str(path.resolve()) for path in parquet_paths],
        "selection_seed": seed,
        "scenes_per_testcase": scenes_per_testcase,
        "splits_per_scene": 3,
        "num_testcases": len(selected),
        "num_scenes": sum(len(value["combined"]) for value in selected.values()),
        "num_annotation_runs": len(output_rows),
        "base_case_stats": base_stats,
        "testcases": {
            testcase: {
                "num_existing_scenes": len(value["existing"]),
                "num_new_scenes": len(value["new"]),
                "num_combined_scenes": len(value["combined"]),
                "existing_scene_ids": [item[0] for item in value["existing"]],
                "new_scene_ids": [item[0] for item in value["new"]],
                "combined_scene_ids": [item[0] for item in value["combined"]],
            }
            for testcase, value in selected.items()
        },
        "output_csv": str(output_path.resolve()),
    }
    write_json(output_path.with_suffix(".manifest.json"), manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keep the original 20 scenes and deterministically add 20 new scenes per testcase."
    )
    parser.add_argument("--base-cases", type=Path, default=PROJECT_ROOT / "rsr_case.csv")
    parser.add_argument("--parquet-root", type=Path, default=DATA_ROOT / "source")
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_ROOT / "source" / "rsr_case_40_per_testcase.csv",
    )
    parser.add_argument("--scenes-per-testcase", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    parquet_paths = [
        args.parquet_root / f"train-{index:05d}-of-00002.parquet"
        for index in range(2)
    ]
    for path in [args.base_cases, *parquet_paths]:
        if not path.exists():
            raise FileNotFoundError(path)
    print(json.dumps(
        select_cases(
            args.base_cases.resolve(),
            [path.resolve() for path in parquet_paths],
            args.output.resolve(),
            scenes_per_testcase=args.scenes_per_testcase,
            seed=args.seed,
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
