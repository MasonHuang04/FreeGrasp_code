from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import DATA_ROOT, read_json, write_json


DEFAULT_MODELS = ("gpt-4o", "gpt-5.5")


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [
        item for item in records
        if item.get("ssr") is not None and item.get("rsr") is not None
    ]
    excluded = [item for item in records if item not in evaluated]
    count = len(evaluated)
    return {
        "count": count,
        "num_result_records": len(records),
        "num_excluded_missing_selection_or_mask": len(excluded),
        "mean_ssr": (
            sum(float(item["ssr"]) for item in evaluated) / count if count else None
        ),
        "mean_rsr": (
            sum(float(item["rsr"]) for item in evaluated) / count if count else None
        ),
        "status_counts": dict(sorted(Counter(item.get("status") for item in records).items())),
    }


def summarize(input_root: Path, output_root: Path, models: list[str]) -> dict[str, Any]:
    manifest = read_json(input_root / "manifest.json")
    groups = [item["name"] for item in manifest["testcases"]]
    expected_by_group = {
        item["name"]: int(item["num_scenes"]) * 3 for item in manifest["testcases"]
    }
    combined_rows = []
    model_summaries = {}
    for model in models:
        model_root = output_root / "models" / model
        group_rows = []
        all_records = []
        for group in groups:
            records = [
                read_json(path)
                for path in sorted(
                    (model_root / "reason" / "molmo" / group).glob("scene_*/split_*.json")
                )
            ]
            all_records.extend(records)
            failure_path = model_root / "reports" / "failures" / f"{group}.json"
            failures = read_json(failure_path).get("failures", []) if failure_path.exists() else []
            aggregate = _aggregate(records)
            row = {
                "model": model,
                "testcase": group,
                "expected_cases": expected_by_group[group],
                **aggregate,
                "num_run_failures_excluded": len(failures),
            }
            row["num_missing_outputs"] = max(
                0,
                row["expected_cases"]
                - row["num_result_records"]
                - row["num_run_failures_excluded"],
            )
            group_rows.append(row)
            combined_rows.append(row)

        overall = _aggregate(all_records)
        expected_total = sum(expected_by_group.values())
        model_summary = {
            "model": model,
            "expected_cases": expected_total,
            "overall": overall,
            "by_testcase": group_rows,
            "metric_definition": "SSR=max GT IoU; RSR=1 if SSR>0.5 else 0",
            "missing_selection_or_mask_policy": "SSR=null, RSR=null, excluded",
            "api_or_infrastructure_failure_policy": "excluded",
        }
        model_summaries[model] = model_summary
        write_json(model_root / "reports" / "model_summary.json", model_summary)
        _write_csv(model_root / "reports" / "model_summary.csv", group_rows)

    summary = {
        "schema_version": 1,
        "input_manifest": str((input_root / "manifest.json").resolve()),
        "num_scenes": int(manifest["num_scenes"]),
        "splits_per_scene": 3,
        "models": models,
        "expected_api_cases_per_model": int(manifest["num_scenes"]) * 3,
        "expected_api_cases_total": int(manifest["num_scenes"]) * 3 * len(models),
        "model_summaries": model_summaries,
    }
    report_root = output_root / "reports"
    write_json(report_root / "two_model_summary.json", summary)
    _write_csv(report_root / "two_model_summary.csv", combined_rows)
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "testcase",
        "expected_cases",
        "count",
        "num_result_records",
        "num_excluded_missing_selection_or_mask",
        "num_run_failures_excluded",
        "num_missing_outputs",
        "mean_ssr",
        "mean_rsr",
        "status_counts",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            value = dict(row)
            value["status_counts"] = json.dumps(value.get("status_counts", {}), sort_keys=True)
            writer.writerow(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize separate GPT-4o/GPT-5.5 FreeGrasp outputs by testcase."
    )
    parser.add_argument("--input-root", type=Path, default=DATA_ROOT / "input")
    parser.add_argument("--output-root", type=Path, default=DATA_ROOT / "output")
    parser.add_argument("--model", action="append", dest="models")
    args = parser.parse_args()
    print(json.dumps(
        summarize(
            args.input_root.resolve(),
            args.output_root.resolve(),
            list(args.models or DEFAULT_MODELS),
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
