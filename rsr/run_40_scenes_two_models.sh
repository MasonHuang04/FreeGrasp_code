#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/freegrasp-matplotlib}"
mkdir -p "$MPLCONFIGDIR"

if [[ -z "${PYTHON:-}" ]]; then
    for candidate in \
        "$HOME/anaconda3/envs/freegrasp/bin/python" \
        "$HOME/miniconda3/envs/freegrasp/bin/python"; do
        if [[ -x "$candidate" ]]; then
            PYTHON="$candidate"
            break
        fi
    done
fi
PYTHON="${PYTHON:-}"
if [[ ! -x "$PYTHON" ]]; then
    echo "freegrasp Python environment was not found." >&2
    exit 1
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is not set. Export it in this shell first." >&2
    exit 1
fi

RESUME=0
if [[ "${1:-}" == "--resume" ]]; then
    RESUME=1
    shift
fi
if [[ $# -ne 0 ]]; then
    echo "Usage: bash rsr/run_40_scenes_two_models.sh [--resume]" >&2
    exit 2
fi

DATA_ROOT="$ROOT_DIR/rsr/data"
SOURCE_ROOT="$DATA_ROOT/source"
INPUT_ROOT="$DATA_ROOT/input"
OUTPUT_ROOT="$DATA_ROOT/output"
GT_ROOT="$DATA_ROOT/ground_truth_masks"
BASE_CASES="$ROOT_DIR/rsr_case.csv"
EXPANDED_CASES="$SOURCE_ROOT/rsr_case_40_per_testcase.csv"
SHARED_OUTPUT="$OUTPUT_ROOT/shared"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="$DATA_ROOT/run_logs/$RUN_ID"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/master.log") 2>&1

TESTCASES=(
    01_hard_ambiguous
    02_medium_ambiguous
    03_easy_ambiguous
    04_easy_unambiguous
    05_medium_unambiguous
    06_hard_unambiguous
)
MODELS=(gpt-4o gpt-5.5)

echo "[experiment] run_id=$RUN_ID resume=$RESUME"
echo "[experiment] pipeline=input -> Molmo -> GPT model -> LangSAM -> evaluation"
echo "[experiment] expected=6 testcases x 40 scenes x 3 splits x 2 models = 1440 API cases"

if [[ "$RESUME" -eq 0 ]]; then
    echo "[prepare] selecting original 20 + deterministic new 20 per testcase"
    "$PYTHON" -u -m rsr.select_40_scene_cases \
        --base-cases "$BASE_CASES" \
        --parquet-root "$SOURCE_ROOT" \
        --output "$EXPANDED_CASES" \
        --scenes-per-testcase 40 \
        --seed 20260716 \
        > "$LOG_ROOT/selection_manifest.json"

    ARCHIVE_ROOT="$DATA_ROOT/archive/$RUN_ID"
    mkdir -p "$ARCHIVE_ROOT"
    if [[ -f "$SOURCE_ROOT/rsr_case.csv" ]]; then
        cp -a "$SOURCE_ROOT/rsr_case.csv" "$ARCHIVE_ROOT/rsr_case.previous.csv"
    fi
    for active_path in "$INPUT_ROOT" "$OUTPUT_ROOT" "$GT_ROOT"; do
        if [[ -e "$active_path" ]]; then
            echo "[archive] $active_path -> $ARCHIVE_ROOT/$(basename "$active_path")"
            mv "$active_path" "$ARCHIVE_ROOT/$(basename "$active_path")"
        fi
    done

    echo "[prepare] rebuilding combined rsr/data/input with 240 scenes"
    "$PYTHON" -u -m rsr.prepare_inputs \
        --project-root "$SOURCE_ROOT" \
        --cases "$EXPANDED_CASES" \
        --input-root "$INPUT_ROOT" \
        --source-root "$SOURCE_ROOT" \
        --force \
        > "$LOG_ROOT/input_manifest.json"

    echo "[prepare] exporting all GT ID+1 masks for the combined input"
    "$PYTHON" -u -m rsr.export_ground_truth_masks \
        --source-root "$SOURCE_ROOT" \
        --output-root "$GT_ROOT" \
        > "$LOG_ROOT/ground_truth_manifest.json"
else
    echo "[resume] keeping current combined input/output and retrying unfinished cases"
fi

"$PYTHON" - "$INPUT_ROOT/manifest.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
assert p["num_scenes"] == 240, p["num_scenes"]
assert p["num_annotation_runs"] == 720, p["num_annotation_runs"]
assert len(p["testcases"]) == 6
for item in p["testcases"]:
    assert item["num_scenes"] == 40, item
print("[validate input] 6 testcases, 40 scenes each, 3 splits each")
PY

mkdir -p "$OUTPUT_ROOT"
echo "[localization] generating one shared Molmo localization for all 240 scenes"
LOCALIZATION_ARGS=(
    --input-root "$INPUT_ROOT"
    --output-root "$SHARED_OUTPUT"
    --localization-mode molmo
    --localization-only
)
if [[ "$RESUME" -eq 0 ]]; then
    LOCALIZATION_ARGS+=(--force-localization)
fi
bash "$ROOT_DIR/rsr/run_rsr.sh" "${LOCALIZATION_ARGS[@]}"

HAD_FAILURES=0
for MODEL in "${MODELS[@]}"; do
    MODEL_ROOT="$OUTPUT_ROOT/models/$MODEL"
    mkdir -p "$MODEL_ROOT/reports/failures" "$MODEL_ROOT/reports/evaluations"
    if [[ -e "$MODEL_ROOT/localization" && ! -L "$MODEL_ROOT/localization" ]]; then
        echo "Expected a localization symlink but found a real path: $MODEL_ROOT/localization" >&2
        exit 1
    fi
    ln -sfn "$SHARED_OUTPUT/localization" "$MODEL_ROOT/localization"

    echo "[model start] $MODEL"
    for GROUP in "${TESTCASES[@]}"; do
        echo "[group start] model=$MODEL testcase=$GROUP cases=40x3"
        RUN_ARGS=(
            --input-root "$INPUT_ROOT"
            --output-root "$MODEL_ROOT"
            --testcase "$GROUP"
            --localization-mode molmo
            --reason-only
            --model "$MODEL"
            --api-transport openai
            --api-timeout 420
            --api-max-attempts 3
            --api-retry-backoff 5
            --iou-threshold 0.5
        )
        if [[ "$RESUME" -eq 0 ]]; then
            RUN_ARGS+=(--fresh)
        fi

        set +e
        bash "$ROOT_DIR/rsr/run_rsr.sh" "${RUN_ARGS[@]}" \
            2>&1 | tee "$LOG_ROOT/${MODEL}_${GROUP}.log"
        RUN_STATUS=${PIPESTATUS[0]}
        set -e
        if [[ "$RUN_STATUS" -ne 0 ]]; then
            HAD_FAILURES=1
            echo "[group warning] model=$MODEL testcase=$GROUP run_status=$RUN_STATUS; continuing"
        fi
        if [[ -f "$MODEL_ROOT/run_failures.json" ]]; then
            cp -a "$MODEL_ROOT/run_failures.json" \
                "$MODEL_ROOT/reports/failures/$GROUP.json"
        else
            HAD_FAILURES=1
            "$PYTHON" - "$MODEL_ROOT/reports/failures/$GROUP.json" <<'PY'
import json, sys
with open(sys.argv[1], "w") as handle:
    json.dump({"failures": [{"failure_type": "missing_run_failure_report", "excluded_from_statistics": True}]}, handle)
PY
        fi

        echo "[evaluation] model=$MODEL testcase=$GROUP"
        "$PYTHON" -u -m rsr.evaluate \
            --input-root "$INPUT_ROOT" \
            --output-root "$MODEL_ROOT" \
            --testcase "$GROUP" \
            --localization-mode molmo \
            --iou-threshold 0.5 \
            > "$MODEL_ROOT/reports/evaluations/$GROUP.json"
    done
done

echo "[summary] computing model/testcase mean SSR and RSR"
"$PYTHON" -u -m rsr.summarize_two_model_experiment \
    --input-root "$INPUT_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --model gpt-4o \
    --model gpt-5.5 \
    > "$LOG_ROOT/two_model_summary.stdout.json"

echo "[done] combined JSON: $OUTPUT_ROOT/reports/two_model_summary.json"
echo "[done] combined CSV:  $OUTPUT_ROOT/reports/two_model_summary.csv"
echo "[done] master log:    $LOG_ROOT/master.log"
if [[ "$HAD_FAILURES" -ne 0 ]]; then
    echo "[done with exclusions] Some API/infrastructure failures were excluded. Use --resume to retry them." >&2
    exit 1
fi
