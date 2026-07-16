# FreeGrasp SSR/RSR evaluation

[中文说明](README.zh.md) | [Detailed 120-scene protocol](FREEGRASP_120_SCENE_EVALUATION.md)

This directory contains an evaluation pipeline built around the original
FreeGrasp code. It prepares selected FreeGraspData scenes, produces numbered
localization images, asks a VLM which object to select, obtains a LangSAM mask,
compares that mask with extracted ground-truth instance masks, and writes
per-episode and aggregate reports.

The original FreeGrasp source and copied dataset archives are inputs. Normal
RSR scripts do not edit the original source archives.

## 1. Current metric names

The current implementation stores the following values for each episode:

```text
ssr = max IoU(predicted mask, every accepted GT object mask)
rsr = 1 if ssr > 0.5, otherwise 0
```

The comparison is strictly `>`; `IoU == 0.5` produces `rsr = 0`.

Important terminology detail: in this code, an episode's `ssr` field is the
best IoU value, `mean_ssr` is the mean IoU over evaluated episodes, and
`mean_rsr` is the proportion of evaluated episodes whose IoU is greater than
the threshold. These names describe the current implementation and should be
kept consistent when reading its reports.

If multiple dataset object IDs are accepted, the best IoU is used. Dataset
object IDs are zero-based, while NPZ instance labels are one-based with zero as
background:

```text
npz_instance_label = dataset_object_id + 1
GT filename = mask_<npz_instance_label:03d>_gt.png
```

## 2. Pipeline

```text
Parquet + npz_file.zip + case CSV
  -> select scenes
  -> prepare rsr/data/input
  -> export GT masks
  -> localization image (gt or molmo)
  -> VLM object selection
  -> LangSAM predicted mask
  -> IoU / ssr / rsr
  -> JSON, CSV, and distribution reports
```

The two localization modes are:

- `gt`: numbered points are derived from visible GT instances.
- `molmo`: numbered points are produced by Molmo. This is the mode used by the
  completed two-model experiment.

The VLM predicts a temporary localization ID shown on the image. For Molmo,
the point `(x, y)` is mapped back through `instances_objects[y, x]`:

```text
predicted_npz_label = instances_objects[y, x]
predicted_object_id = predicted_npz_label - 1
```

LangSAM receives the predicted class phrase and point. When several masks are
returned, the code selects a mask containing the point, otherwise the nearest
non-empty mask; without a point, it uses the highest logit.

## 3. File guide

| File | Responsibility |
|---|---|
| `common.py` | Shared paths, atomic JSON I/O, boolean parsing, VLM response parsing, and point-to-dataset-ID conversion. |
| `prepare_inputs.py` | Reads the selected case CSV, Parquet shards, and NPZ archive; exports scene images, depth, instances, annotations, and metadata into `data/input`. |
| `select_40_scene_cases.py` | Keeps the original 20 scenes and deterministically selects another 20 complete three-split scenes per category. |
| `export_ground_truth_masks.py` | Extracts every non-background NPZ instance as `mask_NNN_gt.png` without running a model. |
| `segmentation.py` | Calls the original FreeGrasp LangSAM actor and selects the mask associated with the predicted phrase/point. |
| `metrics.py` | Parses GT IDs, computes IoU, applies the ID+1 convention, and calculates episode `ssr` and `rsr`. |
| `run.py` | Main orchestrator: scene filtering, GT/Molmo localization, labeled PNG creation, VLM API/cache handling, LangSAM, metrics, failure records, and reports. |
| `evaluate.py` | Recomputes metrics from saved predicted masks and GT PNGs without running Molmo, a VLM API, or LangSAM. It **writes metrics back** to result JSON files and regenerates reports. |
| `summarize_two_model_experiment.py` | Aggregates separate GPT-4o and GPT-5.5 result JSONs into per-model/per-category CSV and JSON summaries. |
| `plot_ssr_distribution.py` | Reads completed evaluation JSONs and creates separate six-panel SSR IoU distributions for GPT-4o and GPT-5.5. It never combines their statistics. |
| `run_rsr.sh` | Environment wrapper for `python -m rsr.run`; configures the offline model cache, GPU, and API-key check. |
| `run_40_scenes_two_models.sh` | Full 6-category x 40-scene x 3-split x 2-model experiment. A clean run archives active data first; `--resume` keeps completed work. |
| `setup_model_cache.sh` | Builds an RSR-local symlink view of Molmo, GroundingDINO, BERT, and Hugging Face module caches; model files are not copied. |
| `tests/` | Unit tests for ID conversion, prompt compatibility, PNG transport, cache validity, retries/timeouts, GT export, IoU, thresholding, and exclusion policy. |

## 4. Data and output layout

```text
rsr/data/
  source/                    copied Parquet/NPZ/case sources
  input/<testcase>/scene_*/  prepared scene inputs and metadata
  ground_truth_masks/        extracted mask_NNN_gt.png files
  model_cache/               symlink-only offline model view
  output/
    shared/localization/     Molmo localization reused by both models
    models/gpt-4o/           GPT-4o reasoning, masks, and reports
    models/gpt-5.5/          GPT-5.5 reasoning, masks, and reports
    reports/                 combined index plus model-separated SSR plots
  run_logs/                  batch logs
  archive/                   snapshots made by clean batch runs
```

One episode result is normally located at:

```text
data/output/models/<model>/reason/molmo/<testcase>/scene_<id>/split_<n>.json
data/output/models/<model>/reason/molmo/<testcase>/scene_<id>/split_<n>_predicted_mask.png
```

Important JSON fields include `model`, `annotation`, `predicted_localization_id`,
`predicted_npz_label`, `predicted_object_id`, `predicted_mask`, `ssr`, `rsr`,
`per_ground_truth_iou`, `metric_status`, and `excluded_from_statistics`.

## 5. Environment

Run commands from the repository root:

```bash
cd /home/admin128/hanhuang/FreeGrasp_code
export PYTHON=/home/admin128/anaconda3/envs/freegrasp/bin/python
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD"
```

Reasoning also requires:

```bash
export OPENAI_API_KEY='...'
```

`run_rsr.sh` uses the local `freegrasp` environment, configures offline model
caches, defaults to GPU 0, clears proxy variables, and sends VLM traffic to the
configured OpenAI-compatible relay. It does not store the API key.

## 6. Tests

### 6.1 Unit tests: no model or API call

This is the first test to run after changing parsing, IDs, metrics, caching, or
request construction:

```bash
$PYTHON -m unittest discover -s rsr/tests -v
```

The tests use temporary directories and do not modify `rsr/data/output`.

### 6.2 Inspect CLI without changing data

```bash
$PYTHON -m rsr.run --help
$PYTHON -m rsr.evaluate --help
$PYTHON -m rsr.plot_ssr_distribution --help
```

### 6.3 Localization-only smoke test: no VLM API

Use a temporary output root so completed experiment data remains untouched:

```bash
bash rsr/run_rsr.sh \
  --input-root rsr/data/input \
  --output-root /tmp/freegrasp_rsr_smoke \
  --testcase 01_hard_ambiguous \
  --scene-id 1365 \
  --localization-mode molmo \
  --localization-only \
  --fail-fast
```

This verifies input loading, model-cache setup, Molmo localization, numbered
image generation, and localization JSON output.

### 6.4 One-episode end-to-end smoke test

This performs localization, one VLM request, LangSAM segmentation, and metric
calculation while writing only to `/tmp`:

```bash
export OPENAI_API_KEY='...'
bash rsr/run_rsr.sh \
  --input-root rsr/data/input \
  --output-root /tmp/freegrasp_rsr_smoke \
  --testcase 01_hard_ambiguous \
  --scene-id 1365 \
  --split 0 \
  --localization-mode molmo \
  --model gpt-4o \
  --fail-fast
```

Inspect the result with:

```bash
find /tmp/freegrasp_rsr_smoke -type f | sort
```

### 6.5 Reason-only test

After localization exists under the same output root:

```bash
bash rsr/run_rsr.sh \
  --input-root rsr/data/input \
  --output-root /tmp/freegrasp_rsr_smoke \
  --testcase 01_hard_ambiguous \
  --scene-id 1365 \
  --split 0 \
  --localization-mode molmo \
  --reason-only \
  --model gpt-4o \
  --fail-fast
```

`--reason-only` reuses localization. Add `--force-reason` only when the cached
reason result for the selected case must be replaced.

### 6.6 Offline metric recomputation

`rsr.evaluate` does not call Molmo, the VLM API, or LangSAM, but it updates
result JSON files and reports. To test without touching completed results,
evaluate a copy:

```bash
cp -a rsr/data/output/models/gpt-4o /tmp/freegrasp_eval_gpt4o
$PYTHON -m rsr.evaluate \
  --input-root rsr/data/input \
  --output-root /tmp/freegrasp_eval_gpt4o \
  --testcase 01_hard_ambiguous \
  --scene-id 1365 \
  --split 0 \
  --localization-mode molmo
```

## 7. Reports

Rebuild the two-model summary from existing result JSONs:

```bash
$PYTHON -m rsr.summarize_two_model_experiment
```

This writes:

```text
rsr/data/output/reports/two_model_summary.csv
rsr/data/output/reports/two_model_summary.json
```

Generate separate SSR IoU distribution plots for the two models:

```bash
$PYTHON -m rsr.plot_ssr_distribution
```

Outputs:

```text
rsr/data/output/reports/ssr_iou_distribution_by_category_gpt-4o.png
rsr/data/output/reports/ssr_iou_distribution_by_category_gpt-5.5.png
rsr/data/output/reports/ssr_iou_distribution_by_model_category.csv
rsr/data/output/reports/ssr_iou_values.csv
```

Each PNG uses only one model's episodes. No mean, sample count, or histogram
bin combines GPT-4o with GPT-5.5.

## 8. Full two-model experiment

The active combined dataset contains 6 categories, 40 scenes per category,
and 3 instructions per scene. The full experiment therefore runs:

```text
6 x 40 x 3 x 2 models = 1440 API episodes
```

Fresh run:

```bash
export OPENAI_API_KEY='...'
time bash rsr/run_40_scenes_two_models.sh
```

Resume completed data and retry unfinished work:

```bash
export OPENAI_API_KEY='...'
time bash rsr/run_40_scenes_two_models.sh --resume
```

Warning: a fresh run archives the active `input`, `output`, and
`ground_truth_masks` directories and rebuilds them. Use `--resume` when the
current experiment must be preserved.

## 9. Cache and overwrite options

- Default run: reuses matching successful localization/API/reason outputs.
- `--force-localization`: replaces selected localization outputs.
- `--force-reason`: replaces selected reasoning and segmentation outputs.
- `--fresh`: bypasses intermediate API cache and recomputes selected stages.
- `--reason-only`: requires localization to already exist.
- `--localization-only`: never runs reasoning or segmentation.
- `--manual-mask-review`: saves masks but leaves automatic metrics disabled.
- `--fail-fast`: stops at the first failure; otherwise failures are recorded.

Use testcase, scene, split, and temporary-output filters before any force flag.

## 10. Failure and denominator policy

The following cases use `ssr=null`, `rsr=null`, and are excluded from aggregate
denominators:

- API/transport failure;
- localization or segmentation infrastructure failure;
- missing/invalid required GT mask;
- no valid selected object ID;
- no corresponding predicted mask.

A normal predicted mask is always included with its measured IoU, including
IoU zero. Exclusion counts, failure types, and metric status are available in
per-episode JSON, `run_failures.json`, evaluation JSON, and summary reports.
