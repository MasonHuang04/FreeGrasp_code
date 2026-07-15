# FreeGrasp revised SSR/RSR evaluation

See [`FREEGRASP_120_SCENE_EVALUATION.md`](FREEGRASP_120_SCENE_EVALUATION.md)
for the paper-aligned FreeGrasp-only evaluation matrix, exact prompts, revised
metric definitions, and the complete path from one input to one result.

Everything added by this evaluation lives under `FreeGrasp_code/rsr/`. The
original FreeGrasp source files and root dataset archives are read-only inputs.
The runner records the predicted object ID, obtains the corresponding LangSAM
mask, and computes the revised metrics requested for this evaluation.

## What is recorded

The paper reports two RSR localization settings. They use the same GPT-4o
reasoning prompt but different numbered images:

- `gt`: object marks come directly from the visible GT instance masks.
- `molmo`: object marks come from Molmo keypoints.

FreeGrasp uses the following ID spaces:

1. Molmo places numbered points on the image. GPT-4o predicts one of these
   temporary `predicted_molmo_id` values as the first object to grasp.
2. At the Molmo point `(x, y)`, `instances_objects[y, x]` is the one-based NPZ
   instance label. The FreeGraspData/parquet object ID is zero-based, so:

   `predicted_object_id = predicted_npz_label - 1`

For GT, the displayed localization ID is the one-based NPZ label, so the same
subtraction gives the zero-based dataset object ID. Both settings preserve the
point, class text, raw GPT response, status, timing, source metadata, selected
mask, and metric details.

## Revised metric and failure policy

For all dataset object IDs accepted as ground truth for an annotation:

```text
SSR = max IoU(predicted mask, ground-truth object mask)
RSR = 1 if SSR > 0.5 else 0
```

The comparison is strictly `>`: an IoU exactly equal to `0.5` has `RSR=0`.

- API timeout, connection/TLS failure, relay error, or response without a
  usable completion: `SSR=null`, `RSR=null`, excluded from the denominator.
- Missing segmentation packages or weights are also infrastructure failures:
  `SSR=null`, `RSR=null`, excluded rather than producing artificial zeros.
- A usable API completion followed by an invalid selection, unparseable answer,
  or a segmentation model that returns no usable mask is an algorithm failure:
  `SSR=0`, `RSR=0`, included.
- A normal predicted mask is included with its measured IoU and thresholded RSR.

This distinction is written to `run_failures.json`, every result JSON, and
`reports/summary.json` so the statistics stage does not turn infrastructure
failures into model failures.

## Prepare selected data

From `/home/qiuguanhe/huanghan/FreeGrasp_code`:

```bash
/home/qiuguanhe/miniconda3/envs/freegrasp/bin/python -m rsr.prepare_inputs
```

`run_rsr.sh` builds an rsr-local symlink view at `rsr/data/model_cache`: Molmo
comes from `/home/data/models/huggingface/hub`, while GroundingDINO and BERT
come from `/home/data/datasets/.cache/hf`. No public weights are copied or
modified, and all model loading remains offline.

The OpenAI-compatible relay is fixed to `https://www.highland-api.top/v1`.
Only export `OPENAI_API_KEY` in the current shell before reasoning; no API
configuration is read from SmartGrasp and no credential is stored in rsr. The
default `--api-transport auto` matches local SmartGrasp: the OpenAI Python SDK
calls Chat Completions at the fixed compatible relay with `model=gpt-4o`.
The request limit is 420 seconds. The key remains in the process environment.
The full numbered RGB PNG remains on disk at the original scene resolution.
Exactly like SmartGrasp's `reason/vlm/helper.py`, it is encoded in memory as
JPEG quality 90, Base64-wrapped as `data:image/jpeg`, and sent with
`detail=high`; neither a JPEG file nor request JSON is written to disk.
`run_rsr.sh` clears proxy variables, so the relay call is direct. Curl remains
available only as an explicit fallback with `--api-transport curl`.

The input is the root `rsr_case.csv` plus the two parquet shards and
`npz_file.zip`. The CSV currently contains four identical copies of every
annotation row (1440 rows). Preparation deduplicates it to 360 unique
annotations: 120 scenes, three splits per scene. A snapshot of the CSV and all
extracted scene inputs are written below `rsr/data/`.

## Run

Smoke test one scene and one annotation in both paper settings:

```bash
bash rsr/run_rsr.sh --limit-scenes 1 --split 0 --fail-fast
```

Run all 120 scenes and 720 reasoning queries (360 each for GT and Molmo;
cached and resumable):

```bash
bash rsr/run_rsr.sh
```

Run stages or settings separately:

```bash
bash rsr/run_rsr.sh --localization-only
bash rsr/run_rsr.sh --reason-only
bash rsr/run_rsr.sh --localization-mode gt
bash rsr/run_rsr.sh --localization-mode molmo
```

Useful filters include repeated `--testcase`, `--scene-id`, and `--split`
arguments. Use `--force-localization` or `--force-reason` only to replace
cached outputs for selected cases.

Recompute metrics from already saved masks without calling the API again:

```bash
/home/qiuguanhe/miniconda3/envs/freegrasp/bin/python -m rsr.evaluate
```

Mask prediction reuses the original FreeGrasp LangSAM actor without modifying
the source tree. The environment must contain GroundingDINO and the configured
GroundingDINO/SAM weights; a missing backend is recorded as an excluded
infrastructure failure, not as model score zero.

## Outputs

```text
rsr/data/output/
  localization/gt/scene<id>/
    <id>.png
    <id>_id.txt
    localization_result.json
  localization/molmo/scene<id>/
    <id>.png              # official FreeGrasp labeled image
    <id>_id.txt           # Molmo point and one-based NPZ label
    localization_result.json
  reason/<gt|molmo>/<testcase>/scene_<id>/split_<split>.json
  reason/<gt|molmo>/<testcase>/scene_<id>/split_<split>_predicted_mask.png
  reason/<gt|molmo>/<testcase>/scene_<id>/api_cache/split_<split>.json
  reports/predicted_object_ids.csv
  reports/predicted_object_ids.jsonl
  reports/summary.json
  run_failures.json
```
