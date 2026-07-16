# FreeGrasp SSR/RSR 评估说明

[English](README.md) | [120-scene 详细实验协议](FREEGRASP_120_SCENE_EVALUATION.md)

该目录是在原始 FreeGrasp 代码基础上增加的评估流程。它负责准备指定的
FreeGraspData scene、生成带编号的定位图、调用 VLM 选择物体、通过 LangSAM
生成预测掩码、将预测掩码与 GT instance mask 比较，并保存逐 episode 与聚合报告。

原始 FreeGrasp 源代码和复制到 `rsr/data/source` 的数据压缩包只作为输入，正常
RSR 流程不会修改原始数据压缩包。

## 1. 当前代码中的指标定义

每个 episode 保存：

```text
ssr = max IoU(预测掩码, 所有可接受 GT 物体掩码)
rsr = 1，当且仅当 ssr > 0.5；否则为 0
```

阈值判断是严格的 `>`，因此 `IoU == 0.5` 时 `rsr = 0`。

需要特别注意当前代码的命名：逐 episode 的 `ssr` 字段是最佳 IoU；报告中的
`mean_ssr` 是有效 episode 的平均 IoU；`mean_rsr` 是有效 episode 中 IoU 大于
阈值的比例。阅读当前实验报告时必须按照这套实现解释字段。

如果一条 annotation 接受多个 GT object ID，代码取其中最大的 IoU。数据集
object ID 从 0 开始，而 NPZ instance label 从 1 开始，0 是背景：

```text
npz_instance_label = dataset_object_id + 1
GT 文件名 = mask_<npz_instance_label:03d>_gt.png
```

## 2. 整体数据流

```text
Parquet + npz_file.zip + case CSV
  -> 选择 scene
  -> 准备 rsr/data/input
  -> 导出 GT mask
  -> 生成定位图（gt 或 molmo）
  -> VLM 选择物体
  -> LangSAM 生成预测 mask
  -> 计算 IoU / ssr / rsr
  -> 生成 JSON、CSV 和分布图
```

两种 localization mode：

- `gt`：带编号的点来自可见 GT instance。
- `molmo`：带编号的点来自 Molmo；当前完成的两模型实验使用该模式。

VLM 返回的是图上的临时 localization ID。Molmo 模式根据对应点 `(x, y)` 映射
回数据集 ID：

```text
predicted_npz_label = instances_objects[y, x]
predicted_object_id = predicted_npz_label - 1
```

LangSAM 接收 VLM 返回的物体类别文本和坐标。如果返回多个 mask，优先选择包含
该坐标的 mask；否则选择距离该坐标最近的非空 mask。没有坐标时选择 logit 最高的
mask。

## 3. 各代码文件的作用

| 文件 | 功能 |
|---|---|
| `common.py` | 公共路径、原子 JSON 读写、布尔解析、VLM 响应解析、坐标到 dataset ID 的转换。 |
| `prepare_inputs.py` | 读取 case CSV、Parquet 和 NPZ；导出 scene 图片、深度、instance map、annotation 和 metadata 到 `data/input`。 |
| `select_40_scene_cases.py` | 每类保留原来的 20 个 scene，并通过固定 seed 再选择 20 个具有完整三个 split 的 scene。 |
| `export_ground_truth_masks.py` | 直接从 NPZ instance map 导出所有非背景 `mask_NNN_gt.png`，不调用模型。 |
| `segmentation.py` | 调用原始 FreeGrasp LangSAM，并根据 VLM 类别文本和坐标选择对应 mask。 |
| `metrics.py` | 解析 GT ID、计算 IoU、处理 ID+1 规则，并计算逐 episode 的 `ssr`、`rsr`。 |
| `run.py` | 主流程：筛选 scene、GT/Molmo 定位、带编号 PNG、VLM API/cache、LangSAM、指标、失败记录和报告。 |
| `evaluate.py` | 使用已保存预测 mask 和 GT PNG 重新计算指标，不运行 Molmo、VLM API 或 LangSAM；但它会把指标写回 result JSON，并重新生成报告。 |
| `summarize_two_model_experiment.py` | 分别汇总 GPT-4o 和 GPT-5.5 的结果，生成按模型、题型区分的 CSV/JSON。 |
| `plot_ssr_distribution.py` | 读取完成的 evaluation JSON，为 GPT-4o、GPT-5.5 分别生成六题型 SSR IoU 分布图，不混合两模型统计。 |
| `run_rsr.sh` | `python -m rsr.run` 的环境包装器；设置离线模型缓存、GPU，并检查 API key。 |
| `run_40_scenes_two_models.sh` | 完整 6 类 × 40 scene × 3 split × 2 模型实验；全新运行会先归档活动数据，`--resume` 保留完成结果。 |
| `setup_model_cache.sh` | 为 Molmo、GroundingDINO、BERT 和 HF modules 建立 rsr 本地软链接视图，不复制模型文件。 |
| `tests/` | 测试 ID 映射、prompt 一致性、PNG 传输、cache 判断、超时/重试、GT 导出、IoU、阈值和排除策略。 |

## 4. 数据与输出目录

```text
rsr/data/
  source/                    复制的 Parquet、NPZ、case 数据
  input/<testcase>/scene_*/  准备好的 scene 输入和 metadata
  ground_truth_masks/        导出的 mask_NNN_gt.png
  model_cache/               只包含软链接的离线模型视图
  output/
    shared/localization/     两个模型共用的 Molmo 定位结果
    models/gpt-4o/           GPT-4o reasoning、mask 和报告
    models/gpt-5.5/          GPT-5.5 reasoning、mask 和报告
    reports/                 汇总文件和两个模型各自的 SSR 图
  run_logs/                  批量运行日志
  archive/                   全新批量实验自动保存的历史快照
```

单个 episode 通常位于：

```text
data/output/models/<model>/reason/molmo/<testcase>/scene_<id>/split_<n>.json
data/output/models/<model>/reason/molmo/<testcase>/scene_<id>/split_<n>_predicted_mask.png
```

重点字段：`model`、`annotation`、`predicted_localization_id`、
`predicted_npz_label`、`predicted_object_id`、`predicted_mask`、`ssr`、
`rsr`、`per_ground_truth_iou`、`metric_status`、`excluded_from_statistics`。

## 5. 环境设置

所有命令从仓库根目录运行：

```bash
cd /home/admin128/hanhuang/FreeGrasp_code
export PYTHON=/home/admin128/anaconda3/envs/freegrasp/bin/python
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD"
```

运行 Reason/VLM 前还需要：

```bash
export OPENAI_API_KEY='...'
```

`run_rsr.sh` 会使用本地 `freegrasp` 环境、配置离线模型缓存、默认使用 GPU 0、
清除代理变量，并向配置好的 OpenAI-compatible relay 发送 VLM 请求。脚本不会保存
API key。

## 6. 如何测试

### 6.1 单元测试：不调用模型和 API

修改 response 解析、ID、指标、cache 或请求结构后，首先运行：

```bash
$PYTHON -m unittest discover -s rsr/tests -v
```

单元测试只使用临时目录，不修改 `rsr/data/output`。

### 6.2 只查看 CLI，不修改数据

```bash
$PYTHON -m rsr.run --help
$PYTHON -m rsr.evaluate --help
$PYTHON -m rsr.plot_ssr_distribution --help
```

### 6.3 仅测试 Localization：不调用 VLM API

使用 `/tmp` 作为输出，避免修改已经完成的实验：

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

该测试验证输入读取、模型缓存、Molmo 定位、带编号图片和 localization JSON。

### 6.4 单 episode 端到端测试

下面只运行一个 scene 的一个 split，包含 localization、一次 VLM 请求、LangSAM
和指标计算；所有结果写到 `/tmp`：

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

查看输出：

```bash
find /tmp/freegrasp_rsr_smoke -type f | sort
```

### 6.5 仅测试 Reason

同一个 output root 中已经存在 localization 后，可以运行：

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

`--reason-only` 会复用 localization。只有确实需要覆盖指定 case 的缓存 Reason 时才加
`--force-reason`。

### 6.6 离线重新计算指标

`rsr.evaluate` 不调用 Molmo、VLM API 或 LangSAM，但会修改 result JSON 和 reports。
为了不影响完成的结果，应对 output 副本测试：

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

## 7. 报告生成

从已有 result JSON 重新生成两模型汇总：

```bash
$PYTHON -m rsr.summarize_two_model_experiment
```

输出：

```text
rsr/data/output/reports/two_model_summary.csv
rsr/data/output/reports/two_model_summary.json
```

分别生成两个模型的 SSR IoU 分布图：

```bash
$PYTHON -m rsr.plot_ssr_distribution
```

输出：

```text
rsr/data/output/reports/ssr_iou_distribution_by_category_gpt-4o.png
rsr/data/output/reports/ssr_iou_distribution_by_category_gpt-5.5.png
rsr/data/output/reports/ssr_iou_distribution_by_model_category.csv
rsr/data/output/reports/ssr_iou_values.csv
```

每张 PNG 只读取对应模型的数据；GPT-4o 和 GPT-5.5 的平均值、样本数和 histogram
bin 均不混合。

## 8. 完整两模型实验

当前组合数据集包含 6 个题型，每类 40 个 scene，每个 scene 有 3 个 instruction：

```text
6 × 40 × 3 × 2 models = 1440 个 API episode
```

全新运行：

```bash
export OPENAI_API_KEY='...'
time bash rsr/run_40_scenes_two_models.sh
```

保留已经完成的结果并继续：

```bash
export OPENAI_API_KEY='...'
time bash rsr/run_40_scenes_two_models.sh --resume
```

注意：全新运行会归档当前活动的 `input`、`output`、`ground_truth_masks` 后重新构建。
需要保留当前实验时必须使用 `--resume`。

## 9. Cache 与覆盖参数

- 默认运行：复用匹配且成功的 localization/API/Reason 输出。
- `--force-localization`：覆盖筛选范围内的定位输出。
- `--force-reason`：覆盖筛选范围内的 Reason 和 segmentation 输出。
- `--fresh`：不读取中间 API cache，重新计算筛选范围内的阶段。
- `--reason-only`：要求同一 output root 中已经存在 localization。
- `--localization-only`：不运行 Reason 和 segmentation。
- `--manual-mask-review`：保存 mask，但不自动计算指标。
- `--fail-fast`：第一次失败时停止；不加时记录失败并继续。

使用任何 force 参数前，应先通过 testcase、scene、split 和临时 output root 限制范围。

## 10. 失败与统计分母策略

以下情况写入 `ssr=null`、`rsr=null`，并从聚合统计分母排除：

- API 或网络传输失败；
- localization/segmentation 基础设施失败；
- 必需 GT mask 缺失或无效；
- 没有有效的 selected object ID；
- 没有对应的 predicted mask。

如果产生了正常预测 mask，则一定使用实际 IoU 参与统计，包括 IoU 为 0 的情况。
排除数量、失败类型和指标状态可在逐 episode JSON、`run_failures.json`、evaluation
JSON 和 summary reports 中查看。
