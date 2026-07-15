# FreeGrasp 120 场景评测说明

本文档定义当前只针对 FreeGrasp 的评测。实验布局参考论文 Table I，但
SSR/RSR 严格使用本项目重新定义的计算方式；不测试 ThinkGrasp。

## 1. 评测范围

- grasp reasoning 模型固定为 `gpt-4o`。
- API 地址固定在 `rsr/run.py`，密钥只通过当前终端的
  `OPENAI_API_KEY` 环境变量传入。
- 共 120 个 scene，六类各 20 个：

| Table I 列 | input testcase | scene 数 |
| --- | --- | ---: |
| Easy w/o Amb. | `04_easy_unambiguous` | 20 |
| Easy w Amb. | `03_easy_ambiguous` | 20 |
| Medium w/o Amb. | `05_medium_unambiguous` | 20 |
| Medium w Amb. | `02_medium_ambiguous` | 20 |
| Hard w/o Amb. | `06_hard_unambiguous` | 20 |
| Hard w Amb. | `01_hard_ambiguous` | 20 |

每个 scene 有三条不同人工指令（`split=0,1,2`），所以主评测共有 360 次
FreeGrasp reasoning。当前自定义 SSR/RSR 只要求实际 FreeGrasp 管线，主评测
固定使用 Molmo localization；GT localization 只保留为可选论文消融。

## 2. 需要验证的 FreeGrasp 模式

主评测只需要 Molmo。GT 和 Molmo 不是两个 reasoning 模型；如果执行可选
消融，两者仍使用同一个 `gpt-4o`、同一套 reasoning prompt 和 LangSAM，
区别只在于编号点如何产生。

### GT localization

`gt` 从 `instances_objects.npy` 读取所有可见实例，为每个实例选择一个可见
点，并把一基的 NPZ label 画到 RGB 图像。这是论文 `FreeGrasp RSR (GT)`
的 localization oracle 消融设置。

### Molmo localization

`molmo` 让 Molmo 指出所有物体，在 RGB 图像上绘制 Molmo 返回的编号点，
再通过 `instances_objects[y, x]` 映射到数据集物体。这是正常 FreeGrasp
完整管线，对应论文 `FreeGrasp RSR (Molmo)`；论文的 FreeGrasp SSR 行也
使用这条正常 Molmo 管线。

因此，当前主评测表格只有两行：

| 结果行 | localization | 六个单元格中的值 |
| --- | --- | --- |
| FreeGrasp SSR | Molmo | 自定义 SSR 的均值 |
| FreeGrasp RSR | Molmo | 自定义 RSR 的均值 |

GT localization 不是计算指标所需的 GT mask。每个 Molmo 结果仍然必须和
GT mask 计算 IoU；只是不给 GPT-4o 使用 GT localization 编号图。

## 3. 本项目的 SSR/RSR 定义

这里不使用论文原始定义。对一条指令，令 `P` 为 FreeGrasp/LangSAM 得到的
预测 mask，`G_j` 为 annotation 允许的第 `j` 个 GT object mask：

```text
IoU(P, G_j) = |P intersection G_j| / |P union G_j|
SSR = max_j IoU(P, G_j)
RSR = 1 if SSR > 0.5 else 0
```

- 阈值运算符严格为 `>`；`SSR == 0.5` 时 `RSR=0`。
- 如果有多个 GT object ID，逐个计算 IoU，取最大值作为 SSR。
- RSR 的分组均值就是 `IoU > 0.5` 的成功比例。

Table I 一个单元格包含 20 个 scene。先分别计算 split 0、1、2 在这 20 个
scene 上的指标均值，再用三个 split 均值得到表格中的 `mean ± std`。同时
必须报告有效样本数，因为 API/基础设施失败会被排除。

失败统计规则：

- API timeout、连接/TLS、中转站错误或缺少模型依赖：
  `SSR=null, RSR=null`，不计入分母。
- API 已正常返回，但选择无效或 segmentation 没有可用 mask：
  `SSR=0, RSR=0`，计入分母。
- 正常得到 mask：记录实际 SSR 和阈值化后的 RSR。

## 4. 每一轮使用的指令

### Localization 指令

GT localization 不调用定位模型。Molmo 的实际指令是：

```text
Point out all objects in the green tray
```

论文使用更泛化的描述 `Point out all objects in the bin`；当前 synthetic
图像是绿色 tray，因此 runner 使用前一种文本。

### GPT-4o reasoning 指令

编号图和用户 annotation 一起发送给 `gpt-4o`。system prompt 要求模型：

1. 如果目标没有被遮挡，直接返回目标物体；
2. 如果目标被遮挡，返回一个遮挡目标且当前可以抓取的物体；
3. 只输出：

```text
[object_id, color class_name]
```

user message 的格式是：

```text
Grasp {annotation}
```

scene 815 的三条指令分别为：

| split | annotation | 实际发送的 user text |
| ---: | --- | --- |
| 0 | `the plyer on the left` | `Grasp the plyer on the left` |
| 1 | `the top left clamp` | `Grasp the top left clamp` |
| 2 | `the pliers on the left` | `Grasp the pliers on the left` |

GPT-4o 返回的 class name 作为 LangSAM 的 semantic prompt；返回的 object
ID 对应编号点，用来从同类的多个 mask 中选择具体实例。

## 5. 一个 input 包含什么

scene 815 的 input 为：

```text
rsr/data/input/01_hard_ambiguous/scene_815/
  image.png
  instances_objects.npy
  metadata.json
  source.npz
```

`metadata.json` 包含难度、ambiguity、三条 annotation、query object ID、
允许的 GT object IDs 和源数据位置。数据集 object ID 从 0 开始；NPZ
instance label 从 1 开始，0 表示背景：

```text
predicted_object_id = predicted_npz_label - 1
```

Molmo 模式下，`predicted_localization_id` 是 Molmo 绘制的临时点编号；GT
模式下，它就是画在图像上的一基 NPZ instance label。

## 6. 从一个 input 到一轮完整结果

对选中的 `(scene, split, localization_mode)`：

1. 读取 `metadata.json`、`image.png` 和 `instances_objects.npy`。
2. 用 GT mask 或 Molmo 点定位所有物体。
3. 在图像上绘制编号并保存 localization image。
4. 把编号图和 `Grasp {annotation}` 发送给 `gpt-4o`。
5. 解析 `[object_id, color class_name]`。
6. 把所选点映射为一基 NPZ label 和零基 dataset object ID。
7. 用 class name 运行 LangSAM，再用该编号点选择对应 instance mask。
8. 将预测 mask 与 annotation 接受的每个 GT object mask 计算 IoU。
9. 保存 predicted object ID、mask、每个 GT IoU、自定义 SSR 和自定义 RSR。

这是一轮静态 FreeGraspData reasoning/segmentation 选择。当前 RSR runner
不执行 GraspNet pose、机器人运动或移除物体后的下一轮；这些属于论文真实
机器人实验，不属于 Table I 的静态数据集评测。

## 7. 执行命令

从 input 完整运行 scene 815、split 0、Molmo 主评测：

```bash
cd /home/qiuguanhe/huanghan/FreeGrasp_code

bash rsr/run_rsr.sh \
  --scene-id 815 \
  --split 0 \
  --localization-mode molmo \
  --model gpt-4o \
  --api-transport openai \
  --api-timeout 420
```

scene 815 的三条指令，共三轮 Molmo 主评测：

```bash
bash rsr/run_rsr.sh \
  --scene-id 815 \
  --localization-mode molmo \
  --model gpt-4o \
  --api-transport openai \
  --api-timeout 420
```

全部 120 个 scene、三条指令，共 360 轮：

```bash
bash rsr/run_rsr.sh \
  --localization-mode molmo \
  --model gpt-4o \
  --api-transport openai \
  --api-timeout 420
```

正常命令支持续跑，会复用成功的 localization、API 和最终结果缓存。如果
要求每个样本从 input 重新执行 Molmo、GPT-4o、LangSAM 和指标计算，且不
读取或写入中间 API cache，使用：

```bash
bash rsr/run_rsr.sh \
  --localization-mode molmo \
  --model gpt-4o \
  --api-transport openai \
  --api-timeout 420 \
  --api-max-attempts 5 \
  --api-retry-backoff 2 \
  --fresh
```

`--fresh` 会覆盖选中样本的 localization 和最终结果，并删除该轮旧 mask
及 API cache。最终 JSON 和 mask 仍会保存，否则无法汇总 SSR/RSR。API
达到最大重试次数仍失败时，不生成该轮结果并按基础设施失败排除。

## 8. 一轮结果在哪里

scene 815、split 0、Molmo 的文件为：

```text
rsr/data/output/localization/molmo/scene815/localization_result.json
rsr/data/output/localization/molmo/scene815/815.png
rsr/data/output/reason/molmo/01_hard_ambiguous/scene_815/api_cache/split_0.json
rsr/data/output/reason/molmo/01_hard_ambiguous/scene_815/split_0.json
rsr/data/output/reason/molmo/01_hard_ambiguous/scene_815/split_0_predicted_mask.png
```

查看关键结果：

```bash
jq '{
  scene_id, split, annotation, localization_mode,
  raw_response, predicted_localization_id,
  predicted_npz_label, predicted_object_id,
  predicted_class_name, per_ground_truth_iou,
  ssr, rsr, status, excluded_from_statistics
}' rsr/data/output/reason/molmo/01_hard_ambiguous/scene_815/split_0.json
```

## 9. 当前能否复现论文 Table I

当前代码能够执行与 Table I 六列对齐的 FreeGrasp-only 主评测，但不能声称
复现论文中的原始数值。

已经具备：

- 六个 difficulty/ambiguity 单元格；
- 每个 scene 三条 free-form instruction；
- 主评测的 Molmo localization，以及可选 GT localization 消融；
- `gpt-4o` reasoning 和 LangSAM segmentation；
- predicted object ID、mask 和自定义 SSR/RSR；
- API/基础设施失败排除以及断点续跑。

与论文原表的差异：

- 当前是 120 scenes（每格 20），论文是 300 scenes（每格 50）；
- SSR/RSR 定义按本项目要求修改，和论文原定义不同；
- 不运行 ThinkGrasp；
- API 未完成的样本仍然缺失；
- 当前 `reports/summary.json` 只有 overall 和 localization mode 均值，还没有
  自动导出六列 `mean ± std` 表格。

因此，360 次 FreeGrasp 主评测全部完成后，现有结果足够生成两行六列的
自定义表格；但还需要增加一个六类分组及 `mean ± std` 导出步骤。得到的是
“Table I 布局下的 120-scene 自定义指标结果”，不是论文原数值复现。
