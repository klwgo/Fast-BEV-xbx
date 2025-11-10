# SynWoodScape 预训练验证手册

本文档记录了 SynWoodScape 数据划分的标注自检方法、三任务（3D 检测 / 多视角 2D 检测 / BEV 语义）评估命令，并简要介绍各指标的含义与注意事项。

## 1. 验证 val 划分的标注完整性

使用新增的 `tools/inspect_synwoodscape_split.py` 可快速统计任意 info.pkl 中 2D/3D/BEV 标注的覆盖率。既可直接指定 `--ann-file`，也可通过 `--config + --split` 自动读取配置中的路径。

```bash
python tools/inspect_synwoodscape_split.py \
    --ann-file data/synwoodscape_infos_full.pkl
# 或者
python tools/inspect_synwoodscape_split.py \
    --config configs/woodscape/fastbev_synwoodscape_pretrain.py \
    --split val
```

示例输出：

```
Loaded 28 samples from data/synwoodscape_infos_full.pkl
2D annotations: 28/28 (100.0%)
3D annotations: 28/28 (100.0%)
BEV masks: 28/28 (100.0%)
Examples:
  two_d: {'num_views': 4, 'per_view_shapes': [(42, 4), (60, 4), (34, 4), (31, 4)]}
  three_d: {'num_boxes': 1, 'sample_names': ['ego-vehicle']}
  bev: /mnt/.../semantic_annotations/gtLabels/00007_...
```

如果某一项覆盖率为 0，应先排查 info 生成流程（`tools/build_synwoodscape_info.py`）或配置里的 `ann_file` 是否指向正确路径，再进行训练/测试。

## 2. 训练与验证命令

### 2.1 训练（4 × RTX 4090 示例）

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun --nproc_per_node=4 --master_port=29531 \
    tools/train.py configs/woodscape/fastbev_synwoodscape_pretrain.py \
    --launcher pytorch \
    --work-dir work_dirs/synwoodscape_pretrain
```

配置文件已在 `evaluation` 字段中同时启用 `eval_3d=True, eval_2d=True, eval_bev=True`，每 5 个 epoch 自动输出三组指标。

### 2.2 单独测试

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun --nproc_per_node=4 --master_port=29531 \
    tools/test.py configs/woodscape/fastbev_synwoodscape_pretrain.py \
    work_dirs/synwoodscape_pretrain/latest.pth \
    --launcher pytorch \
    --eval bbox \
    --eval-options eval_3d=True eval_2d=True eval_bev=True score_thr=0.0
```

`--eval bbox` 仅用于满足脚本的操作检查，真正控制 3D/2D/BEV 评估的开关在 `--eval-options` 中。若只需其中部分指标，可将对应布尔值改为 `False`。

## 3. 指标说明

| 任务 | 指标 | 含义 | 备注 |
| --- | --- | --- | --- |
| 3D 检测 | `AP_{cls}@0.5`、`mAP@0.5` | 对 `gt_bboxes_3d` 与预测 3D 框在 IoU=0.5 下计算 AP | 需 `gt_boxes`/`gt_labels_3d`，若 SynWoodScape val 不含真实 3D，可在 config 中设置 `eval_3d=False` |
| 多视角 2D | `mAP_2d@0.5` | 将所有视角的 2D 预测（`mv_bboxes/mv_scores/mv_labels`）拼接后，以 IoU=0.5 统计 P-R 曲线面积 | 依赖 `PrepareSynWoodscape2DTargets` 在 pipeline 中整理 `mv_bboxes`，同时要求模型 simple_test 输出 per-view 2D 框 |
| BEV 语义 | `IoU_bev_*`、`mIoU_bev` | 对 `bev_seg_classes` 中的每个类别计算像素 IoU，并给出平均值 | 预测来自 `seg_head` 的 logits，评估阶段自动 argmax 后与 `gt_bev_seg` 对比；若 info 内未提供 `gt_bev_seg`，IoU 结果会是 NaN |

### 3D 检测

Fast-BEV 延续 FreeAnchor3D 头，以 IoU=0.5 的最近邻指标 (`bbox_overlaps_nearest_3d`) 评估。若 val split 缺失 3D GT，建议：

1. 将 `evaluation.eval_3d=False`，避免无意义的 0 分；
2. 或切换到 WoodScape 实采数据进行验证。

### 多视角 2D 检测

训练阶段利用 `bbox_head_2d` (FCOS) 对每个视角单独监督；推理阶段新增的 `_simple_test_multiview_2d` 会：

1. 根据 `img_metas` 构造逐视角的 `img_metas_2d`；
2. 调用 FCOS `simple_test_bboxes` 获取每个视角的框/置信度/类别；
3. 将结果打包为 `mv_bboxes/mv_scores/mv_labels`，供 `WoodScapeDataset._evaluate_multiview_bbox` 计算 mAP。

### BEV 语义分割

`seg_head` 输出的 BEV logits 会被写入 `result['bev_seg']`。数据集在评估时会：

1. 自动 `argmax` 到类别维；
2. 与 `ann_info['gt_bev_seg']` 对齐；
3. 对每个类别累计交并比并输出 `IoU_bev_<class>` 与 `mIoU_bev`。

若 val 数据只含少数类别或 mask 缺失，请先用 `tools/inspect_synwoodscape_split.py` 检查，或在 info 生成阶段补全。

## 4. 常见问题

1. **mAP 和 mIoU 始终为 0 / NaN**：多半是 val info 中缺少相应 GT 或 pipeline 未加载（例如 `PrepareSynWoodscape2DTargets` 仅用于训练）。使用第 1 节的脚本确认后，再检查 config 中的 pipeline 是否覆盖到 val/test。
2. **`tools/test.py` 报 “Address already in use”**：更换 `--master_port` 或结束占用同端口的旧进程。
3. **`tools/test.py` 报 “Please specify at least one operation”**：必须指定 `--eval`、`--out`、`--show` 等任意选项。

按照上述流程，即可在 SynWoodScape 预训练阶段同步获得 3D/2D/BEV 三项指标，便于后续迁移或 ablation。
