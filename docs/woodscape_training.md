# WoodScape 多任务训练与评估指南

本文档说明如何使用当前代码库在 WoodScape 数据集上完成 **2D 检测 + 3D 检测 + BEV 语义分割** 的联合训练，并给出常用命令、关键参数解释以及输出结果格式。

---

## 1. 数据准备

1. **解压原始资源（首次执行）**
   ```bash
   mkdir -p /mnt/new_data/woodspace/woodscape_extracted
   unzip -qo /mnt/new_data/woodspace/WoodScape_ICCV19-*.zip \
     -d /mnt/new_data/woodspace/woodscape_extracted
   unzip -qo /mnt/new_data/woodspace/rgb_images-006.zip \
     -d /mnt/new_data/woodspace/woodscape_extracted
   unzip -qo "/mnt/new_data/woodspace/rgb_images(test_set)-005.zip" \
     -d /mnt/new_data/woodspace/woodscape_extracted
   unzip -qo /mnt/new_data/woodspace/previous_images-002.zip \
     -d /mnt/new_data/woodspace/woodscape_extracted
   unzip -qo "/mnt/new_data/woodspace/previous_images(test_set)-001.zip" \
     -d /mnt/new_data/woodspace/woodscape_extracted
   unzip -qo /mnt/new_data/woodspace/soiling_dataset-004.zip \
     -d /mnt/new_data/woodspace/woodscape_extracted/soiling_dataset_tmp
   mv /mnt/new_data/woodspace/woodscape_extracted/soiling_dataset_tmp/{train,test} \
      /mnt/new_data/woodspace/woodscape_extracted/soiling_dataset
   rm -rf /mnt/new_data/woodspace/woodscape_extracted/soiling_dataset_tmp
   unzip -qo /mnt/new_data/woodspace/woodscape_extracted/WoodScape_ICCV19/instance_annotations/instance_annotations.zip \
     -d /mnt/new_data/woodspace/woodscape_extracted/WoodScape_ICCV19/instance_annotations
   unzip -qo /mnt/new_data/woodspace/woodscape_extracted/WoodScape_ICCV19/motion_annotations/motion_annotations.zip \
     -d /mnt/new_data/woodspace/woodscape_extracted/WoodScape_ICCV19/motion_annotations
   ```

2. **构建基础 info（全量 43 类多边形）**
   ```bash
   python tools/build_woodscape_full_info.py \
     --raw-root /mnt/new_data/woodspace/woodscape_extracted/WoodScape_ICCV19 \
     --rgb-root /mnt/new_data/woodspace/woodscape_extracted/rgb_images \
     --output data/woodscape_infos_base_all.pkl \
     --split full
   ```

3. **准备 BEV 掩码（若暂无，可先写零矩阵）**
   ```bash
   mkdir -p /mnt/new_data/woodspace/bev_masks
   python - <<'PY'
import mmcv, numpy as np, os
info = mmcv.load('data/woodscape_infos_base_all.pkl')
for item in info['infos']:
    token = item['token']
    path = os.path.join('/mnt/new_data/woodspace/bev_masks', f'{token}.npy')
    if not os.path.exists(path):
        np.save(path, np.zeros((6, 200, 200), dtype=np.uint8))
PY
   ```

4. **生成增强版 info（写入 2D/BEV 路径）**
   ```bash
   python tools/convert_woodscape_full.py \
     --base-info data/woodscape_infos_base_all.pkl \
     --raw-root /mnt/new_data/woodspace/woodscape_extracted/WoodScape_ICCV19 \
     --bev-root /mnt/new_data/woodspace/bev_masks \
     --output data/woodscape_infos_full_all.pkl \
     --split full --overwrite
   ```

5. **验证 info 是否完整**
   ```bash
   python tools/verify_woodscape_info.py \
     data/woodscape_infos_full_all.pkl \
     --ref-root /mnt/new_data/woodspace/WoodScape_ICCV19-20251024T032614Z-1-003.zip
   ```
   - 脚本会对照原始数据检查 3D/2D/语义类别覆盖情况，并列出缺失项。

---

## 2. 训练命令

```bash
CUDA_VISIBLE_DEVICES=0,1 \
tools/dist_train.sh configs/woodscape/fastbev_woodscape_fisheye.py 2 \
    --cfg-options \
    data.train.ann_file=data/woodscape_infos_full_all.pkl \
    data.val.ann_file=data/woodscape_infos_full_all.pkl \
    data.test.ann_file=data/woodscape_infos_full_all.pkl
```

### 关键参数说明
- `configs/woodscape/fastbev_woodscape_fisheye.py`  
  - 已启用三分支：`bbox_head`（3D）、`bbox_head_2d`（2D）、`seg_head`（BEV 语义）。
  - `bbox2d_classes`、`bev_seg_classes` 在配置开头定义。若添加类别需同步修改。
  - `loss` 参数可在配置内直接调换，详细可参考 `docs/woodscape_loss_options.md`。
- `tools/dist_train.sh ... 2`  
  - 第二个参数为 GPU 数量；若单卡训练可改用 `tools/train.py`。
- `--cfg-options`  
  - 允许在命令行覆盖配置项，如更换 ann_file、调低 `samples_per_gpu` 等。
  - 常见示例：
    - 修改 batch size：`data.samples_per_gpu=1`
    - 替换损失：`model.seg_head.loss_seg.type=DiceLoss`
    - 关闭 2D 分支：`model.bbox_head_2d=None`

### 训练日志
- `log_config.interval` 默认 10，可在 config 中调整以打印更细粒度 loss。
- `evaluation.interval` 默认 5，训练第 5、10、… epoch 会在验证集上计算指标。

---

## 3. 测试与导出

1. **标准测试**
   ```bash
   CUDA_VISIBLE_DEVICES=0 tools/dist_test.sh \
     configs/woodscape/fastbev_woodscape_fisheye.py \
     work_dirs/fastbev_woodscape_fisheye/latest.pth \
     1 \
     --cfg-options data.test.ann_file=data/woodscape_infos_full_all.pkl \
     --eval mAP
   ```
   - 输出示例：
     ```
     2025-xx-xx INFO - WoodScape evaluation (IoU 0.50): {
       'AP_pedestrian@0.5': ...,
       'AP_four-wheeler vehicle@0.5': ...,
       'AP_two-wheeler vehicle@0.5': ...,
       'mAP@0.5': ...,
       'mAP_2d@0.5': ... (若 eval_2d=True)
     }
     ```

2. **导出结果文件**  
   - `dist_test.sh` 默认会在 `work_dirs/.../evaluation/` 下生成 json/pkl，可用于后续分析。

---

## 4. Head 模块详解

### 3D 检测：FreeAnchor3DHead
- **负责任务**：障碍物识别（车辆、行人、骑行者等）与近场立体目标；输出张量形状约为 `[B, sum_levels, num_classes/box_code]`，经过 decode 后生成每个体素候选的 3D 框集合。
- 配置中启用 `FreeAnchor3DHead`，共享 BEV 特征，并通过四组 anchor size 覆盖 WoodScape 常见目标（`configs/woodscape/fastbev_woodscape_fisheye.py:66`）。`pre_anchor_topk=25` 与 `bbox_thr=0.5` 控制正样本筛选强度，使自由锚点策略在点云稀疏场景下更稳。
- 头部同时输出分类、框回归与方向二分类，损失由 `FocalLoss + SmoothL1 + CrossEntropy` 组成（`configs/woodscape/fastbev_woodscape_fisheye.py:95`）。方向分支有助于约束航向角翻转问题。
- 训练阶段沿用 FreeAnchor“正负 bag”选择逻辑，并在 IoU 计算前对 GT/预测框维度对齐，兼容 WoodScape 9 维速度标注（`mmdet3d/models/dense_heads/free_anchor3d_head.py:15`）。

### BEV 语义分割：WoodscapeBEVSegHead
- **负责任务**：可行驶区域识别（Road / Free space 等），输出尺寸为 `[B, num_classes, H_bev, W_bev]`，可将每个像素理解为 `H × W × C` 的栅格概率。
- 结构为若干 3×3 Conv-BN-ReLU 堆叠 + 1×1 分类头，默认两层中间通道 128，输出类别数与 `bev_seg_classes` 一致（`configs/woodscape/fastbev_woodscape_fisheye.py:60`）。
- `loss_seg` 默认使用交叉熵，脚本中包含上采样逻辑，自动对齐标签尺寸；也可在配置中替换为 Dice/Focal 等损失（`mmdet3d/models/decode_heads/bev_seg_head.py:15`）。
- 若只做检测，可将 `model.seg_head=None` 禁用该分支，节省显存。

### 多视角 2D 检测：FCOSHead
- **负责任务**：标志物识别（交通标志/信号灯等）与多视角 2D 目标检测，输出特征为四个金字塔级别 `[B, C, H_i, W_i]`，经 decode 得到图像坐标系下的框列表。
- 使用 FCOS 单阶段策略，四个 stride（4/8/16/32）协同回归多尺度框（`configs/woodscape/fastbev_woodscape_fisheye.py:104`）。每层特征通道 64，可覆盖车辆/行人/标志等前视任务。
- 损失包含分类 Focal、IoU 回归、centerness CrossEntropy，兼顾定位质量；`train_cfg_2d`/`test_cfg_2d` 定义 IoU 门限、NMS 参数，便于快速裁剪目标库（`configs/woodscape/fastbev_woodscape_fisheye.py:115`）。
- 若部署时不需要 2D 监督，可在配置里将 `bbox_head_2d`、`train_cfg_2d`、`test_cfg_2d` 置空，训练脚本会跳过正负样本准备。

> 备注：车位识别当前未接入独立 head，如需后续扩展，可复用 BEV 特征新增语义/检测分支。

---

## 5. 输出格式说明

训练过程中 `EvaluateHook` 会输出：
- `AP_<class>@0.5`：3D BEV mAP（IoU=0.5）。
- `mAP@0.5`：3 类平均。
- `mAP_2d@0.5`：多视角 2D 简化指标（如启用 `eval_2d=True`）。
- 其他任务（如 BEV 语义）通常通过 loss 查看；若需 IoU/F1 指标，可在 dataset 里进一步扩展。

模型推理结果 `results[idx]` 包含：
- `boxes_3d`：`BaseInstance3DBoxes` 对象（`tensor` shape [N, 9]）。
- `scores_3d`、`labels_3d`：对应得分与类别索引。
- 若 pipeline 收集了多视角 2D，可额外包含 `mv_bboxes`、`mv_scores`、`mv_labels`。

---

## 6. 常见问题
1. **无 BEV 掩码**：`convert_woodscape_full.py` 未找到 `--bev-root` 时会只记录路径，需要在训练时自行加载或改写脚本。
2. **显存不足**：调低 `samples_per_gpu` 或裁剪输入尺寸。
3. **类别缺失**：使用 `tools/verify_woodscape_info.py` 检查是否所有类别都在 info 中出现。

---

如需新增任务（例如动态掩码、车位分类），可在生成 info 时补充相应字段，并在 pipeline / config 中扩展分支与损失。
