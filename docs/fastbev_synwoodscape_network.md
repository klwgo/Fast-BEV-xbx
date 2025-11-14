# Fast-BEV (SynWoodScape Fisheye) 模型结构概览

下文整理当前 `configs/woodscape/fastbev_synwoodscape_multitask.py` 中 Fast-BEV 的完整数据流与子模块，方便在迭代损失、结构或调参时快速定位关键组件。

## 输入与数据预处理

- **输入模态**：4 路鱼眼相机 (`CAM_FRONT/L/R/B`)，每帧只取一个时间戳 (`n_times=1`)。
- **Pipeline（训练）**
  1. `MultiViewPipeline`：并行读取 4 张原始图像。
  2. `LoadAnnotations3D`：只加载 2D 框 (`gt_bboxes/gt_labels`) 与 BEV 掩码 (`gt_bev_seg`)，SynWoodScape 没有真实 3D 框，因此 `with_bbox_3d=False`。
  3. `NormalizeMultiviewImage`：使用 ImageNet 均值方差。
  4. `DefaultFormatBundle3D` & `Collect3D`：打包成张量，输出 `img / gt_bboxes / gt_labels / gt_bev_seg` 及相关 meta。
- **测试 pipeline** 与训练相同但不加载 GT。

## 模型总览

```
多视角图像
   ↓ ResNet-50 (共享权重)
FPN (2D Feature Pyramid, 输出 4 个尺度 × 64 channel)
   ↓ neck_fuse (concat + 1×1 conv 拼接多视角)
   ↓ M2BevNeck (6 层 3D 卷积 -> BEV feature, 输出 192 channel, H×W≈520×540)
   ↓ WoodscapeBEVSegHead (Head-A/B)
      ↳ 主 BEV 语义预测 (6 类)
      ↳ Dice 辅助 & 多个 Binary BCELoss
   ↓ 2D FCOS Head (Head-D)
(可选) FreeAnchor3DHead 在 SynWoodScape 配置里被禁用
```

### Backbone & 2D Neck
- **Backbone**：ResNet-50 (ImageNet 预训练，SyncBN，冻结 stage-1)；对 4 个视角共享参数。
- **FPN**：将每个视角的 C2~C5 映射为 64 通道，形成 `[P2,P3,P4,P5]`。
- **neck_fuse**：直接把 4 视角同尺度特征在通道维上拼接（`base_bev_channels * num_views`），再用 1×1 Conv 降回 `fuse_out_channels=256`，为 3D BEV 颈部做输入。

### BEV 颈部 (M2BevNeck)
- `type='M2BevNeck'`，6 层 3D 卷积/上采样，将 `[num_views*base_bev_channels, H, W]` 映射到 192 通道的 BEV 特征，空间尺寸约为 `n_voxels=[[520,540,6]]`。
- 内部 `fuse` 模块把 2D 融合特征和 3D 累积特征再次融合；`is_transpose=False`，所以是普通卷积堆叠。

### WoodscapeBEVSegHead（Head-A/B）
- **Encoder**：两层 `Conv3×3 + BN + ReLU`，输出 96 通道。
- **预上采样**：`pre_upsample=dict(size=(675, 650))`，把 BEV 特征插值到与 GT 掩码一致的分辨率。
- **Refine head**：额外一层 3×3 Conv + BN + ReLU，抹平插值带来的锯齿。
- **分类层**：1×1 Conv → 6 通道 logits（road/free_space/lane/parking_line/other_ground_marking/zebra_crossing）。
- **Loss 体系**
  - 主 `CrossEntropyLoss`：类权重 `[9,11,8,8,6,0.5]`，整体 loss weight 0.8，配合 `SegLossWarmupHook` 在前 4k iter 内将权重从 0.3→0.8。
  - `DiceLoss`：与 CE 共用同一组权重，loss weight 0.25。
  - `binary_aux_losses`
    - lane_aux：`class_index=2`, `loss_weight=0.08`, `pos_weight=3.0`
    - veg_aux：`class_index=3`, `loss_weight=0.06`, `pos_weight=2.0`
    - ground_aux：`class_index=4`, `loss_weight=0.06`, `pos_weight=2.0`
    - 每个辅助损失仅对对应类别像素计算 BCEWithLogits，并继承 warmup 缩放。

### 2D FCOS Head（Head-D）
- 复用基础配置：4 层 tower，检测 SynWoodScape 提供的三个 2D 类别（行人/四轮/两轮）。
- `loss_cls = FocalLoss`，`loss_bbox = IoULoss`，`loss_centerness = BCE`。

### 3D Head
- 在 SynWoodScape 配置中 `bbox_head=None`，即关闭 Head-C（FreeAnchor3D），只训练 BEV + 2D。

## Loss 汇总
| 模块 | Loss | 说明 |
| --- | --- | --- |
| WoodscapeBEVSegHead | `loss_bev_seg` (CE) | 主语义分割，带类权重与 warmup |
|  | `loss_bev_seg_aux` (Dice) | 同步权重、防止稀有类梯度稀释 |
|  | `loss_lane_aux` | Lane 二值 BCE |
|  | `loss_veg_aux` | Vegetation 二值 BCE |
|  | `loss_ground_aux` | Ground 二值 BCE |
| FCOS 2D | `loss_cls`/`loss_bbox`/`loss_centerness` | 与原 Fast-BEV 相同 |

## 优化与调度
- **优化器**：AdamW2，基础 lr=1.5e-4，backbone 使用 0.1 × lr；`weight_decay=0.01`。
- **梯度裁剪**：`max_norm=25`.
- **LR 调度**：CosineAnnealing（按 epoch 迭代），500 iter 线性 warmup，最小 lr=5e-5。
- **自定义 Hook**：`SegLossWarmupHook` 在前 4000 iter 逐步放大 CE/Dice/BinaryLoss 的类权重，防止早期梯度爆炸。

## 输出
- **BEV 语义图**：形状 `[B, 6, 675, 650]`，经 softmax/argmax 后写入 `bev_seg`，用于 IoU/mIoU 评测以及后处理。
- **2D 检测结果**：FCOS 输出多视角 2D 框，可在 `eval_2d=True` 时计算 mAP。

## 备注
- `n_voxels` 与 GT 分辨率不同，SegHead 中的 `pre_upsample` 和 `refine_head` 负责对齐。
- 若需要继续调整类权重或新增辅助支路，只需在 `binary_aux_losses` 中追加配置，或修改 `class_weight` 与 `SegLossWarmupHook` 即可。
