# WoodScape 多任务损失组合说明

本文档总结了当前 Fast-BEV WoodScape 方案中可选的损失函数，方便后续实验进行对比与调参。

## 3D 检测（FreeAnchor3DHead）
- `loss_cls`：FocalLoss（可选参数：`gamma`、`alpha`、`loss_weight`）。可替换为 CrossEntropyLoss、GHM 等。
- `loss_bbox`：SmoothL1Loss（`beta`、`loss_weight` 可调），可改为 L1Loss、IoULoss、GIoULoss。
- `loss_dir`：CrossEntropyLoss（方向分类），可替换为 FocalLoss、BCEWithLogitsLoss。

## 2D 检测（FCOSHead）
- `loss_cls`：FocalLoss，与 3D 检测类似。
- `loss_bbox`：IoULoss，可替换为 GIoU / DIoU / CIoU。
- `loss_centerness`：CrossEntropyLoss（sigmoid 形式），可替换为 BCEWithLogitsLoss。

## BEV 语义分割（WoodscapeBEVSegHead）
- 默认 `CrossEntropyLoss`。可通过 config 中的 `loss_seg` 替换为 DiceLoss、FocalLoss、LovaszLoss 等。

### 调参示例
```python
seg_head=dict(
    type='WoodscapeBEVSegHead',
    in_channels=256,
    num_classes=len(bev_seg_classes),
    loss_seg=dict(type='DiceLoss', loss_weight=1.0)
)
```

### 注意事项
1. 修改损失时，请同步调整 `loss_weight`，确保各任务梯度规模相近。
2. 若新增损失实现，需要在 `mmdet/models/losses/__init__.py` 注册。
