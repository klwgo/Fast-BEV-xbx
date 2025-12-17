# FastBEV-Fish 模型结构说明

本文梳理仓库内 FastBEV-Fish 的主干结构、关键开关以及多任务 BEV 头。流程图使用 Mermaid 描述。

## 总体流程

```mermaid
flowchart LR
    imgs[多视角鱼眼/透视图像 (B×V×3×H×W)] --> reshape[视角展开\n(B·V)×3×H×W]
    reshape --> backbone[2D Backbone\nResNet/VOVNet]
    backbone --> fpn[FPN 多尺度特征\nP2-P5]
    fpn --> fuse[neck_fuse (可选)\n跨尺度拼接+3×3 conv]
    fuse --> backproj[Back-Projection\n透视/鱼眼 LUT]
    backproj --> bev_feat[BEV 3D体素\n(B, C·Vz, X, Y, 1)]
    bev_feat --> neck3d[M2BevNeck\n3D→2D BEV 聚合]
    neck3d --> heads{{任务 Heads}}
    heads --> det3d[3D 检测头 (可选)]
    heads --> segbev[BEV 分割头 (可选)]
    heads --> multitask[BEV 多任务头\n(drivable/marking/slot/obstacle)]
    heads --> det2d[多视角 2D 检测头 (可选)]
```

### 模块要点
- **多视角 2D 编码**：`backbone` + `FPN`，支持 `neck_fuse` 对指定层做特征重采样并通道融合。
- **回投到 BEV**：
  - 透视：用相机外参计算投影矩阵，`backproject_inplace/backproject_vanilla` 将每层特征写入体素。
  - 鱼眼：先按标定生成/缓存 LUT（`FisheyeLUTCache + build_fisheye_lut`），直接索引回填，支持 `fusion_mode=mean/max`。
  - `style` 选项决定是否多尺度 BEV：`v1/v2` 单尺度，`v3/v4` 多尺度 BEV 并按 `multi_scale_3d_scaler` 对齐。
- **3D→2D 聚合**：`neck_3d`(M2BevNeck) 将体素特征压平到 BEV 平面，得到统一 `feature_bev`。
- **任务头**：
  - `bbox_head`：3D 检测（FreeAnchor3D 等）。
  - `seg_head`：BEV 语义分割。
  - `multitask_head`：FisheyeBEVMultiTaskHead，输出可行驶/标线/停车位/障碍/遮挡。
  - `bbox_head_2d`：多视角 2D 检测（可选）。

## Fisheye BEV 多任务头结构

```mermaid
flowchart TB
    feat[输入 BEV 特征 (B,C,H,W)] --> ctx[多尺度空洞卷积\n(d=1/2/3) 聚合上下文]
    ctx --> shared[Shared Block\n两层 3×3 Conv-BN-ReLU]
    shared --> drivable_adp[Drivable Adapter\n3×3 Conv-BN-ReLU]
    drivable_adp --> drivable_cls[Drivable Head\n3×3×2]
    shared --> marking_dec[Marking Decoder\n两层 3×3]
    marking_dec --> vh[Vertical/Horizontal\n(line_kernel conv)]
    vh --> mark_fuse[标线浅层融合\n(含 shallow skip)]
    mark_fuse --> mark_refine[Refine + High-res upsample]
    mark_refine --> marking_cls[Marking Head\n1×1×K]
    mark_refine --> boundary[Boundary Head\n1×1×1]
    shared --> slot[Slot Head (可选)]
    shared --> obstacle[Obstacle Head (可选)]
    shared --> occlusion[Occlusion Head (可选)]
```

- 损失：drivable/marking 支持 Sigmoid/CE；配置中可通过 `weight/ignore_index` 调整难例平衡；可选 Dice/Boundary 辅助。
- 正负采样：`pos_topk_ratio/neg_pos_ratio` 控制 top-k 选取，设为 `None` 时全量像素参与。

## 关键配置/开关

- `style`: `v1/v2`（单尺度 BEV，`v2` 含图像多尺度），`v3/v4`（多尺度 BEV）。
- `multi_scale_id`: 指定要保留/融合的 FPN 层；对应 `neck_fuse_{i}` 动态创建。
- `backproject`: `inplace`（快，默认）/`vanilla`（计算均值+显式 valid 掩码）。
- `fisheye_lut`：`camera_model='fisheye'` 时启用 LUT；可设置 `cache_dir/force_rebuild/fusion_mode` 等。
- `n_voxels` / `voxel_size`: 控制 BEV 范围与分辨率；多尺度时为列表。
- 任务开关：在配置 `enable_heads` 与对应损失段落启用/关闭具体分支。

## 训练/推理流程简述
1. 读入多视角图像，按视角展开送入 2D 主干 + FPN（可选 neck_fuse）。
2. 按相机标定将特征投影到 BEV 体素（鱼眼优先 LUT，透视走投影矩阵）。
3. M2BevNeck 聚合体素得到 BEV 特征。
4. 根据启用的头计算损失或推理输出：3D 框、BEV 分割、多任务 BEV（drivable/marking/...），可选多视角 2D 检测。

如需快速查看标签对齐/预测，可用 `tools/visualize_bev.py` 或 `tools/visualize_multitask_gt.py` 进行可视化。 
