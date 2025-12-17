# SynWoodScape FastBEV 模型结构说明

## 总览
针对合成 WoodScape（SynWoodScape）数据的多任务 BEV 语义模型，当前默认启用 Head-A（可行驶区），保留 Head-B/C 的标注加载，后续可开启标线/其他分支。

```mermaid
flowchart LR
    CAMS[4 路鱼眼图像] --> |MultiViewPipeline| IMG[多路图像张量]
    IMG --> |ResNet50| FEAT2D[FPN 多尺度特征]
    FEAT2D --> |NeckFuse concat| FUSE2D[融合特征<br/>64x4=256 通道]
    FUSE2D --> |Fisheye LUT 投影<br/>voxelize (675x650x6)| BEV3D[体素特征]
    BEV3D --> |M2BevNeck| BEV2D[BEV 平面特征 256c]
    BEV2D --> |MultiTask Head| OUTA[HeadA Drivable mask]
    OUTA --- GT_A[GT Drivable]
    %% HeadB/HeadC 标线等分支暂未启用
```

## 数据与标注
- 数据集：SynWoodScape（若切换 `dataset_choice='woodscape'` 可用真值 WoodScape；`hybrid` 可拼接）。
- 相机：4 路鱼眼（CAM_FRONT/FRONT_LEFT/FRONT_RIGHT/BACK）。
- 标注加载：`with_bev_seg=True`，同时加载 drivable / marking / boundary（即便标线头未开，标注仍加载以便后续开启）。
- BEV 范围与网格：`point_cloud_range=[-50, -50, -5, 50, 50, 3]`，`voxel_size=[0.35, 0.35, 1.0]`，`n_voxels=[675, 650, 6]`。

## 模型组件
- **Backbone**：ResNet-50（全训练，SyncBN）。
- **Neck(FPN)**：4 级输出，通道 64。
- **NeckFuse**：拼接 4 视角，通道 256。
- **Fisheye LUT 投影**：按相机内外参将 2D 特征投到 BEV 体素（默认缓存目录 `work_dirs/lut_cache`，若多进程写冲突可改为 `cache_dir=None`）。
- **3D Neck (M2BevNeck)**：输入 256 通道体素，1×1 融合 256×6→256，残差堆叠+stride=2 下采样，输出 BEV 平面 256c。
- **MultiTask Head**：FisheyeBEVMultiTaskHead，当前启用 drivable（Head-A）；标线/停车等分支参数保留但 `enable_heads.marking/slot/obstacle/occlusion=False`。
  - Drivable 损失：BCE，`loss_weight=3.0`，背景/前景权重 `[0.6, 1.0]`，支持 hard negative (`pos_topk_ratio=0.3`, `neg_pos_ratio=3`)，膨胀核 `drivable_dilate_kernel=11`。
  - Marking/Boundary/Focal 等损失配置保留，需开启头后生效。

## 训练与验证流水线
1. **MultiViewPipeline** 读取 4 路图像。
2. **LoadAnnotations3D**：加载 BEV 分割、3D 框（框可选；`filter_empty_gt=False`）。
3. **GenerateBEVMultitaskTargets**：生成 drivable/marking/boundary 目标。
4. **KittiSetOrigin**：对齐坐标原点。
5. **Data Aug**：`RandomScaleImageMultiViewImage` 轻量缩放。
6. **Normalize/Format/Collect**：输出键 `img`、`gt_drivable_mask`、`gt_marking_mask`、`gt_marking_boundary`。

## 训练配置要点
- 批大小：`samples_per_gpu=1`，`workers_per_gpu=2`（可按资源调）。
- 优化器：AdamW，`lr=6e-4`，weight_decay=0.01，`grad_clip max_norm=35`。
- 学习率：poly，warmup 1000 iters。
- 训练周期：40 epoch；`evaluation.interval=2`，默认 metric 关注 `headA`。
- 混合精度：`fp16 loss_scale='dynamic'`。
- 自定义可视化 Hook：`UndistortVisHook` 用于鱼眼去畸变可视化。

## 关键文件与路径
- 配置：`configs/woodscape/fastbev_syn_headabc.py`
- 数据 info：`./data/synwoodscape_infos_train.pkl`、`./data/synwoodscape_infos_val.pkl`（或 woodscape/hybrid 变体）。
- LUT 缓存：`work_dirs/lut_cache`（若关闭落盘，将 `cache_dir=None`）。
- 工作目录：`work_dirs/syn_headabc`

## 开启 Head-B/C 的提示
- 将 `enable_heads.marking=True`（必要时 `slot/obstacle/occlusion`）并保留对应 GT；调整损失权重（marking `loss_marking`/`loss_boundary` 已预设）。
- 留意显存：开启更多分支需适当降低 batch 或通道配置。
