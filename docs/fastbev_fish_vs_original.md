# Fast-BEV-Fish vs. 原始 Fast-BEV（dev 分支）差异详解

本文从模块结构、类继承关系、数据集读取、任务 Head、损失选择以及配置改动等角度，整理 Fast-BEV-Fish 相较于 Sense-GVT/Fast-BEV（dev）版本的主要差异，帮助开发者快速了解二次开发的重点。

---

## 1. 项目结构与模块概览

### 1.1 原始项目核心层次（dev 分支）
- `mmdet3d/models/detectors/fastbev.py`：主模型入口，继承自 `BaseDetector`，负责图像特征提取、BEV 构建以及多任务头部的 forward。
- `mmdet3d/models/dense_heads/free_anchor3d_head.py`：3D 检测 head，基于 FreeAnchor 思想实现 BEV 上的 anchor 匹配与损失。
- `mmdet3d/models/seg_heads/woodscape_bev_seg_head.py`：BEV 语义分割 head。
- `configs/`：提供 NuScenes/WoodScape 等数据集的配置示例。
- 数据集部分以 NuScenes 为主，WoodScape 的支持较为基础，SynWoodScape 未直接集成。

### 1.2 Fast-BEV-Fish 新增/改动模块
- **数据转换脚本**  
  - `tools/build_synwoodscape_info.py`：新增 SynWoodScape -> Fast-BEV info 的全量转换脚本。  
  - `tools/build_woodscape_full_info.py` + `tools/convert_woodscape_full.py`：在原流程基础上补齐多任务字段。

- **数据集类增强**  
  - `mmdet3d/datasets/woodscape_dataset.py`：继承自 `NuScenesMultiViewDataset`，增加了 WoodScape/SynWoodScape 专属字段处理（详见 3.2）。

- **配置与训练流程**  
  - `configs/woodscape/fastbev_synwoodscape_pretrain.py`：新增 SynWoodScape 预训练配置，重启 2D/BEV 任务。  
  - 支持通过环境变量切换数据路径，使集群部署更灵活。

- **文档与工具**  
  - `docs/woodscape_training.md`、`docs/fastbev_fish_enhancements.md` 等：新增训练指南与改动说明。

---

## 2. 模型结构与类层次

### 2.1 Fast-BEV 主体（`mmdet3d/models/detectors/fastbev.py`）
- 原始结构保留：`FastBEV` 继承自 `BaseDetector`，在 `__init__` 中构建 backbone、FPN、neck_fuse、neck_3d、seg_head、bbox_head、bbox_head_2d。
- **Fast-BEV-Fish 改动重点**：
  - 在 `extract_feat()` 中加入对 `lidar2img` 结构的兼容：读取每个相机的内参（`intrinsics`）、畸变（`distortions`）、鱼眼模型（`models`），为 `fisheye_lut` 构建提供完整数据。
  - `forward_train()` 支持同时返回 3D、2D、BEV 三任务 losses，驱动多任务训练。
  - 保留 SyncBN + 鱼眼 LUT 的流程，确保 WoodScape/SynWoodScape 能正确转换特征。

### 2.2 3D Head（`mmdet3d/models/dense_heads/free_anchor3d_head.py`）
- 原始 Fast-BEV 基于 FreeAnchor，将 LiDAR/Radar 的 3D 检测适配到 BEV 特征。
- **Fast-BEV-Fish** 保持原逻辑，仅在配置中将 `class_names` 固定为三类（`('pedestrian', 'four-wheeler vehicle', 'two-wheeler vehicle')`），与数据转换后的类别一致。

### 2.3 2D Head（FCOS）
- 原始 dev 分支中的 WoodScape 示例同样使用 FCOS；Fast-BEV-Fish 复用但更新了类别数与训练流水线，确保 `mv_bboxes` 和 `mv_labels` 被加载。

### 2.4 BEV Segmentation Head
- `WoodscapeBEVSegHead` 结构保持不变；Fast-BEV-Fish 更新了类别列表与掩码路径，使用六类（road_surface, lane_marking, sidewalk, vegetation, ground, background）。

---

## 3. 数据集读取与增强

### 3.1 SynWoodScape 转换（`tools/build_synwoodscape_info.py`）
- 主要函数：
  - `load_calibrations()`：读取 `calibration_data/*.json`，转换为 Fast-BEV 坐标系的姿态矩阵。
  - `load_vehicle_pose()`：解析 `vehicle_data/rgb_images/*.txt`，提取帧级位姿，写入 `ego2global_rotation/translation`。
  - `corners_to_box()`：将 8 点轮廓还原为 BEV 3D 框（包含中心、长宽高、朝向）。
  - `convert_bev_semantic()`：将 BEV 语义 PNG 重映射为六类栅格，同时保存 `.npy`。
  - 主循环中整合 3D/2D/BEV、前后帧图像路径、标定数据，并输出 `infos + metadata`。

### 3.2 WoodScape 数据集类（`mmdet3d/datasets/woodscape_dataset.py`）
- `WoodScapeMultiViewDataset` 重载 `load_annotations()`，对 `cams`、`ann_info` 进行深拷贝处理：  
  - `_process_cams()`：补齐相机内外参、畸变参数，转换为 numpy 数组；若缺失则填充默认值。  
  - `get_data_info()`：在返回字典中增加 `lidar2img=...`，包含 `intrinsics/distortions/models` 字段，便于前向阶段使用。  
  - `get_ann_info()`：在父类基础上合并多任务标注（MV 2D 框、BEV 掩码、motion 路径），为流水线提供完整输入。

### 3.3 加载流水线（`configs/woodscape/fastbev_synwoodscape_pretrain.py`）
- `MultiViewPipeline`：从 info 中挑选多相机图像，并保持与标注的一致性。
- `LoadAnnotations3D(with_bbox=True, with_bev_seg=True, with_bbox_mv=True)`：一次性加载 3D 框、2D 框、BEV 掩码与多视角标注。
- `Collect3D`：同步 `gt_bboxes`、`gt_labels`、`gt_bboxes_3d`、`gt_labels_3d`、`gt_bev_seg`，满足三任务训练需求。

---

## 4. 任务 Head 与损失配置

### 4.1 3D 检测 (FreeAnchor3DHead)
- Anchor 配置 (`configs/woodscape/fastbev_woodscape_fisheye.py`)：
  ```python
  anchor_generator=dict(
      type='AlignedAnchor3DRangeGenerator',
      ranges=[[-50, -50, -1.8, 50, 50, -1.8]],
      sizes=[
          [0.8660, 2.5981, 1.],
          [0.5774, 1.7321, 1.],
          [1., 1., 1.],
          [0.4, 0.4, 1],
      ],
      rotations=[0, 1.57],
      reshape_out=True)
  ```
- 损失：  
  - `FocalLoss` (分类)  
  - `SmoothL1Loss` (bbox)  
  - `CrossEntropyLoss` (方向)

### 4.2 多视角 2D 检测 (FCOSHead)
- 通过 `bbox_head_2d` 定义，共三类。  
- 损失组合：Focal (cls) + IoULoss (bbox) + CrossEntropy (centerness)。

### 4.3 BEV 语义分割
- `WoodscapeBEVSegHead` 输出六类。  
- 默认使用 `CrossEntropyLoss`；可在配置中改为 Dice/Focal。  
- 输入是聚合后的 BEV 特征（`feature_bev`），标签来自 `gt_bev_seg`。

---

## 5. 后期修改指引

- **新增/调整数据字段**：  
  - 在 `build_synwoodscape_info.py` 中添加字段（如动态掩码、光流），再在 `WoodScapeMultiViewDataset.get_ann_info()` 中取回。  
  - 修改配置管线中的 `LoadAnnotations3D` 参数，决定是否加载新字段。

- **增删任务分支**：  
  - 在 `configs/woodscape/fastbev_synwoodscape_pretrain.py` 中设置 `model.seg_head=None` 或 `model.bbox_head_2d=None` 可以快速关闭任务。  
  - 若需要新增 head，可在 `mmdet3d/models/detectors/fastbev.py` 中添加 forward 逻辑，并在配置里加入相应 head 定义。

- **损失函数调整**：  
  - 可在配置中直接覆盖 `model.<head>.loss_*`，如 `model.seg_head.loss_seg=dict(type='DiceLoss', loss_weight=1.0)`。

- **SyncBN 与单卡调试**：  
  - 如需在单卡上运行，需将 `norm_cfg=dict(type='SyncBN', ...)` 改为 `dict(type='BN', ...)`，或者改用 `tools/dist_train.sh` 多卡启动。

- **验证与调试**：  
  - 生成 info 之后，建议使用 `python tools/verify_woodscape_info.py`（WoodScape）或 `python - <<'PY' ...` 快速检查 SynWoodScape info 的关键字段。  
  - 训练前可运行一次配置加载和前向冒烟测试，确认路径、标注、损失输出正常。

- **目录结构约定**：  
  - 数据整理后默认位于 `/mnt/new_data/woodspace/{woodscape,synwoodscape}`，项目通过环境变量适配。  
  - 自生成的 BEV 掩码放在 `<pkl>_bev_masks/` 目录中，训练时需确保路径可读。

---

## 6. 后续优化方向

1. **模态扩展**：在 SynWoodScape info 中加入光流、深度图等模态的读取与转换，探索多模态训练。  
2. **自动化测试**：为数据生成脚本加入最小化样例与 pytest，确保升级或重构后字段不会缺失。  
3. **训练脚本封装**：将 SynWoodScape 预训练 → WoodScape 微调 → 评估统一封装为 bash/python 脚本，降低使用门槛。  
4. **指标体系完善**：针对多视角 2D 检测，提供 per-camera 指标；对 BEV 语义增补 IoU、F1 等评估函数。  
5. **配置模板化**：拆分公共配置（模型结构）与数据差异配置（数据集、任务标签），提升维护效率。

---

如需深入了解某一模块，可在项目中搜索对应脚本或类名称，结合本文档定位改动位置，进一步阅读源码。欢迎继续完善本说明。  
