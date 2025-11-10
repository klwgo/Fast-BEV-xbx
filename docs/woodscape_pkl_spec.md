# WoodScape Info（PKL）数据格式规范

本文档定义 Fast-BEV 在 WoodScape 数据集上的 `woodscape_infos_*.pkl` 必须包含的字段及取值约束，便于生成与校验。

## 1. 顶层结构
```python
{
    'metadata': {...},   # 数据集元信息
    'infos': [ ... ]     # 每个样本的详细标注
}
```

- `metadata`：
  - `version`：字符串，推荐使用 `woodscape-v1`、`woodscape-full-v1` 等。
  - `dataset`：字符串，例如 `WoodScape`。
  - `camera_types`：按顺序列出可用视角，需覆盖 Fast-BEV 使用的 4 目：`['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK']`。
  - `bev_seg`：字典，描述 BEV 语义掩码信息：
    - `class_names`：列表，约定顺序（例如 `['road_surface','free_space','lane_marking','parking_line','other_ground_marking','zebra_crossing']`）。
    - `mask_shape`：列表或元组，指定掩码尺寸 `[H, W, C]`，如 `[200, 200, 6]`。
  - `bbox2d_classes`：2D 检测类别列表，默认 `['vehicles','person','bicycle','traffic_light','traffic_sign']`。
  - 可附加 `split`（train/val/test）、`description` 等信息。

## 2. infos 列表
每个元素 `info` 代表一帧样本，必须包含如下键：

### 2.1 基本信息
| 键名 | 类型 | 说明 |
| --- | --- | --- |
| `token` | str | 唯一编号，建议使用 `woodscape-xxxxx`。 |
| `scene_name` / `frame_id` / `timestamp` | 可选，用于跟踪时序关系。 |
| `prev` / `next` | 可选，指向上一帧 / 下一帧 token。 |

### 2.2 传感器与姿态
- `cams`: dict，键为相机名（参考 `metadata.camera_types`）。每个相机包含：
  - `data_path`: 图像相对路径，例如 `data/woodscape/rgb_images/CAM_FRONT/xxx.png`。
  - `height` / `width`: 原始图像尺寸。
  - `timestamp`: 与帧同步的时间戳。
  - `sensor2ego_rotation` / `sensor2ego_translation`: 相机 → 车体位姿。
  - `ego2global_rotation` / `ego2global_translation`: 车体 → 全球坐标。
  - `sensor2lidar_rotation` / `sensor2lidar_translation`: 便于对齐 BEV。
  - `cam_intrinsic` / `cam_distortion`: 内参与畸变系数（鱼眼同场景设置相同）。
  - `annos`: 2D 标注字典（见下）。
- `lidar2ego_translation` / `lidar2ego_rotation`、`ego2global_*`: 若使用虚拟 LiDAR，可保持全零或单位矩阵。

### 2.3 3D 障碍物
- `gt_boxes`: `np.ndarray`，形状 `[N, 9]`，格式 `(x, y, z, w, l, h, yaw, vx, vy)`，单位与坐标系需与 Fast-BEV 配置一致（默认 LiDAR 坐标）。
- `gt_names`: `np.ndarray`，长度 N，类别名（`pedestrian`/`two-wheeler vehicle`/`four-wheeler vehicle`）。
- `gt_velocity`: `[N, 2]`，若无有效速度可填零。
- `num_lidar_pts` / `num_radar_pts` / `valid_flag`: `np.ndarray`，可保留原始统计或填默认值。

### 2.4 多视角 2D 标注（可选但推荐）
在 `info['ann_info']` 中提供：
- `mv_bboxes`: List[np.ndarray]，长度等于视角数，每个为 `[Mi, 4]` 的 2D 框（`[x1,y1,x2,y2]`）。
- `mv_labels`: List[np.ndarray]，与 `mv_bboxes` 对应的类别索引（0~K-1 对应 `metadata['bbox2d_classes']`）。
- `bboxes` / `labels`: 兼容传统 pipeline 的字段，内容同 `mv_bboxes` / `mv_labels`。
- 每个相机 `annos` 中需同步 `bbox`、`category_id`、`category_name`，便于调试可视化。

### 2.5 BEV / 语义标注（可选）
- `gt_bev_seg`: `np.ndarray` 或文件路径。默认使用 `[H, W, C]` 或 `[C, H, W]` 格式，训练时会自动识别并转为 `[N, C, H, W]`。
- `bev_seg_classes`: 与 `metadata['bev_seg']['class_names']` 保持一致。
- 若使用 PNG 存储，可在 `gt_bev_seg` 中保存路径，加载端负责读取。

### 2.6 动态掩码 / 其他（可选）
- `motion_seg_path`: 动态区域掩码路径（PNG），如需训练额外任务时使用。
- `semantic_paths`: dict，记录每个相机的语义标注 PNG 路径。
- 其他任务可按需扩展，例如车位属性、污渍标注等。

## 3. 与原始数据对齐
- 推荐使用 `tools/convert_woodscape_full.py` 脚本由原始 zip + 基础 info 自动生成增强版 `infos`，避免手工遗漏。
- 生成后应运行 `tools/verify_woodscape_info.py --ref-root ...` 校验：
  - 3D 框字段完整；
  - 2D 类别覆盖 `metadata['bbox2d_classes']`；
  - BEV 掩码存在且类别匹配；
  - 与原始数据比较，提示缺失类别。

## 4. 注意事项
1. 数组类型全部需使用 `np.ndarray`，不可混用 list；数据读取阶段会做类型检查。
2. 所有路径建议写为相对工程根目录的相对路径，方便跨环境协作。
3. 若训练时启用 `filter_empty_gt=False`，保持与配置一致即可；若设置为 True，应在生成 info 时过滤无 GT 的帧。
4. 元信息中记录 `split`、`bev_seg`、`bbox2d_classes` 等，便于后续脚本直接读取。

遵循上述规范，即可确保 Fast-BEV 多任务训练顺利执行。若扩展其他任务，只需在 `ann_info` 中增加对应字段并对应修改数据管线。***
