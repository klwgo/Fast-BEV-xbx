# WoodScape 与 SynWoodScape 数据集分析

## 信息来源
- DeepWiki《WoodScape Dataset》页面（2025-09 索引），提供数据总览、任务定义与多摄像机说明。
- 本地路径 `/mnt/new_data/woodspace/woodscape_extracted/` 中的 WoodScape 与 SynWoodScape 数据快照（分析日期：2025-11-05）。
- `/mnt/new_data/woodspace` 内全部 `.zip` 包已解压至对应目录（使用 `unzip -n` 保留原档案），以下统计均基于解压后的最新文件结构。

> 提示：本文将 DeepWiki 的官方描述与本地文件统计结果对照，可直接用于训练前的资料共享或配置文档。

## WoodScape（真实采集数据）

### 目录与内容概览
- `rgb_images/`、`rgb_images_test/`：四路环视（FV、RV、MVL、MVR）RGB 图，分为训练与测试子集，命名规则 `{frame_id}_{camera}.png`。
- `previous_images/`、`previous_images_test/`：与当前帧配对的上一时刻图像 `{frame_id}_{camera}_prev.png`，用于运动/光流任务。
- `WoodScape_ICCV19/`：官方标注包，包含 2D 框、实例多边形、语义/运动分割（部分仍为 `.zip`）、车辆 CAN 数据及标定参数。
- 其余压缩包（如 `SynWoodScape_V0.1.1-002.zip`、`soiling_dataset-004.zip`）保留原始发布内容，可按需解压。

### 核心模态规模（本地统计）

| 子集 | 文件总数 | FV | MVL | MVR | RV |
| --- | --- | --- | --- | --- | --- |
| `rgb_images` | 8,234 | 2,037 | 2,066 | 2,145 | 1,986 |
| `rgb_images_test` | 1,766 | 442 | 441 | 441 | 442 |
| `previous_images` | 8,234 | 2,037 | 2,066 | 2,145 | 1,986 |
| `previous_images_test` | 1,766 | 442 | 441 | 441 | 442 |
| `WoodScape_ICCV19/box_2d_annotations` | 8,234 TXT | 2,037 | 2,066 | 2,145 | 1,986 |
| `WoodScape_ICCV19/instance_annotations` | 8,234 JSON | 2,037 | 2,066 | 2,145 | 1,986 |
| `WoodScape_ICCV19/semantic_annotations` | 8,234 PNG（gtLabels/rgbLabels） | 2,037 | 2,066 | 2,145 | 1,986 |
| `WoodScape_ICCV19/motion_annotations` | 8,234 PNG（gtLabels/rgbLabels） | 2,037 | 2,066 | 2,145 | 1,986 |

> DeepWiki 指出 WoodScape 含 10,000 帧、四路环视图像，其中约 8,200 帧配有时间配对前帧、1,800 帧为测试集；本地统计与官方规模一致，其中部分任务（语义/运动分割）仍保留在压缩档中。

### 标注类别小结

#### 实例多边形（41 类，772,167 个对象）

以 `instance_annotations/instance_annotations/*.json` 直接统计得到的对象数量如下（四路环视分布与 RGB 样本一致）：

| 类别 | 对象数 |
| --- | --- |
| animal | 128 |
| bicycle | 19,759 |
| bus | 1,006 |
| car | 60,666 |
| caravan | 27 |
| cats_eyes_and_botts_dots | 662 |
| construction | 106,036 |
| curb | 35,338 |
| ego_vehicle | 18,037 |
| fence | 7,100 |
| free_space | 45,618 |
| green_strip | 15,011 |
| grouped_animals | 2 |
| grouped_botts_dots | 33 |
| grouped_pedestrian_and_animals | 3,219 |
| grouped_vehicles | 6,888 |
| lane_marking | 62,662 |
| motorcycle | 5,755 |
| movable_object | 641 |
| nature | 45,808 |
| other_ground_marking | 35,865 |
| other_wheeled_transport | 11,627 |
| parking_line | 43,468 |
| parking_marking | 417 |
| person | 24,759 |
| pole | 79,328 |
| rider | 7,881 |
| road_surface | 50,773 |
| sky | 25,939 |
| traffic_light_green | 748 |
| traffic_light_red | 680 |
| traffic_light_yellow | 68 |
| traffic_sign | 6,992 |
| trafficsign_indistingushable | 9,272 |
| trailer | 200 |
| train/tram | 435 |
| truck | 1,904 |
| unknown_traffic_light | 3,837 |
| van | 4,576 |
| void | 8,754 |
| zebra_crossing | 20,248 |

#### 2D 边界框（5 类，72,063 个框）

基于 `box_2d_annotations/box_2d_annotations/*.txt` 统计全部框数量：

| 类别 | 框数 |
| --- | --- |
| bicycle | 7,053 |
| person | 16,249 |
| traffic_light | 1,500 |
| traffic_sign | 2,857 |
| vehicles | 44,404 |

#### 语义分割掩码（10 类，8,234 张）

- `semantic_annotations/semantic_annotations/gtLabels` 与 `rgbLabels` 已全部解压，可直接读取；各视角文件数与 RGB 样本一致（FV 2,037 / MVL 2,066 / MVR 2,145 / RV 1,986）。
- 标签集合及其在图像中的出现次数（按 `seg_annotation_info.json` 定义）：

| 类别 | 帧数 |
| --- | --- |
| void (0) | 8,234 |
| road (1) | 8,233 |
| lanemarks (2) | 7,966 |
| curb (3) | 7,804 |
| person (4) | 6,336 |
| rider (5) | 3,341 |
| vehicles (6) | 8,137 |
| bicycle (7) | 4,895 |
| motorcycle (8) | 2,905 |
| traffic_sign (9) | 3,243 |

- 语义掩码为类别索引图，不携带实例 ID，无法直接统计实例数量；如需实例级信息需结合多边形标注。

#### 运动分割掩码（8,234 张）

- `motion_annotations/motion_annotations/gtLabels` / `rgbLabels` 已解压，视角分布与 RGB 样本一致。
- 掩码同样以类别索引表示动态目标，本地统计到的标签及出现帧数如下（0 表示背景）：

| 标签值 | 类别 | 帧数 |
| --- | --- | --- |
| 0 | background | 8,234 |
| 1 | animal | 19 |
| 2 | rider | 1 |
| 3 | grouped_pedestrian_and_animals | 357 |
| 4 | grouped_animals | 2,501 |
| 5 | bicycle | 4,121 |
| 6 | person | 2,253 |
| 7 | motorcycle | 706 |
| 9 | car | 5,200 |
| 11 | truck | 709 |
| 12 | caravan | 4 |
| 13 | bus | 267 |
| 14 | van | 333 |
| 15 | dynamic_van | 22 |
| 16 | train_tram | 126 |
| 17 | grouped_vehicles | 260 |
| 19 | other_wheeled_transport | 65 |

- 未检测到 `dynamic_car`、`moveable_objects` 等标签，`dynamic_van` 仅在 22 帧中出现；掩码不包含实例 ID。

#### Soiling
- `soiling_dataset` 仅包含 `soiling_annotation_info.json`，训练/测试划分列表仍位于未解压的源压缩包中。

### 标定与传感器信息
- `calibration_data/*.json` 采用 4 次多项式径向模型，包含每个相机的内外参、分辨率、主点偏移；车辆坐标系遵循 ISO 8855。
- `vehicle_data/vehicle_info.zip` 打包 CAN 总线信息、车速/转角等，可用于时序感知任务。

## SynWoodScape（仿真数据）

### 目录与内容概览
- `rgb_images/`、`previous_images/`：每帧包含 BEV + 四路环视五个视角的 PNG 图像（前者为当前帧，后者为 `_prev` 时序帧）。
- `box_2d_annotations/`、`box_3d_annotations/`：2D 矩形框 TXT 与 3D 包围盒 `.pkl`，后一者按帧聚合。
- `semantic_annotations/`、`instance_annotations/`、`motion_annotations/`：均拆分为 `gtLabels/`（索引/实例 ID 掩码）与 `rgbLabels/`（彩色可视化）。
- `optical_flow/`、`depth_maps/`、`dvs_signals/`：分别提供原始数据（`.flo`/`.npy`）与彩色渲染 PNG。
- `lidar_data/`、`distances_traveled/`、`vehicle_data/`：记录点云、行驶里程及载具状态参数。

### 核心模态规模（本地统计）

| 模态 | 文件总数 | 视角分布（BEV/FV/MVL/MVR/RV） | 说明 |
| --- | --- | --- | --- |
| `rgb_images` | 2,500 | 500 / 500 / 500 / 500 / 500 | 1280×966（环视）+ 1024×1024（BEV）PNG |
| `previous_images` | 2,500 | 500 / 500 / 500 / 500 / 500 | `_prev` 对应上一帧 |
| `box_2d_annotations` | 2,500 | 500 / 500 / 500 / 500 / 500 | 每视角 TXT |
| `box_3d_annotations` | 500 | 每帧 1 份 | `.pkl`，包含多对象矩阵 |
| `semantic_annotations/gtLabels` | 2,500 | 500 / 500 / 500 / 500 / 500 | 单通道 PNG |
| `semantic_annotations/rgbLabels` | 2,500 | 500 / 500 / 500 / 500 / 500 | 彩色可视化 PNG |
| `instance_annotations/gtLabels` | 2,500 | 500 / 500 / 500 / 500 / 500 | 实例 ID（8-bit）PNG |
| `instance_annotations/rgbLabels` | 2,500 | 500 / 500 / 500 / 500 / 500 | 彩色实例可视化 |
| `motion_annotations/gtLabels` | 2,500 | 500 / 500 / 500 / 500 / 500 | 前景/背景掩码 |
| `motion_annotations/rgbLabels` | 2,500 | 500 / 500 / 500 / 500 / 500 | 彩色示意 |
| `optical_flow/raw_data` | 2,500 | 500 / 500 / 500 / 500 / 500 | `.flo` |
| `optical_flow/color_coded` | 2,500 | 500 / 500 / 500 / 500 / 500 | `.png` |
| `depth_maps/raw_data` | 2,500 | 500 / 500 / 500 / 500 / 500 | 浮点 `.npy` |
| `depth_maps/colormap` | 2,500 | 500 / 500 / 500 / 500 / 500 | 伪彩 `.png` |
| `dvs_signals/raw_data` | 2,500 | 500 / 500 / 500 / 500 / 500 | 事件张量 `.npy` |
| `dvs_signals/color_coded` | 2,500 | 500 / 500 / 500 / 500 / 500 | 可视化 `.png` |
| `lidar_data` | 500 | 每帧 1 份 | 点云 `.pkl` |
| `distances_traveled` | 500 | 每帧 1 份 | 里程 `.txt` |

### 标注类别小结

#### 2D 边界框（3 类，113,018 个框）

基于 `box_2d_annotations/*.txt` 统计全部框数量：

| 类别 | 框数 |
| --- | --- |
| four-wheeler vehicle | 20,578 |
| pedestrian | 85,211 |
| two-wheeler vehicle | 7,229 |

#### 语义掩码（22 个标签值，2,500 张）

- `semantic_annotations/gtLabels` 与 `rgbLabels` 已平整解压；每个视角 500 张。
- 掩码仅提供类别索引，下表列出了实际出现的标签及其覆盖帧数（可结合项目内映射表还原名称）：

| 标签值 | 覆盖帧数 |
| --- | --- |
| 1 | 2,301 |
| 2 | 1,258 |
| 3 | 2,411 |
| 4 | 2,473 |
| 5 | 2,500 |
| 6 | 2,499 |
| 7 | 2,500 |
| 8 | 2,500 |
| 9 | 2,500 |
| 10 | 2,297 |
| 11 | 1,059 |
| 12 | 2,339 |
| 13 | 2,416 |
| 14 | 1,368 |
| 15 | 157 |
| 16 | 444 |
| 18 | 1,696 |
| 19 | 767 |
| 20 | 1,857 |
| 21 | 2,018 |
| 22 | 2,402 |
| 24 | 2,500 |

#### 实例掩码
- `instance_annotations/gtLabels` 共 2,500 张实例 ID 掩码（8-bit），跨数据集出现 233 个不同的实例 ID。
- 实例统计（非零 ID）：

| 视角 | 图像数 | 实例总数 | 单帧平均实例数 |
| --- | --- | --- | --- |
| BEV | 500 | 7,069 | 14.14 |
| FV | 500 | 20,443 | 40.89 |
| MVL | 500 | 32,744 | 65.49 |
| MVR | 500 | 24,773 | 49.55 |
| RV | 500 | 30,489 | 60.98 |
| **合计** | 2,500 | 115,518 | 46.21（平均） |

- 单帧实例数范围：1–162，中位数 42；`rgbLabels` 与 `gtLabels` 完全对齐，可用于可视化。

#### 运动掩码

- `motion_annotations/gtLabels` 与 `rgbLabels` 含 2,500×5 视角共 12,500 张两值掩码（0＝静态、255＝动态），适合用于运动前景提取；掩码不区分实例。

#### 其他模态
- `optical_flow/raw_data` 与 `optical_flow/color_coded` 提供光流原始 `.flo` 与彩色图，帧数与视角与 RGB 一致。
- `depth_maps/raw_data`、`depth_maps/colormap`、`dvs_signals/raw_data`、`dvs_signals/color_coded` 分别保存深度与事件相机的数值/可视化版本。
- `box_3d_annotations/*.pkl` 结合 `lidar_data/*.pkl` 可直接用于 3D 目标检测或多传感器融合。

### 标定与传感器信息
- `calibration_data/*.json` 提供 BEV/FV/MVL/MVR/RV 五个视角的内外参及多项式畸变系数，与 WoodScape 真实数据的参数格式保持一致。
- `vehicle_data/`、`distances_traveled/` 等文本可用于重建车辆姿态、速度等补充信息。

## 实拍 vs 仿真对比要点

| 维度 | WoodScape | SynWoodScape |
| --- | --- | --- |
| RGB 图像数量 | 8,234（FV 2,037 / MVL 2,066 / MVR 2,145 / RV 1,986） | 2,500（BEV + 四路，各 500） |
| 时间配对帧 | 8,234 `_prev` 图 | 2,500 `_prev` 图 |
| 2D 边界框 | 5 类 / 72,063 框 | 3 类 / 113,018 框 |
| 实例掩码 | 8,234 JSON，多边形共 772,167 个对象 | 2,500 `gtLabels`，233 个实例 ID（总结统计见上） |
| 语义掩码 | 10 类 gtLabels/rgbLabels（帧数统计已列） | 22 个标签索引（帧数统计已列） |
| 运动掩码 | 17 个有效标签（帧数统计已列，掩码无实例 ID） | 0 / 255 两值掩码（不区分实例） |
| 附加模态 | soiling、vehicle_data（部分在压缩包） | 光流、深度、DVS、点云、3D 框等均展开 |
| 标定 | `calibration_data/*.json`（FV/MVL/MVR/RV） | `calibration_data/*.json`（新增 BEV） |

- WoodScape 更贴近真实采集，但存在视角不均衡与压缩档依赖；SynWoodScape 则提供高度对称、模态丰富的仿真数据，适合多任务快速实验。
- 组合使用时，可利用 WoodScape 的真实场景多样性校验模型泛化，再借助 SynWoodScape 扩充极端场景或多传感器训练。

## 使用建议
1. 若重新同步官方 WoodScape 包，请确认 `semantic_annotations` 与 `motion_annotations` ZIP 已解压，并按项目既定标签映射整理掩码。
2. 进行多模态训练时，可将两套数据的外参 JSON 对齐到统一坐标系，并在采样策略中显式区分真实/仿真来源。
3. 若需同步维护 2D/3D 检测任务，可在 SynWoodScape 的 `.pkl` 3D 标注与 WoodScape 的 JSON 多边形之间建立转换脚本，确保标签空间可互操作。
