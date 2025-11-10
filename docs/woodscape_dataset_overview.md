# WoodScape 数据整理与统计

## 目录结构
已在 `/mnt/new_data/woodspace/woodscape_extracted/` 下完成解压与组织，结构如下：

```
woodscape_extracted/
├── WoodScape_ICCV19/
│   ├── box_2d_annotations/box_2d_annotations/*.txt
│   ├── instance_annotations/instance_annotations/*.json
│   ├── semantic_annotations/semantic_annotations/gtLabels/*.png
│   ├── motion_annotations/motion_annotations/gtLabels/*.png
│   └── soiling_dataset/...
├── rgb_images/
├── rgb_images_test/
├── previous_images/
├── previous_images_test/
└── soiling_dataset/{train,test}/
```

## 原始数据统计（WoodScape IC19）

| 类型 | 文件数 | 说明 |
| ---- | ------ | ---- |
| 实例多边形 JSON | 8,234 | 约 2,058 帧 × 4 视角；覆盖 43 个类别 |
| 2D 框 TXT | 8,234 | 仅提供前视（FV）相机的 2D 框 |
| 语义掩码 PNG | 8,234 | `semantic_annotations/.../gtLabels/` |
| 动态掩码 PNG | 8,234 | `motion_annotations/.../gtLabels/` |
| RGB 图像（train） | 8,234 | `rgb_images/` |
| RGB 图像（test） | 5,000 | `rgb_images_test/`

### 2D 框类别分布（原始统计）
```
vehicles        44,404
person          16,249
bicycle          7,053
traffic_sign     2,857
traffic_light    1,500
```

### 实例多边形类别 TOP20
```
construction     25,263    parking_line         9,582
pole             20,992    other_ground_marking 8,066
car              14,482    sky                  7,857
lane_marking     14,267    person               6,103
road_surface     12,159    bicycle              4,819
nature           11,893    zebra_crossing       4,816
free_space       11,495    ego_vehicle          3,737
curb              9,753    green_strip          3,679
```

## 增强版 info 生成

```bash
python tools/convert_woodscape_full.py \
  --base-info data/woodscape_infos_train.pkl \
  --raw-root /mnt/new_data/woodspace/woodscape_extracted/WoodScape_ICCV19 \
  --output data/woodscape_infos_full_train.pkl \
  --split train --overwrite
```

生成的 `metadata` 包含：
- `bev_seg.class_names = ['road_surface','free_space','lane_marking','parking_line','other_ground_marking','zebra_crossing']`
- `bbox2d_classes = ['vehicles','person','bicycle','traffic_light','traffic_sign']`
- `instance_classes`：与原始数据一致的 43 类列表。

> 基础 pkl 只覆盖 500 帧 mock 数据，目前仍仅包含三类 3D 障碍与前视 2D 框，如需全量类别需重新制作底层 info。

### info 统计（`data/woodscape_infos_full_train.pkl`）
- 样本数：500
- 3D 类别计数：
  - 四轮车：70,014
  - 行人：18,187
  - 两轮车：1,381
- 2D FV 框计数：
  - vehicles：2,290
  - person：1,181
  - bicycle：470
  - traffic_light：182
  - traffic_sign：260

## 校验脚本

```bash
python tools/verify_woodscape_info.py \
  data/woodscape_infos_full_train.pkl \
  --ref-root /mnt/new_data/woodspace/WoodScape_ICCV19-20251024T032614Z-1-003.zip
```

脚本会对比原始数据与 pkl 中的 3D/2D/BEV 标注覆盖情况，并报告缺失类别。当前 info 仍缺少大部分实例类别与非前视 2D 框，属预期告警。

## SynWoodScape 资源
- 目录：`woodscape_extracted/SynWoodScape_V0.1.1/`
- 包含：`box_3d_annotations`、`depth_maps`、`optical_flow`、`dvs_signals` 等仿真数据以及扩展文档。
- 可结合官方 README / dataset description，将 SynWoodScape 的多传感器标注融入上述流水线（同样需要转换为 info.pkl 结构）。
