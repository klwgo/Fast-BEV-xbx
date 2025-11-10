# WoodScape 任务覆盖与差距分析

| 任务 | 数据来源 | 当前支持情况 | 差距 & TODO |
| ---- | -------- | ------------ | ---------- |
| 3D/BEV 障碍检测 | `instance_annotations/*.json` | 已收集 43 类多边形；`build_woodscape_full_info.py` 会保存 `gt_polygons` | 未生成真实 3D box；需将多边形编码为 BEV 栅格或 `gt_boxes`（可参考论文或 SynWoodScape 标注） |
| 多视角 2D 检测 | `box_2d_annotations/*.txt` | 仅 FRONTVIEW (FV) 提供 5 类；已写入 `mv_bboxes`/`mv_labels` | 其他视角缺框；可结合多边形投影或 SynWoodScape 生成 MVL/MVR/RV 2D 标注 |
| BEV 语义分割 | `semantic_annotations` 或多边形栅格化 | 当前示例只写入零填充 mask | 需根据语义 PNG 或多边形 raster 生成实际 mask，再在 `gt_bev_seg` 填写路径 |
| 动态物体掩码 | `motion_annotations/gtLabels/*.png` | 路径已整理；尚未写入 info | 将路径写入 `ann_info['motion_seg_path']`，并在 pipeline 中加载 |
| 汽车/车位线任务 | 实例类别中已有 `parking_line`、`parking_marking` | 多边形已统计 | 需要单独 raster 生成车位栅格或训练专用 head |
| SynWoodScape 扩展 | `SynWoodScape_V0.1.1/SynWoodScape_V0.1.0/` | 包含 3D boxes、深度、光流等 | 尚未集成；可复用同样的 info 架构或自定义转换脚本 |

### 建议路线
1. **3D 框生成**：根据 WoodScape/SynWoodScape 文档，将多边形转为 BEV 占据图或拟合外接矩形填充 `gt_boxes`。  
2. **BEV Mask**：利用多边形栅格化（`cv2.fillPoly`）生成 label，并替换当前的零填充数据。  
3. **多视角 2D**：若只有 FV，可以在模型中将其他视角标记为空；若需全视角检测，可用投影/合成数据补齐。  
4. **SynWoodScape 融合**：将其仿真标注也转换为 info.pkl，用于预训练或数据增强。  
5. **持续校验**：每次生成新的 info 后运行 `tools/verify_woodscape_info.py --ref-root ...` 确认类别覆盖情况。
