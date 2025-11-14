# Head-A/B/D 与传统 3D/2D/BEV 三任务对比

## 1. 任务映射概览

| 功能诉求 | 传统方案（3D/2D/BEV） | 现行 Head-A/B/D | 核心差异 |
| --- | --- | --- | --- |
| 障碍物识别 | FreeAnchor3DHead 直接回归 3D 框，辅以 FCOS 多视角 2D（`configs/woodscape/fastbev_woodscape_fisheye.py:45-149`、`docs/woodscape_head_design.md:7-37`） | FisheyeBEVMultiTaskHead 的 obstacle 支路（Head-D）将 3D 框栅格化为 BEV 分类图（`configs/woodscape/fastbev_multitask_fisheye.py:129-198`、`mmdet3d/models/decode_heads/bev_multitask_head.py:165-240`） | 旧方案输出精确几何，但依赖锚框和稀缺 3D 标注；Head-D 仅需 BEV 网格，更易与语义共享特征但丢失高度信息 |
| 路面标志/标线 | 仅作为 BEV 语义中的若干类别，缺少专门建模 | Head-B 采用纵横卷积 + boundary 辅助监督聚焦细线（`mmdet3d/models/decode_heads/bev_multitask_head.py:114-147`） | 新方案针对性特征提取与边界损失，对细长结构更友好 |
| 可行驶区域 | WoodscapeBEVSegHead 输出 6 类 BEV 语义（`docs/woodscape_head_design.md:20-26`） | Head-A 将语义聚合为「不可行驶/可行驶/不确定」三级，并与 Head-B 共享标签生成链路（`mmdet3d/datasets/pipelines/preprocess.py:140-245`） | Head-A 降低类别不平衡、可直接服务规划，但依赖可靠 BEV mask |

## 2. 分任务深入分析

### 2.1 障碍物（旧 3D 检测 vs Head-D）

- **传统 3D/2D 组合**：FreeAnchor3DHead 在 BEV 栅格上匹配 anchor，回归 9 维框并评估方向，辅以 FCOS 处理前视 2D 细目标（`docs/woodscape_head_design.md:7-37`）。优点是输出具备真实尺寸和朝向，适合下游规划；缺点是锚点配置与 IoU bag 使训练复杂，且 WoodScape 目前仍缺真实 3D 框，仅由多边形推断（`docs/woodscape_task_gap_analysis.md:3-10`）。
- **Head-D（Obstacle BEV）**：`GenerateBEVMultitaskTargets` 将 `gt_bboxes_3d` 投影成离散 BEV 类别图，再由 obstacle 分支学习（`mmdet3d/datasets/pipelines/preprocess.py:218-364`，`mmdet3d/models/decode_heads/bev_multitask_head.py:165-240`）。共享 BEV Backbone，使障碍物预测与道路语义统一；回归难度更低，也规避了 anchor 设计。
- **优劣**：
  - 优势：与 Head-A/B 共享特征/标签生成链路；输出与可行驶区域同尺寸，易做规则融合；可在 BEV Mask 完备前先用粗略投影占位。
  - 劣势：失去高度与形状，难区分叠层目标；精度受限于 3D 框质量与栅格分辨率；若 `gt_bboxes_3d` 仍缺失则无法监督。

### 2.2 路面标志

- **传统方案**：Road/lane/parking 只能作为 BEV 语义中的若干像素类别，卷积核各向同性，难以捕捉细长形态（`docs/woodscape_head_design.md:20-31`）。
- **Head-B**：专门的纵横卷积 + boundary head 结构（`mmdet3d/models/decode_heads/bev_multitask_head.py:114-147`）将线/端点拆分监督，并可追加停车位通道（`configs/woodscape/fastbev_multitask_fisheye.py:36-55`）。与 Head-A 共用标签生成逻辑，可从统一 BEV mask 推导标线、端点、边界。
- **优劣**：
  - 优势：卷积核方向性 + boundary BCE 提升细线召回；同源标签减少额外标注成本；可扩展 slot/occlusion 任务。
  - 劣势：完全依赖 BEV mask，当前示例仍用零矩阵（`docs/woodscape_training.md:42-55`），需要先完成多边形栅格化；网格分辨率不足时，线宽会被侵蚀。

### 2.3 可行驶区域

- **传统 BEV Seg**：WoodscapeBEVSegHead 直接输出 6 类语义（`docs/woodscape_head_design.md:20-26`），类别多、正负极度失衡（free space vs. zebra crossing），并与标线共享一组参数，难在规划中直接使用。
- **Head-A**：在标签生成阶段即将类别聚合为「可行驶/不可/不确定」，并输出更轻量的三类预测（`mmdet3d/datasets/pipelines/preprocess.py:170-205`）。预测结果同栅格直接可用于栅格规划或与障碍/标线叠加。
- **优劣**：
  - 优势：类别更均衡、loss 稳定；可直接切换到规划接口；与 Head-B 共享 backbone 提升一致性。
  - 劣势：仍依赖高质量 BEV mask；若只拥有语义 PNG 而未投影到 BEV，则需额外转换。

## 3. 数据满足程度

- **障碍物**：实例多边形齐全但尚未量化为真实 3D Boxes（`docs/woodscape_task_gap_analysis.md:3-10`），目前 info 只覆盖 500 帧 mock 数据的三类 3D 标签（`docs/woodscape_dataset_overview.md:68-81`）。Head-D 只能在先完成 3D 框生成或投影后才有监督信号。
- **路面标志**：多边形统计充足（lane_marking、parking_line、other_ground_marking 等均有上万样本，见 `docs/woodscape_dataset_overview.md:41-51`），适合转换为 BEV mask 供 Head-B 使用；但转换流程尚未落地（`docs/woodscape_task_gap_analysis.md:7-9`）。
- **可行驶区域**：语义 PNG 已存在，但当前 `gt_bev_seg` 仍是占位零矩阵（`docs/woodscape_training.md:42-55`）。Head-A 要发挥作用需完成语义/多边形 → BEV 栅格化。
- **多视角 2D 支撑**：原始 2D 框仅覆盖前视（`docs/woodscape_dataset_overview.md:25-39`），即便保留 FCOS 作为补充，也难以训练真正的多视角检测，需借助 SynWoodScape 或投影补标。

## 4. 结论与建议

1. **先补标签，再启用 Head-A/B/D**：在 `convert_woodscape_full.py` 流程中完成语义/多边形栅格化，保证 `gt_bev_seg` 和 `gt_bboxes_3d` 不为空，否则多任务 head 将失去监督。
2. **障碍物策略混合**：短期仍可保留 FreeAnchor3DHead/FCOS 作为精细输出，Head-D 则提供低分辨率占据图，二者可通过蒸馏或后处理融合。
3. **路面标志优先落地**：已有充足多边形类别，建议先从 Head-B 切入，验证 anisotropic conv + boundary loss 对线形目标的收益，再扩展至 slot/occlusion。
4. **规划端接口**：Head-A/B/D 全部输出固定 BEV 栅格，利于与规划共享；完成数据补齐后，可逐步弱化昂贵的 3D anchor-based 头，以降低部署成本。
