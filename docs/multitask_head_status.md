# 多任务 Head 现状、问题与计划调整

> 参考 `docs/head_abd_vs_legacy.md`、`docs/woodscape_multitask_delivery.md` 与 `docs/multitask_task_plan.md`，结合当前实现进度，更新多任务 Head 设计与排期说明。

---

## 1. 已完成里程碑

1. **鱼眼 LUT 改造**：`mmdet3d/models/utils/fisheye_lut.py` 已适配 WoodScape/SynWoodScape 的多项式畸变参数，反投影链路通过单测，满足 Sprint 1“几何底座”目标。
2. **多视角鱼眼模型**：`mmdet3d/models/detectors/fastbev.py`、`woodscape_dataset.py` 中的相机建模已扩展为四目鱼眼，支持 `MultiViewPipeline` 加载与 LUT/BEV 对齐。
3. **原版三任务 Head（2D/3D/分割）**：`fastbev_woodscape_fisheye.py` 下的 FCOS、FreeAnchor3DHead、WoodscapeBEVSegHead 均可训练，满足 Sprint 2 基线要求。

---

## 2. 为什么需要新的多任务 Head 设计

1. **标签不完备**：WoodScape 原始 3D 数据只有多边形/投影，缺乏可靠 `gt_bboxes_3d`，导致 FreeAnchor3DHead 难以稳定收敛。`docs/head_abd_vs_legacy.md` 中所述 Head-D（obstacle mask）可以绕过锚框依赖，但需要新的 head 结构与损失。
2. **多任务耦合**：LUT/BEV 管线完成后，可行驶区域、标线、停车位、障碍物均共享同一 BEV 栅格，旧式“各任务独立 head”导致特征/标签重复，无法复用 `GenerateBEVMultitaskTargets` 输出。
3. **扩展性需求**：未来要加入 slot graph/occlusion/污渍等任务，单纯的 FCOS + FreeAnchor + Seg 组合可维护性差，亟需统一的 Fisheye BEV MultiTask Head 设计。

---

## 3. 当前突出问题

| 问题 | 现状 | 影响 | 解决办法 |
| --- | --- | --- | --- |
| 3D 数据缺失 | WoodScape 3D 框仍在拟合阶段，仅 mock/Syn 有有效标签 | FreeAnchor3DHead 训练不稳定，Head-D 开发被阻塞 | 继续执行 `tools/convert_woodscape_full.py` 的多边形拟合流程，结合 `verify_bev_from_boxes.py` 抽查；同时采用 SynWoodScape 预训练 + Wood 微调，让 obstacle 分支先在合成数据上成型 |
| 多视角 2D 标注不足 | 官方仅提供前视 2D 框，其余视角为空 | 即便 LUT 支持四目，2D head 实际只利用一目，影响多任务蒸馏 | 借助 LUT 将实例多边形投影到各视角，自动生成 `mv_bboxes/mv_labels`；缺乏纹理处使用伪标签 + 可视化校对。若短期无法补齐，则在训练中对缺失视角屏蔽 loss |
| 多任务 Head 设计未落地 | `FisheyeBEVMultiTaskHead` 仅完成骨架，slot/obstacle 未验证 | Sprint 3 之后的 slot/obstacle 目标无法按计划推进 | 已新增 `refine_head + binary_aux_losses` 版本，并在 SynWoodScape 上测试；后续需要在 WoodScape 数据补齐后继续验证，并根据梯度情况调节各 loss 权重与 warmup |
| 任务并行复杂 | BEV mask、slot 标签、3D 框需同时补齐，对脚本/算力要求高 | 预计排期需再扩展 1~2 周 | 按 `docs/multitask_task_plan.md` 拆分阶段：先完成标签 + 可视化；再执行 Head 调参；训练阶段统一用 8×A100，通过 Cosine LR + 梯度裁剪控制爆炸 |

---

## 4. WoodScape 多视角数据缺口的应对策略

1. **SynWoodScape 预训练**：使用 Syn 数据的全视角 2D/3D/BEV 标签，先训练多任务 head，再将权重迁移到 WoodScape（参考 Sprint 5 “合成→真实”路线）。
2. **投影补标**：利用已完成的鱼眼 LUT + instance 多边形，将前视以外的视角投影生成伪 2D 框，写入 `mv_bboxes`/`mv_labels`（`tools/convert_woodscape_full.py` 扩展）。
3. **可视化验证**：增强 `tools/debug_woodscape_samples.py`，一键查看四目框/栅格，快速淘汰出错样本。

---

## 5. 多任务设计的主要困难

1. **标签一致性**：需要将 road/free_space/lane_marking/parking_line 等语义映射到 drivable/marking/slot 多个标签空间，容错空间小。
2. **损失权重与梯度冲突**：Head-A/B/D 共享特征，若 loss 权重不平衡，将导致某个分支 dominate，其余任务发散。
3. **算力与显存**：多任务 head + 高分辨率 BEV 训练对显存要求极高，需提前规划多卡资源。
4. **评测缺乏**：slot/obstacle 没有成熟指标，必须同步构建 `ps20_metrics.py`、obstacle IoU 评估工具。

---

## 6. 预计用时调整

结合 `docs/woodscape_multitask_delivery.md` 的六个 Sprint 规划，预估需在 Sprint 3~4 之间追加约 **+1.5 周** 来完成多任务 head 验证与标签补齐，具体如下：

| 原计划 | 原时间 | 新增耗时 | 新交付时间 |
| --- | --- | --- | --- |
| Sprint 3：BEV 分割 & 3D 检测原型 | 2025-12-08 ~ 12-21 | +5 天（BEV mask、3D 拟合、Head-D 蒸馏） | 延至 12-26 |
| Sprint 4：车位检测 v1 | 2025-12-22 ~ 2026-01-04 | +5 天（slot 标签 + head 验证） | 延至 2026-01-09 |
| Sprint 5/6 | 2026-01-05 ~ 02-01 | 受前两 Sprint 延迟，整体顺延约 1 周 | 2026-02-07 完成 |

---

## 7. 下一步行动（按优先级）

### 7.1 任务分解与预计用时

| 阶段 | 任务 | 说明 | 预计用时 |
| --- | --- | --- | --- |
| 数据补齐 | BEV 栅格生产 | 批处理语义 PNG、多边形到 `[C,H,W]`，生成新掩码并在 200 帧上抽检 | 2 天 |
|  | Slot 标签派生 | 从 parking_line/zebra_crossing 等类别生成 slot line/endpoint 通道，写入 `bev_mask_shape` | 1 天 |
|  | 3D 框拟合 | 通过多边形拟合外接矩形 + 高度估计，更新 `gt_bboxes_3d` 并用 `verify_bev_from_boxes.py` 检查 | 2 天 |
| 模型验证 | Head-A (Drivable) 调参 | 调整 `drivable_positive`、loss，确保 IoU 收敛；基于 Syn 预训练 + Wood 微调 | 1 天 |
|  | Head-B (Marking/Slot) 联调 | 打通 marking/boundary/slot 输出，测试不同 line_kernel/dilation 组合 | 1.5 天 |
|  | Head-D (Obstacle) 蒸馏 | 先用 Syn 数据训练 obstacle mask，再与 FreeAnchor3DHead 双向蒸馏 | 2 天 |
| 工具链 | 可视化扩展 | 在 `tools/debug_woodscape_samples.py` 增加四层热力图显示、交互开关 | 1 天 |
|  | 指标实现 | 在 `tools/test.py` 中实现 slot F1、obstacle IoU、drivable/marking IoU 报表 | 1 天 |
| 文档同步 | Sprint 更新 | 将新增排期、风险写入 `docs/woodscape_multitask_delivery.md` 并同步周报 | 0.5 天 |

> 若多任务训练需要额外算力排期，建议在 Head-D 蒸馏前提前申请 8×A100 资源（约 0.5 天协调成本）。

### 7.2 行动顺序

1. **补齐标签**（进行中）：完成 BEV 栅格、slot 标签、3D 拟合 → 更新 `woodscape_infos_full_*.pkl`。  
2. **验证多任务 head**：在 SynWoodScape 上启用 `FisheyeBEVMultiTaskHead` 的 drivable/marking/slot/obstacle，记录 loss 梯度。  
3. **扩展工具链**：完善 `tools/debug_woodscape_samples.py`、`tools/verify_bev_from_boxes.py`，形成可视化/指标闭环。  
4. **更新 Sprint 文档**：将本文件的延迟与风险同步写入 `docs/woodscape_multitask_delivery.md` 和周会纪要。  

如遇新的数据或排期风险，请先更新本文件并通知项目干系人，保持多任务设计与实现路径的一致性。***
