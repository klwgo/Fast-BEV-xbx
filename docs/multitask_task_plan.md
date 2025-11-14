# Fast-BEV-Fish 多任务阶段性任务重构

基于当前仓库成果（已具备 Fisheye 多任务 Head、`GenerateBEVMultitaskTargets` 标签链路与 WoodScape/SynWoodScape 转换脚本），重新整理接下来一轮可执行任务。若无特殊说明，均以木星主仓为默认路径。

---

## 1. 当前成果速览

1. **多任务 Head 能力**：`FisheyeBEVMultiTaskHead` 已集成可行驶区域、标线、slot、障碍分支（`mmdet3d/models/decode_heads/bev_multitask_head.py:49`），可通过 `enable_heads` 精简配置。
2. **标签生成链路**：`GenerateBEVMultitaskTargets` 支持 drivable/marking/slot/obstacle 的统一生成（`mmdet3d/datasets/pipelines/preprocess.py:140`），并已纳入 `fastbev_multitask_fisheye.py` 训练管线。
3. **数据工具**：`tools/convert_woodscape_full.py`、`tools/verify_woodscape_info.py`、`docs/woodscape_dataset_overview.md` 等提供多模态数据统计与校验。
4. **策略文档**：`docs/head_abd_vs_legacy.md`、`docs/woodscape_multitask_delivery.md` 已阐述 Head-A/B/D 与传统 head 的互补关系及长线 Sprint 规划。

---

## 2. 方向假设

1. **短期**：完成 BEV mask/3D 框补全，打通 Head-A/B/D 四任务的基础监督。
2. **中期**：以 SynWoodScape 预训练 + WoodScape 微调为主线，验证多任务训练策略和 slot/obstacle 精度。
3. **长期**：围绕 Sprint 5/6 目标（迁移、发布）构建增强、导出、拓扑规整流程。

---

## 3. 新任务拆解

### 3.1 数据与标签

| 任务 | 描述 | 依赖 | Owner/时长 |
| --- | --- | --- | --- |
| WoodScape BEV 栅格化 | 将多边形/语义 PNG 转为 `[C,H,W]` mask，更新 `woodscape_infos_full_*.pkl` | 原始 `semantic_annotations`、`cv2.fillPoly` | 2 天 |
| 3D 框拟合/验证 | 根据实例多边形计算外接矩形 + 高度，填充 `gt_bboxes_3d`，并用 `verify_bev_from_boxes.py` 可视化 | `tools/convert_woodscape_full.py` | 2 天 |
| Slot 标签提炼 | 由 parking_line/zebra_crossing 生成 slot_line/endpoint 通道，补充 `bev_mask_shape` 元数据 | 上述 BEV 栅格化产物 | 1 天 |

### 3.2 模型与训练

| 任务 | 描述 | 依赖 | Owner/时长 |
| --- | --- | --- | --- |
| Head-A 调参 | 调整 `drivable_positive/ambiguous`、loss 权重，确保可行驶区域收敛 | 新 BEV mask | 1 天 |
| Head-B/Slot 调试 | 启用 `with_slot=True`，微调 line_kernel/dilation，联动 `slot_head.py`（若需要单独模块） | Slot 标签 | 1.5 天 |
| Head-D 蒸馏策略 | 保留 FreeAnchor3DHead 输出，探索 obstacle mask ←→ 3D 框互相蒸馏 | 补全 3D 框 | 2 天 |
| 多任务权重搜索 | 引入自适应 loss 或手动权重 grid search，观察四任务梯度平衡 | 上述三头稳定后 | 1 天 |

### 3.3 工具与评测

| 任务 | 描述 | 依赖 | Owner/时长 |
| --- | --- | --- | --- |
| 可视化整合 | 扩展 `tools/debug_woodscape_samples.py` 展示 drivable/marking/slot/obstacle 四层栅格 | 新标签 | 1 天 |
| Slot/Obstacle 指标 | 在 `tools/test.py` 中加入 slot F1、obstacle IoU，输出 CSV/Markdown 报告 | 模型预测 | 1 天 |
| Sprint 回顾模板 | 基于 `docs/woodscape_multitask_delivery.md` 制作实际 vs 计划清单，供复盘 | 当前文档 | 0.5 天 |

---

## 4. 排期建议

1. **Week 1**：完成数据标签三个任务 + 可视化整合，确保 Head-A/B/D 训练输入就绪。
2. **Week 2**：并行 Head-A/B/Slot 调试 + Slot/Obstacle 指标实现。
3. **Week 3**：Head-D 蒸馏 + 多任务权重搜索；产出第一版四任务联合模型。
4. **Week 4**：对照 Sprint 文档进行回顾，决定是否进入迁移/增强阶段。

---

## 5. 风险与缓解

1. **标签质量不稳定**：建议在每个转换脚本后立即调用 `verify_woodscape_info.py` 并保存样例可视化，防止无监督训练。
2. **算力排期冲突**：多任务训练显存开销大，优先安排 8×A100；若不足，先在 SynWoodScape 缩小分辨率进行冒烟。
3. **指标缺乏**：在 Slot/Obstacle 任务落地前务必完成指标模块，否则难以迭代。

---

后续若任务完成或方向调整，请同步更新本文件并在 Sprint 例会上汇报，保持计划与实际一致。***
