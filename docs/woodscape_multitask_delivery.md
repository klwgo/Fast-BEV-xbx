# Fast-BEV-Fish 六个 Sprint 任务拆解与时间规划

以下路线基于 2025-11-10 起的六个 Sprint（每个两周），目标是在 2026-02-01 前完成停车位、障碍物、路面标志、可行驶区域四大任务的端到端联调。所有交付物默认落在主仓 `Fast-BEV-Fish`，并与 `docs/head_abd_vs_legacy.md` 和多任务配置保持一致。

---

## Sprint 总览

| Sprint | 时间 | 核心目标 | 关键交付 | 验收标准 |
| --- | --- | --- | --- | --- |
| S1 | 2025-11-10 ~ 11-23 | 仓库初始化、鱼眼/BEV 底座、数据 Reader v0 | 基础配置、鱼眼 LUT/IPM 原型、可视化 Demo、单测集 | 单测通过；反投影误差 < 2px |
| S2 | 2025-11-24 ~ 12-07 | Fast-BEV 适配 + 2D/语义基线 | 视角变换模块、2D 检测 & BEV Seg 头、评测脚本 | 2D mAP / mIoU 达基线 ±10% |
| S3 | 2025-12-08 ~ 12-21 | BEV 分割 & 3D 检测原型 | `seg_bev_head.py`、`det3d_head.py`、多视角可视化 | BEV mIoU 稳定上升，3D mAP 持续提升 |
| S4 | 2025-12-22 ~ 2026-01-04 | 车位检测 v1（PS2.0） | `slot_head.py`、`slot_graph_match.py`、`ps20_metrics.py` | 车位 F1 ≥0.95，类型 Acc ≥0.95 |
| S5 | 2026-01-05 ~ 01-18 | 合成→真实迁移 + 强增强 | 合成预训练、真实微调、风格/污渍增强、阶段报告 | 三任务（2D/BEV/3D）指标各提升 ≥+2 mAP/mIoU |
| S6 | 2026-01-19 ~ 02-01 | 多任务联合 & 发布 | Loss 加权、NMS/拓扑规整、ONNX 导出、终版报告 | 四任务全部达标，端到端推理稳定 |

---

## Sprint 1：仓库初始化 + 几何底座（2025-11-10 ~ 11-23）

| 任务 | 描述 | 依赖 | 负责人 | 预估 |
| --- | --- | --- | --- | --- |
| 仓库骨架 | 梳理 `configs/`、`tools/`、`docs/` 结构，输出模板 config 与 `fastbev_fish_env.yml` | Git 基础模板 | Infra | 2d |
| 数据 Reader v0 | 在 `woodscape_dataset.py` 中实现最小 info pipeline，打通 `MultiViewPipeline` → `Collect3D` | WoodScape mock 数据 | Data | 2d |
| 鱼眼 LUT & IPM | 改造 `fisheye_lut.py`、IPM demo，支持鱼眼多项式参数与缓存 | 标定文件 | Vision | 3d |
| 可视化 Demo | `tools/visualize_synwoodscape_raw.py` 显示投影前后差异 + GUI | 上述 LUT | Tooling | 1d |
| 单元测试 | 对 LUT 精度、数据 Reader 建单测并接入 CI；验证反投影误差 < 2px | 上述实现 | QA | 1d |

---

## Sprint 2：Fast-BEV 适配 + 2D/BEV 分割基线（2025-11-24 ~ 12-07）

| 任务 | 描述 | 依赖 | 负责人 | 预估 |
| --- | --- | --- | --- | --- |
| 视角变换模块 | 在 `fastbev.py` 完成鱼眼到 BEV 多尺度特征融合，联通 `neck_fuse`/`neck_3d` | Sprint1 几何底座 | Model | 3d |
| 2D 检测/BEV Seg | 启用 FCOS、WoodscapeBEVSegHead，配置损失与数据键 | 同上 | Model | 2d |
| 评测脚本 | 扩展 `tools/test.py`、`verify_woodscape_info.py`，输出 2D mAP / BEV mIoU | 数据 Reader | Tooling | 1d |
| 冒烟训练 | 在 mock/Syn 数据上跑通 2D/BEV baseline，记录日志、收敛曲线 | 上述模块 | Training | 3d |
| 验收 | 对比基线指标，确认 mAP/mIoU 在 ±10% 范围 | 同上 | PM | 0.5d |

---

## Sprint 3：BEV 分割 + 3D 检测原型（2025-12-08 ~ 12-21）

| 任务 | 描述 | 依赖 | 负责人 | 预估 |
| --- | --- | --- | --- | --- |
| Head-A（seg_bev_head.py） | 结合 `GenerateBEVMultitaskTargets`，实现可行驶区域三分类 + loss 调参 | Sprint2 baseline | Model | 3d |
| Head-D 原型 | 构建 3D obstacle head（FreeAnchor/anchor-free），并与 BEV 特征对齐 | Syn 3D 标签 | Model | 4d |
| 可视化升级 | `verify_bev_from_boxes.py`/`debug_woodscape_samples.py` 输出 BEV mask + 3D 框叠加 | Head-A/D 输出 | Tooling | 2d |
| 指标看板 | 在 `tools/test.py` 增加 BEV mIoU、3D mAP 曲线导出 | 上述模块 | Tooling | 1d |
| 验收 | 记录至少两阶段结果，mIoU/mAP 呈上升趋势 | 训练产物 | PM | 0.5d |

---

## Sprint 4：车位检测 v1 / PS2.0（2025-12-22 ~ 2026-01-04）

| 任务 | 描述 | 依赖 | 负责人 | 预估 |
| --- | --- | --- | --- | --- |
| slot_head | 在 `FisheyeBEVMultiTaskHead` 中启用 slot 分支，调试 dilation/attention | Head-A/B 结果 | Model | 3d |
| slot_graph_match | 将 BEV slot mask 转为 PS2.0 拓扑（车位编号、端点连通性） | slot_head 输出 | Tooling | 2d |
| ps20_metrics | 实现车位 F1、类型 Acc、合法性指标并接入评测脚本 | 同上 | Tooling | 1d |
| 训练与调参 | 在 Syn + Wood 数据上训练 slot，优化 F1/Acc | 上述模块 | Training | 4d |
| 验收 | 验证集 F1≥0.95、类型 Acc≥0.95，提交报告 & demo | Training 结果 | PM | 1d |

---

## Sprint 5：合成→真实迁移与强增强（2026-01-05 ~ 01-18）

| 任务 | 描述 | 依赖 | 负责人 | 预估 |
| --- | --- | --- | --- | --- |
| 合成预训练 | 使用 SynWoodScape 跑多任务联合训练，保留权重与日志 | Sprint4 模型 | Training | 5d |
| 真实微调 | 在 WoodScape 上做多阶段微调，记录 2D/BEV/3D 指标 | 上述权重 | Training | 4d |
| 增强算子 | 在 `preprocess.py` 添加风格化/污渍/照明增强，支持可配置开关 | Dataset | Data | 3d |
| 对比分析 | 评估增强对三任务指标的影响，形成报告与可视化 | 训练结果 | PM | 2d |
| 验收 | 确认 2D/BEV/3D 指标较 Sprint4 ≥+2 mAP/mIoU | 报告 | PM | 0.5d |

---

## Sprint 6：多任务联合训练、后处理与发布（2026-01-19 ~ 02-01）

| 任务 | 描述 | 依赖 | 负责人 | 预估 |
| --- | --- | --- | --- | --- |
| 多任务加权 | 设计 loss 权重策略（自适应/手动），记录梯度趋势 | Sprint5 模型 | Model | 3d |
| 后处理/拓扑 | 实现障碍物 NMS、slot 拓扑清洗、drivable 约束 | 同上 | Tooling | 3d |
| 导出与部署 | 完成 ONNX/TensorRT 导出，编写 Benchmark & 推理脚本 | 优化模型 | Infra | 4d |
| 最终报告 | 汇总四任务指标、资源占用、场景案例，准备发布材料 | 所有结果 | PM | 2d |
| 验收 | 终端到端推理通过，发布包齐备 | 同上 | PM/QA | 1d |

---

## 资源与风险提示

- **数据依赖**：S1-S3 必须拿到可靠的标定/语义/多边形，相应转换脚本需提前规划；若真实 3D 框缺失，可用 SynWoodScape 预训练 + 伪标签过渡。  
- **算力排期**：建议每个 Sprint 预留至少 2 台 8×A100 服务器用于长时间训练，避免 CPU 上游脚本阻塞。  
- **工具链**：`tools/debug_woodscape_samples.py`、`verify_woodscape_info.py`、`visualize_synwoodscape_gt.py` 将贯穿所有 Sprint，需持续维护。  
- **沟通机制**：每周至少一次里程碑回顾，并在 docs 中更新本文件，保持计划与实际同步。

若后续需求变动，可在 Sprint 评审时回写该文档，确保所有成员共享最新的任务拆解与时间安排。
