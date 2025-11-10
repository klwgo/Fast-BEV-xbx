# WoodScape Head 设计与技术对比

本文档整理 WoodScape 多任务模型中三个核心 head（FreeAnchor3DHead、WoodscapeBEVSegHead、FCOSHead）的设计思路、正负样本策略与输出形式，方便在项目内对照理解与后续迭代。

---

## 1. FreeAnchor3DHead vs FCOSHead

| 对比维度 | FreeAnchor3DHead（3D 障碍物） | FCOSHead（2D 标志物） |
| --- | --- | --- |
| **输入空间** | BEV 特征图，代表回投影后的三维体素；输出 anchor-based 3D 框。 | 相机图像域的 FPN 多尺度特征；输出 2D 像素坐标框。 |
| **锚框依赖** | Anchor-based，自带 anchor 先验；为每个 GT 动态挑选一组候选 anchor。 | 完全 anchor-free，无需预设锚框，由像素点直接回归目标边界。 |
| **正负样本策略** | FreeAnchor “bag-of-anchors”\uFF1A对每个 GT 组建正 bag，按置信度重加权，避免硬性 IoU 阈值；负样本同样以 bag 形式聚合。 | FCOS 中心度逻辑：落在 GT 内的像素为正，距离越近 centerness 越高；边界外像素自动视为负样本。 |
| **回归参数化** | 基于 anchor 的残差 (Δx,Δy,Δz,Δw,Δl,Δh,Δθ, …)，支持 7/9 维框与朝向估计。 | 回归像素到目标边界的四个距离 (l,r,t,b)，再结合 centerness 重建 2D 框。 |
| **损失设计** | Focal 正/负 bag Loss + SmoothL1 + 朝向 CrossEntropy；重点提升稀疏 3D 目标的匹配质量。 | Focal 分类 + IoU/GIoU 框损失 + centerness BCE；突出图像域密集目标的定位精度。 |
| **选择理由** | 3D 障碍物数量少、尺寸跨度大，需借助 anchor 先验与自适应 bag 减少错配；输出直接服务规划与碰撞检测。 | 2D 标志物密集、尺度极端，anchor-free 可省去大量 anchor 设计并提升泛化，对标志、信号灯等细节目标更友好。 |

---

## 2. WoodscapeBEVSegHead（可行驶区域）

- **任务定位**：BEV 栅格语义分割，预测道路/可行驶区域等类别（输出 `[B, num_classes, H_bev, W_bev]`）。
- **结构概览**：多层 3×3 Conv-BN-ReLU 堆叠 + 1×1 分类层；默认两层中间通道 128，可按需求加深或改用 ASPP/Transformer 模块。
- **正负样本与损失**：语义分割使用全像素监督，默认交叉熵 (`loss_seg`)，可切换 Dice/Focal 等组合以适配类别不平衡。
- **与检测 Head 协同**：共用 Fast-BEV 提供的 BEV 特征；推理时输出概率图可直接用于路径规划或道路约束。

---

## 3. 设计要点与扩展方向

1. **任务覆盖**：当前 3D head 负责障碍物，2D head 拓展标志物细粒度识别，BEV head 在平面上输出道路约束。车位识别尚未接入，可复用 BEV 特征新增检测或语义分支。
2. **正负样本一致性**：3D 与 2D 在匹配策略上差异明显，训练配置（IoU 阈值、损失权重）需分别调优，避免互相干扰。
3. **推理融合**：三路 head 输出面向不同坐标系。部署时建议在后处理阶段统一转换到车辆坐标或全局地图，减少信息孤岛。
4. **进一步优化**：可考虑
   - 在 3D head 中引入动态锚点或 query-based 方法，以减轻 anchor 依赖；
   - 在 2D head 中加入轻量 Transformer/Deformable 模块，改善小目标召回；
   - 为 BEV 语义分割引入多尺度上下文（如 ASPP/Multi-head attention）提升鲁棒性。

如需在项目中扩展其他任务（车位、动态物体属性等），建议优先评估数据形态与正负样本分布，再选择合适的 head 框架与损失组合，以保持与现有系统的兼容性。
