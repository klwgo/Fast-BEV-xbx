# Fast-BEV-Fish 项目改进概览

本文档整理 Fast-BEV-Fish 相较于原始 Fast-BEV 开源版本所做的关键改动，涵盖数据处理、模型配置、多任务训练与工程化支持四大部分，便于团队成员快速了解项目特性与使用方式。

---

## 1. 数据集与标注构建

- **SynWoodScape 全量信息生成**  
  - 新增 `tools/build_synwoodscape_info.py`，支持从官方仿真数据一键生成符合 MMDetection3D 管线的 `info.pkl`。  
  - 主要实现细节：  
    - `load_calibrations()` 从 `calibration_data/*.json` 解析 CARLA 坐标系下的外参并转换为 Fast-BEV 所需的 `sensor2ego_rotation / sensor2ego_translation`。  
    - `corners_to_box()` 将 3D 八点框还原成中心、尺寸与朝向，并附带零速度占位。  
    - `convert_bev_semantic()` 将 BEV 语义 PNG 重新映射到 6 类栅格，生成 `.npy` 掩码文件并保存在 `<pkl>_bev_masks/` 目录下。  
    - 在 `info['ann_info']` 中新增 `bboxes`、`labels`、`mv_bboxes`、`mv_labels`、`gt_bev_seg`、`bev_seg_classes` 等字段，保证三任务齐备。  
  - 自动导出 BEV 栅格 (`synwoodscape_infos_*_bev_masks/*.npy`)，并将路径写入 `ann_info['gt_bev_seg']`，解决原版缺失 BEV 标签导致无法联合训练的问题。

- **WoodScape 多任务增强管线**  
  - 保留 JSON/TXT -> info 的转换链条：`build_woodscape_full_info.py` 负责基础信息，`convert_woodscape_full.py:106-153` 将 2D/语义/动态标注补全到 `ann_info`。  
  - `verify_woodscape_info.py` 扩展校验逻辑：  
    - 支持读取 `WoodScape_ICCV19*.zip`，对照官方 class info 统计 3D/2D/BEV 覆盖情况。  
    - 通过 `--skip-bev` 或 `--ref-root` 自定义验证范围，简化快速检查。

- **数据结构整理脚本**  
  - 调整 `/mnt/new_data/woodspace` 目录，将所有 WoodScape/SynWoodScape 资源统一归档到 `woodscape/`、`synwoodscape/`，便于迁移与同步。

---

## 2. 模型与配置改动

- **SynWoodScape 预训练配置 `configs/woodscape/fastbev_synwoodscape_pretrain.py`**  
  - 继承原 `fastbev_woodscape_fisheye` 基线，重新启用 2D FCOS 头与 BEV 分割头，支持三分支联合训练。  
  - `model` 段显式更新 `bbox_head_2d.num_classes` 与 `seg_head.num_classes`，保证与新生成的 info 类别数量一致。  
  - 通过 `SYN_DATA_ROOT / SYN_TRAIN_INFO / SYN_VAL_INFO` 环境变量切换数据位置，适配不同服务器路径；若未设置则使用本地默认路径。  
  - 训练流水线恢复加载 `gt_bboxes`、`gt_labels`、`gt_bboxes_3d`、`gt_labels_3d`、`gt_bev_seg` 等字段，并将 `with_bbox_mv=True` 以便评估阶段统计 2D mAP。  
  - `Collect3D` 的键集合同步更新，确保 DataLoader 输出包含多任务所需的标签。

- **多视角相机与标定支持**  
  - `build_synwoodscape_info.py` 中读取 `calibration_data/*.json`，统一写入每个相机的内参、畸变、外参，满足 Fast-BEV 鱼眼 LUT 的需求。  
  - `mmdet3d/datasets/woodscape_dataset.py` 复写 `get_data_info`，将 `lidar2img` 组织成带有 `intrinsics/distortions/models` 的结构，兼容鱼眼投影。

- **任务类别定义**  
  - 在配置与脚本中对齐 3D/2D/BEV 三类/五(六)类标签，确保 `DefaultFormatBundle3D` / `WoodScapeMultiViewDataset` 能正确映射标签索引。

---

## 3. 训练与部署流程

- **多 GPU 与单机调试指南**  
  - 建议使用 `torchrun` 或 `tools/dist_train.sh` 在多卡环境启动（SyncBN 依赖分布式初始化）。  
  - 单卡调试需将 `norm_cfg` 替换为普通 BN；文档中提供了明确的命令序列和环境变量设置方法。

- **冒烟测试脚本**  
  - 提供加载配置、构建 dataset、执行一次前向的 Python 片段，便于验证路径与标注是否成功读取。

- **验证工具**  
  - `verify_woodscape_info.py` 用于真实 WoodScape 数据集的完整性检查；SynWoodScape 侧推荐使用快速脚本验证 2D/3D/BEV 标签与路径。

---

## 4. 其他工程化提升

- **路径抽象与环境变量**：配置文件支持通过环境变量覆盖数据路径，减少跨服务器部署的修改成本。  
- **目录归档**：遗留的扩展数据、权重等统一移动到 `woodscape/archive/`，保持根目录整洁。  
- **文档补充**：新增本篇说明文档，并在 `docs/woodscape_training.md` 中更新多任务训练指南与常见问题。

---

### 后续规划

- **完备模态支持**：在 `build_synwoodscape_info.py` 中补充动态掩码、光流（`.flo`）、深度图（`.npy`）的转换逻辑，为未来多模态实验预留字段。  
- **单元测试与 CI**：构建最小样例（若干样本的精简版 info），用 pytest 验证关键字段（`gt_bev_seg`、`mv_bboxes`、相机标定）是否按预期写入。  
- **训练管线脚本化**：整理 SynWoodScape 预训练 → WoodScape 微调的命令序列，封装成 shell/python 工具，降低上线成本。  
- **指标体系扩展**：在 `woodscape_dataset.py` 内进一步细化 2D per-camera 评估、BEV IoU 指标输出，为模型对比提供更丰富的数据。  
- **配置模板统一化**：后续将 Syn/Wood 配置拆分为公共模板 + 数据差异补丁（如 `base_fastbev.py` + `*_dataset.py`），减少重复字段，方便后续升级。

如对某一部分仍有疑问，可在项目内搜索上述脚本或配置文件，查看具体实现细节。欢迎继续完善本文档。  
