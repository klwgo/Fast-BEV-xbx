"""对单路相机将 BEV 网格点投影到像平面，直观检查标定/LUT 问题。

用法（fastbev-fish 环境）:
python tools/vis_cam_grid.py \
  --config configs/woodscape/fastbev_syn_headabc.py \
  --ann-file data/synwoodscape_infos_train_small_fixed.pkl \
  --idx 0 --cam CAM_FRONT --out work_dirs/cam_proj

功能：
- 从模型配置获取 voxel_size / n_voxels / origin 构建 BEV 网格点。
- 可选 z 范围（默认 [-2, 2] m）。
- 使用相机外参将点变换到相机坐标，再按 fisheye/radial_poly 模型投影。
- 可视化有效投影点，若只出现在中心/三角，则外参/轴可能有误。
"""

import argparse
import os
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from mmcv import Config
from mmdet3d.apis import init_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models.detectors.fastbev import get_points
from mmdet3d.models.utils.fisheye_lut import project_fisheye_points


def quat_to_rot(q):
    q = np.array(q, dtype=np.float32).flatten()
    if q.size != 4:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def build_extrinsic(cam, data_info):
    # 优先使用已计算好的 extrinsic
    E = cam.get("extrinsic")
    if E is not None:
        return np.array(E, dtype=np.float32)
    # 退化到 sensor2ego + lidar2ego 链式
    s2e_Rq = cam.get("sensor2ego_rotation")
    s2e_t = cam.get("sensor2ego_translation")
    l2e_q = data_info.get("lidar2ego_rotation")
    l2e_t = data_info.get("lidar2ego_translation")
    if s2e_Rq is None or s2e_t is None or l2e_q is None or l2e_t is None:
        return None
    R_se = quat_to_rot(s2e_Rq)
    t_se = np.array(s2e_t, dtype=np.float32).reshape(3, 1)
    R_le = quat_to_rot(l2e_q)
    t_le = np.array(l2e_t, dtype=np.float32).reshape(3, 1)
    R_ec = R_se.T
    t_ec = -R_ec @ t_se
    R_lc = R_ec @ R_le
    t_lc = R_ec @ t_le + t_ec
    E = np.eye(4, dtype=np.float32)
    E[:3, :3] = R_lc
    E[:3, 3:4] = t_lc
    return E


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--cam", default="CAM_FRONT")
    parser.add_argument("--out", default="work_dirs/cam_proj")
    parser.add_argument("--z-min", type=float, default=-2.0)
    parser.add_argument("--z-max", type=float, default=2.0)
    parser.add_argument("--max-points", type=int, default=200000, help="可视化点数上限")
    parser.add_argument("--drop-distortion", action="store_true", help="忽略畸变")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    cfg = Config.fromfile(args.config)
    cfg.data.train.ann_file = args.ann_file
    dataset = build_dataset(cfg.data.train, dict(test_mode=False))
    data_info = dataset.data_infos[args.idx]
    sample = dataset[args.idx]
    img_metas = sample["img_metas"]
    if hasattr(img_metas, "data"):
        raw = img_metas.data
        if isinstance(raw, (list, tuple)) and len(raw) > 0:
            img_metas = raw[0]
        elif isinstance(raw, dict):
            img_metas = raw
    if isinstance(img_metas, list) and len(img_metas) > 0:
        img_metas = img_metas[0]
    img_metas = copy.deepcopy(img_metas)

    cams = img_metas.get("cams", {})
    if not cams and isinstance(img_metas.get("cam_names"), list):
        cams = {name: img_metas.get(name, {}) for name in img_metas["cam_names"]}
    if not cams:
        cams = data_info.get("cams", {})
    if args.cam not in cams:
        raise ValueError(f"未找到相机 {args.cam}")
    cam = cams[args.cam]

    model = init_model(cfg, checkpoint=None, device="cpu")
    voxel_size = torch.tensor(model.voxel_size[0])
    n_voxels = torch.tensor(model.n_voxels[0])
    origin = torch.tensor(img_metas.get("lidar2img", {}).get("origin", [0.0, 0.0, -1.0]))
    points = get_points(n_voxels=n_voxels, voxel_size=voxel_size, origin=origin).view(3, -1)  # [3,N]

    # 只保留指定高度范围
    z = points[2]
    mask_z = (z >= args.z_min) & (z <= args.z_max)
    points = points[:, mask_z]

    # 限制点数，便于渲染
    if points.shape[1] > args.max_points:
        idx = torch.randperm(points.shape[1])[: args.max_points]
        points = points[:, idx]

    E = build_extrinsic(cam, data_info)
    if E is None:
        raise ValueError("缺少外参，无法投影")
    R = torch.tensor(E[:3, :3])
    t = torch.tensor(E[:3, 3:4])
    pts_cam = (R @ points) + t  # [3,N]

    # 相机内参/畸变
    intrinsic = cam.get("cam_radial_params") or cam.get("cam_intrinsic")
    if intrinsic is None:
        intrinsic = np.eye(3, dtype=np.float32)
    model_name = cam.get("cam_model", "polynomial")
    distortion = None if args.drop_distortion else cam.get("cam_distortion", None)
    radial_params = intrinsic if isinstance(intrinsic, dict) else None
    intrinsic_tensor = torch.tensor(intrinsic, dtype=torch.float32) if radial_params is None else torch.eye(3)
    distortion_tensor = (
        None if distortion is None else torch.tensor(distortion, dtype=torch.float32)
    )

    u, v, in_front = project_fisheye_points(
        pts_cam,
        intrinsic_tensor,
        distortion_tensor,
        "equidistant" if args.drop_distortion else model_name,
        radial_params=radial_params,
    )

    height = cam.get("height") or img_metas.get("img_shape", [0, 0])[0]
    width = cam.get("width") or img_metas.get("img_shape", [0, 0])[1]
    height = int(height)
    width = int(width)
    inside = (u >= 0) & (v >= 0) & (u < width) & (v < height)
    valid = in_front & inside

    print(
        f"cam={args.cam}, points={points.shape[1]}, valid={valid.sum().item()} "
        f"({valid.float().mean().item()*100:.2f}%)"
    )

    # 绘制
    plt.figure(figsize=(8, 8))
    plt.scatter(u[valid].numpy(), v[valid].numpy(), s=0.5, c="red", alpha=0.6)
    plt.xlim([0, width])
    plt.ylim([height, 0])
    plt.title(f"{args.cam} projected points")
    plt.tight_layout()
    out_path = os.path.join(args.out, f"{args.cam}_idx{args.idx}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
