"""扫轴向重映射/直接传感器外参的 LUT 覆盖验证脚本。

用法示例（在 fastbev-fish 环境）:
python tools/sweep_lut_axis.py \
  --config configs/woodscape/fastbev_syn_headabc.py \
  --ann-file data/synwoodscape_infos_train_small_fixed.pkl \
  --idx 0 \
  --remap-list none cam_rhs cam_y_up cam_front \
  --out work_dirs/lut_sweep

作用:
- 直接使用 sensor2lidar_rotation/translation 反算 lidar->cam（不经过 ego），
  并尝试不同轴映射，检查覆盖率是否能变成均衡的四周扇形。
- 可选 --drop-distortion 做等距对比。
输出:
- 控制台打印各 remap + stride 的覆盖率。
- 在 out 目录保存各相机的 valid_map 可视化。
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
from mmdet3d.models.utils.fisheye_lut import (
    prepare_calibrations,
    build_fisheye_lut,
)


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


def axis_matrix(name: str):
    if name == 'none':
        return np.eye(3, dtype=np.float32)
    if name == 'cam_rhs':   # x右 y下 z前
        return np.array([[0, -1, 0],
                         [0, 0, -1],
                         [1, 0, 0]], dtype=np.float32)
    if name == 'cam_y_up':  # x右 y上 z前
        return np.array([[0, 1, 0],
                         [0, 0, -1],
                         [1, 0, 0]], dtype=np.float32)
    if name == 'cam_front':  # x前 y右 z下
        return np.array([[1, 0, 0],
                         [0, 0, 1],
                         [0, -1, 0]], dtype=np.float32)
    return np.eye(3, dtype=np.float32)


def extract_cam_dict(img_metas):
    cams = img_metas.get('cams', {})
    if not cams and isinstance(img_metas.get('cam_names'), list):
        cams = {name: img_metas.get(name, {}) for name in img_metas['cam_names']}
    return cams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--idx', type=int, default=0)
    parser.add_argument('--out', default='work_dirs/lut_sweep')
    parser.add_argument('--strides', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--remap-list', nargs='+', default=['none', 'cam_rhs', 'cam_y_up', 'cam_front'])
    parser.add_argument('--drop-distortion', action='store_true', help='忽略畸变，使用等距模型')
    parser.add_argument('--intrinsic-key', default='cam_intrinsic')
    parser.add_argument('--distortion-key', default='cam_distortion')
    parser.add_argument('--model-key', default='cam_model')
    parser.add_argument('--sensor2lidar-rot', default='sensor2lidar_rotation')
    parser.add_argument('--sensor2lidar-trans', default='sensor2lidar_translation')
    parser.add_argument('--assume-lidar-to-sensor', action='store_true',
                        help='若 sensor2lidar_* 实际已是 lidar->sensor，则不再转置求逆（直接用即可）')
    parser.add_argument('--use-ego-chain', action='store_true',
                        help='不使用 sensor2lidar，改用 sensor2ego + lidar2ego 链式反算 lidar->cam')
    parser.add_argument('--sensor-pose-mode', default='sensor_to_ego',
                        choices=['sensor_to_ego', 'ego_to_sensor'],
                        help='传感器位姿键的含义，默认 sensor_to_ego，如为 ego_to_sensor 则先求逆得到 cam->ego')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    cfg = Config.fromfile(args.config)
    cfg.data.train.ann_file = args.ann_file
    dataset = build_dataset(cfg.data.train, dict(test_mode=False))
    data_info = dataset.data_infos[args.idx]
    sample = dataset[args.idx]
    img_metas = sample['img_metas']
    # 解包 DataContainer / list / dict
    if hasattr(img_metas, 'data'):
        raw = img_metas.data
        if isinstance(raw, (list, tuple)) and len(raw) > 0:
            img_metas = raw[0]
        elif isinstance(raw, dict):
            img_metas = raw
    if isinstance(img_metas, list) and len(img_metas) > 0:
        img_metas = img_metas[0]
    img_metas = copy.deepcopy(img_metas)

    # 取模型参数
    model = init_model(cfg, checkpoint=None, device='cpu')
    voxel_size = torch.tensor(model.voxel_size[0])
    n_voxels = torch.tensor(model.n_voxels[0])
    origin = torch.tensor(img_metas.get('lidar2img', {}).get('origin', [0.0, 0.0, -1.0]))
    points = get_points(n_voxels=n_voxels, voxel_size=voxel_size, origin=origin)
    vx, vy, vz = n_voxels.tolist()

    cams = extract_cam_dict(img_metas)
    if not cams:
        cams = data_info.get('cams', {})
    if not cams:
        raise ValueError('未找到 cams 信息（img_metas 和 data_info 均缺失 cams）')

    # 图像尺寸取第一路
    first_cam = next(iter(cams.values()))
    img_shape_meta = img_metas.get('img_shape')
    if isinstance(img_shape_meta, (list, tuple)) and len(img_shape_meta) > 0:
        img_shape_meta = img_shape_meta[0]
    height = first_cam.get('height', None)
    width = first_cam.get('width', None)
    if height is None or width is None:
        if isinstance(img_shape_meta, (list, tuple)) and len(img_shape_meta) >= 2:
            height = height or img_shape_meta[0]
            width = width or img_shape_meta[1]
    if height is None or width is None:
        raise ValueError(f'无法确定图像尺寸: {img_shape_meta}')
    height = int(height)
    width = int(width)

    for remap_name in args.remap_list:
        Rm = axis_matrix(remap_name)
        cover_lines = []
        for stride in args.strides:
            intrinsics = []          # pinhole K list (only用于 prepare_calibrations when非radial)
            radial_params_list = []  # radial_poly 参数列表
            extrinsics = []
            distortions = []
            models = []
            for cam in cams.values():
                K = cam.get(args.intrinsic_key)
                if K is None:
                    K = np.eye(3, dtype=np.float32)
                is_radial = isinstance(K, dict)
                # 优先选择构造外参的方式
                E = None
                if args.use_ego_chain:
                    s2e_Rq = cam.get('sensor2ego_rotation')
                    s2e_t = cam.get('sensor2ego_translation')
                    l2e_q = data_info.get('lidar2ego_rotation')
                    l2e_t = data_info.get('lidar2ego_translation')
                    if s2e_Rq is not None and s2e_t is not None and l2e_q is not None and l2e_t is not None:
                        R_se = quat_to_rot(s2e_Rq)
                        t_se = np.array(s2e_t, dtype=np.float32).reshape(3, 1)
                        if args.sensor_pose_mode == 'ego_to_sensor':
                            # 已是 ego->cam，求逆得到 cam->ego
                            R_ce = R_se.T
                            t_ce = -R_ce @ t_se
                        else:
                            # sensor_to_ego: cam->ego 直接用
                            R_ce = R_se
                            t_ce = t_se
                        R_le = quat_to_rot(l2e_q)
                        t_le = np.array(l2e_t, dtype=np.float32).reshape(3, 1)
                        R_ec = R_ce.T
                        t_ec = -R_ec @ t_ce
                        R_lc = R_ec @ R_le
                        t_lc = R_ec @ t_le + t_ec
                        R_lc = Rm @ R_lc
                        t_lc = Rm @ t_lc
                        E = np.eye(4, dtype=np.float32)
                        E[:3, :3] = R_lc
                        E[:3, 3:4] = t_lc
                else:
                    R_sl_q = cam.get(args.sensor2lidar_rot)
                    t_sl = cam.get(args.sensor2lidar_trans)
                    if R_sl_q is not None and t_sl is not None:
                        R_sl = quat_to_rot(R_sl_q)
                        t_sl = np.array(t_sl, dtype=np.float32).reshape(3, 1)
                        if args.assume_lidar_to_sensor:
                            R_lc = R_sl
                            t_lc = t_sl
                        else:
                            R_lc = R_sl.T
                            t_lc = -R_lc @ t_sl
                        R_lc = Rm @ R_lc
                        t_lc = Rm @ t_lc
                        E = np.eye(4, dtype=np.float32)
                        E[:3, :3] = R_lc
                        E[:3, 3:4] = t_lc
                if E is None:
                    continue
                if is_radial:
                    radial_params_list.append(K)
                else:
                    intrinsics.append(np.array(K, dtype=np.float32))
                    radial_params_list.append(None)
                extrinsics.append(E)
                d = cam.get(args.distortion_key)
                models.append(cam.get(args.model_key, 'polynomial'))
                distortions.append(None if args.drop_distortion or is_radial else d)

            if not intrinsics or not extrinsics:
                print(f"[{remap_name}] 缺少标定，跳过 stride={stride}")
                continue

            # 快速打印首个相机的外参平移与几个锚点投影（忽略畸变，仅 pinhole 便于定位朝向）
            def debug_anchors():
                # 若当前使用 radial 参数，调试时改用 cam_intrinsic 矩阵
                if radial_params_list and radial_params_list[0] is not None:
                    K_dbg = cams[list(cams.keys())[0]].get('cam_intrinsic', np.eye(3, dtype=np.float32))
                else:
                    K_dbg = intrinsics[0]
                K0 = torch.tensor(K_dbg, dtype=torch.float32)
                E0 = torch.tensor(extrinsics[0], dtype=torch.float32)
                R0 = E0[:3, :3]; t0 = E0[:3, 3:4]
                anchors = torch.tensor([
                    [0, 0, 0, 1],
                    [10, 0, 0, 1],
                    [-10, 0, 0, 1],
                    [0, 10, 0, 1],
                    [0, -10, 0, 1],
                    [0, 0, 2, 1],
                    [0, 0, -2, 1],
                ], dtype=torch.float32).t()  # [4,7]
                cam_pts = (E0 @ anchors)[:3]  # [3,7]
                uv = K0 @ cam_pts
                uv = uv[:2] / uv[2:].clamp(min=1e-6)
                in_front = cam_pts[2] > 0
                inside = (uv[0] >= 0) & (uv[1] >= 0) & (uv[0] < width) & (uv[1] < height) & in_front
                print(f"[{remap_name}] stride={stride} cam0 t(m)={t0.view(-1).tolist()} anchors in_front={in_front.tolist()} inside={inside.tolist()} uv={uv.t().tolist()}")
            debug_anchors()

            # 组装输入：有 radial 参数则直接传 list（prepare_calibrations 支持）
            if any(rp is not None for rp in radial_params_list):
                intrinsic_input = []
                for rp in radial_params_list:
                    intrinsic_input.append(rp if rp is not None else np.eye(3, dtype=np.float32))
            else:
                intrinsic_input = np.stack(intrinsics)

            calibrations = prepare_calibrations(
                intrinsic=intrinsic_input,
                extrinsics=extrinsics,
                distortion=distortions,
                model_per_cam=models if not args.drop_distortion else ['equidistant'] * len(extrinsics),
            )
            with torch.no_grad():
                pixel_idx, valid_mask = build_fisheye_lut(
                    points=points,
                    cameras=calibrations,
                    height=height,
                    width=width,
                    stride=stride,
                )
            total = valid_mask.numel()
            nz = int(valid_mask.sum().item())
            cover_lines.append(f"stride={stride}: {nz}/{total} ({nz/total*100:.4f}%)")

            vm = valid_mask.view(len(calibrations), vx, vy, vz).any(dim=3).cpu().numpy()
            for cid in range(vm.shape[0]):
                plt.figure(figsize=(6, 6))
                plt.imshow(vm[cid], cmap='gray')
                plt.title(f'{remap_name} cam{cid} stride{stride}')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(args.out, f'{remap_name}_cam{cid}_s{stride}.png'), bbox_inches='tight')
                plt.close()
        print(f"[{remap_name}] coverage:\n  " + "\n  ".join(cover_lines))

    print(f'done. results saved to {args.out}')


if __name__ == '__main__':
    main()
