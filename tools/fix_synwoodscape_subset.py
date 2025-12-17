"""修复 SynWoodScape 数据：生成 gt_drivable_mask，补充 lidar2img，并截取小集."""
import argparse
import os
import pickle
import numpy as np

def quat_to_rot(q):
    """Quaternion (w, x, y, z) -> 3x3 rotation."""
    q = np.array(q, dtype=np.float32)
    if q.shape[-1] != 4:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = q
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    r = np.array([
        [ww + xx - yy - zz,     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), ww - xx + yy - zz,     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), ww - xx - yy + zz],
    ], dtype=np.float32)
    return r

def build_drivable_mask(gt_bev_seg, bev_seg_classes, drivable_positive):
    """从多类 bev_seg 生成二值可行驶 mask."""
    cls_map = {name: idx for idx, name in enumerate(bev_seg_classes)}
    pos_ids = [cls_map[c] for c in drivable_positive if c in cls_map]
    mask = np.isin(gt_bev_seg, pos_ids).astype(np.uint8)
    return mask


def load_bev_seg(path, data_root):
    if not isinstance(path, str):
        return path
    # 依次尝试：原路径 -> data_root 拼接 -> 仓库根目录拼接
    candidates = [path]
    if not os.path.isabs(path):
        candidates.append(os.path.join(data_root, path))
        candidates.append(os.path.join(os.getcwd(), path))
    for p in candidates:
        if p.endswith('.npy') and os.path.exists(p):
            return np.load(p)
    # 若是 png/jpg，可自行扩展
    return None


def main():
    parser = argparse.ArgumentParser(description='修复 SynWoodScape pkl，生成可行驶掩码 + lidar2img，并截取小集')
    parser.add_argument('--in-pkl', default='data/synwoodscape_infos_train.pkl', help='输入原始 pkl')
    parser.add_argument('--out-pkl', default='data/synwoodscape_infos_train_small_fixed.pkl', help='输出修复后 pkl')
    parser.add_argument('--num', type=int, default=50, help='截取条数')
    parser.add_argument('--data-root', default='data/synwoodscape', help='原始数据根目录（用于相对路径解析）')
    parser.add_argument('--drivable-positive', nargs='+', default=[
        'road_surface', 'free_space', 'lane_marking', 'parking_line', 'ground'
    ], help='视为可行驶的类别名')
    parser.add_argument('--intrinsic-key', default='cam_intrinsic', help='cam 内参键名')
    parser.add_argument(
        '--extrinsic-key',
        default=None,
        help='cam 外参键名（若缺失则由 sensor2lidar_rotation/translation 反算 lidar->cam）',
    )
    parser.add_argument(
        '--use-sensor2lidar',
        action='store_true',
        help='优先使用 sensor2lidar_rotation/translation 直接反算 lidar->cam（若缺失则回退链式 cam->ego + lidar->ego）',
    )
    parser.add_argument(
        '--axis-remap',
        default='cam_rhs',
        choices=['none', 'cam_rhs', 'cam_y_up', 'cam_front'],
        help='在 lidar->cam 外参上再施加轴向重映射：'
             'none: 不变; '
             'cam_rhs: x右 y下 z前; '
             'cam_y_up: x右 y上 z前; '
             'cam_front: x前 y右 z下',
    )
    args = parser.parse_args()

    if not os.path.exists(args.in_pkl):
        raise FileNotFoundError(args.in_pkl)
    with open(args.in_pkl, 'rb') as f:
        data = pickle.load(f)
    infos = data.get('infos', [])
    bev_seg_classes = None
    fixed_infos = []
    for info in infos[:args.num]:
        ann = info.get('ann_info', {})
        if bev_seg_classes is None:
            bev_seg_classes = ann.get('bev_seg_classes', [])
        gt_bev_seg = ann.get('gt_bev_seg')
        if gt_bev_seg is None:
            continue
        gt_bev_seg_arr = load_bev_seg(gt_bev_seg, args.data_root)
        if gt_bev_seg_arr is None:
            continue
        gt_drv = ann.get('gt_drivable_mask')
        if gt_drv is None:
            gt_drv = build_drivable_mask(gt_bev_seg_arr, bev_seg_classes, args.drivable_positive)
            ann['gt_drivable_mask'] = gt_drv
        # 补充 lidar2img：若原本缺失，从每个相机的 intrinsics/extrinsics 构建
        if info.get('lidar2img') is None:
            cams = info.get('cams', {})
            intrinsics = []
            extrinsics = []
            distortions = []
            models = []
            img_shape_list = []
            def apply_axis_remap(R, t):
                if args.axis_remap == 'none':
                    return R, t
                if args.axis_remap == 'cam_rhs':          # x右 y下 z前
                    R_map = np.array([[0, -1, 0],
                                      [0, 0, -1],
                                      [1, 0, 0]], dtype=np.float32)
                elif args.axis_remap == 'cam_y_up':       # x右 y上 z前
                    R_map = np.array([[0, 1, 0],
                                      [0, 0, -1],
                                      [1, 0, 0]], dtype=np.float32)
                elif args.axis_remap == 'cam_front':      # x前 y右 z下
                    R_map = np.array([[1, 0, 0],
                                      [0, 0, 1],
                                      [0, -1, 0]], dtype=np.float32)
                else:
                    R_map = np.eye(3, dtype=np.float32)
                return R_map @ R, R_map @ t

            for _, cam_info in cams.items():
                K = cam_info.get(args.intrinsic_key)
                # 优先使用现成的外参；否则用 sensor2ego / lidar2ego 链式反算 lidar->cam
                E = cam_info.get(args.extrinsic_key) or cam_info.get('extrinsic')
                if E is None:
                    s2e_Rq = cam_info.get('sensor2ego_rotation')
                    s2e_t = cam_info.get('sensor2ego_translation')
                    l2e_q = info.get('lidar2ego_rotation')
                    l2e_t = info.get('lidar2ego_translation')
                    s2l_Rq = cam_info.get('sensor2lidar_rotation')
                    s2l_t = cam_info.get('sensor2lidar_translation')
                    # 1) 直接使用 sensor2lidar 反算
                    if args.use_sensor2lidar and s2l_Rq is not None and s2l_t is not None:
                        R_sl = quat_to_rot(s2l_Rq)
                        t_sl = np.array(s2l_t, dtype=np.float32).reshape(3, 1)
                        # lidar->cam = (sensor->lidar)^-1
                        R_lc = R_sl.T
                        t_lc = -R_lc @ t_sl
                        R_lc, t_lc = apply_axis_remap(R_lc, t_lc)
                        E = np.eye(4, dtype=np.float32)
                        E[:3, :3] = R_lc
                        E[:3, 3:4] = t_lc
                    # 2) 链式 cam->ego + lidar->ego
                    elif s2e_Rq is not None and s2e_t is not None and l2e_q is not None and l2e_t is not None:
                        R_ce = quat_to_rot(s2e_Rq)  # cam->ego
                        t_ce = np.array(s2e_t, dtype=np.float32).reshape(3, 1)
                        R_ec = R_ce.T
                        t_ec = -R_ec @ t_ce
                        R_le = quat_to_rot(l2e_q)
                        t_le = np.array(l2e_t, dtype=np.float32).reshape(3, 1)
                        R_lc = R_ec @ R_le
                        t_lc = R_ec @ t_le + t_ec
                        R_lc, t_lc = apply_axis_remap(R_lc, t_lc)
                        E = np.eye(4, dtype=np.float32)
                        E[:3, :3] = R_lc
                        E[:3, 3:4] = t_lc
                if K is None or E is None:
                    # 若缺失，填充单位矩阵，至少保证维度正确
                    K = np.eye(3, dtype=np.float32)
                    E = np.eye(4, dtype=np.float32)
                intrinsics.append(np.array(K, dtype=np.float32))
                extrinsics.append(np.array(E, dtype=np.float32))
                # 畸变 & 模型（若存在）
                if 'cam_distortion' in cam_info:
                    distortions.append(np.array(cam_info['cam_distortion'], dtype=np.float32))
                else:
                    distortions.append(None)
                models.append(cam_info.get('cam_model', 'polynomial'))
                h = cam_info.get('height')
                w = cam_info.get('width')
                if h is not None and w is not None:
                    img_shape_list.append((int(h), int(w), 3))
            if intrinsics and extrinsics:
                info['lidar2img'] = dict(
                    intrinsic=np.stack(intrinsics),
                    extrinsic=np.stack(extrinsics),
                    distortion=distortions,
                    model=models,
                    origin=np.array([0.0, 0.0, -1.0], dtype=np.float32),
                )
            if img_shape_list:
                info['img_shape'] = img_shape_list
        info['ann_info'] = ann
        fixed_infos.append(info)

    data['infos'] = fixed_infos
    os.makedirs(os.path.dirname(args.out_pkl), exist_ok=True)
    with open(args.out_pkl, 'wb') as f:
        pickle.dump(data, f)
    print(f'Saved {len(fixed_infos)} infos to {args.out_pkl}')


if __name__ == '__main__':
    main()
