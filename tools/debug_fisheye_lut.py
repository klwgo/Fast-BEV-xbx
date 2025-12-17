"""快速定位鱼眼 LUT 覆盖率问题。

用法示例（在 fastbev-fish 环境）:
python tools/debug_fisheye_lut.py \
    --config configs/woodscape/fastbev_syn_headabc.py \
    --ann-file data/synwoodscape_infos_train_small_fixed.pkl \
    --idx 0 \
    --out work_dirs/lut_debug

输出:
- 打印各 stride 下的有效体素数量 / 总体素数，以及非零比例。
- 保存每路相机的有效掩码投影图 (vx, vy) 到 out 目录，便于查看覆盖范围。
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
    FisheyeLUTCache,
    project_fisheye_points,
)


def prepare_meta(meta: dict):
    """确保 lidar2img/ img_shape 数值化。"""
    lidar2img = meta.get('lidar2img', {})
    if isinstance(lidar2img, dict):
        if 'origin' not in lidar2img:
            lidar2img['origin'] = np.zeros(3, dtype=np.float32)
        for k in ['intrinsic', 'extrinsic']:
            if isinstance(lidar2img.get(k), list):
                lidar2img[k] = np.array(lidar2img[k], dtype=np.float32)
        # 兼容原始 meta 仅存 models 的情况
        if (lidar2img.get('model') is None) and 'models' in lidar2img:
            lidar2img['model'] = lidar2img['models']
    if meta.get('img_shape') is None and isinstance(meta.get('cam_names'), list):
        h_list = []
        for cam_name in meta['cam_names']:
            cam = meta.get(cam_name) or meta.get('cams', {}).get(cam_name, {})
            h = cam.get('height'); w = cam.get('width')
            if h is not None and w is not None:
                h_list.append((int(h), int(w), 3))
        if h_list:
            meta['img_shape'] = h_list[0]
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='配置文件路径')
    parser.add_argument('--checkpoint', default=None, help='可选权重')
    parser.add_argument('--ann-file', required=True, help='pkl 路径')
    parser.add_argument('--idx', type=int, default=0, help='样本索引')
    parser.add_argument('--out', default='work_dirs/lut_debug', help='输出目录')
    parser.add_argument('--strides', type=int, nargs='+', default=[1, 2, 4, 8], help='测试的 stride 列表')
    parser.add_argument('--drop-distortion', action='store_true',
                        help='调试用：忽略畸变并统一使用等距模型，便于定位畸变问题')
    parser.add_argument('--no-anchor-print', action='store_true',
                        help='不打印锚点投影信息')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    cfg = Config.fromfile(args.config)
    cfg.data.train.ann_file = args.ann_file
    dataset = build_dataset(cfg.data.train, dict(test_mode=False))
    data = dataset[args.idx]

    # 构建模型（无权重也可，用于标定/体素参数）
    model = init_model(cfg, args.checkpoint, device='cpu')
    model.eval()
    # 即便当前配置为 perspective，也为调试强制创建 LUT cache，避免未初始化报错
    if getattr(model, 'fisheye_lut_cache', None) is None:
        model.fisheye_lut_cache = FisheyeLUTCache(cache_dir=args.out, save_to_disk=False)

    img_metas = data['img_metas']
    # 解包 DataContainer / list
    if hasattr(img_metas, 'data'):
        raw = img_metas.data
        if isinstance(raw, (list, tuple)) and len(raw) > 0:
            img_metas = raw[0]
        elif isinstance(raw, dict):
            img_metas = raw
    if isinstance(img_metas, list) and len(img_metas) > 0:
        img_metas = img_metas[0]
    img_metas = prepare_meta(copy.deepcopy(img_metas))
    # 可选：忽略畸变并改用等距模型，便于与真实畸变对比
    if args.drop_distortion:
        if 'distortion' in img_metas.get('lidar2img', {}):
            n_cam = len(img_metas['lidar2img']['extrinsic'])
            img_metas['lidar2img']['distortion'] = [None] * n_cam
        img_metas['lidar2img']['model'] = ['equidistant'] * len(img_metas['lidar2img']['extrinsic'])

    voxel_size = torch.tensor(model.voxel_size[0])
    n_voxels = torch.tensor(model.n_voxels[0])
    origin = torch.tensor(img_metas['lidar2img']['origin'])
    points = get_points(n_voxels=n_voxels, voxel_size=voxel_size, origin=origin)  # [3,vx,vy,vz]

    img_shape = img_metas.get('img_shape')
    if isinstance(img_shape, (list, tuple)):
        img_shape = img_shape[0] if len(img_shape) > 0 else img_shape
    if isinstance(img_shape, (list, tuple)) and len(img_shape) >= 2:
        height = int(img_shape[0]); width = int(img_shape[1])
    else:
        raise ValueError(f"img_shape 不合法: {img_shape}")

    print(f"vx,vy,vz={tuple(n_voxels.tolist())}, voxel_size={voxel_size.tolist()}, origin={origin.tolist()}")
    print(f"img HxW={height}x{width}")

    # 额外：打印若干 anchor 点的投影，便于核对外参/轴向
    if not args.no_anchor_print:
        anchors = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [-10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, -10.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=torch.float32,
        )  # [N,3] in lidar frame
        anchors_h = torch.cat([anchors, torch.ones(len(anchors), 1)], dim=1)  # [N,4]
        lidar2img = img_metas['lidar2img']
        Ks = lidar2img.get('intrinsic', [])
        Es = lidar2img.get('extrinsic', [])
        models = lidar2img.get('model', [])
        distortions = lidar2img.get('distortion', [])
        radial_params_list = lidar2img.get('radial_params', [])
        cam_names = lidar2img.get('cam_names') or img_metas.get('cam_names') or [f'cam{i}' for i in range(len(Es))]

        def _get(lst, idx, default=None):
            if isinstance(lst, list) and idx < len(lst):
                return lst[idx]
            return lst if not isinstance(lst, list) else default

        for cid, (K, E) in enumerate(zip(Ks, Es)):
            name = cam_names[cid] if cid < len(cam_names) else f'cam{cid}'
            model_name = _get(models, cid, 'equidistant')
            distortion = _get(distortions, cid, None)
            radial_params = _get(radial_params_list, cid, None)
            K_t = torch.as_tensor(K, dtype=torch.float32)
            E_t = torch.as_tensor(E, dtype=torch.float32)
            pts_cam = (E_t @ anchors_h.T).T[:, :3].T  # [3,N]
            u, v, in_front = project_fisheye_points(
                points_cam=pts_cam,
                intrinsic=K_t,
                distortion=torch.as_tensor(distortion) if distortion is not None else None,
                model=model_name,
                radial_params=radial_params,
            )
            inside = (u >= 0) & (u < width) & (v >= 0) & (v < height) & in_front
            t = E_t[:3, 3].cpu().numpy().tolist()
            uv_list = torch.stack([u, v], dim=1).cpu().numpy().tolist()
            print(f"[anchors] cam{cid} {name} t(m)={t} model={model_name}")
            print(f"  in_front={in_front.cpu().numpy().tolist()} inside={inside.cpu().numpy().tolist()}")
            print(f"  uv={uv_list}")

    for stride in args.strides:
        with torch.no_grad():
            pixel_idx, valid_mask = model._get_fisheye_lut(
                img_meta=img_metas,
                points=points,
                stride=stride,
                height=height,
                width=width,
                voxel_size=voxel_size,
            )
        total = valid_mask.numel()
        nz = int(valid_mask.sum().item())
        print(f"stride={stride}: valid={nz} / {total} ({nz/total*100:.4f}%)")

        # 将 valid_mask reshape 为 [n_cam, vx, vy, vz] 并投影到 (vx,vy)
        n_cam = valid_mask.shape[0]
        vx, vy, vz = n_voxels.tolist()
        vm = valid_mask.view(n_cam, vx, vy, vz).any(dim=3).cpu().numpy()  # [n_cam,vx,vy]
        for cid in range(n_cam):
            plt.figure(figsize=(6, 6))
            plt.imshow(vm[cid], cmap='gray')
            plt.title(f'cam{cid} stride{stride} valid_map')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, f'cam{cid}_stride{stride}.png'), bbox_inches='tight')
            plt.close()

    print(f'done. plots saved to {args.out}')


if __name__ == '__main__':
    main()
