"""在透视投影下估算 BEV 覆盖率（不经过 LUT），验证标定/范围是否合理。

用法:
python tools/check_perspective_coverage.py \
  --config configs/woodscape/fastbev_syn_headabc.py \
  --ann-file data/synwoodscape_infos_train_small_fixed.pkl \
  --idx 0
"""

import argparse
import numpy as np
import torch
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.models.detectors.fastbev import get_points


def compute_projection(img_meta, stride):
    projection = []
    intrinsic = torch.tensor(img_meta["lidar2img"]["intrinsic"][:3, :3])
    intrinsic[:2] /= stride
    extrinsics = [torch.tensor(ex) for ex in img_meta["lidar2img"]["extrinsic"]]
    for ex in extrinsics:
        proj = intrinsic @ ex[:3]  # [3,4]
        projection.append(proj)
    return torch.stack(projection)  # [n_cam,3,4]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--idx', type=int, default=0)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.data.train.ann_file = args.ann_file
    dataset = build_dataset(cfg.data.train, dict(test_mode=False))
    data = dataset[args.idx]
    metas = data['img_metas']
    # 解包 DataContainer / list
    if hasattr(metas, 'data'):
        raw = metas.data
        if isinstance(raw, (list, tuple)) and len(raw) > 0:
            metas = raw[0]
        elif isinstance(raw, dict):
            metas = raw
    if isinstance(metas, list) and len(metas) > 0:
        metas = metas[0]

    voxel_size = torch.tensor(cfg.model.voxel_size[0])
    n_voxels = torch.tensor(cfg.model.n_voxels[0])
    origin = torch.tensor(metas['lidar2img'].get('origin', [0, 0, 0]))
    points = get_points(n_voxels=n_voxels, voxel_size=voxel_size, origin=origin)  # [3,vx,vy,vz]
    flat = points.reshape(3, -1)

    img_shape = metas.get('img_shape')
    if isinstance(img_shape, (list, tuple)) and len(img_shape) > 0:
        if isinstance(img_shape[0], (list, tuple)):
            img_shape = img_shape[0]
    if not (isinstance(img_shape, (list, tuple)) and len(img_shape) >= 2):
        raise ValueError(f"img_shape 不合法: {img_shape}")
    # 转成标量 Tensor，避免与 tuple 比较时报错
    height_t = torch.as_tensor(img_shape[0], dtype=torch.float32)
    width_t = torch.as_tensor(img_shape[1], dtype=torch.float32)
    stride = 1
    proj = compute_projection(metas, stride)

    homo = torch.cat([flat, torch.ones(1, flat.shape[1])], dim=0)  # [4,N]
    pts = proj @ homo  # [n_cam,3,N]
    x = pts[:, 0] / pts[:, 2]
    y = pts[:, 1] / pts[:, 2]
    z = pts[:, 2]

    valid = (x >= 0) & (y >= 0) & (x < width_t) & (y < height_t) & (z > 0)
    valid_any = valid.any(dim=0)
    nz = int(valid_any.sum().item())
    total = valid_any.numel()

    print(f"perspective coverage: {nz} / {total} ({nz/total*100:.4f}%)")


if __name__ == '__main__':
    main()
