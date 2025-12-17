"""打印 BEV 网格范围，帮助调整 origin/voxel_size/n_voxels。

用法:
python tools/check_bev_range.py --config configs/woodscape/fastbev_syn_headabc.py
"""

import argparse
from mmcv import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    voxel_size = cfg.model.get('voxel_size')[0]
    n_voxels = cfg.model.get('n_voxels')[0]
    origin = cfg.model.get('origin', [0, 0, 0])

    x_min = origin[0] - n_voxels[0] * voxel_size[0] / 2
    x_max = origin[0] + n_voxels[0] * voxel_size[0] / 2
    y_min = origin[1] - n_voxels[1] * voxel_size[1] / 2
    y_max = origin[1] + n_voxels[1] * voxel_size[1] / 2
    z_min = origin[2] - n_voxels[2] * voxel_size[2] / 2
    z_max = origin[2] + n_voxels[2] * voxel_size[2] / 2

    print(f"voxel_size={voxel_size}, n_voxels={n_voxels}, origin={origin}")
    print(f"x range: [{x_min:.2f}, {x_max:.2f}] m")
    print(f"y range: [{y_min:.2f}, {y_max:.2f}] m")
    print(f"z range: [{z_min:.2f}, {z_max:.2f}] m")


if __name__ == '__main__':
    main()

