"""检查标定矩阵是否为占位身份矩阵。

用法（fastbev-fish 环境）:
python tools/check_calib_identity.py \
    --config configs/woodscape/fastbev_syn_headabc.py \
    --ann-file data/synwoodscape_infos_train_small_fixed.pkl \
    --idx 0
"""

import argparse
import numpy as np
import pickle
from mmcv import Config
from mmdet3d.datasets import build_dataset


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
    l2i = metas['lidar2img']
    intr = np.array(l2i['intrinsic'])
    extr = np.array(l2i['extrinsic'])

    def stat(mat, name):
        eye = np.eye(mat.shape[-1])
        diff = np.abs(mat - eye).mean()
        print(f"{name}: shape={mat.shape}, mean={mat.mean():.4f}, std={mat.std():.4f}, |mat-I|_mean={diff:.4f}")

    print("=== intrinsic ===")
    stat(intr, "intrinsic")
    print("=== extrinsic ===")
    stat(extr, "extrinsic")


if __name__ == '__main__':
    main()
