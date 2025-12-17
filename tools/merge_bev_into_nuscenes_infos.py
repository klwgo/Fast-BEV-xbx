# -*- coding: utf-8 -*-
"""
将 BEV 掩码路径合并到原始 nuScenes info pkl，保留原有字段（timestamp 等）。

用法示例：
python tools/merge_bev_into_nuscenes_infos.py \
  --base-pkl data/nuscenes_infos_train.pkl \
  --mask-dir work_dirs/nuscenes_bev_masks_full_v1 \
  --out work_dirs/nuscenes_infos_train_bev_merged.pkl
"""

import argparse
import pickle
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='合并 BEV 掩码到 nuScenes info pkl')
    parser.add_argument('--base-pkl', required=True, help='原始 info pkl（含 timestamp 等完整字段）')
    parser.add_argument('--mask-dir', required=True, help='BEV 掩码目录，含 <token>_drivable.npy 等')
    parser.add_argument('--out', required=True, help='输出 pkl 路径')
    args = parser.parse_args()

    mask_dir = Path(args.mask_dir)
    with open(args.base_pkl, 'rb') as f:
        base = pickle.load(f)
    infos = base['infos']
    meta = base.get('metadata', {})

    merged = 0
    for info in infos:
        token = info.get('token')
        if not token:
            continue
        drv = mask_dir / f'{token}_drivable.npy'
        if not drv.exists():
            continue
        info.setdefault('ann_info', {})
        info['ann_info']['gt_bev_seg'] = str(drv)
        info['ann_info']['bev_seg_classes'] = ['non_drivable', 'drivable']
        mark = mask_dir / f'{token}_marking.npy'
        if mark.exists():
            info['ann_info']['bev_marking_path'] = str(mark)
        obs = mask_dir / f'{token}_obstacle.npy'
        if obs.exists():
            info['ann_info']['bev_obstacle_path'] = str(obs)
        merged += 1

    payload = dict(infos=infos, metadata=meta)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump(payload, f)
    print(f'Merged {merged} samples into {args.out} (total {len(infos)})')


if __name__ == '__main__':
    main()
