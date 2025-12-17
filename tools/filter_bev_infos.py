# -*- coding: utf-8 -*-
"""
过滤 info pkl 中缺失 BEV 掩码的样本，避免加载 gt_bev_seg 报 KeyError。

用法：
python tools/filter_bev_infos.py \
  --in-pkl work_dirs/nuscenes_infos_train_bev_merged.pkl \
  --out-pkl work_dirs/nuscenes_infos_train_bev_filtered.pkl
"""

import argparse
import pickle
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='过滤缺少 gt_bev_seg 的样本')
    parser.add_argument('--in-pkl', required=True)
    parser.add_argument('--out-pkl', required=True)
    args = parser.parse_args()

    with open(args.in_pkl, 'rb') as f:
        data = pickle.load(f)
    infos = data['infos']
    kept = []
    for info in infos:
        ann = info.get('ann_info', {})
        bev_path = ann.get('gt_bev_seg')
        if bev_path is None:
            continue
        if not Path(bev_path).exists():
            continue
        kept.append(info)
    payload = dict(infos=kept, metadata=data.get('metadata', {}))
    Path(args.out_pkl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_pkl, 'wb') as f:
        pickle.dump(payload, f)
    print(f'Filtered {len(kept)} / {len(infos)} samples -> {args.out_pkl}')


if __name__ == '__main__':
    main()
