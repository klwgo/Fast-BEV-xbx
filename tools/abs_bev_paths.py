# -*- coding: utf-8 -*-
"""
将 BEV 掩码路径改为绝对路径，防止不同工作目录下相对路径失效。

用法示例：
python tools/abs_bev_paths.py \
  --in-pkl work_dirs/nuscenes_infos_train_bev_filtered_exist.pkl \
  --out-pkl work_dirs/nuscenes_infos_train_bev_abs.pkl
"""

import argparse
import pickle
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='将 bev 路径改为绝对路径')
    parser.add_argument('--in-pkl', required=True)
    parser.add_argument('--out-pkl', required=True)
    args = parser.parse_args()

    base = Path('.').resolve()
    data = pickle.load(open(args.in_pkl, 'rb'))
    infos = data['infos']
    updated = 0
    for info in infos:
        ann = info.get('ann_info', {})
        for key in ['gt_bev_seg', 'bev_marking_path', 'bev_obstacle_path']:
            if key in ann:
                p = Path(ann[key])
                if not p.is_absolute():
                    ann[key] = str((base / p).resolve())
                    updated += 1
    payload = dict(infos=infos, metadata=data.get('metadata', {}))
    Path(args.out_pkl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_pkl, 'wb') as f:
        pickle.dump(payload, f)
    print(f'Updated {updated} paths -> {args.out_pkl}')


if __name__ == '__main__':
    main()
