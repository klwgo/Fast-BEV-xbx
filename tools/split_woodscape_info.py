#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将统一的 woodscape_infos_all.pkl 按比例拆分为 train / val。

示例：
    python tools/split_woodscape_info.py \
        --base-info data/woodscape_infos_all.pkl \
        --train-output data/woodscape_infos_train.pkl \
        --val-output data/woodscape_infos_val.pkl \
        --train-ratio 0.85 \
        --seed 42
"""
import argparse
import copy
import random

import mmcv


def parse_args():
    parser = argparse.ArgumentParser(description='WoodScape info 拆分脚本')
    parser.add_argument('--base-info', required=True, help='包含所有样本的 info.pkl')
    parser.add_argument('--train-output', required=True, help='输出的训练集 info.pkl')
    parser.add_argument('--val-output', required=True, help='输出的验证集 info.pkl')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='训练集占比（0~1），默认 0.8')
    parser.add_argument('--seed', type=int, default=0,
                        help='随机种子，保证拆分可复现')
    parser.add_argument('--min-val-samples', type=int, default=1,
                        help='验证集至少包含的样本数')
    return parser.parse_args()


def split_infos(infos, train_ratio, seed, min_val):
    """按比例随机划分 infos 列表。"""
    total = len(infos)
    if total == 0:
        raise ValueError('base info 中不包含任何样本')
    rand = random.Random(seed)
    indices = list(range(total))
    rand.shuffle(indices)
    train_count = int(total * train_ratio)
    # 确保验证集样本数不少于 min_val
    train_count = min(train_count, total - min_val)
    train_indices = set(indices[:train_count])
    train_infos = [infos[i] for i in range(total) if i in train_indices]
    val_infos = [infos[i] for i in range(total) if i not in train_indices]
    return train_infos, val_infos


def build_split_output(base_meta, infos, split_name):
    """构造新的 pkl 数据结构，补充 split 字段。"""
    metadata = copy.deepcopy(base_meta) if base_meta else {}
    metadata['split'] = split_name
    return dict(metadata=metadata, infos=infos)


def main():
    args = parse_args()
    base = mmcv.load(args.base_info)
    infos = base.get('infos', [])
    base_meta = base.get('metadata', {})
    train_infos, val_infos = split_infos(
        infos=infos,
        train_ratio=args.train_ratio,
        seed=args.seed,
        min_val=args.min_val_samples)

    train_output = build_split_output(base_meta, train_infos, 'train')
    val_output = build_split_output(base_meta, val_infos, 'val')

    mmcv.dump(train_output, args.train_output)
    mmcv.dump(val_output, args.val_output)

    print(f'总样本 {len(infos)} -> 训练 {len(train_infos)}，验证 {len(val_infos)}')
    print(f'训练集已写入: {args.train_output}')
    print(f'验证集已写入: {args.val_output}')


if __name__ == '__main__':
    main()
