# -*- coding: utf-8 -*-
"""
简单可视化 BEV 预测与 GT：可行驶 (HeadA) + 标线 (HeadB)。

用法（默认加载 syn_headabc_retry 最优/最新权重）:
python tools/visualize_bev.py \
    --config configs/woodscape/fastbev_syn_headabc.py \
    --checkpoint work_dirs/syn_headabc_retry/latest.pth \
    --out-dir work_dirs/syn_headabc_retry/vis_bev \
    --num-samples 4 \
    --device cuda:0
"""

import argparse
import os
import os.path as osp

import matplotlib
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmdet3d.apis import init_model
from mmdet3d.datasets import build_dataloader, build_dataset

matplotlib.use('Agg')


def to_numpy(arr):
    if arr is None:
        return None
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def build_datasets(cfg):
    cfg = cfg.copy()
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=min(2, cfg.data.get('workers_per_gpu', 2)),
        shuffle=False,
        dist=False,
        drop_last=False)
    return dataset, data_loader


def visualize_one(idx, res, targets, out_path, class_names):
    """绘制预测/GT 对比，支持可行驶与标线。"""
    pred_drivable = to_numpy(res.get('bev_drivable'))
    if pred_drivable is not None and pred_drivable.ndim == 3:
        pred_drivable = pred_drivable.argmax(axis=0)
    gt_drivable = targets.get('gt_drivable_mask')

    pred_marking = to_numpy(res.get('bev_marking'))
    if pred_marking is not None and pred_marking.ndim == 3:
        pred_marking = pred_marking.argmax(axis=0)
    gt_marking = targets.get('gt_marking_mask')

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    drivable_cmap = matplotlib.colors.ListedColormap(['tab:red', 'tab:green', 'tab:gray'])
    marking_cmap = matplotlib.colors.ListedColormap(['black', 'yellow', 'cyan', 'magenta', 'orange'])

    def imshow(ax, img, title, cmap):
        if img is None:
            ax.axis('off')
            ax.set_title(f'{title}: None')
            return
        ax.imshow(img, cmap=cmap, interpolation='nearest')
        ax.set_title(title)
        ax.axis('off')

    imshow(axes[0, 0], pred_drivable, 'Pred Drivable', drivable_cmap)
    imshow(axes[0, 1], gt_drivable, 'GT Drivable', drivable_cmap)
    imshow(axes[1, 0], pred_marking, 'Pred Marking', marking_cmap)
    imshow(axes[1, 1], gt_marking, 'GT Marking', marking_cmap)

    fig.suptitle(f'Sample {idx} | Drivable classes: {class_names["drivable"]} | '
                 f'Marking classes: {class_names["marking"]}')
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='可视化 Fast-BEV BEV 预测')
    parser.add_argument('--config', default='configs/woodscape/fastbev_syn_headabc.py', help='配置文件路径')
    parser.add_argument('--checkpoint', default='work_dirs/syn_headabc_retry/latest.pth', help='权重路径')
    parser.add_argument('--out-dir', default='work_dirs/syn_headabc_retry/vis_bev', help='输出目录')
    parser.add_argument('--num-samples', type=int, default=4, help='可视化样本数')
    parser.add_argument('--device', default='cuda:0', help='推理设备')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None  # 避免 init_model 再次下载
    cfg.model.train_cfg = None

    dataset, data_loader = build_datasets(cfg)
    model = init_model(cfg, args.checkpoint, device=args.device)
    model.eval()

    # 类名供图标题参考
    bev_gen = getattr(dataset, 'bev_target_generator', None)
    class_names = dict(
        drivable=getattr(bev_gen, 'drivable_class_names', ('non_drivable', 'drivable')),
        marking=['background'] + list(getattr(bev_gen, 'marking_classes', [])) if bev_gen else []
    )

    for i, data in enumerate(data_loader):
        if i >= args.num_samples:
            break
        # mmcv.scatter 需要 GPU id（int），而 init_model 支持 'cuda:0' 字符串。
        if isinstance(args.device, str) and args.device.startswith('cuda'):
            device_id = int(args.device.split(':')[-1]) if ':' in args.device else int(args.device)
            data_gpu = scatter(data, [device_id])[0]
        else:
            data_gpu = data  # CPU 情况直接使用原数据
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data_gpu)
        # result 是 batch 列表
        res = result[0] if isinstance(result, (list, tuple)) else result

        ann = dataset.get_ann_info(i)
        targets = dataset._prepare_bev_targets(ann) or {}

        out_path = osp.join(args.out_dir, f'{i:04d}.png')
        visualize_one(i, res, targets, out_path, class_names)
        print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
