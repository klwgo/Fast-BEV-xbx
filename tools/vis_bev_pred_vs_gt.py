import argparse
import os
import pickle

import cv2
import numpy as np
import torch


def to_label(mask):
    """将预测或 GT 压成单通道标签."""
    arr = np.asarray(mask)
    if isinstance(mask, torch.Tensor):
        t = mask
        if t.dtype == torch.float16:
            t = t.float()
        arr = t.detach().cpu().numpy()
    if arr.ndim == 4:  # [B,C,H,W]
        arr = arr[0]
    if arr.ndim == 3:
        # 若是 one-hot/logits，取 softmax 后 argmax；否则假定已经是单通道
        if arr.shape[0] in (2, 3, 4):  # C,H,W
            t = torch.from_numpy(arr.astype(np.float32))
            arr = torch.softmax(t, dim=0).argmax(dim=0).numpy()
        elif arr.shape[-1] in (2, 3, 4):  # H,W,C
            t = torch.from_numpy(arr.astype(np.float32))
            arr = torch.softmax(t, dim=-1).argmax(dim=-1).numpy()
        else:
            arr = arr.squeeze()
    return arr.astype(np.uint8)


def colorize(mask, fg_color):
    """简单上色：0 为深灰，其余为指定颜色."""
    canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    canvas[mask == 0] = (30, 30, 30)
    canvas[mask != 0] = fg_color
    return canvas


def main():
    parser = argparse.ArgumentParser(
        description="对比 BEV drivable 预测与 GT，自动对齐尺寸后拼图保存")
    parser.add_argument("--ann-file", required=True, help="包含 GT 路径的 pkl")
    parser.add_argument("--res-file", required=True, help="测试结果 pkl（列表，每条含 bev_drivable 或 bev_seg）")
    parser.add_argument("--out-dir", default="work_dirs/vis_pred_imgs", help="输出目录")
    parser.add_argument("--max-num", type=int, default=50, help="最多可视化多少条")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.ann_file, "rb") as f:
        ann = pickle.load(f)
    infos = ann["infos"]

    with open(args.res_file, "rb") as f:
        results = pickle.load(f)
    # 若结果被包装成 (outputs, metrics)，取第一个
    if isinstance(results, tuple) and len(results) >= 1 and isinstance(results[0], list):
        results = results[0]

    total = min(len(results), len(infos), args.max_num)
    for i in range(total):
        res = results[i]
        info = infos[i]

        # 预测：按 key 顺序取，避免 numpy 在 or 时触发真假判断
        pred = None
        for key in ["bev_drivable", "bev_seg", "pred"]:
            if key in res:
                pred = res[key]
                break
        if pred is None:
            print(f"[{i}] 缺少 bev_drivable/bev_seg，跳过")
            continue
        # 统一成标签；若是概率/分数，阈值 0.5 取二值
        if isinstance(pred, (list, tuple)) and len(pred) == 1:
            pred = pred[0]
        if isinstance(pred, np.ndarray) and pred.ndim == 2 and pred.dtype != np.uint8:
            pred = (pred >= 0.5).astype(np.uint8)
        pred = to_label(pred)

        # GT
        ann_info = info.get("ann_info", info)
        gt_path = ann_info.get("gt_bev_seg") or ann_info.get("bev_seg_path")
        if gt_path is None:
            print(f"[{i}] 缺少 gt_bev_seg 路径，跳过")
            continue
        if not os.path.isabs(gt_path):
            gt_path = os.path.join(os.path.dirname(args.ann_file), "..", gt_path)
            gt_path = os.path.normpath(gt_path)
        if not os.path.exists(gt_path):
            print(f"[{i}] GT 路径不存在 {gt_path}，跳过")
            continue
        gt = np.load(gt_path)
        gt = to_label(gt)

        # 对齐尺寸
        gh, gw = gt.shape[:2]
        ph, pw = pred.shape[:2]
        if (ph, pw) != (gh, gw):
            pred = cv2.resize(pred.astype(np.uint8), (gw, gh), interpolation=cv2.INTER_NEAREST)

        gt_vis = colorize(gt, (0, 0, 255))      # GT 红色
        pred_vis = colorize(pred, (0, 255, 0))  # 预测绿色
        canvas = np.concatenate([gt_vis, pred_vis], axis=1)

        out_path = os.path.join(args.out_dir, f"{i:05d}.png")
        cv2.imwrite(out_path, canvas)
        # 打印一些统计，便于发现“所有预测一样”情况
        print(f"saved {out_path} | pred sum={int(pred.sum())}, shape={pred.shape}")


if __name__ == "__main__":
    main()
