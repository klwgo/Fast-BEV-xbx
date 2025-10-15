#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遍历 vis_fastbev_mini_offline 的所有子文件夹，按“三行布局”合成两个总视频：GT / Pred
布局（每帧）：
  第一行：前、前左、前右   （等高横排3张）
  第二行：后、后左、后右   （等高横排3张）
  第三行：BEV 图（bev_gt.png / bev_pred.png）
相机源与重排（与用户提供一致）：
  组0(0..5)   : 前, 前右, 前左, 后, 后左, 后右  → 目标 [前,前左,前右, 后,后左,后右]  = [0,2,1,3,4,5]
  组1/2/3     : 前, 前右, 前左, 后, 后右, 后左  → 目标 [前,前左,前右, 后,后左,后右]  = [0,2,1,3,5,4]
GT/Pred 的相机图：分别取每张相机图的上半/下半。
"""

import os, glob, argparse
import cv2
import numpy as np

try:
    import imageio.v2 as imageio  # pip install -U imageio imageio-ffmpeg
    HAS_IMAGEIO = True
except Exception:
    HAS_IMAGEIO = False

# ---- 基础工具 ----
def even_wh(w, h):
    return (w + w % 2, h + h % 2)

def limit_resize(img, max_w, max_h):
    """只缩小不放大；保持比例；输出偶数尺寸"""
    h, w = img.shape[:2]
    scale = min(max_w / float(w), max_h / float(h), 1.0)
    nw, nh = even_wh(int(w * scale), int(h * scale))
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return img

def list_numeric_pngs(folder):
    """按文件名中的数字排序，排除 bev_*.png"""
    def key(p):
        base = os.path.basename(p)
        num = ''.join([c for c in os.path.splitext(base)[0] if c.isdigit()])
        return int(num) if num != '' else 999999
    cand = [p for p in glob.glob(os.path.join(folder, "*.png"))
            if not os.path.basename(p).lower().startswith("bev_")]
    return sorted(cand, key=key)

def crop_half(im, mode):  # mode in {"gt","pred"}
    if im is None: return None
    h = im.shape[0]
    mid = h // 2
    return im[:mid] if mode == "gt" else im[mid:]

def row_equal_height_concat(imgs, target_h=None):
    """把3张图做成等高一行横排；缺图用黑图占位"""
    valid = [im for im in imgs if im is not None]
    if target_h is None:
        target_h = min(im.shape[0] for im in valid) if valid else 480
    row = []
    for im in imgs:
        if im is None:
            im = np.zeros((target_h, target_h, 3), dtype=np.uint8)
        else:
            if im.shape[0] != target_h:
                w = int(im.shape[1] * (target_h / float(im.shape[0])))
                im = cv2.resize(im, (max(2, w), target_h), interpolation=cv2.INTER_AREA)
        row.append(im)
    return np.concatenate(row, axis=1), target_h

def letterbox(img, target_w, target_h):
    """把 img 等比缩放后居中贴到 target_w×target_h 画布（不拉伸）"""
    if img is None:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    scale = min(target_w / float(w), target_h / float(h), 1.0)
    nw, nh = int(w * scale), int(h * scale)
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x0 = (target_w - img.shape[1]) // 2
    y0 = (target_h - img.shape[0]) // 2
    canvas[y0:y0+img.shape[0], x0:x0+img.shape[1]] = img
    return canvas

def stack_three_rows(row1, row2, row3, max_w, max_h):
    """把三行垂直拼接；先控制 row1+row2 总高，再 letterbox row3（BEV）到同宽"""
    # 限制前两行高度总和不超过 max_h 的 2/3
    r1 = limit_resize(row1, max_w, max_h // 3)
    r2 = limit_resize(row2, max_w, max_h // 3)
    # 宽度对齐到两者的最小宽
    tgt_w = min(r1.shape[1], r2.shape[1])
    if r1.shape[1] != tgt_w:
        r1 = cv2.resize(r1, (tgt_w, int(r1.shape[0] * tgt_w / r1.shape[1])), interpolation=cv2.INTER_AREA)
    if r2.shape[1] != tgt_w:
        r2 = cv2.resize(r2, (tgt_w, int(r2.shape[0] * tgt_w / r2.shape[1])), interpolation=cv2.INTER_AREA)
    # 第三行（BEV）letterbox 到与前两行同宽，且高度不超过 1/3
    r3 = letterbox(row3, tgt_w, max_h - r1.shape[0] - r2.shape[0])
    return np.concatenate([r1, r2, r3], axis=0)

def safe_write_video(frames_bgr, out_path, fps=10, max_w=1920, max_h=1080):
    valid = [f for f in frames_bgr if f is not None]
    if not valid:
        print("No frames to write:", out_path); return
    # 统一每帧到相同分辨率 & 偶数尺寸
    saned, tgt_w, tgt_h = [], None, None
    for f in valid:
        f2 = limit_resize(f, max_w, max_h)
        if tgt_w is None:
            tgt_w, tgt_h = f2.shape[1], f2.shape[0]
        elif (f2.shape[1], f2.shape[0]) != (tgt_w, tgt_h):
            f2 = cv2.resize(f2, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)
        saned.append(f2)
    # imageio 优先，失败回退 OpenCV
    if HAS_IMAGEIO:
        try:
            rgb = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in saned]
            imageio.mimsave(out_path, rgb, fps=float(fps))
            print("Saved:", out_path); return
        except Exception as e:
            print("[warn] imageio failed:", e, "-> fallback to OpenCV")
    H, W = saned[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, float(fps), (W, H))
    for fr in saned:
        vw.write(fr)
    vw.release()
    print("Saved:", out_path)

# ---- 重排与组装 ----
REORDER_GROUP0 = [0,2,1,3,4,5]  # 组0：前, 前右, 前左, 后, 后左, 后右 → 前,前左,前右, 后,后左,后右
REORDER_GROUPX = [0,2,1,3,5,4]  # 组1/2/3：前, 前右, 前左, 后, 后右, 后左 → 前,前左,前右, 后,后左,后右
OFFSETS = [0, 6, 12, 18]       # 四个时刻

def build_frames_for_folder(folder, mode, max_w, max_h):
    """
    返回该子文件夹的4帧（四个时刻），每帧为：
      第一行：前/前左/前右（等高三张）
      第二行：后/后左/后右（等高三张）
      第三行：bev_{mode}.png
    mode: "gt" or "pred"（决定相机图裁上半/下半 & 选择 bev_gt.png / bev_pred.png）
    """
    imgs = list_numeric_pngs(folder)
    if len(imgs) < 24:
        print(f"[warn] {folder}: found {len(imgs)} imgs (<24), skip.")
        return []
    # 读取 BEV
    bev_path = os.path.join(folder, f"bev_{mode}.png")
    bev_img = cv2.imread(bev_path) if os.path.exists(bev_path) else np.zeros((480,640,3), np.uint8)

    frames = []
    for t, base in enumerate(OFFSETS):
        order = REORDER_GROUP0 if t == 0 else REORDER_GROUPX
        idxs = [base + k for k in order]  # 6 张相机的目标顺序
        cams = [cv2.imread(imgs[i]) if i < len(imgs) else None for i in idxs]
        cams = [crop_half(im, mode) if im is not None else None for im in cams]

        # 拆成两行：上(前三张=前/前左/前右)，中(后三张=后/后左/后右)
        row1, h1 = row_equal_height_concat(cams[:3])
        row2, h2 = row_equal_height_concat(cams[3:])

        # 三行垂直拼接（第三行是 BEV）
        frame = stack_three_rows(row1, row2, bev_img, max_w=max_w, max_h=max_h)
        frames.append(frame)
    return frames

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/vis_fastbev_mini_offline", help="含 81 个子文件夹的根目录")
    ap.add_argument("--out-gt",   default="runs/all_folders_gt_3rows.mp4")
    ap.add_argument("--out-pred", default="runs/all_folders_pred_3rows.mp4")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--maxw", type=int, default=1920)
    ap.add_argument("--maxh", type=int, default=1080)
    args = ap.parse_args()

    subdirs = sorted([os.path.join(args.root, d) for d in os.listdir(args.root)
                      if os.path.isdir(os.path.join(args.root, d))])

    all_gt_frames, all_pred_frames = [], []

    for d in subdirs:
        gt_frames   = build_frames_for_folder(d, mode="gt",   max_w=args.maxw, max_h=args.maxh)
        pred_frames = build_frames_for_folder(d, mode="pred", max_w=args.maxw, max_h=args.maxh)
        all_gt_frames.extend(gt_frames)
        all_pred_frames.extend(pred_frames)

    os.makedirs(os.path.dirname(args.out_gt), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_pred), exist_ok=True)

    if all_gt_frames:
        safe_write_video(all_gt_frames, args.out_gt, fps=args.fps, max_w=args.maxw, max_h=args.maxh)
    else:
        print("[info] no GT frames found; skip writing", args.out_gt)

    if all_pred_frames:
        safe_write_video(all_pred_frames, args.out_pred, fps=args.fps, max_w=args.maxw, max_h=args.maxh)
    else:
        print("[info] no Pred frames found; skip writing", args.out_pred)

if __name__ == "__main__":
    main()
