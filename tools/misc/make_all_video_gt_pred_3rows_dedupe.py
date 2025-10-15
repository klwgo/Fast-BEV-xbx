#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遍历 vis_fastbev_mini_offline 的所有子文件夹，按“三行布局”合成两个总视频：GT / Pred
每帧三行：
  1) 前、前左、前右（等高横排3张）
  2) 后、后左、后右（等高横排3张）
  3) bev_gt.png / bev_pred.png
严格只读 26 张（0.png..25.png + bev_*.png），并支持连续帧去重。
"""

import os, glob, argparse
import cv2
import numpy as np

try:
    import imageio.v2 as imageio  # pip install -U imageio imageio-ffmpeg
    HAS_IMAGEIO = True
except Exception:
    HAS_IMAGEIO = False

# -------- 基础工具 --------
def even_wh(w, h):
    return (w + w % 2, h + h % 2)

def limit_resize(img, max_w, max_h):
    h, w = img.shape[:2]
    scale = min(max_w / float(w), max_h / float(h), 1.0)  # 只缩小不放大
    nw, nh = even_wh(int(w * scale), int(h * scale))
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return img

def crop_half(im, mode):  # mode in {"gt","pred"}
    if im is None: return None
    h = im.shape[0]
    mid = h // 2
    return im[:mid] if mode == "gt" else im[mid:]

def row_equal_height_concat(imgs, target_h=None):
    """等高一行横排；缺图用黑图占位"""
    valid = [im for im in imgs if im is not None]
    if target_h is None:
        target_h = min(im.shape[0] for im in valid) if valid else 480
    row = []
    for im in imgs:
        if im is None:
            im = np.zeros((target_h, target_h, 3), dtype=np.uint8)
        elif im.shape[0] != target_h:
            w = int(im.shape[1] * (target_h / float(im.shape[0])))
            im = cv2.resize(im, (max(2, w), target_h), interpolation=cv2.INTER_AREA)
        row.append(im)
    return np.concatenate(row, axis=1)

def letterbox(img, target_w, target_h):
    """等比缩放后居中贴到指定画布（不拉伸）"""
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
    """三行垂直拼接；前两行各不超过 1/3 高，第三行占剩余高度"""
    r1 = limit_resize(row1, max_w, max_h // 3)
    r2 = limit_resize(row2, max_w, max_h // 3)
    tgt_w = min(r1.shape[1], r2.shape[1])
    if r1.shape[1] != tgt_w:
        r1 = cv2.resize(r1, (tgt_w, int(r1.shape[0] * tgt_w / r1.shape[1])), interpolation=cv2.INTER_AREA)
    if r2.shape[1] != tgt_w:
        r2 = cv2.resize(r2, (tgt_w, int(r2.shape[0] * tgt_w / r2.shape[1])), interpolation=cv2.INTER_AREA)
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
    if HAS_IMAGEIO:
        try:
            rgb = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in saned]
            imageio.mimsave(out_path, rgb, fps=float(fps))
            print("Saved:", out_path); return
        except Exception as e:
            print("[warn] imageio failed:", e, "-> fallback to OpenCV")
    H, W = saned[0].shape[:2]
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (W, H))
    for fr in saned:
        vw.write(fr)
    vw.release()
    print("Saved:", out_path)

# -------- 读取固定 26 张 & 重排 --------
REORDER_GROUP0 = [0,2,1,3,4,5]  # 组0(0..5)：前, 前右, 前左, 后, 后左, 后右 → 前,前左,前右, 后,后左,后右
REORDER_GROUPX = [0,2,1,3,5,4]  # 组1/2/3：前, 前右, 前左, 后, 后右, 后左 → 前,前左,前右, 后,后左,后右
OFFSETS        = [0, 6, 12, 18]

def read_fixed_26(folder):
    """严格读取 0.png..25.png；缺就警告并跳过该序列"""
    files = [os.path.join(folder, f"{i}.png") for i in range(26)]
    if not all(os.path.exists(p) for p in files[:24]):
        # 兜底：退回到“数字排序”的方式，但会提示有风险
        cand = [p for p in glob.glob(os.path.join(folder, "*.png"))
                if not os.path.basename(p).lower().startswith("bev_")]
        def key(p):
            base = os.path.basename(p)
            num = ''.join([c for c in os.path.splitext(base)[0] if c.isdigit()])
            return int(num) if num != '' else 999999
        files = sorted(cand, key=key)
        if len(files) < 24:
            return None
    return files

def build_frames_for_folder(folder, mode, max_w, max_h):
    """
    返回该子文件夹的 4 帧（四个时刻），每帧三行：
      上：前/前左/前右
      中：后/后左/后右
      下：bev_{mode}.png
    mode: "gt" or "pred"（决定相机图裁上半/下半 & 选择 bev_gt.png / bev_pred.png）
    """
    files26 = read_fixed_26(folder)
    if files26 is None:
        print(f"[warn] {folder}: missing 0..23 pngs, skip.")
        return []

    # 读取 BEV（文件名固定）
    bev_path = os.path.join(folder, f"bev_{mode}.png")
    bev_img = cv2.imread(bev_path) if os.path.exists(bev_path) else np.zeros((480,640,3), np.uint8)

    frames = []
    for t, base in enumerate(OFFSETS):
        order = REORDER_GROUP0 if t == 0 else REORDER_GROUPX
        idxs = [base + k for k in order]  # 6 张目标顺序
        cams = [cv2.imread(files26[i]) if i < len(files26) else None for i in idxs]
        cams = [crop_half(im, mode) if im is not None else None for im in cams]
        # 上行 & 中行（各 3 张）
        row1 = row_equal_height_concat(cams[:3])
        row2 = row_equal_height_concat(cams[3:])
        frame = stack_three_rows(row1, row2, bev_img, max_w=max_w, max_h=max_h)
        frames.append(frame)
    return frames

# -------- 连续帧去重 --------
def frame_fingerprint(img, size=(64, 36)):
    """把帧缩到小图做粗糙指纹（灰度），返回 float32 向量"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    return g

def is_near_duplicate(fprev, fcur, thr=2.0):
    """
    连续两帧是否近似重复：
      计算缩小灰度后的 MAE（0~255），小于阈值 thr 视为重复
    """
    if fprev is None or fcur is None: return False
    a = frame_fingerprint(fprev)
    b = frame_fingerprint(fcur)
    mae = float(np.mean(np.abs(a - b)))
    return mae < thr

def dedupe_consecutive(frames, thr):
    if not frames: return frames, 0
    out = [frames[0]]
    removed = 0
    for i in range(1, len(frames)):
        if is_near_duplicate(out[-1], frames[i], thr=thr):
            removed += 1
            continue
        out.append(frames[i])
    return out, removed

# -------- 主流程 --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/vis_fastbev_mini_offline", help="含多个子文件夹的根目录")
    ap.add_argument("--out-gt",   default="runs/all_folders_gt_3rows.mp4")
    ap.add_argument("--out-pred", default="runs/all_folders_pred_3rows.mp4")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--maxw", type=int, default=1920)
    ap.add_argument("--maxh", type=int, default=1080)
    ap.add_argument("--dedupe", action="store_true", help="启用连续帧去重")
    ap.add_argument("--dedupe-thr", type=float, default=2.0, help="去重阈值(灰度MAE, 0~255)")
    args = ap.parse_args()

    subdirs = sorted([os.path.join(args.root, d) for d in os.listdir(args.root)
                      if os.path.isdir(os.path.join(args.root, d))])

    gt_frames_all, pred_frames_all = [], []

    for d in subdirs:
        gt_frames   = build_frames_for_folder(d, mode="gt",   max_w=args.maxw, max_h=args.maxh)
        pred_frames = build_frames_for_folder(d, mode="pred", max_w=args.maxw, max_h=args.maxh)
        gt_frames_all.extend(gt_frames)
        pred_frames_all.extend(pred_frames)

    # 去重（可选）
    if args.dedupe:
        gt_frames_all, rm_gt   = dedupe_consecutive(gt_frames_all, thr=args.dedupe_thr)
        pred_frames_all, rm_pr = dedupe_consecutive(pred_frames_all, thr=args.dedupe_thr)
        print(f"[dedupe] GT removed {rm_gt} frames; Pred removed {rm_pr} frames (thr={args.dedupe_thr}).")

    os.makedirs(os.path.dirname(args.out_gt), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_pred), exist_ok=True)

    if gt_frames_all:
        safe_write_video(gt_frames_all, args.out_gt, fps=args.fps, max_w=args.maxw, max_h=args.maxh)
    else:
        print("[info] no GT frames to write.")

    if pred_frames_all:
        safe_write_video(pred_frames_all, args.out_pred, fps=args.fps, max_w=args.maxw, max_h=args.maxh)
    else:
        print("[info] no Pred frames to write.")

if __name__ == "__main__":
    import numpy as np
    main()
