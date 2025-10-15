#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make videos like tools/misc/visualize_results.py:
- Cameras: 6 imgs → 3x2 mosaic, each row reordered by [2,0,1] (same as Fast-BEV visualize_results)
- Stack: (Cameras mosaic) on top + (BEV) at bottom
- Read from vis_fastbev_mini_offline/<sample_dir>/ with 0..23 (4*6 cameras) + bev_pred.png / bev_gt.png
- Output two big videos: GT(uses top half of cam imgs) & Pred(uses bottom half)
"""

import os, glob, argparse
import cv2
import numpy as np

try:
    import imageio.v2 as imageio  # pip install -U imageio imageio-ffmpeg
    HAS_IMAGEIO = True
except Exception:
    HAS_IMAGEIO = False

REORDER_ROW = [2,0,1]          # same as visualize_results.py
OFFSETS     = [0,6,12,18]      # 4 timesteps per folder

def even_wh(w,h): return (w+w%2, h+h%2)

def limit_resize(img, max_w, max_h):
    h,w = img.shape[:2]
    scale = min(max_w/float(w), max_h/float(h), 1.0)  # only downscale
    nw,nh = even_wh(int(w*scale), int(h*scale))
    if (nw,nh)!=(w,h):
        img = cv2.resize(img,(nw,nh), interpolation=cv2.INTER_AREA)
    return img

def crop_half(img, mode):  # 'gt' = top-half, 'pred' = bottom-half
    if img is None: return None
    h = img.shape[0]; mid = h//2
    return img[:mid] if mode=='gt' else img[mid:]

def row_equal_height_concat(imgs):
    valid = [im for im in imgs if im is not None]
    if not valid:
        return np.zeros((480,640,3), dtype=np.uint8)
    H = min(im.shape[0] for im in valid)
    row = []
    for im in imgs:
        if im is None:
            im = np.zeros((H,H,3), dtype=np.uint8)
        elif im.shape[0]!=H:
            w = int(im.shape[1]*(H/float(im.shape[0])))
            im = cv2.resize(im,(max(2,w),H), interpolation=cv2.INTER_AREA)
        row.append(im)
    return np.concatenate(row, axis=1)

def mosaic_3x2(cam6):
    assert len(cam6)==6
    # reorder rows like visualize_results.py: sort_list([..], [2,0,1])
    top3 = [cam6[i] for i in [0,1,2]]
    bot3 = [cam6[i] for i in [3,4,5]]
    top3 = [top3[i] for i in REORDER_ROW]
    bot3 = [bot3[i] for i in REORDER_ROW]
    r1 = row_equal_height_concat(top3)
    r2 = row_equal_height_concat(bot3)
    # width align
    if r2.shape[1] < r1.shape[1]:
        pad = np.zeros((r2.shape[0], r1.shape[1]-r2.shape[1], 3), dtype=r2.dtype)
        r2 = np.concatenate([r2,pad], axis=1)
    elif r2.shape[1] > r1.shape[1]:
        pad = np.zeros((r1.shape[0], r2.shape[1]-r1.shape[1], 3), dtype=r1.dtype)
        r1 = np.concatenate([r1,pad], axis=1)
    return np.concatenate([r1,r2], axis=0)

def stack_cams_bev(cams_mosaic, bev_img, max_w=1920, max_h=1080):
    top = limit_resize(cams_mosaic, max_w, max_h//2)
    bottom = limit_resize(bev_img, top.shape[1], max_h - top.shape[0])
    tgt_w = min(top.shape[1], bottom.shape[1])
    if top.shape[1]!=tgt_w:
        top = cv2.resize(top,(tgt_w, int(top.shape[0]*tgt_w/top.shape[1])), interpolation=cv2.INTER_AREA)
    if bottom.shape[1]!=tgt_w:
        bottom = cv2.resize(bottom,(tgt_w, int(bottom.shape[0]*tgt_w/bottom.shape[1])), interpolation=cv2.INTER_AREA)
    return np.concatenate([top,bottom], axis=0)

def read_0_25(folder):
    """Prefer strict 0.png..25.png; fallback to numeric sort (excluding bev_*.png)."""
    strict = [os.path.join(folder, f"{i}.png") for i in range(26)]
    if all(os.path.exists(p) for p in strict[:24]):
        return strict
    cand = [p for p in glob.glob(os.path.join(folder,"*.png"))
            if not os.path.basename(p).lower().startswith("bev_")]
    def key(p):
        b=os.path.basename(p); n=''.join(c for c in os.path.splitext(b)[0] if c.isdigit())
        return int(n) if n!='' else 999999
    cand = sorted(cand, key=key)
    return cand

def build_frames_for_folder(folder, mode, max_w, max_h):
    files = read_0_25(folder)
    if len(files) < 24:
        return []
    bev_name = f"bev_{mode}.png"
    bev = cv2.imread(os.path.join(folder, bev_name)) if os.path.exists(os.path.join(folder, bev_name)) \
          else np.zeros((480,640,3), np.uint8)

    frames=[]
    for base in OFFSETS:
        cam6 = [ cv2.imread(files[base+k]) if base+k < len(files) else None for k in range(6) ]
        cam6 = [ crop_half(im, mode) if im is not None else None for im in cam6 ]
        cams_mosaic = mosaic_3x2(cam6)
        frame = stack_cams_bev(cams_mosaic, bev, max_w=max_w, max_h=max_h)
        frames.append(frame)
    return frames

def safe_write(frames, out_path, fps=8, max_w=1920, max_h=1080):
    if not frames: return
    saned=[]; tw=th=None
    for f in frames:
        f2 = limit_resize(f, max_w, max_h)
        if tw is None: tw,th=f2.shape[1],f2.shape[0]
        elif (f2.shape[1],f2.shape[0])!=(tw,th):
            f2 = cv2.resize(f2,(tw,th), interpolation=cv2.INTER_AREA)
        saned.append(f2)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if HAS_IMAGEIO:
        try:
            rgb=[cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in saned]
            imageio.mimsave(out_path, rgb, fps=float(fps))
            print("Saved:", out_path); return
        except Exception as e:
            print("[warn] imageio failed:", e, "-> fallback OpenCV")
    H,W = saned[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, float(fps), (W,H))
    for fr in saned: vw.write(fr)
    vw.release(); print("Saved:", out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-dir", required=True, help="vis_fastbev_mini_offline 根目录（含多个样例子目录）")
    ap.add_argument("--out-gt",   default="runs/like_visualize_gt.mp4")
    ap.add_argument("--out-pred", default="runs/like_visualize_pred.mp4")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--maxw", type=int, default=1920)
    ap.add_argument("--maxh", type=int, default=1080)
    args = ap.parse_args()

    subdirs = sorted([os.path.join(args.show_dir,d) for d in os.listdir(args.show_dir)
                      if os.path.isdir(os.path.join(args.show_dir,d))])

    all_gt, all_pred = [], []
    for sd in subdirs:
        all_gt.extend(  build_frames_for_folder(sd, mode="gt",   max_w=args.maxw, max_h=args.maxh) )
        all_pred.extend(build_frames_for_folder(sd, mode="pred", max_w=args.maxw, max_h=args.maxh) )

    if all_gt:   safe_write(all_gt,   args.out_gt,   fps=args.fps, max_w=args.maxw, max_h=args.maxh)
    if all_pred: safe_write(all_pred, args.out_pred, fps=args.fps, max_w=args.maxw, max_h=args.maxh)

if __name__ == "__main__":
    main()
