"""基于当前 PKL 校验畸变：显示原图与去畸变后的图（每路相机）。

支持 radial_poly（WoodScape 原始模型，完全不用 cv2.fisheye）以及常规 fisheye。
仅用于直观验证模型是否正常。

用法：
conda activate fastbev-fish
export PYTHONPATH=.
python tools/vis_undistort_synwoodscape.py \
  --ann-file data/synwoodscape_infos_train_small_fixed.pkl \
  --idx 0 \
  --out work_dirs/undistort_check
"""

import argparse
import os
import copy
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_image_path(cam_info):
    for key in ["data_path", "img_path", "filename", "img_filename"]:
        if key in cam_info:
            return cam_info[key]
    return None


def build_k_d(cam, use_size_k=False):
    """返回 (K, D, width, height, radial_params)。

    - 优先 radial_poly；若 cam_model=polynomial 且 cam_distortion 给出，则把 cam_distortion 当作 k1..k4 构造 radial_params。
    - 否则走常规 cam_intrinsic+cam_distortion。
    若 use_size_k=True，使用图像尺寸构造 K（fx=fy=min(w,h)/2，cx,cy=中心），常用于极端 fisheye。"""
    width = cam.get("width")
    height = cam.get("height")
    if width is None or height is None:
        raise ValueError("缺少 width/height")

    cam_model = cam.get("cam_model", "")
    rp = None
    if isinstance(cam.get("cam_radial_params"), dict):
        rp = cam["cam_radial_params"]
    elif cam_model == "polynomial" and cam.get("cam_distortion") is not None:
        # 数据集中使用 polynomial 模型但未显式给出 radial_params，直接用 cam_distortion 作为 k1..k4
        dist = cam.get("cam_distortion")
        rp = {
            "k1": float(dist[0]),
            "k2": float(dist[1]),
            "k3": float(dist[2]),
            "k4": float(dist[3]),
            "cx_offset": 0.0,
            "cy_offset": 0.0,
            "aspect_ratio": 1.0,
            "width": width,
            "height": height,
        }

    if rp is not None:
        camK = None if use_size_k else cam.get("cam_intrinsic")
        if camK is not None and not use_size_k:
            fx = float(camK[0][0])
            fy = float(camK[1][1])
            cx = float(camK[0][2])
            cy = float(camK[1][2])
        else:
            base_f = min(width, height) * 0.5
            fx = base_f
            fy = base_f
            cx = width * 0.5 - 0.5
            cy = height * 0.5 - 0.5
        aspect = float(rp.get("aspect_ratio", 1.0) or 1.0)
        fy = fy / aspect
        cx += rp.get("cx_offset", 0.0)
        cy += rp.get("cy_offset", 0.0)
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        # 保留原始 k1..k4，供 radial_poly 反投影使用
        D = np.array([[rp.get("k1", 0.0), rp.get("k2", 0.0), rp.get("k3", 0.0), rp.get("k4", 0.0)]],
                     dtype=np.float32)
    else:
        K = np.array(cam.get("cam_intrinsic"), dtype=np.float32)
        dist = cam.get("cam_distortion")
        if dist is None:
            dist = [0, 0, 0, 0]
        D = np.array([dist], dtype=np.float32)
    return K, D, int(width), int(height), rp


def _solve_theta_for_rho(target_rho, rp):
    """给定目标 rho，求 radial_poly 下的 theta（简单二分避免牛顿发散）"""
    k1, k2, k3, k4 = [float(rp.get(f"k{i}", 0.0)) for i in range(1, 5)]
    # 在 0..1.7rad(>97deg) 搜索
    lo, hi = 0.0, 1.7
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        rho = k1 * mid + k2 * mid ** 2 + k3 * mid ** 3 + k4 * mid ** 4
        if rho < target_rho:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def undistort_radial_poly(img, rp, out_size=None, rect_scale=0.12, auto_scale=False, intrinsic_K=None, use_intrinsic_f=False):
    """基于 WoodScape radial_poly 正向模型的去畸变（完全摆脱 cv2.fisheye）。

    做法：把目标视图看成简单 pinhole，相机中心对准车体坐标 +Z；为目标图每个像素生成射线，
    再用 radial_poly 正向投影到源图，最后用 remap 采样。
    这样不需要解方程/牛顿迭代，避免黑屏。
    """
    h, w = img.shape[:2]
    out_w, out_h = out_size or (w, h)
    k1, k2, k3, k4 = [float(rp.get(f"k{i}", 0.0)) for i in range(1, 5)]
    cx_off = float(rp.get("cx_offset", 0.0))
    cy_off = float(rp.get("cy_offset", 0.0))
    aspect = float(rp.get("aspect_ratio", 1.0) or 1.0)

    # 目标（去畸变）pinhole 内参
    if use_intrinsic_f and intrinsic_K is not None:
        # 使用标定内参的 fx/fy（取均值）作为基础焦距，可避免视场过宽带来的拉伸
        fx_base = float(intrinsic_K[0][0])
        fy_base = float(intrinsic_K[1][1])
        base_f = rect_scale * max(1.0, (fx_base + fy_base) * 0.5)
    elif auto_scale:
        # 自动估计 pinhole 焦距：让源图半高映射到 pinhole 的 45~90 度区间，避免过度拉伸
        # 以 half_min_rho 作为限制
        half_min_rho = 0.5 * min(w, h)
        theta_limit = _solve_theta_for_rho(half_min_rho, rp)
        # 控制不要超过 85deg，以免视场过广导致边缘拉伸
        theta_cap = np.deg2rad(85)
        theta_limit = min(theta_limit, theta_cap)
        base_f = (out_h * 0.5) / np.tan(theta_limit)
        base_f = max(base_f, 1.0)
    else:
        base_f = rect_scale * min(out_w, out_h)
    fx_rect = fy_rect = base_f
    cx_rect = out_w * 0.5 - 0.5
    cy_rect = out_h * 0.5 - 0.5

    xs = np.arange(out_w, dtype=np.float32)
    ys = np.arange(out_h, dtype=np.float32)
    u_rect, v_rect = np.meshgrid(xs, ys)

    # 归一化射线（目标 pinhole）
    x_n = (u_rect - cx_rect) / fx_rect
    y_n = (v_rect - cy_rect) / fy_rect
    chi = np.sqrt(x_n ** 2 + y_n ** 2)
    # 防止除零
    chi_safe = np.maximum(chi, 1e-8)
    theta = np.arctan2(chi_safe, 1.0)

    # radial_poly 正向投影到源鱼眼图
    rho = k1 * theta + k2 * theta ** 2 + k3 * theta ** 3 + k4 * theta ** 4
    rho = np.clip(rho, -2.0 * max(w, h), 2.0 * max(w, h))  # 防止极端数值把坐标推到无穷远
    u_src = rho * x_n / chi_safe + (w * 0.5 - 0.5) + cx_off
    v_src = rho * y_n / chi_safe
    v_src = v_src * aspect + (h * 0.5 - 0.5) + cy_off

    # 避免极端大值：出界区域直接置为 -1，让 remap 返回 0
    map1 = u_src.astype(np.float32)
    map2 = v_src.astype(np.float32)
    invalid = (~np.isfinite(map1)) | (~np.isfinite(map2)) | (map1 < -1) | (map1 > w + 1) | (map2 < -1) | (map2 > h + 1)
    map1[invalid] = -1
    map2[invalid] = -1
    return cv2.remap(
        img,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def undistort_fisheye(img, K, D, balance=0.0, dim=None):
    h, w = img.shape[:2]
    dim = dim or (w, h)
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, dim, np.eye(3), balance=balance
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, dim, cv2.CV_16SC2
    )
    undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/woodscape/fastbev_syn_headabc.py")
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--out", default="work_dirs/undistort_check")
    parser.add_argument("--data-root", default=None, help="若 info 中 data_path 为相对路径，可指定原始数据根目录")
    parser.add_argument("--use-size-k", action="store_true",
                        help="忽略 cam_intrinsic，改用基于 width/height 的 fx=fy=min(w,h)/2, cx,cy 为图中心")
    parser.add_argument("--negate-dist", action="store_true",
                        help="将 k1..k4 取反再去畸变（若明显过度畸变可尝试）")
    parser.add_argument("--rect-scale", type=float, default=0.12,
                        help="去畸变目标焦距 = min(w,h)*rect_scale；若开启 auto-rect 则忽略该值")
    parser.add_argument("--rect-w", type=int, default=None, help="输出宽，默认用相机 width")
    parser.add_argument("--rect-h", type=int, default=None, help="输出高，默认用相机 height")
    parser.add_argument("--auto-rect", action="store_true",
                        help="根据 radial_poly 自动估计 pinhole 焦距，减轻拉伸")
    parser.add_argument("--use-intrinsic-f", action="store_true",
                        help="用 cam_intrinsic 的 fx/fy 做基准焦距，再乘以 rect-scale（适合想减少裁切/拉伸）")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    with open(args.ann_file, "rb") as f:
        ann = pickle.load(f)
    infos = ann["infos"]
    if args.idx >= len(infos):
        raise IndexError(f"idx {args.idx} 越界，样本数={len(infos)}")
    cams = copy.deepcopy(infos[args.idx].get("cams", {}))
    if not cams:
        raise ValueError("未找到 cams 信息")

    for name, cam in cams.items():
        img_path = load_image_path(cam)
        if img_path is None:
            print(f"{name}: 未找到 data_path/img_path 字段，跳过")
            continue
        if not os.path.isabs(img_path) and args.data_root:
            candidate = os.path.join(args.data_root, img_path)
            if os.path.exists(candidate):
                img_path = candidate
        if not os.path.exists(img_path):
            print(f"{name}: 找不到图像路径 {img_path}, 跳过")
            continue
        img = cv2.imread(img_path)
        K, D, width, height, rp = build_k_d(cam, use_size_k=args.use_size_k)
        out_w = args.rect_w or width
        out_h = args.rect_h or height
        if args.negate_dist:
            D = -D
        img = cv2.resize(img, (width, height))
        if isinstance(rp, dict):
            undist = undistort_radial_poly(
                img,
                rp,
                out_size=(out_w, out_h),
                rect_scale=args.rect_scale,
                auto_scale=args.auto_rect,
                intrinsic_K=K,
                use_intrinsic_f=args.use_intrinsic_f,
            )
        else:
            undist = undistort_fisheye(img, K, D, balance=0.0, dim=(out_w, out_h))

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"{name} original")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(undist, cv2.COLOR_BGR2RGB))
        plt.title(f"{name} undistorted")
        plt.axis("off")
        plt.tight_layout()
        out_path = os.path.join(args.out, f"{name}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
