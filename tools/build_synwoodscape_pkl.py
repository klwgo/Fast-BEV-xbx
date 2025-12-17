#!/usr/bin/env python3
# Copyright (c) OpenAI
"""
根据 SynWoodScape 原始数据重新生成 PKL。
- 读取 calibration_data 下的 JSON，保留 cam_model=radial_poly 及其 radial_params(k1~k4,cx_offset,cy_offset,aspect_ratio,width,height)
- 将 extrinsic 四元数/平移转换为 4x4 矩阵，填入 cam 的 sensor2ego_rotation/sensor2ego_translation，并同步到 extrinsic 4x4
- 仅保存路径，不加载图像内容；可通过 --img-pattern 自定义文件名模式（默认 {idx:05d}_{cam}.png，可包含 {cam}/{idx:05d}.png 等）
- 生成一个小集合（默认 50 条），同时包含 BEV mask 路径。

用法示例：
python tools/build_synwoodscape_pkl.py \\
  --raw-root data/synwoodscape \\
  --output data/synwoodscape_infos_train_small.pkl \\
  --limit 50 \\
  --bev-mask-root data/synwoodscape_bev_masks \\
  --calib-root data/synwoodscape/SynWoodScape_V0.1.1/calibration_data \\
  --img-pattern \"SynWoodScape_V0.1.1/rgb_images/{cam}/{idx:05d}.png\"
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import mmcv

# cam 与校准文件名的映射
DEFAULT_CAM_FILES = {
    "CAM_FRONT": "FV.json",
    "CAM_FRONT_LEFT": "MVL.json",
    "CAM_FRONT_RIGHT": "MVR.json",
    "CAM_BACK": "RV.json",
}

BEV_CLASSES = ["road_surface", "lane_marking", "sidewalk", "vegetation", "ground", "background"]


def _normalize_quaternion(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / norm).astype(np.float32)


def quaternion_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """四元数 wxyz -> 3x3 旋转矩阵"""
    w, x, y, z = q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    return R


def load_calibration(calib_root: Path, cam_files: Dict[str, str]) -> Dict[str, dict]:
    if not cam_files:
        raise ValueError("未提供有效的相机-标定文件映射，请检查 --cams 与 DEFAULT_CAM_FILES 是否匹配。")

    def _arr(x, dtype=np.float32):
        return np.array(x, dtype=dtype)

    calibrations = {}
    for cam, fname in cam_files.items():
        cfile = calib_root / fname
        if not cfile.is_file():
            raise FileNotFoundError(f"标定文件不存在: {cfile}")

        with open(cfile, "r") as f:
            j = json.load(f)

        intr = j.get("intrinsic", {})
        cam_model = intr.get("model", j.get("cam_model", "radial_poly"))
        radial_params = None

        # 1) radial_poly: k1~k4 + cx_offset/cy_offset/aspect_ratio/width/height
        if isinstance(intr, dict) and all(k in intr for k in ["k1", "k2", "k3", "k4"]):
            radial_params = {
                "k1": float(intr["k1"]),
                "k2": float(intr["k2"]),
                "k3": float(intr["k3"]),
                "k4": float(intr["k4"]),
                "cx_offset": float(intr.get("cx_offset", 0.0)),
                "cy_offset": float(intr.get("cy_offset", 0.0)),
                "aspect_ratio": float(intr.get("aspect_ratio", 1.0)),
                "width": float(intr.get("width", 1280.0)),
                "height": float(intr.get("height", 966.0)),
            }
            cam_model = cam_model or "radial_poly"
            cx = radial_params["width"] * 0.5 + radial_params["cx_offset"] - 0.5
            cy = radial_params["height"] * 0.5 + radial_params["cy_offset"] - 0.5
            # SynWoodScape 的 radial_poly 未显式给出 fx/fy，使用 width/height 近似，同时保留为 float32
            fx = float(intr.get("fx", radial_params["width"]))
            fy = float(intr.get("fy", radial_params["height"]))
            cam_intrinsic = _arr([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            distortion = _arr(
                [radial_params["k1"], radial_params["k2"], radial_params["k3"], radial_params["k4"]]
            )
        # 2) pinhole/常规模型: fx,fy,cx,cy 或 cam_K
        elif isinstance(intr, dict) and all(k in intr for k in ["fx", "fy", "cx", "cy"]):
            fx, fy, cx, cy = float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"])
            cam_model = cam_model or intr.get("model", "pinhole")
            cam_intrinsic = _arr([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            dist_raw = intr.get("distortion") or intr.get("D") or [0, 0, 0, 0]
            distortion = _arr(dist_raw)[:4]
        elif isinstance(intr, dict) and "cam_K" in intr:
            K = _arr(intr["cam_K"]).reshape(3, 3)
            cam_model = cam_model or intr.get("model", "pinhole")
            cam_intrinsic = K
            dist_raw = intr.get("distortion") or intr.get("D") or [0, 0, 0, 0]
            distortion = _arr(dist_raw)[:4]
        else:
            # 尝试 radial_params 嵌套字段
            rp = intr.get("radial_params") if isinstance(intr, dict) else j.get("radial_params")
            if rp and all(k in rp for k in ["k1", "k2", "k3", "k4"]):
                cam_model = cam_model or "radial_poly"
                radial_params = {
                    "k1": float(rp["k1"]),
                    "k2": float(rp["k2"]),
                    "k3": float(rp["k3"]),
                    "k4": float(rp["k4"]),
                    "cx_offset": float(rp.get("cx_offset", 0.0)),
                    "cy_offset": float(rp.get("cy_offset", 0.0)),
                    "aspect_ratio": float(rp.get("aspect_ratio", 1.0)),
                    "width": float(rp.get("width", 1280.0)),
                    "height": float(rp.get("height", 966.0)),
                }
                cx = radial_params["width"] * 0.5 + radial_params["cx_offset"] - 0.5
                cy = radial_params["height"] * 0.5 + radial_params["cy_offset"] - 0.5
                fx = float(rp.get("fx", radial_params["width"]))
                fy = float(rp.get("fy", radial_params["height"]))
                cam_intrinsic = _arr([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                distortion = _arr(
                    [radial_params["k1"], radial_params["k2"], radial_params["k3"], radial_params["k4"]]
                )
            else:
                raise ValueError(f"不支持的 intrinsic 结构: {intr}")

        if radial_params is None:
            # 保证 radial_params 非空以满足可视化与 LUT 生成
            radial_params = {
                "k1": float(distortion[0]),
                "k2": float(distortion[1]),
                "k3": float(distortion[2]),
                "k4": float(distortion[3]),
                "cx_offset": float(cam_intrinsic[0, 2] - 0.5 * intr.get("width", cam_intrinsic[0, 2] * 2 + 1)),
                "cy_offset": float(cam_intrinsic[1, 2] - 0.5 * intr.get("height", cam_intrinsic[1, 2] * 2 + 1)),
                "aspect_ratio": float(intr.get("aspect_ratio", 1.0)),
                "width": float(intr.get("width", cam_intrinsic[0, 2] * 2 + 1)),
                "height": float(intr.get("height", cam_intrinsic[1, 2] * 2 + 1)),
            }

        if np.allclose(distortion, 0):
            # 兜底：将畸变设为 radial 参数，避免后续检查失败
            distortion = _arr(
                [radial_params["k1"], radial_params["k2"], radial_params["k3"], radial_params["k4"]]
            )

        ext = j.get("extrinsic", {})
        quat = ext.get("quaternion", [1, 0, 0, 0])
        quat = _normalize_quaternion(np.array(quat, dtype=np.float32))
        trans = ext.get("translation used in CARLA (CARLA reference)", ext.get("translation", [0, 0, 0]))
        trans = np.array(trans, dtype=np.float32)
        rot_mat = quaternion_wxyz_to_matrix(quat)
        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, :3] = rot_mat
        extrinsic[:3, 3] = trans

        calibrations[cam] = {
            "cam_model": cam_model,
            "cam_intrinsic": cam_intrinsic,
            "cam_distortion": distortion,
            "cam_radial_params": radial_params,
            "sensor2ego_rotation": quat.tolist(),  # 仍然使用 wxyz
            "sensor2ego_translation": trans.tolist(),
            "extrinsic": extrinsic,
        }
    return calibrations


def build_info(
    idx: int,
    cams: List[str],
    calib: Dict[str, dict],
    raw_root: Path,
    bev_mask_root: Path,
    img_pattern: str,
    cam_files: Dict[str, str],
) -> dict:
    cams_dict = {}
    lidar_intrinsics = []
    lidar_extrinsics = []
    distortions = []
    models = []
    for cam in cams:
        c = calib[cam]
        cam_file_name = cam_files.get(cam, cam)
        cam_file_base = Path(cam_file_name).stem
        data_path_fmt = img_pattern.format(
            cam=cam, idx=idx, cam_file=cam_file_base, cam_file_name=cam_file_name
        )
        img_path = Path(data_path_fmt)
        if not img_path.is_absolute():
            img_path = raw_root / img_path
        if not img_path.is_file():
            raise FileNotFoundError(f"找不到图像: {img_path}")
        data_path = str(img_path)
        rel_data_path = None
        try:
            rel_data_path = str(img_path.relative_to(raw_root))
        except Exception:
            rel_data_path = data_path_fmt
        # 读取尺寸
        img = mmcv.imread(str(img_path), channel_order="rgb")
        h, w = img.shape[:2]

        R_ce = c["extrinsic"][:3, :3]
        t_ce = c["extrinsic"][:3, 3]
        R_lc = R_ce.T  # lidar(ego)->cam
        t_lc = -R_lc @ t_ce

        cams_dict[cam] = {
            "data_path": data_path,
            "img_rel_path": rel_data_path,
            "prev_data_path": data_path,  # 小集，直接复用
            "timestamp": idx,
            "height": h,
            "width": w,
            "sensor2ego_rotation": c["sensor2ego_rotation"],
            "sensor2ego_translation": c["sensor2ego_translation"],
            "ego2global_rotation": [1, 0, 0, 0],
            "ego2global_translation": [0, 0, 0],
            "sensor2lidar_rotation": R_lc,
            "sensor2lidar_translation": t_lc,
            "cam_intrinsic": c["cam_intrinsic"],
            "cam_distortion": c["cam_distortion"],
            "cam_model": c["cam_model"],
            "cam_radial_params": c["cam_radial_params"],
        }
        lidar_intrinsics.append(c["cam_intrinsic"])
        lidar_extrinsics.append(c["extrinsic"])
        distortions.append(c["cam_distortion"])
        models.append(c["cam_model"])

    ann_info = {
        "mv_bboxes": [],
        "mv_labels": [],
        "semantic_paths": [],
        "motion_paths": [],
        "bbox_class_names": [],
        "bboxes": [],
        "labels": [],
        "bev_seg_classes": BEV_CLASSES,
        "gt_bev_seg": str(bev_mask_root / f"{idx:05d}.npy"),
    }

    info = {
        "token": f"{idx:06d}",
        "lidar_path": "",
        "sweeps": [],
        "timestamp": idx,
        "cams": cams_dict,
        "ann_info": ann_info,
        "lidar2img": {
            "intrinsic": np.stack(lidar_intrinsics),
            "extrinsic": np.stack(lidar_extrinsics),
            "distortion": np.stack(distortions),
            "model": models,
            "models": models,
            "radial_params": [calib[c]["cam_radial_params"] for c in cams],
            "cam_names": cams,
            "origin": [0.0, 0.0, 0.0],
        },
    }
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True, help="SynWoodScape 根目录，例如 data/synwoodscape")
    parser.add_argument("--output", required=True, help="输出 PKL 路径")
    parser.add_argument("--limit", type=int, default=50, help="样本数量")
    parser.add_argument("--start", type=int, default=0, help="起始 idx")
    parser.add_argument("--bev-mask-root", default="data/synwoodscape_bev_masks", help="BEV mask 目录")
    parser.add_argument(
        "--calib-root",
        default="data/synwoodscape/SynWoodScape_V0.1.1/calibration_data",
        help="标定文件目录",
    )
    parser.add_argument(
        "--cams",
        nargs="+",
        default=["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK"],
        help="相机列表，需与标定文件映射一致",
    )
    parser.add_argument(
        "--img-pattern",
        default="rgb_images/{idx:05d}_{cam_file}.png",
        help="图像相对 raw-root 的路径模式，支持 {cam},{idx},{cam_file},{cam_file_name}。若提供绝对路径则不会再拼接 raw-root。",
    )
    args = parser.parse_args()

    if len(args.cams) == 1 and " " in args.cams[0]:
        # 兼容 "--cams \"CAM_FRONT CAM_BACK\"" 的写法
        args.cams = args.cams[0].split()

    raw_root = Path(args.raw_root)
    bev_mask_root = Path(args.bev_mask_root)
    calib_root = Path(args.calib_root)

    cam_files = {cam: DEFAULT_CAM_FILES[cam] for cam in args.cams if cam in DEFAULT_CAM_FILES}
    missing = [cam for cam in args.cams if cam not in cam_files]
    if missing:
        raise ValueError(f"缺少标定文件映射的相机: {missing}，请在 DEFAULT_CAM_FILES 中补充或调整 --cams")
    calib = load_calibration(calib_root, cam_files)

    infos = []
    for idx in range(args.start, args.start + args.limit):
        info = build_info(idx, args.cams, calib, raw_root, bev_mask_root, args.img_pattern, cam_files)
        infos.append(info)

    meta = {
        "version": "synwoodscape-v0.1.1",
        "cams": args.cams,
        "camera_models": {k: calib[k]["cam_model"] for k in args.cams},
        "bev_seg_classes": BEV_CLASSES,
    }

    mmcv.dump({"infos": infos, "meta": meta}, args.output)
    print(f"Saved {len(infos)} infos to {args.output}")


if __name__ == "__main__":
    main()
