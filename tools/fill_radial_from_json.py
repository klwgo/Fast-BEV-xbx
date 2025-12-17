import argparse
import json
import os
import pickle
from typing import Dict, Any


CAM_JSON_MAP = {
    "CAM_FRONT": "FV.json",
    "CAM_FRONT_LEFT": "MVL.json",
    "CAM_FRONT_RIGHT": "MVR.json",
    "CAM_BACK": "RV.json",
}


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def extract_radial_params(calib: Dict[str, Any], fallback_w: float = None, fallback_h: float = None) -> Dict[str, float]:
    """
    从标定 json 中提取 radial_poly 所需的参数。
    期望键：k1,k2,k3,k4,cx_offset,cy_offset,aspect_ratio,width,height
    首选 calib['intrinsic']，若无再找 radial_params。
    """
    intrinsic = calib.get("intrinsic")
    candidates = [
        intrinsic if isinstance(intrinsic, dict) else None,
        calib.get("radial_params"),
        calib.get("intrinsic_parameters", {}).get("radial_params") if isinstance(calib.get("intrinsic_parameters"), dict) else None,
        calib,
    ]
    radial = None
    for c in candidates:
        if isinstance(c, dict) and all(k in c for k in ("k1", "k2", "k3", "k4")):
            radial = c
            break
    if radial is None:
        raise ValueError("未在 json 中找到 radial 参数 (k1~k4)")

    def pick(key: str, alt_keys=()):
        for k in (key, *alt_keys):
            if k in radial:
                return float(radial[k])
            if k in calib:
                return float(calib[k])
        return None

    k1 = pick("k1")
    k2 = pick("k2")
    k3 = pick("k3")
    k4 = pick("k4")
    cx_offset = pick("cx_offset", ("cxOff", "cxoff", "cx_off"))
    cy_offset = pick("cy_offset", ("cyOff", "cyoff", "cy_off"))
    aspect_ratio = pick("aspect_ratio", ("aspectRatio",))
    width = pick("width", ("img_width", "image_width"))
    height = pick("height", ("img_height", "image_height"))

    if width is None and fallback_w is not None:
        width = float(fallback_w)
    if height is None and fallback_h is not None:
        height = float(fallback_h)

    missing = [k for k, v in [
        ("k1", k1), ("k2", k2), ("k3", k3), ("k4", k4),
        ("cx_offset", cx_offset), ("cy_offset", cy_offset),
        ("aspect_ratio", aspect_ratio), ("width", width), ("height", height),
    ] if v is None]
    if missing:
        raise ValueError(f"radial_params 缺少字段: {missing}")

    return {
        "k1": k1, "k2": k2, "k3": k3, "k4": k4,
        "cx_offset": cx_offset, "cy_offset": cy_offset,
        "aspect_ratio": aspect_ratio,
        "width": width, "height": height,
    }


def main():
    parser = argparse.ArgumentParser(description="将 synwoodscape 标定 json 的 radial_poly 参数写入 pkl 的 cams 中")
    parser.add_argument("--ann-file", required=True, help="输入 pkl 路径")
    parser.add_argument("--calib-root", required=True, help="calibration_data 目录，例如 data/synwoodscape/SynWoodScape_V0.1.1/calibration_data")
    parser.add_argument("--out", required=True, help="输出 pkl 路径")
    args = parser.parse_args()

    with open(args.ann_file, "rb") as f:
        ann = pickle.load(f)

    infos = ann["infos"] if isinstance(ann, dict) and "infos" in ann else ann

    # 预加载标定
    calib_cache = {}
    for cam, fname in CAM_JSON_MAP.items():
        path = os.path.join(args.calib_root, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到标定文件 {path}")
        calib_cache[cam] = load_json(path)

    updated = 0
    for item in infos:
        cams = item.get("cams", {})
        for cam_name, cam_info in cams.items():
            if cam_name not in calib_cache:
                continue
            calib = calib_cache[cam_name]
            w = cam_info.get("width") or cam_info.get("img_shape", [None, None])[-1]
            h = cam_info.get("height") or cam_info.get("img_shape", [None, None])[0]
            try:
                radial = extract_radial_params(calib, fallback_w=w, fallback_h=h)
            except Exception as e:
                raise RuntimeError(f"{cam_name} 提取 radial_params 失败: {e}")

            cam_info["cam_radial_params"] = radial
            cam_info["cam_model"] = "radial_poly"
            updated += 1

    with open(args.out, "wb") as f:
        pickle.dump(ann, f)

    print(f"已更新 {updated} 个相机参数，输出到 {args.out}")


if __name__ == "__main__":
    main()
