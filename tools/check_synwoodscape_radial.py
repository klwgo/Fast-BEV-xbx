"""校验 synwoodscape pkl 中是否正确写入 radial_poly 参数。

使用方法：
  python tools/check_synwoodscape_radial.py \
    --ann-file data/synwoodscape_infos_train_small_fixed.pkl \
    --max-samples 50

检查项：
  - cams 是否存在
  - cam_model 是否 radial_poly / polynomial
  - cam_radial_params 是否为 dict，且包含 k1..k4、cx_offset、cy_offset、aspect_ratio、width、height
  - 若缺失/为 None，会列出样本索引和相机名
"""

import argparse
import pickle


REQUIRED_KEYS = [
    "k1",
    "k2",
    "k3",
    "k4",
    "cx_offset",
    "cy_offset",
    "aspect_ratio",
    "width",
    "height",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-file", required=True, help="synwoodscape pkl 文件")
    parser.add_argument("--max-samples", type=int, default=100, help="最多检查多少条样本")
    args = parser.parse_args()

    with open(args.ann_file, "rb") as f:
        ann = pickle.load(f)

    infos = ann.get("infos") or []
    total = len(infos)
    print(f"共 {total} 条，检查前 {min(total, args.max_samples)} 条")

    missing = []
    bad_model = []
    ok = 0

    for idx, info in enumerate(infos[: args.max_samples]):
        cams = info.get("cams", {})
        if not cams:
            missing.append((idx, "<no cams>"))
            continue
        for name, cam in cams.items():
            model = cam.get("cam_model", "")
            if model not in {"radial_poly", "polynomial"}:
                bad_model.append((idx, name, model))
                continue
            rp = cam.get("cam_radial_params")
            if not isinstance(rp, dict):
                missing.append((idx, name))
                continue
            if not all(k in rp and rp[k] is not None for k in REQUIRED_KEYS):
                missing.append((idx, name))
                continue
            ok += 1

    print(f"通过相机数量: {ok}")
    if bad_model:
        print("模型非 radial_poly / polynomial 的条目:")
        for item in bad_model:
            print("  idx=%s cam=%s model=%s" % item)
    if missing:
        print("缺少 cam_radial_params 关键字段的条目:")
        for idx, name in missing:
            print(f"  idx={idx} cam={name}")
    if not bad_model and not missing:
        print("全部满足要求")


if __name__ == "__main__":
    main()
