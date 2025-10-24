import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from mmdet3d.models.detectors.fastbev import get_points
from mmdet3d.models.utils import (
    build_fisheye_lut,
    make_lut_cache_key,
    prepare_calibrations,
)


def _as_tensor(data: Any) -> torch.Tensor:
    return torch.as_tensor(data, dtype=torch.float32)


def _load_payload(path: Path) -> Dict[str, Any]:
    with path.open('r') as handle:
        payload = json.load(handle)
    required_keys = [
        'intrinsic',
        'extrinsic',
        'origin',
        'voxel_size',
        'n_voxels',
        'img_shape',
        'stride',
    ]
    missing = [k for k in required_keys if k not in payload]
    if missing:
        raise KeyError(f'Missing keys in calibration json: {missing}')
    return payload


def _prepare_calibration(payload: Dict[str, Any]):
    # 支持多相机输入，因此内参既可以是单个矩阵也可以是数组
    intrinsic = torch.as_tensor(payload['intrinsic'], dtype=torch.float32)
    extrinsic = [torch.as_tensor(ex, dtype=torch.float32) for ex in payload['extrinsic']]
    distortion = None
    if 'distortion' in payload:
        distortion = [torch.as_tensor(d, dtype=torch.float32) for d in payload['distortion']]
    model = payload.get('models')
    return prepare_calibrations(
        intrinsic=intrinsic,
        extrinsics=extrinsic,
        distortion=distortion,
        model_per_cam=model,
    )


def main():
    parser = argparse.ArgumentParser(
        description='预计算鱼眼相机 BEV LUT，供 Fast-BEV 在 WoodScape 等数据集上使用',
    )
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='包含相机标定与 BEV 网格设定的 JSON 文件路径',
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='输出 LUT 的保存路径（.pt）',
    )
    parser.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='执行 LUT 计算的设备，默认为首个可用 GPU',
    )
    args = parser.parse_args()

    payload = _load_payload(args.config)

    device = torch.device(args.device)
    # 构建 BEV 体素网格
    n_voxels = _as_tensor(payload['n_voxels']).long()
    voxel_size = _as_tensor(payload['voxel_size'])
    origin = _as_tensor(payload['origin'])
    points = get_points(
        n_voxels=n_voxels,
        voxel_size=voxel_size,
        origin=origin,
    ).to(device)

    stride = int(payload['stride'])
    img_h, img_w = payload['img_shape']
    feature_h = math.ceil(img_h / stride)
    feature_w = math.ceil(img_w / stride)

    # 将 JSON 中的相机参数封装为 CameraCalibration 列表
    calibrations = _prepare_calibration(payload)

    # 计算 LUT 与有效性掩码
    lut, valid = build_fisheye_lut(
        points=points,
        cameras=calibrations,
        height=feature_h,
        width=feature_w,
        stride=stride,
    )

    cache_key = make_lut_cache_key(
        calibrations=calibrations,
        stride=stride,
        voxel_size=voxel_size,
        origin=origin,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 保存 LUT、有效性掩码及元信息，便于训练阶段直接加载
    torch.save(
        {
            'lut': lut.cpu(),
            'valid': valid.cpu(),
            'metadata': {
                'n_voxels': payload['n_voxels'],
                'voxel_size': payload['voxel_size'],
                'origin': payload['origin'],
                'stride': stride,
                'img_shape': payload['img_shape'],
                'cache_key': cache_key,
                'camera_model': payload.get('models', 'polynomial'),
                'n_cams': len(calibrations),
            },
        },
        args.output,
    )
    print(f"LUT 已保存至 {args.output}，缓存 key: {cache_key}")


if __name__ == '__main__':
    main()
