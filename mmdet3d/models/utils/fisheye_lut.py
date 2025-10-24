import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch


@dataclass
class CameraCalibration:
    """描述单个相机的标定数据容器，所有张量默认使用 float32。"""

    intrinsic: torch.Tensor  # shape (3, 3)
    extrinsic: torch.Tensor  # shape (4, 4) or (3, 4)
    distortion: Optional[torch.Tensor] = None  # shape (N,), typically 4
    model: str = "polynomial"


def _normalise_calib_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """将张量搬运到 CPU 并展平，用于生成缓存哈希。"""

    if tensor.dim() == 0:
        return tensor.detach().cpu().reshape(1)
    return tensor.detach().cpu().reshape(-1)


def _build_camera_key(
    calibrations: Iterable[CameraCalibration],
    stride: int,
    voxel_size: torch.Tensor,
    origin: torch.Tensor,
) -> str:
    """针对特定相机设置生成稳定的缓存 key，保证 LUT 可复用。"""

    buffer: List[torch.Tensor] = [
        torch.tensor([float(stride)], dtype=torch.float32),
        voxel_size.detach().cpu().float().reshape(-1),
        origin.detach().cpu().float().reshape(-1),
    ]
    # 将每个相机的内参、外参及畸变参数依次拼接，形成唯一特征向量
    for calib in calibrations:
        buffer.append(_normalise_calib_tensor(calib.intrinsic))
        buffer.append(_normalise_calib_tensor(calib.extrinsic))
        if calib.distortion is not None:
            buffer.append(_normalise_calib_tensor(calib.distortion))
        buffer.append(torch.tensor([hash(calib.model) % (2**16)], dtype=torch.float32))

    payload = torch.cat(buffer).numpy().tobytes()
    return hashlib.sha1(payload).hexdigest()


def project_fisheye_points(
    points_cam: torch.Tensor,
    intrinsic: torch.Tensor,
    distortion: Optional[torch.Tensor],
    model: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """使用指定鱼眼模型将相机坐标系下的 3D 点投影到像素平面。"""

    eps = 1e-8
    # 鱼眼投影在半球坐标下展开，首先归一化到成像平面
    x = points_cam[0] / torch.clamp(points_cam[2], min=eps)
    y = points_cam[1] / torch.clamp(points_cam[2], min=eps)
    z = points_cam[2]

    r = torch.sqrt(x * x + y * y)
    theta = torch.atan(r)

    # 不同模型对 theta 的处理不同，OpenCV 多项式需考虑高阶畸变
    if model == "polynomial":
        if distortion is None:
            distortion = torch.zeros(4, device=points_cam.device, dtype=points_cam.dtype)
        else:
            distortion = torch.nn.functional.pad(
                distortion.to(points_cam.device, points_cam.dtype),
                (0, max(0, 4 - distortion.numel())),
                mode="constant",
                value=0,
            )
        theta2 = theta * theta
        theta_d = theta * (
            1
            + distortion[0] * theta2
            + distortion[1] * theta2 * theta2
            + distortion[2] * theta2 * theta2 * theta2
            + distortion[3] * theta2 * theta2 * theta2 * theta2
        )
    elif model == "equidistant":
        theta_d = theta
    elif model == "equisolid":
        theta_d = 2.0 * torch.sin(theta * 0.5)
    elif model == "stereographic":
        theta_d = 2.0 * torch.tan(theta * 0.5)
    else:
        raise ValueError(f"Unsupported fisheye model: {model}")

    scale = torch.where(r > eps, theta_d / torch.clamp(r, min=eps), torch.ones_like(r))

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    xd = x * scale
    yd = y * scale
    u = fx * xd + cx
    v = fy * yd + cy

    valid = z > 0
    return u, v, valid


def transform_world_to_camera(extrinsic: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """将世界坐标系下的点通过外参矩阵转换到相机坐标系。"""

    if extrinsic.shape == (3, 4):
        rot = extrinsic[:, :3]
        trans = extrinsic[:, 3:]
    else:
        rot = extrinsic[:3, :3]
        trans = extrinsic[:3, 3:]
    return rot @ points + trans


def build_fisheye_lut(
    points: torch.Tensor,
    cameras: List[CameraCalibration],
    height: int,
    width: int,
    stride: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """构建鱼眼相机下的体素到图像索引表。

    返回结果按相机维度排列，便于后续执行“多视角融合到单体素”的特征聚合：
        - pixel_indices: [n_cams, n_voxels]，保存像素的线性索引 (y * width + x)，无效位置填 -1
        - valid_mask:    [n_cams, n_voxels]，布尔掩码，指示该体素是否被对应相机覆盖
    """

    device = points.device
    dtype = points.dtype
    n_voxels = points.shape[1] * points.shape[2] * points.shape[3]
    n_cameras = len(cameras)

    pixel_indices = torch.full(
        (n_cameras, n_voxels), -1, dtype=torch.int32, device=device
    )
    valid_mask = torch.zeros(
        (n_cameras, n_voxels), dtype=torch.bool, device=device
    )

    # 将体素网格展平，方便统一投影
    flat_points = points.reshape(3, -1)

    for cam_id, calib in enumerate(cameras):
        # 针对每路相机独立完成一次投影映射
        intrinsic = calib.intrinsic.clone().to(device=device, dtype=dtype)
        intrinsic[:2, :3] /= float(stride)

        extrinsic = calib.extrinsic.clone().to(device=device, dtype=dtype)
        cam_points = transform_world_to_camera(extrinsic, flat_points)

        u, v, in_front = project_fisheye_points(
            cam_points,
            intrinsic,
            calib.distortion.to(device=device, dtype=dtype) if calib.distortion is not None else None,
            calib.model,
        )

        x = torch.round(u).to(torch.long)
        y = torch.round(v).to(torch.long)
        within_bounds = (x >= 0) & (y >= 0) & (x < width) & (y < height)
        # 仅保留在成像平面内且位于相机前方的体素
        mask = in_front & within_bounds

        if mask.any():
            linear_idx = y * width + x
            pixel_indices[cam_id, mask] = linear_idx[mask].to(torch.int32)
            valid_mask[cam_id, mask] = True

    return pixel_indices, valid_mask


class FisheyeLUTCache:
    """管理鱼眼 LUT 的内存/磁盘缓存，可复用离线生成结果。"""

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        save_to_disk: bool = True,
    ) -> None:
        self.memory_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.save_to_disk = bool(cache_dir) and save_to_disk
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_lut(
        self,
        key: str,
        build_fn,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """按 key 获取 LUT，如不存在则调用构建函数并可选落盘。"""

        if key in self.memory_cache:
            return self.memory_cache[key]

        disk_path = None
        if self.cache_dir is not None:
            disk_path = self.cache_dir / f"{key}.pt"
            if disk_path.is_file():
                payload = torch.load(disk_path, map_location="cpu")
                lut = payload["lut"]
                valid = payload["valid"]
                self.memory_cache[key] = (lut, valid)
                return lut, valid

        lut, valid = build_fn()

        if self.save_to_disk and disk_path is not None:
            torch.save({"lut": lut.cpu(), "valid": valid.cpu()}, disk_path)

        self.memory_cache[key] = (lut, valid)
        return lut, valid


def make_lut_cache_key(
    calibrations: Iterable[CameraCalibration],
    stride: int,
    voxel_size: torch.Tensor,
    origin: torch.Tensor,
) -> str:
    """公开接口：计算 LUT 缓存 key，方便离线脚本复用。"""

    return _build_camera_key(calibrations, stride, voxel_size, origin)


def prepare_calibrations(
    intrinsic: torch.Tensor,
    extrinsics: Iterable[torch.Tensor],
    distortion: Optional[Iterable[torch.Tensor]],
    model_per_cam: Optional[Iterable[str]],
) -> List[CameraCalibration]:
    """将原始标定数组转成 CameraCalibration 列表，便于统一处理。"""

    calib_list: List[CameraCalibration] = []
    distortions: Optional[List[torch.Tensor]] = None
    if distortion is not None:
        distortions = [torch.as_tensor(d) for d in distortion]
    models: Optional[List[str]] = None
    if model_per_cam is not None:
        models = list(model_per_cam)

    intrinsic = torch.as_tensor(intrinsic)
    for idx, extrinsic in enumerate(extrinsics):
        calib_list.append(
            CameraCalibration(
                intrinsic=torch.as_tensor(intrinsic[idx] if intrinsic.ndim == 3 else intrinsic),
                extrinsic=torch.as_tensor(extrinsic),
                distortion=None if distortions is None else distortions[idx],
                model="polynomial" if models is None else models[idx],
            )
        )
    return calib_list
