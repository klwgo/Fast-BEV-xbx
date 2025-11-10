#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为 Fast-BEV 派生模型导出 ONNX 结构图的辅助脚本。

目前支持两种导出模式：
1. image2bev (2d)：从多视角图像经过 backbone + FPN + neck_fuse 输出融合后的特征图；
2. bev2head (3d)：从已经回投影到 BEV 的体素特征进入 M2BevNeck 及检测头、分割头。

示例：
    python tools/export_fastbev_onnx.py \
        --config configs/woodscape/fastbev_woodscape_fisheye.py \
        --branch image2bev \
        --output work_dirs/export/fastbev_image2bev.onnx

    python tools/export_fastbev_onnx.py \
        --config configs/woodscape/fastbev_woodscape_fisheye.py \
        --branch bev2head \
        --bev-shape 200 100 6 \
        --output work_dirs/export/fastbev_bev2head.onnx
"""

import argparse
import importlib.util
import os
from os import path as osp
import sys
import types
from typing import Tuple


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch


def _ensure_mmcv_stub_ops() -> None:
    """为缺失的 mmcv CUDA 算子注册占位实现，便于仅做结构导出。"""

    class _DummyOp(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, *args, **kwargs):
            raise RuntimeError(
                "该算子依赖 mmcv-full CUDA 扩展，当前环境仅提供占位符。"
            )

    if "mmcv._ext" not in sys.modules:
        sys.modules["mmcv._ext"] = types.ModuleType("mmcv._ext")

    mmcv_ops = sys.modules.get("mmcv.ops")
    if mmcv_ops is None:
        mmcv_ops = types.ModuleType("mmcv.ops")
        sys.modules["mmcv.ops"] = mmcv_ops
    if not hasattr(mmcv_ops, "__path__"):
        mmcv_ops.__path__ = []

    if not hasattr(mmcv_ops, "carafe"):
        carafe_mod = types.ModuleType("mmcv.ops.carafe")
        carafe_mod.CARAFEPack = _DummyOp
        carafe_mod.CARAFEPackFast = _DummyOp
        sys.modules["mmcv.ops.carafe"] = carafe_mod
        mmcv_ops.carafe = carafe_mod

    if not hasattr(mmcv_ops, "multi_scale_deform_attn"):
        msda_mod = types.ModuleType("mmcv.ops.multi_scale_deform_attn")
        msda_mod.MultiScaleDeformableAttention = _DummyOp
        sys.modules["mmcv.ops.multi_scale_deform_attn"] = msda_mod
        mmcv_ops.multi_scale_deform_attn = msda_mod

    if not hasattr(mmcv_ops, "merge_cells"):
        merge_mod = types.ModuleType("mmcv.ops.merge_cells")
        merge_mod.GlobalPoolingCell = _DummyOp
        merge_mod.SumCell = _DummyOp
        merge_mod.ConcatCell = _DummyOp
        sys.modules["mmcv.ops.merge_cells"] = merge_mod
        mmcv_ops.merge_cells = merge_mod

    if not hasattr(mmcv_ops, "nms_match"):
        def _nms_match_stub(*args, **kwargs):
            raise RuntimeError("nms_match 依赖 mmcv-full，当前为占位实现。")

        mmcv_ops.nms_match = _nms_match_stub

    if not hasattr(mmcv_ops, "roi_align"):
        roi_align_mod = types.ModuleType("mmcv.ops.roi_align")

        def _roi_align_stub(*args, **kwargs):
            raise RuntimeError("roi_align 依赖 mmcv-full，当前为占位实现。")

        roi_align_mod.roi_align = _roi_align_stub
        sys.modules["mmcv.ops.roi_align"] = roi_align_mod
        mmcv_ops.roi_align = roi_align_mod

    if not hasattr(mmcv_ops, "nms"):
        nms_mod = types.ModuleType("mmcv.ops.nms")

        def _batched_nms_stub(*args, **kwargs):
            raise RuntimeError("batched_nms 依赖 mmcv-full，当前为占位实现。")

        nms_mod.batched_nms = _batched_nms_stub
        sys.modules["mmcv.ops.nms"] = nms_mod
        mmcv_ops.nms = nms_mod
        mmcv_ops.batched_nms = _batched_nms_stub

    if not hasattr(mmcv_ops, "DeformConv2d"):
        mmcv_ops.DeformConv2d = _DummyOp

    if not hasattr(mmcv_ops, "CornerPool"):
        mmcv_ops.CornerPool = _DummyOp

    if not hasattr(mmcv_ops, "sigmoid_focal_loss"):
        def _sigmoid_focal_stub(*args, **kwargs):
            raise RuntimeError("sigmoid_focal_loss 依赖 mmcv-full，当前为占位实现。")

        mmcv_ops.sigmoid_focal_loss = _sigmoid_focal_stub

    if not hasattr(mmcv_ops, "MaskedConv2d"):
        mmcv_ops.MaskedConv2d = _DummyOp

    if not hasattr(mmcv_ops, "point_sample"):
        def _point_sample_stub(*args, **kwargs):
            raise RuntimeError("point_sample 依赖 mmcv-full，当前为占位实现。")

        mmcv_ops.point_sample = _point_sample_stub

    if not hasattr(mmcv_ops, "rel_roi_point_to_rel_img_point"):
        def _rel_roi_stub(*args, **kwargs):
            raise RuntimeError("rel_roi_point_to_rel_img_point 依赖 mmcv-full，当前为占位实现。")

        mmcv_ops.rel_roi_point_to_rel_img_point = _rel_roi_stub

    if not hasattr(mmcv_ops, "CrissCrossAttention"):
        mmcv_ops.CrissCrossAttention = _DummyOp

    if not hasattr(mmcv_ops, "PSAMask"):
        mmcv_ops.PSAMask = _DummyOp


_ensure_mmcv_stub_ops()

import mmcv

import mmdet3d  # 基础包可直接导入

# 某些模块意外依赖 ipdb，仅用于调试，这里提供占位实现
if "ipdb" not in sys.modules:
    ipdb_stub = types.ModuleType("ipdb")

    def _ipdb_set_trace(*args, **kwargs):
        raise RuntimeError("ipdb 未安装，当前脚本仅提供占位实现。")

    ipdb_stub.set_trace = _ipdb_set_trace
    sys.modules["ipdb"] = ipdb_stub

if "mmdet3d.ops" not in sys.modules:
    ops_stub = types.ModuleType("mmdet3d.ops")
    ops_stub.__path__ = []

    def _unsupported(*args, **kwargs):
        raise RuntimeError("mmdet3d.ops 中的算子依赖 CUDA 扩展，当前为占位实现。")

    sys.modules["mmdet3d.ops"] = ops_stub
    setattr(mmdet3d, "ops", ops_stub)

    for attr in [
        "RoIAlign",
        "SigmoidFocalLoss",
        "NaiveSyncBatchNorm1d",
        "NaiveSyncBatchNorm2d",
        "DynamicScatter",
        "Voxelization",
        "SparseBasicBlock",
        "SparseBottleneck",
        "RoIAwarePool3d",
        "PAConv",
        "PAConvCUDA",
        "PAConvSAModule",
        "PAConvSAModuleMSG",
        "PAConvCUDASAModule",
        "PAConvCUDASAModuleMSG",
        "PointSAModule",
        "PointSAModuleMSG",
        "PointFPModule",
        "Points_Sampler",
    ]:
        setattr(ops_stub, attr, _unsupported)

    for func_name in [
        "get_compiler_version",
        "get_compiling_cuda_version",
        "dynamic_scatter",
        "voxelization",
        "sigmoid_focal_loss",
        "assign_score_withk",
        "build_sa_module",
        "furthest_point_sample",
        "furthest_point_sample_with_dist",
        "three_interpolate",
        "three_nn",
        "gather_points",
        "group_points",
        "grouping_operation",
        "ball_query",
        "knn",
        "points_in_boxes_gpu",
        "points_in_boxes_cpu",
        "points_in_boxes_batch",
        "make_sparse_convmodule",
    ]:
        setattr(ops_stub, func_name, _unsupported)

    iou3d_mod = types.ModuleType("mmdet3d.ops.iou3d")
    sys.modules["mmdet3d.ops.iou3d"] = iou3d_mod
    setattr(ops_stub, "iou3d", iou3d_mod)
    setattr(iou3d_mod, "iou3d_cuda", _unsupported)

    iou3d_utils_mod = types.ModuleType("mmdet3d.ops.iou3d.iou3d_utils")
    sys.modules["mmdet3d.ops.iou3d.iou3d_utils"] = iou3d_utils_mod
    for fn in ["nms_gpu", "nms_normal_gpu", "boxes_overlap_bev_gpu"]:
        setattr(iou3d_utils_mod, fn, _unsupported)

    spconv_mod = types.ModuleType("mmdet3d.ops.spconv")
    sys.modules["mmdet3d.ops.spconv"] = spconv_mod
    setattr(spconv_mod, "SparseConvTensor", _unsupported)

    roiaware_mod = types.ModuleType("mmdet3d.ops.roiaware_pool3d")
    sys.modules["mmdet3d.ops.roiaware_pool3d"] = roiaware_mod
    for fn in ["points_in_boxes_gpu", "points_in_boxes_cpu", "points_in_boxes_batch", "roiaware_pool3d"]:
        setattr(roiaware_mod, fn, _unsupported)

if "lyft_dataset_sdk" not in sys.modules:
    lyft_sdk = types.ModuleType("lyft_dataset_sdk")
    sys.modules["lyft_dataset_sdk"] = lyft_sdk

    eval_mod = types.ModuleType("lyft_dataset_sdk.eval")
    sys.modules["lyft_dataset_sdk.eval"] = eval_mod
    setattr(lyft_sdk, "eval", eval_mod)

    detection_mod = types.ModuleType("lyft_dataset_sdk.eval.detection")
    sys.modules["lyft_dataset_sdk.eval.detection"] = detection_mod
    setattr(eval_mod, "detection", detection_mod)

    map_mod = types.ModuleType("lyft_dataset_sdk.eval.detection.mAP_evaluation")
    sys.modules["lyft_dataset_sdk.eval.detection.mAP_evaluation"] = map_mod
    setattr(detection_mod, "mAP_evaluation", map_mod)

    class _LyftBox3D:
        pass

    map_mod.Box3D = _LyftBox3D

    def _lyft_stub(*args, **kwargs):
        raise RuntimeError("lyft_dataset_sdk 未安装，当前为占位实现。")

    map_mod.get_ap = _lyft_stub
    map_mod.get_classwise_aps = _lyft_stub
    map_mod.get_class_names = _lyft_stub
    map_mod.get_ious = _lyft_stub
    map_mod.group_by_key = _lyft_stub
    map_mod.wrap_in_box = _lyft_stub

if "trimesh" not in sys.modules:
    trimesh_stub = types.ModuleType("trimesh")
    sys.modules["trimesh"] = trimesh_stub

    def _trimesh_stub(*args, **kwargs):
        raise RuntimeError("trimesh 未安装，当前为占位实现。")

    trimesh_stub.creation = types.SimpleNamespace(box=_trimesh_stub)

    class _Scene:
        def __init__(self, *args, **kwargs):
            self._items = []

        def add_geometry(self, geom):
            self._items.append(geom)

        def dump(self):
            return self._items

    trimesh_stub.scene = types.SimpleNamespace(Scene=_Scene)
    trimesh_stub.util = types.SimpleNamespace(concatenate=_trimesh_stub)
    trimesh_stub.io = types.SimpleNamespace(export=types.SimpleNamespace(export_mesh=_trimesh_stub))

# 限制性地注册 mmdet3d.models 相关子模块，避免触发大量 C++ 扩展依赖
models_pkg = sys.modules.get("mmdet3d.models")
if models_pkg is None:
    models_pkg = types.ModuleType("mmdet3d.models")
    models_pkg.__path__ = [os.path.join(PROJECT_ROOT, "mmdet3d/models")]
    sys.modules["mmdet3d.models"] = models_pkg
    setattr(mmdet3d, "models", models_pkg)

detectors_pkg = sys.modules.get("mmdet3d.models.detectors")
if detectors_pkg is None:
    detectors_pkg = types.ModuleType("mmdet3d.models.detectors")
    detectors_pkg.__path__ = [os.path.join(PROJECT_ROOT, "mmdet3d/models/detectors")]
    sys.modules["mmdet3d.models.detectors"] = detectors_pkg
    setattr(models_pkg, "detectors", detectors_pkg)

necks_pkg = sys.modules.get("mmdet3d.models.necks")
if necks_pkg is None:
    necks_pkg = types.ModuleType("mmdet3d.models.necks")
    necks_pkg.__path__ = [os.path.join(PROJECT_ROOT, "mmdet3d/models/necks")]
    sys.modules["mmdet3d.models.necks"] = necks_pkg
    setattr(models_pkg, "necks", necks_pkg)

dense_heads_pkg = sys.modules.get("mmdet3d.models.dense_heads")
if dense_heads_pkg is None:
    dense_heads_pkg = types.ModuleType("mmdet3d.models.dense_heads")
    dense_heads_pkg.__path__ = [os.path.join(PROJECT_ROOT, "mmdet3d/models/dense_heads")]
    sys.modules["mmdet3d.models.dense_heads"] = dense_heads_pkg
    setattr(models_pkg, "dense_heads", dense_heads_pkg)

decode_heads_pkg = sys.modules.get("mmdet3d.models.decode_heads")
if decode_heads_pkg is None:
    decode_heads_pkg = types.ModuleType("mmdet3d.models.decode_heads")
    decode_heads_pkg.__path__ = [os.path.join(PROJECT_ROOT, "mmdet3d/models/decode_heads")]
    sys.modules["mmdet3d.models.decode_heads"] = decode_heads_pkg
    setattr(models_pkg, "decode_heads", decode_heads_pkg)

# 按需加载 utils 子模块
utils_spec = importlib.util.spec_from_file_location(
    "mmdet3d.models.utils", os.path.join(PROJECT_ROOT, "mmdet3d/models/utils/__init__.py")
)
utils_module = importlib.util.module_from_spec(utils_spec)
assert utils_spec.loader is not None
utils_spec.loader.exec_module(utils_module)
sys.modules["mmdet3d.models.utils"] = utils_module
setattr(models_pkg, "utils", utils_module)

# 注册自定义 neck，确保构建器可用
m2bev_spec = importlib.util.spec_from_file_location(
    "mmdet3d.models.necks.m2bev_neck",
    os.path.join(PROJECT_ROOT, "mmdet3d/models/necks/m2bev_neck.py"),
)
m2bev_module = importlib.util.module_from_spec(m2bev_spec)
assert m2bev_spec.loader is not None
m2bev_spec.loader.exec_module(m2bev_module)
sys.modules["mmdet3d.models.necks.m2bev_neck"] = m2bev_module

free_anchor_spec = importlib.util.spec_from_file_location(
    "mmdet3d.models.dense_heads.free_anchor3d_head",
    os.path.join(PROJECT_ROOT, "mmdet3d/models/dense_heads/free_anchor3d_head.py"),
)
free_anchor_module = importlib.util.module_from_spec(free_anchor_spec)
assert free_anchor_spec.loader is not None
free_anchor_spec.loader.exec_module(free_anchor_module)
sys.modules["mmdet3d.models.dense_heads.free_anchor3d_head"] = free_anchor_module

bev_seg_spec = importlib.util.spec_from_file_location(
    "mmdet3d.models.decode_heads.bev_seg_head",
    os.path.join(PROJECT_ROOT, "mmdet3d/models/decode_heads/bev_seg_head.py"),
)
bev_seg_module = importlib.util.module_from_spec(bev_seg_spec)
assert bev_seg_spec.loader is not None
bev_seg_spec.loader.exec_module(bev_seg_module)
sys.modules["mmdet3d.models.decode_heads.bev_seg_head"] = bev_seg_module
setattr(decode_heads_pkg, "bev_seg_head", bev_seg_module)
try:
    from mmseg.models.builder import HEADS as MMSEG_HEADS  # type: ignore
    MMSEG_HEADS.register_module(force=True)(bev_seg_module.WoodscapeBEVSegHead)
except Exception:
    pass

# mmcv-lite 环境下可能缺少 MultiScaleDeformableAttention，提前打桩避免导入失败
try:
    from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention  # type: ignore
except (ImportError, AttributeError):
    import mmcv.cnn.bricks.transformer as mmcv_transformer

    class _DummyMSDA(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise RuntimeError(
                "MultiScaleDeformableAttention 依赖 mmcv-full，请安装包含 CUDA 扩展的版本。"
            )

    mmcv_transformer.MultiScaleDeformableAttention = _DummyMSDA

from mmcv import Config
from mmcv.runner import load_checkpoint

FASTBEV_SPEC = importlib.util.spec_from_file_location(
    "fastbev_module", os.path.join(PROJECT_ROOT, "mmdet3d/models/detectors/fastbev.py")
)
fastbev_module = importlib.util.module_from_spec(FASTBEV_SPEC)
assert FASTBEV_SPEC.loader is not None
FASTBEV_SPEC.loader.exec_module(fastbev_module)
FastBEV = fastbev_module.FastBEV


BRANCH_CHOICES = ("image2bev", "bev2head")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Fast-BEV model (or subgraph) to ONNX for structure inspection."
    )
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="可选的权重文件路径，若不提供则使用随机初始化参数。",
    )
    parser.add_argument(
        "--branch",
        default="image2bev",
        choices=BRANCH_CHOICES,
        help="选择导出的子图：image2bev（图像到融合特征）或 bev2head（BEV 到检测/分割头）。",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="导出的 ONNX 文件路径。",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="使用的设备，例如 cpu、cuda、cuda:0 等。默认 cpu 以保证兼容。",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=11,
        help="ONNX opset 版本，默认 11。",
    )
    parser.add_argument(
        "--image-shape",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help="图像输入尺寸 (H W)，默认读取 config.data_config.test_input_size 或 (320, 640)。",
    )
    parser.add_argument(
        "--bev-shape",
        type=int,
        nargs=3,
        metavar=("VX", "VY", "VZ"),
        default=None,
        help="BEV 体素网格形状 (VX VY VZ)，默认读取 config.model.n_voxels[0]。",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="bev2head 输入的序列长度（如时序融合帧数），默认读取 config.n_times 或 1。",
    )
    parser.add_argument(
        "--disable-seg-head",
        action="store_true",
        help="导出 bev2head 时跳过 BEV segmentation head（如不需要对应输出）。",
    )
    return parser.parse_args()


def init_model(cfg: Config, checkpoint: str, device: torch.device) -> torch.nn.Module:
    cfg = cfg.copy()
    model_cfg = cfg.model.copy()
    model_cfg.setdefault("pretrained", None)
    model_type = model_cfg.pop("type")
    assert model_type == "FastBEV", f"当前脚本只支持 FastBEV，收到 {model_type}"
    model = FastBEV(**model_cfg)
    if checkpoint:
        load_checkpoint(model, checkpoint, map_location="cpu")
    model.cfg = cfg  # 有些模块在 forward 中需要访问
    model.to(device)
    if not hasattr(model, "neck_fuse"):
        for attr in dir(model):
            if attr.startswith("neck_fuse_"):
                setattr(model, "neck_fuse", getattr(model, attr))
                break
    model.eval()
    return model


def prepare_image_dummy(cfg: Config, args: argparse.Namespace, device: torch.device) -> Tuple[torch.Tensor, list]:
    num_views = cfg.model.get("num_views", 6)
    if args.image_shape is not None:
        img_h, img_w = args.image_shape
    else:
        data_cfg = cfg.get("data_config", {})
        img_h, img_w = data_cfg.get("test_input_size", data_cfg.get("input_size", (320, 640)))
    dummy = torch.randn(num_views, 3, img_h, img_w, device=device)
    img_metas = [dict()]  # onnx_export_2d 不会实际读取 meta
    return dummy, img_metas


def prepare_bev_dummy(cfg: Config, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    bev_shape = args.bev_shape
    if bev_shape is None:
        n_voxels = cfg.model.get("n_voxels", [[200, 200, 6]])
        bev_shape = n_voxels[0]
    assert len(bev_shape) == 3, "BEV 体素形状需提供 3 个维度 (VX VY VZ)"
    seq_len = args.seq_len if args.seq_len is not None else cfg.get("n_times", 1)
    seq_len = max(1, int(seq_len))
    in_channels = cfg.model["neck_3d"]["in_channels"]
    vx, vy, vz = bev_shape
    dummy = torch.randn(seq_len, in_channels, vx, vy, vz, device=device)
    return dummy


def export_image_branch(model: torch.nn.Module, cfg: Config, args: argparse.Namespace, device: torch.device) -> None:
    dummy, img_metas = prepare_image_dummy(cfg, args, device)

    class _Wrapper(torch.nn.Module):
        def __init__(self, inner_model, metas):
            super().__init__()
            self.inner_model = inner_model
            self.metas = metas

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
            return self.inner_model(
                x,
                self.metas,
                return_loss=False,
                export_2d=True,
                export_3d=False,
            )

    wrapper = _Wrapper(model, img_metas)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        args.output,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["multiview_img"],
        output_names=["fused_bev_tokens"],
        dynamic_axes={
            "multiview_img": {0: "num_views", 2: "input_h", 3: "input_w"},
            "fused_bev_tokens": {0: "num_views", 2: "feat_h", 3: "feat_w"},
        },
    )


def export_bev_branch(model: torch.nn.Module, cfg: Config, args: argparse.Namespace, device: torch.device) -> None:
    dummy = prepare_bev_dummy(cfg, args, device)

    class _Wrapper(torch.nn.Module):
        def __init__(self, inner_model, disable_seg: bool):
            super().__init__()
            self.inner_model = inner_model
            self.disable_seg = disable_seg

        def forward(self, x: torch.Tensor):  # type: ignore
            outputs = self.inner_model(
                x,
                None,
                return_loss=False,
                export_2d=False,
                export_3d=True,
            )
            if self.disable_seg and isinstance(outputs, (list, tuple)):
                return outputs[0]
            return outputs

    wrapper = _Wrapper(model, args.disable_seg_head)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        args.output,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["bev_volume"],
        output_names=["head_outputs"],
        dynamic_axes={
            "bev_volume": {0: "seq_len", 2: "vx", 3: "vy", 4: "vz"},
        },
    )


def _load_py_config(path: str) -> Config:
    placeholder_globals = dict(
        __file__=osp.abspath(path),
        bev_seg_classes=(),
        bbox2d_classes=(),
    )
    cfg_locals: dict = {}
    with open(path, "r", encoding="utf-8") as f:
        code = compile(f.read(), path, "exec")
    exec(code, placeholder_globals, cfg_locals)
    merged = {
        k: v
        for k, v in {**placeholder_globals, **cfg_locals}.items()
        if not k.startswith("__")
    }
    cfg = Config(merged, filename=path)
    with open(path, "r", encoding="utf-8") as f:
        cfg.text = f.read()
    return cfg


def main():
    args = parse_args()
    device = torch.device(args.device)
    cfg = _load_py_config(args.config)
    model = init_model(cfg, args.checkpoint, device)

    with torch.no_grad():
        if args.branch == "image2bev":
            export_image_branch(model, cfg, args, device)
        else:
            export_bev_branch(model, cfg, args, device)

    print(f"ONNX 已导出到: {args.output}")


if __name__ == "__main__":
    main()
