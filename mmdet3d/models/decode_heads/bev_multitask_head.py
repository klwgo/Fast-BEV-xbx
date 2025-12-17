# -*- coding: utf-8 -*-
"""Fast-BEV 多任务 Head：Drivable / Marking / Slot / Obstacle / Occlusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import kaiming_init
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from mmdet.models import build_loss
from mmdet.models.builder import HEADS
from mmseg.models.builder import HEADS as MMSEG_HEADS


class _WeightedLoss(nn.Module):
    """Wrap torch loss to support loss_weight similar到 MMCV."""

    def __init__(self, criterion, loss_weight=1.0):
        super().__init__()
        self.criterion = criterion
        self.loss_weight = loss_weight

    def forward(self, *args, **kwargs):
        return self.loss_weight * self.criterion(*args, **kwargs)


class _MaskedLoss(nn.Module):
    """Wrap a criterion to support ignore_index by masking reduction='none' output."""

    def __init__(self, criterion, ignore_index, reduction='mean'):
        super().__init__()
        self.criterion = criterion
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, pred, target, *args, **kwargs):
        loss = self.criterion(pred, target, *args, **kwargs)
        if loss.dim() > target.dim():
            # align loss shape to target when criterion returns per-class logits
            loss = loss.squeeze(1)
        valid = target != self.ignore_index
        if not valid.any():
            return loss.sum() * 0
        loss = loss * valid
        if self.reduction == 'mean':
            return loss.sum() / valid.sum()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class _SigmoidFocalLoss(nn.Module):
    """Simplified sigmoid focal loss, supports arbitrary shape, reduction handled here."""

    def __init__(self, gamma=2.0, alpha=0.25, reduction='none'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):
        # pred/target same shape
        prob = torch.sigmoid(pred)
        pt = target * prob + (1 - target) * (1 - prob)
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        focal = alpha_t * torch.pow(1 - pt, self.gamma)
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        loss = focal * bce
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class SEBlock(nn.Module):
    """Squeeze-and-Excitation for channel attention."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x)
        return x * w


class ASPPModule(nn.Module):
    """简单 ASPP，用多尺度空洞卷积汇聚上下文。"""

    def __init__(self, in_channels, out_channels, dilations=(1, 3, 6)):
        super().__init__()
        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=d, dilation=d, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * len(dilations), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        return self.fuse(torch.cat(feats, dim=1))


class SpatialAttention(nn.Module):
    """空间注意力，强调显著区域。"""

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.conv(torch.cat([avg_out, max_out], dim=1))
        attn = self.sigmoid(attn)
        return x * attn


def _dice_loss(logits,
               target,
               ignore_index=255,
               eps=1e-6,
               class_weight=None):
    """Multi-class soft dice with ignore mask."""
    if logits.dim() == 4:
        # N,C,H,W ; target N,H,W
        probs = F.softmax(logits, dim=1)
    else:
        raise ValueError('Dice loss expects shape (N,C,H,W)')
    num_classes = probs.size(1)
    # mask ignore
    valid = target != ignore_index
    if not valid.any():
        return probs.sum() * 0
    one_hot = F.one_hot(torch.clamp(target, min=0), num_classes=num_classes)  # (N,H,W,C)
    one_hot = one_hot.permute(0, 3, 1, 2).float()
    probs = probs * valid.unsqueeze(1)
    one_hot = one_hot * valid.unsqueeze(1)
    dims = (0, 2, 3)
    intersection = (probs * one_hot).sum(dims)
    union = probs.sum(dims) + one_hot.sum(dims) + eps
    dice = 2 * intersection / union
    if class_weight is not None:
        weight = torch.as_tensor(class_weight,
                                 device=dice.device,
                                 dtype=dice.dtype)
        dice = dice * weight
        norm = weight.sum().clamp(min=eps)
    else:
        norm = float(num_classes)
    loss = 1 - dice.sum() / norm
    return loss


def _build_loss_module(cfg, default_type=None):
    """支持 mmcv 配置或 PyTorch 原生损失."""
    if cfg is None:
        return None
    loss_cfg = cfg.copy()
    loss_type = loss_cfg.pop('type', default_type)
    if loss_type in (None,):
        return None
    # 兼容 class_weight 写法
    class_weight = loss_cfg.pop('class_weight', None)
    weight = loss_cfg.get('weight')
    if weight is None and class_weight is not None:
        weight = class_weight
    if isinstance(weight, (list, tuple)):
        loss_cfg['weight'] = torch.tensor(weight, dtype=torch.float32)
    if 'pos_weight' in loss_cfg:
        pw = loss_cfg['pos_weight']
        if not torch.is_tensor(pw):
            loss_cfg['pos_weight'] = torch.as_tensor(pw, dtype=torch.float32)
    ignore_index = loss_cfg.pop('ignore_index', None)
    loss_weight = loss_cfg.pop('loss_weight', 1.0)
    # 对于 BCE/SigmoidFocal，期望外部自行处理 ignore_index / class_weight，这里强制 reduction='none'
    if loss_type in ('BCEWithLogitsLoss', 'SigmoidFocalLoss'):
        loss_cfg.setdefault('reduction', 'none')
        # weight 用于外部 class_weight，避免传入 criterion
        loss_cfg.pop('weight', None)
    if loss_type == 'BCEWithLogitsLoss':
        # 清理不属于 BCE 的多余参数（如 gamma/alpha）避免构造报错
        loss_cfg.pop('gamma', None)
        loss_cfg.pop('alpha', None)
        return _WeightedLoss(nn.BCEWithLogitsLoss(**loss_cfg), loss_weight)
    if loss_type == 'SigmoidFocalLoss':
        gamma = loss_cfg.pop('gamma', 2.0)
        alpha = loss_cfg.pop('alpha', 0.25)
        reduction = loss_cfg.pop('reduction', 'none')
        return _WeightedLoss(_SigmoidFocalLoss(gamma=gamma, alpha=alpha, reduction=reduction),
                             loss_weight)
    if loss_type == 'MSELoss':
        return _WeightedLoss(nn.MSELoss(**loss_cfg), loss_weight)
    if loss_type == 'CrossEntropyLoss':
        return _WeightedLoss(nn.CrossEntropyLoss(**loss_cfg), loss_weight)
    loss_cfg['type'] = loss_type
    # 对不支持 ignore_index 的损失（如 FocalLoss）改用外部 mask
    if ignore_index is not None:
        loss_cfg.setdefault('reduction', 'none')
    criterion = build_loss(loss_cfg)
    if ignore_index is not None and not hasattr(criterion, 'ignore_index'):
        reduction = loss_cfg.get('reduction', 'mean')
        criterion = _MaskedLoss(criterion, ignore_index, reduction)
    if isinstance(criterion, nn.Module) and hasattr(criterion, 'loss_weight'):
        criterion.loss_weight *= loss_weight
        return criterion
    return _WeightedLoss(criterion, loss_weight)


@HEADS.register_module()
@MMSEG_HEADS.register_module()
class FisheyeBEVMultiTaskHead(BaseModule):
    """统一输出 BEV 多任务预测，并支持按需开关."""

    def __init__(self,
                 in_channels,
                 shared_channels=128,
                 drivable_classes=3,
                 marking_classes=5,
                 slot_channels=2,
                 obstacle_classes=4,
                 enable_heads=None,
                 line_kernel=5,
                 loss_drivable=dict(
                     type='BCEWithLogitsLoss', ignore_index=255, loss_weight=1.0),
                 loss_marking=dict(
                     type='BCEWithLogitsLoss', ignore_index=255, loss_weight=1.0),
                 loss_boundary=None,
                 loss_slot=dict(
                     type='BCEWithLogitsLoss', reduction='mean', loss_weight=1.0),
                 loss_obstacle=dict(
                     type='BCEWithLogitsLoss', ignore_index=255, loss_weight=1.0),
                 loss_occlusion=dict(
                     type='BCEWithLogitsLoss', reduction='mean', loss_weight=1.0),
                 pos_topk_ratio=0.3,
                 neg_pos_ratio=3.0,
                 balance_drivable=False,
                 balance_marking=True,
                 balance_obstacle=False,
                 max_balance_factor=200.0,
                 marking_dilate_kernel=3,
                 marking_dice_weight=2.0,
                 drivable_dilate_kernel=None,
                 use_simple_drivable=False):
        super().__init__()
        enable_heads = enable_heads or dict(
            drivable=True,
            marking=True,
            slot=True,
            obstacle=False,
            occlusion=False,
        )
        self.enable_drivable = enable_heads.get('drivable', True)
        self.enable_marking = enable_heads.get('marking', True)
        self.enable_slot = enable_heads.get('slot', True)
        self.enable_obstacle = enable_heads.get('obstacle', False)
        self.enable_occlusion = enable_heads.get('occlusion', False)

        self.drivable_classes = drivable_classes
        self.marking_classes = marking_classes
        self.slot_channels = slot_channels
        self.obstacle_classes = obstacle_classes
        self.pos_topk_ratio = pos_topk_ratio
        self.neg_pos_ratio = neg_pos_ratio
        self.balance_drivable = balance_drivable
        self.balance_marking = balance_marking
        self.balance_obstacle = balance_obstacle
        self.max_balance_factor = max_balance_factor
        self.marking_dilate_kernel = marking_dilate_kernel
        self.marking_dice_weight = marking_dice_weight
        self.drivable_dilate_kernel = drivable_dilate_kernel
        self.use_simple_drivable = use_simple_drivable
        # 记录 ignore_index 与 class_weight 供 sigmoid/BCE 使用
        self.drivable_ignore = loss_drivable.get('ignore_index', 255)
        self.marking_ignore = loss_marking.get('ignore_index', 255)
        self.obstacle_ignore = loss_obstacle.get('ignore_index', 255)
        self.drivable_class_weight = torch.as_tensor(
            loss_drivable.get('weight', [1.0] * drivable_classes), dtype=torch.float32)
        self.marking_class_weight = torch.as_tensor(
            loss_marking.get('weight', [1.0] * marking_classes), dtype=torch.float32)
        self.obstacle_class_weight = torch.as_tensor(
            loss_obstacle.get('weight', [1.0] * obstacle_classes), dtype=torch.float32)

        # 多尺度上下文增强（空洞卷积 ASPP 风格）
        ctx_planes = in_channels // 2
        self.context_branches = nn.ModuleList([
            nn.Conv2d(in_channels, ctx_planes, 3, padding=1, dilation=1, bias=False),
            nn.Conv2d(in_channels, ctx_planes, 3, padding=2, dilation=2, bias=False),
            nn.Conv2d(in_channels, ctx_planes, 3, padding=3, dilation=3, bias=False),
        ])
        self.context_fuse = nn.Sequential(
            nn.Conv2d(ctx_planes * 3, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.se_shared_in = SEBlock(in_channels, reduction=8)
        self.sa_shared_in = SpatialAttention(kernel_size=7)
        # 轻量 Pyramid Pooling，补充全局上下文
        ppm_out = in_channels // 2
        self.ppm_convs = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_channels, ppm_out, 1, bias=False),
                nn.GroupNorm(32, ppm_out),
                nn.ReLU(inplace=True))
            for scale in (1, 2, 3)
        ])
        self.ppm_fuse = nn.Sequential(
            nn.Conv2d(in_channels + ppm_out * 3, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, shared_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(shared_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_channels, shared_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(shared_channels),
            nn.ReLU(inplace=True),
        )

        if self.enable_drivable:
            self.drivable_shallow = ConvModule(
                in_channels,
                shared_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU', inplace=True))
            # 额外高分辨率分支（保持原分辨率细节）
            self.drivable_highres = nn.Sequential(
                nn.Conv2d(in_channels, shared_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(shared_channels),
                nn.ReLU(inplace=True),
            )
            self.drivable_adapter = nn.Sequential(
                nn.Conv2d(shared_channels, shared_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(shared_channels),
                nn.ReLU(inplace=True),
            )
            self.drivable_fuse = nn.Sequential(
                nn.Conv2d(shared_channels * 3, shared_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(shared_channels),
                nn.ReLU(inplace=True),
            )
            self.drivable_refine = nn.Sequential(
                nn.Conv2d(shared_channels, shared_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(shared_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_channels, shared_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(shared_channels),
                nn.ReLU(inplace=True),
            )
            # 大核/小核双分支强化
            self.drivable_big = nn.Conv2d(shared_channels, shared_channels, 5, padding=2, bias=False)
            self.drivable_small = nn.Conv2d(shared_channels, shared_channels, 1, bias=False)
            # PPM 特征融合 + refine
            self.drivable_ppm = ASPPModule(shared_channels, shared_channels, dilations=(1, 3, 6))
            self.drivable_ppm_refine = nn.Sequential(
                nn.Conv2d(shared_channels, shared_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(shared_channels),
                nn.ReLU(inplace=True),
            )
            # ASPP + 空间注意力增强可行驶上下文
            self.drivable_aspp = ASPPModule(shared_channels, shared_channels, dilations=(1, 3, 6))
            self.drivable_sa = SpatialAttention(kernel_size=7)
            # 简化版 head（用于快速验证，避免复杂分支导致梯度消失）
            self.drivable_simple = nn.Sequential(
                nn.Conv2d(shared_channels, shared_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(shared_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_channels, shared_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(shared_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_channels, drivable_classes, kernel_size=1),
            )
            self.drivable_head = nn.Sequential(
                nn.Conv2d(shared_channels, shared_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(shared_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_channels, shared_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(shared_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_channels, drivable_classes, kernel_size=1))
            self.loss_drivable = _build_loss_module(loss_drivable)
            self.loss_drivable_dice = self._build_dice_loss(
                class_weight=getattr(self.loss_drivable, 'weight', None),
                loss_weight=1.0)
            # 额外浅层辅助头，鼓励早期特征学习 drivable，权重为主头的一半
            aux_cfg = loss_drivable.copy()
            aux_cfg['loss_weight'] = loss_drivable.get('loss_weight', 1.0) * 0.5
            self.drivable_aux_head = nn.Conv2d(shared_channels, drivable_classes, kernel_size=1)
            self.loss_drivable_aux = _build_loss_module(aux_cfg)
        else:
            self.drivable_head = None
            self.loss_drivable = None
            self.loss_drivable_dice = None
            self.drivable_adapter = None
            self.drivable_aux_head = None
            self.loss_drivable_aux = None
            self.drivable_shallow = None
            self.drivable_fuse = None
            self.drivable_refine = None
            self.drivable_aspp = None
            self.drivable_sa = None
        # 前景先验，避免初始全背景（logit prior）
        self.drivable_prior = 0.01

        if self.enable_marking:
            padding_v = (line_kernel // 2, 0)
            padding_h = (0, line_kernel // 2)
            self.marking_decoder = nn.Sequential(
                nn.Conv2d(shared_channels, shared_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(shared_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_channels, shared_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(shared_channels),
                nn.ReLU(inplace=True),
            )
            self.marking_vertical = nn.Sequential(
                nn.Conv2d(shared_channels, shared_channels, (line_kernel, 1),
                          padding=padding_v, bias=False),
                nn.BatchNorm2d(shared_channels),
                nn.ReLU(inplace=True),
            )
            self.marking_horizontal = nn.Sequential(
                nn.Conv2d(shared_channels, shared_channels, (1, line_kernel),
                          padding=padding_h, bias=False),
                nn.BatchNorm2d(shared_channels),
                nn.ReLU(inplace=True),
            )
            self.marking_adapter = nn.Sequential(
                nn.Conv2d(shared_channels, shared_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(shared_channels),
                nn.ReLU(inplace=True),
            )
            # 融合浅层 BEV 细节（来自原始 bev_feat）
            self.marking_shallow = nn.Sequential(
                nn.Conv2d(in_channels, shared_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(shared_channels),
                nn.ReLU(inplace=True),
            )
            self.marking_fuse = nn.Sequential(
                nn.Conv2d(shared_channels * 2, shared_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(shared_channels),
                nn.ReLU(inplace=True),
            )
            # 额外细化一层，增强纹理与线条
            self.marking_refine = nn.Sequential(
                nn.Conv2d(shared_channels, shared_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(shared_channels),
                nn.ReLU(inplace=True),
            )
            # 高分辨率解码：再做一次卷积后上采样
            self.marking_highres = nn.Sequential(
                nn.Conv2d(shared_channels, shared_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(shared_channels),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            )
            # 高分辨率分支再融合一次
            self.marking_highres_refine = nn.Sequential(
                nn.Conv2d(shared_channels, shared_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(shared_channels),
                nn.ReLU(inplace=True),
            )
            # 上采样后再细化一层
            self.marking_post = nn.Sequential(
                nn.Conv2d(shared_channels, shared_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(shared_channels),
                nn.ReLU(inplace=True),
            )
            self.marking_cls = nn.Conv2d(shared_channels, marking_classes, kernel_size=1)
            self.boundary_head = nn.Conv2d(shared_channels, 1, kernel_size=1)
            self.loss_marking = _build_loss_module(loss_marking)
            self.loss_boundary = _build_loss_module(
                loss_boundary, default_type='BCEWithLogitsLoss')
            self.loss_marking_dice = self._build_dice_loss(
                class_weight=getattr(self.loss_marking, 'weight', None),
                loss_weight=self.marking_dice_weight)
        else:
            self.marking_decoder = None
            self.marking_vertical = None
            self.marking_horizontal = None
            self.marking_fuse = None
            self.marking_shallow = None
            self.marking_highres = None
            self.marking_highres_refine = None
            self.marking_post = None
            self.marking_cls = None
            self.loss_marking = None
            self.loss_boundary = None
            self.loss_marking_dice = None
            self.marking_refine = None
            self.boundary_head = None
            self.marking_adapter = None

        if self.enable_slot:
            self.slot_decoder = nn.Sequential(
                nn.Conv2d(shared_channels, shared_channels, 3, padding=2, dilation=2, bias=False),
                nn.BatchNorm2d(shared_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_channels, shared_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(shared_channels),
                nn.ReLU(inplace=True),
            )
            self.slot_head = nn.Conv2d(shared_channels, slot_channels, kernel_size=1)
            self.loss_slot = _build_loss_module(
                loss_slot, default_type='BCEWithLogitsLoss')
        else:
            self.slot_decoder = None
            self.slot_head = None
            self.loss_slot = None

        if self.enable_obstacle:
            self.obstacle_decoder = nn.Sequential(
                nn.Conv2d(shared_channels, shared_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(shared_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_channels, shared_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(shared_channels),
                nn.ReLU(inplace=True),
            )
            self.obstacle_head = nn.Conv2d(shared_channels, obstacle_classes, kernel_size=1)
            self.loss_obstacle = _build_loss_module(loss_obstacle)
            self.loss_obstacle_dice = self._build_dice_loss(
                class_weight=getattr(self.loss_obstacle, 'weight', None),
                loss_weight=2.0)
        else:
            self.obstacle_decoder = None
            self.obstacle_head = None
            self.loss_obstacle = None
            self.loss_obstacle_dice = None

        if self.enable_occlusion:
            self.occlusion_head = nn.Sequential(
                nn.Conv2d(shared_channels, shared_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(shared_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_channels, 1, kernel_size=1))
            self.loss_occlusion = _build_loss_module(
                loss_occlusion, default_type='BCEWithLogitsLoss')
        else:
            self.occlusion_head = None
            self.loss_occlusion = None

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                kaiming_init(module)
        # 为 drivable 输出设置前景先验 bias，防止初始全背景
        if self.enable_drivable and self.drivable_head is not None:
            last = self.drivable_head[-1]
            if hasattr(last, 'bias') and last.bias is not None:
                # bias = log(p/(1-p))
                import math
                p = 0.5  # 拉高前景先验，避免预测坍塌到全背景
                last.bias.data.fill_(math.log(p / (1 - p)))

    def forward(self, bev_feat):
        """返回启用分支的预测字典."""
        if isinstance(bev_feat, (list, tuple)):
            bev_feat = bev_feat[0]
        # 上下文增强
        ctx_feats = [branch(bev_feat) for branch in self.context_branches]
        ctx = torch.cat(ctx_feats, dim=1)
        bev_feat = self.context_fuse(ctx)
        bev_feat = self.se_shared_in(bev_feat)
        bev_feat = self.sa_shared_in(bev_feat)
        # PPM 追加全局语义
        ppm_feats = [bev_feat]
        for conv in self.ppm_convs:
            ppm_feats.append(F.interpolate(conv(bev_feat), size=bev_feat.shape[-2:], mode='bilinear', align_corners=False))
        bev_feat = self.ppm_fuse(torch.cat(ppm_feats, dim=1))

        shared = self.shared(bev_feat)
        preds = {}
        if self.enable_drivable:
            if getattr(self, 'use_simple_drivable', False):
                preds['drivable'] = self.drivable_simple(shared)
            else:
                drivable_feat = shared
                if self.drivable_adapter is not None:
                    drivable_feat = self.drivable_adapter(drivable_feat)
                if self.drivable_shallow is not None:
                    shallow = self.drivable_shallow(bev_feat)
                    highres = self.drivable_highres(bev_feat) if hasattr(self, 'drivable_highres') else shallow
                    drivable_feat = self.drivable_fuse(torch.cat([drivable_feat, shallow, highres], dim=1))
                if self.drivable_refine is not None:
                    drivable_feat = self.drivable_refine(drivable_feat)
                if hasattr(self, 'drivable_big') and hasattr(self, 'drivable_small'):
                    drivable_feat = self.drivable_big(drivable_feat) + self.drivable_small(drivable_feat)
                if hasattr(self, 'drivable_ppm'):
                    drivable_feat = self.drivable_ppm(drivable_feat)
                    drivable_feat = self.drivable_ppm_refine(drivable_feat)
                if self.drivable_aspp is not None:
                    drivable_feat = self.drivable_aspp(drivable_feat)
                if self.drivable_sa is not None:
                    drivable_feat = self.drivable_sa(drivable_feat)
                preds['drivable'] = self.drivable_head(drivable_feat)
                preds['drivable_aux'] = self.drivable_aux_head(drivable_feat)
        if self.enable_marking:
            feat_v = self.marking_vertical(shared)
            feat_h = self.marking_horizontal(shared)
            marking_feat = self.marking_fuse(torch.cat([feat_v, feat_h], dim=1))
            # 浅层 BEV 细节加和
            if self.marking_shallow is not None:
                marking_feat = marking_feat + self.marking_shallow(bev_feat)
            if self.marking_refine is not None:
                marking_feat = self.marking_refine(marking_feat)
            marking_feat = self.marking_decoder(marking_feat)
            if self.marking_adapter is not None:
                marking_feat = self.marking_adapter(marking_feat)
            marking_feat = self.marking_highres(marking_feat)
            if self.marking_highres_refine is not None:
                marking_feat = self.marking_highres_refine(marking_feat)
            if self.marking_post is not None:
                marking_feat = self.marking_post(marking_feat)
            preds['marking'] = self.marking_cls(marking_feat)
            if getattr(self, 'boundary_head', None) is not None:
                preds['marking_boundary'] = self.boundary_head(marking_feat)
        if self.enable_slot:
            slot_feat = self.slot_decoder(shared)
            preds['slot'] = self.slot_head(slot_feat)
        if self.enable_obstacle:
            obstacle_feat = self.obstacle_decoder(shared)
            preds['obstacle'] = self.obstacle_head(obstacle_feat)
        if self.enable_occlusion:
            preds['occlusion'] = self.occlusion_head(shared)
        return preds

    def _sigmoid_class_loss(self,
                            pred,
                            target,
                            criterion,
                            class_weight,
                            ignore_index,
                            pos_topk_ratio=None,
                            neg_pos_ratio=None,
                            balance_pos_neg=False,
                            max_balance_factor=200.0):
        """多通道 sigmoid 损失，支持 ignore；可选 hard mining；可选自适应前景放大。"""
        # 强制形状为 N,C,H,W，通道维优先
        if pred.dim() == 3:  # (N,H,W) -> 单通道
            pred = pred.unsqueeze(1)
        if pred.dim() != 4:
            raise ValueError(f'pred shape 期望 4 维 (N,C,H,W)，当前 {pred.shape}')
        num_classes = class_weight.numel() if class_weight is not None else pred.size(1)
        channels_last = pred.shape[-1] == num_classes and pred.shape[1] != num_classes
        if channels_last:
            pred = pred.permute(0, 3, 1, 2).contiguous()
        if pred.size(1) != num_classes:
            raise ValueError(f'pred 通道数与类别数不符: {pred.size(1)} vs {num_classes}')
        # 兼容字符串 None/空值
        def _to_float(val, default):
            if val is None:
                return default
            if isinstance(val, str):
                if val.lower() == 'none':
                    return None
                try:
                    return float(val)
                except ValueError:
                    return default
            return float(val)

        pos_topk_ratio = _to_float(pos_topk_ratio if pos_topk_ratio is not None else getattr(self, 'pos_topk_ratio', None), None)
        neg_pos_ratio = _to_float(neg_pos_ratio if neg_pos_ratio is not None else getattr(self, 'neg_pos_ratio', None), None)
        valid = target != ignore_index
        if not valid.any():
            return pred.sum() * 0
        one_hot = F.one_hot(torch.clamp(target, min=0), num_classes=num_classes)  # (N,H,W,C)
        one_hot = one_hot.permute(0, 3, 1, 2).float()
        # 对齐尺寸（仅调整 one_hot/valid 到 pred 的空间尺寸，不再交换 pred 维度）
        if one_hot.shape[-2:] != pred.shape[-2:]:
            one_hot = F.interpolate(one_hot, size=pred.shape[-2:], mode='nearest')
            valid = F.interpolate(valid.unsqueeze(1).float(),
                                  size=pred.shape[-2:], mode='nearest').bool().squeeze(1)
        # 确保形状完全一致
        if one_hot.shape != pred.shape:
            raise ValueError(
                f'pred/target 形状不匹配: pred={pred.shape}, target(one_hot)={one_hot.shape}, '
                f'raw_target={target.shape}')
        loss_raw = criterion(pred, one_hot)
        # BCE/SigmoidFocal 输出同 pred 形状
        if loss_raw.shape != pred.shape:
            loss_raw = loss_raw.view_as(pred)
        if class_weight is not None:
            cw = class_weight.to(loss_raw.device).view(1, -1, 1, 1)
            loss_raw = loss_raw * cw
        loss_raw = loss_raw * valid.unsqueeze(1)

        pos_mask = one_hot.bool() & valid.unsqueeze(1)
        neg_mask = (~one_hot.bool()) & valid.unsqueeze(1)
        if balance_pos_neg:
            pos_count = pos_mask.sum().clamp(min=1)
            balance = (valid.sum() / pos_count).clamp(max=max_balance_factor)
            loss_raw = loss_raw.clone()
            loss_raw[pos_mask] = loss_raw[pos_mask] * balance

        if pos_topk_ratio is None or neg_pos_ratio is None:
            denom = valid.sum().clamp(min=1)
            return loss_raw.sum() / denom

        pos_losses = loss_raw[pos_mask]
        neg_losses = loss_raw[neg_mask]

        pos_keep = max(1, int(pos_losses.numel() * pos_topk_ratio)) if pos_losses.numel() > 0 else 0
        if pos_keep > 0:
            pos_topk = torch.topk(pos_losses, k=pos_keep, largest=True).values
            pos_loss = pos_topk.sum()
        else:
            pos_loss = loss_raw.new_tensor(0.0)

        neg_keep = 0
        if pos_keep > 0 and neg_losses.numel() > 0:
            neg_keep = min(int(pos_keep * neg_pos_ratio), neg_losses.numel())
            neg_topk = torch.topk(neg_losses, k=neg_keep, largest=True).values
            neg_loss = neg_topk.sum()
        else:
            neg_loss = loss_raw.new_tensor(0.0)

        denom = max(pos_keep + neg_keep, 1)
        return (pos_loss + neg_loss) / denom

    def loss(self, preds, targets):
        """多头联合损失."""
        losses = {}
        if self.loss_drivable is not None and 'gt_drivable_mask' in targets:
            gt = self._prepare_class_target(targets['gt_drivable_mask'])
            gt = self._dilate_marking(gt, kernel=self.drivable_dilate_kernel, ignore_index=self.drivable_ignore)
            # 如果是 CE，走 softmax 分支避免形状不匹配
            if hasattr(self.loss_drivable, 'criterion') and isinstance(self.loss_drivable.criterion, nn.CrossEntropyLoss):
                gt_ce = self._resize_class_target(gt, preds['drivable'])
                losses['loss_drivable'] = self._cross_entropy_class_loss(
                    preds['drivable'], gt_ce, self.loss_drivable,
                    self.drivable_class_weight, self.drivable_ignore)
                if getattr(self, 'loss_drivable_aux', None) is not None and 'drivable_aux' in preds:
                    gt_aux = self._resize_class_target(gt, preds['drivable_aux'])
                    losses['loss_drivable_aux'] = self._cross_entropy_class_loss(
                        preds['drivable_aux'], gt_aux, self.loss_drivable_aux,
                        self.drivable_class_weight, self.drivable_ignore)
            else:
                losses['loss_drivable'] = self._sigmoid_class_loss(
                    preds['drivable'], gt, self.loss_drivable,
                    self.drivable_class_weight, self.drivable_ignore,
                    self.pos_topk_ratio, self.neg_pos_ratio,
                    balance_pos_neg=self.balance_drivable,
                    max_balance_factor=self.max_balance_factor)
                if getattr(self, 'loss_drivable_aux', None) is not None and 'drivable_aux' in preds:
                    losses['loss_drivable_aux'] = self._sigmoid_class_loss(
                        preds['drivable_aux'], gt, self.loss_drivable_aux,
                        self.drivable_class_weight, self.drivable_ignore,
                        self.pos_topk_ratio, self.neg_pos_ratio,
                        balance_pos_neg=self.balance_drivable,
                        max_balance_factor=self.max_balance_factor)
        if getattr(self, 'loss_drivable_dice', None) is not None and 'gt_drivable_mask' in targets:
            gt = self._prepare_class_target(targets['gt_drivable_mask'])
            gt = self._resize_class_target(gt, preds['drivable'])
            losses['loss_drivable_dice'] = self.loss_drivable_dice(preds['drivable'], gt)
        if self.loss_marking is not None and 'gt_marking_mask' in targets:
            gt_mark = self._prepare_class_target(targets['gt_marking_mask'])
            gt_mark = self._dilate_marking(gt_mark, kernel=self.marking_dilate_kernel)
            # 如果是 CrossEntropyLoss 走 softmax 分支，避免 one-hot 尺寸错配
            if hasattr(self.loss_marking, 'criterion') and isinstance(
                    self.loss_marking.criterion, nn.CrossEntropyLoss):
                gt_mark = self._resize_class_target(gt_mark, preds['marking'])
                losses['loss_marking'] = self._cross_entropy_class_loss(
                    preds['marking'], gt_mark, self.loss_marking,
                    self.marking_class_weight, self.marking_ignore)
            else:
                losses['loss_marking'] = self._sigmoid_class_loss(
                    preds['marking'], gt_mark, self.loss_marking,
                    self.marking_class_weight, self.marking_ignore,
                    self.pos_topk_ratio, self.neg_pos_ratio,
                    balance_pos_neg=self.balance_marking,
                    max_balance_factor=self.max_balance_factor)
        if getattr(self, 'loss_marking_dice', None) is not None and 'gt_marking_mask' in targets:
            gt_mark = self._prepare_class_target(targets['gt_marking_mask'])
            gt_mark = self._dilate_marking(gt_mark, kernel=self.marking_dilate_kernel)
            gt_mark = self._resize_class_target(gt_mark, preds['marking'])
            losses['loss_marking_dice'] = self.loss_marking_dice(preds['marking'], gt_mark)
        if self.loss_boundary is not None and 'gt_marking_boundary' in targets and 'marking_boundary' in preds:
            boundary = targets['gt_marking_boundary']
            boundary = self._resize_binary_target(boundary, preds['marking_boundary'])
            losses['loss_marking_boundary'] = self.loss_boundary(
                preds['marking_boundary'], boundary)
        if self.loss_slot is not None and 'gt_slot_masks' in targets:
            slot = self._prepare_slot_target(targets['gt_slot_masks'])
            slot = self._resize_binary_target(slot, preds['slot'])
            losses['loss_slot'] = self.loss_slot(preds['slot'], slot)
        if self.loss_obstacle is not None and 'gt_obstacle_mask' in targets:
            obstacle = self._prepare_class_target(targets['gt_obstacle_mask'])
            losses['loss_obstacle'] = self._sigmoid_class_loss(
                preds['obstacle'], obstacle, self.loss_obstacle,
                self.obstacle_class_weight, self.obstacle_ignore,
                self.pos_topk_ratio, self.neg_pos_ratio,
                balance_pos_neg=self.balance_obstacle,
                max_balance_factor=self.max_balance_factor)
        if getattr(self, 'loss_obstacle_dice', None) is not None and 'gt_obstacle_mask' in targets:
            obstacle = self._prepare_class_target(targets['gt_obstacle_mask'])
            obstacle = self._resize_class_target(obstacle, preds['obstacle'])
            losses['loss_obstacle_dice'] = self.loss_obstacle_dice(preds['obstacle'], obstacle)
        if self.loss_occlusion is not None and 'gt_occlusion_mask' in targets:
            occlusion = self._prepare_binary_target(targets['gt_occlusion_mask'])
            occlusion = self._resize_binary_target(occlusion, preds['occlusion'])
            losses['loss_occlusion'] = self.loss_occlusion(preds['occlusion'], occlusion)
        return losses

    @staticmethod
    def _prepare_class_target(target):
        if target.dim() == 4 and target.size(1) == 1:
            target = target[:, 0]
        return target.long()

    @staticmethod
    def _prepare_binary_target(target):
        if target.dim() == 3:
            target = target.unsqueeze(1)
        if target.dtype != torch.float32:
            target = target.float()
        return target

    @staticmethod
    def _prepare_slot_target(target):
        if target.dim() == 3:
            target = target.unsqueeze(1)
        if target.dtype != torch.float32:
            target = target.float()
        return target

    def _build_dice_loss(self, class_weight=None, loss_weight=1.0):
        if class_weight is not None:
            class_weight = torch.tensor(class_weight, dtype=torch.float32)

        def dice(pred, target):
            return loss_weight * _dice_loss(
                pred, target, ignore_index=255, class_weight=class_weight)

        return dice

    @staticmethod
    def _resize_class_target(target, pred):
        """将 int mask 按预测的空间尺寸对齐（最近邻）。"""
        if target.shape[-2:] == pred.shape[-2:]:
            return target
        # target: (N,H,W) -> (N,1,H,W) for interpolate
        t = target.unsqueeze(1).float()
        t = F.interpolate(t, size=pred.shape[-2:], mode='nearest')
        return t[:, 0].long()

    @staticmethod
    def _resize_binary_target(target, pred):
        """将二值/多通道 float mask 对齐到预测尺寸（最近邻）。"""
        if target.shape[-2:] == pred.shape[-2:]:
            return target
        t = target
        if t.dim() == 3:  # (N,H,W)
            t = t.unsqueeze(1)
        t = F.interpolate(t, size=pred.shape[-2:], mode='nearest')
        return t

    @staticmethod
    def _cross_entropy_class_loss(pred, target, criterion, class_weight, ignore_index):
        """普通 CE，多通道 softmax，用于避免 one-hot 维度错配。"""
        if pred.dim() == 4 and target.dim() == 3:
            pass
        elif pred.dim() == 4 and target.dim() == 4 and target.size(1) == 1:
            target = target[:, 0]
        else:
            raise ValueError(f'CE 期望 pred (N,C,H,W) / target (N,H,W)，得到 pred={pred.shape}, target={target.shape}')
        if target.shape[-2:] != pred.shape[-2:]:
            target = F.interpolate(target.unsqueeze(1).float(),
                                   size=pred.shape[-2:],
                                   mode='nearest')[:, 0].long()
        if class_weight is not None:
            cw = class_weight.to(pred.device)
            criterion.criterion.weight = cw
        return criterion(pred, target)

    @staticmethod
    def _dilate_marking(target, kernel=3, ignore_index=255):
        """对标线前景做轻微膨胀，缓解极端稀疏。仅支持二分类：0 背景 / 1 前景。"""
        if isinstance(kernel, str):
            try:
                kernel = int(kernel)
            except Exception:
                kernel = None
        if kernel is None or kernel <= 1:
            return target
        if target.dim() != 3:
            return target
        fg = (target == 1).float()
        fg = F.max_pool2d(fg, kernel_size=kernel, stride=1, padding=kernel // 2)
        dilated = (fg > 0).long()
        out = target.clone()
        valid = target != ignore_index
        out[valid] = dilated[valid]
        return out
