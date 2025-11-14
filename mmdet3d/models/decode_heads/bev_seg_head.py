# -*- coding: utf-8 -*-
"""WoodScape BEV 语义分割头实现."""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init
from mmcv.runner import BaseModule

from mmdet.models import build_loss
from mmdet.models.builder import HEADS
from mmseg.models.builder import HEADS as MMSEG_HEADS


@HEADS.register_module()
@MMSEG_HEADS.register_module()
class WoodscapeBEVSegHead(BaseModule):
    """一个轻量级的 BEV 语义分割头。

    设计目标：
    1. 输入为 Fast-BEV 生成的 BEV 特征图（N, C, H, W）。
    2. 通过若干卷积层提取上下文后输出语义概率图。
    3. 提供 ``losses`` 函数，返回标准交叉熵损失，便于多任务联合训练。
    """

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 mid_channels: int = 128,
                 num_convs: int = 2,
                 loss_seg: Dict = dict(
                     type='CrossEntropyLoss', loss_weight=1.0),
                 loss_aux: Dict = None,
                 loss_aux_class_weight=None,
                 binary_aux_losses=None,
                 refine_head: bool = False,
                 pre_upsample: Dict = None):
        super().__init__()

        assert num_convs >= 1, '至少需要一个卷积层'

        layers = []  # 保存卷积模块
        channels = in_channels  # 记录当前通道数
        for _ in range(num_convs - 1):  # 逐层堆叠卷积
            layers.append(nn.Conv2d(channels, mid_channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(mid_channels))
            layers.append(nn.ReLU(inplace=True))
            channels = mid_channels

        layers.append(nn.Conv2d(channels, mid_channels, kernel_size=3, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(mid_channels))
        layers.append(nn.ReLU(inplace=True))

        self.encoder = nn.Sequential(*layers)
        self.pre_upsample = pre_upsample
        self.refine = None
        if refine_head:
            self.refine = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True))
        self.cls_layer = nn.Conv2d(mid_channels, num_classes, kernel_size=1)

        self.loss_seg = self._build_loss(loss_seg)
        self.loss_aux = self._build_loss(
            loss_aux, class_weight_override=loss_aux_class_weight) if loss_aux is not None else None

        self.binary_aux = []
        if binary_aux_losses:
            for aux_cfg in binary_aux_losses:
                cfg = aux_cfg.copy()
                class_idx = cfg.pop('class_index', None)
                assert class_idx is not None and 0 <= class_idx < num_classes, \
                    'binary_aux_losses 需要合法的 class_index'
                ignore_idx = cfg.pop('ignore_index', 255)
                loss_name = cfg.pop('loss_name', f'class_{class_idx}')
                loss_module = _BinarySegLoss(**cfg)
                self.binary_aux.append(
                    dict(class_index=class_idx,
                         ignore_index=ignore_idx,
                         loss_name=loss_name,
                         loss=loss_module))

        class_weight = getattr(self.loss_seg, 'class_weight', None)
        if class_weight is not None:
            base_weight = torch.as_tensor(class_weight, dtype=torch.float32)
            self.register_buffer('loss_seg_base_class_weight', base_weight.clone())
            self.loss_seg.class_weight = base_weight.clone()
        else:
            self.loss_seg_base_class_weight = None

        self.num_classes = num_classes
        self._init_weights()

    def _init_weights(self):
        for m in self.encoder.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
        kaiming_init(self.cls_layer)

    def forward(self, bev_feat: torch.Tensor) -> torch.Tensor:
        if isinstance(bev_feat, (list, tuple)):
            bev_feat = bev_feat[0]
        x = self.encoder(bev_feat)
        if self.pre_upsample is not None:
            x = F.interpolate(x, **self.pre_upsample)
        if self.refine is not None:
            x = self.refine(x)
        x = self.cls_layer(x)
        return x

    def losses(self, seg_pred: torch.Tensor, seg_gt: torch.Tensor) -> Dict[str, torch.Tensor]:
        if seg_pred.shape[-2:] != seg_gt.shape[-2:]:
            seg_pred = F.interpolate(
                seg_pred, size=seg_gt.shape[-2:], mode='bilinear', align_corners=False)
        if seg_gt.ndim == 4:
            seg_gt = seg_gt.squeeze(1)
        loss = self.loss_seg(seg_pred, seg_gt)
        losses = {'loss_bev_seg': loss}
        if self.loss_aux is not None:
            aux_loss = self.loss_aux(seg_pred, seg_gt)
            losses['loss_bev_seg_aux'] = aux_loss
        for aux in self.binary_aux:
            aux_loss = self._compute_binary_aux_loss(seg_pred, seg_gt, aux)
            losses[f"loss_{aux['loss_name']}"] = aux_loss
        return losses

    def _compute_binary_aux_loss(self, seg_pred, seg_gt, aux_cfg):
        logits = seg_pred[:, aux_cfg['class_index'], ...]
        target = (seg_gt == aux_cfg['class_index']).float()
        valid_mask = (seg_gt != aux_cfg['ignore_index'])
        if not valid_mask.any():
            return logits.new_tensor(0.0)
        logits = logits[valid_mask]
        target = target[valid_mask]
        return aux_cfg['loss'](logits, target)

    def _build_loss(self, loss_cfg: Dict, class_weight_override=None):
        if loss_cfg is None:
            return None
        cfg = loss_cfg.copy()
        loss_type = cfg.pop('type', 'CrossEntropyLoss')
        if loss_type == 'DiceLoss':
            class_weight = cfg.pop('class_weight', None)
            if class_weight_override is not None:
                class_weight = class_weight_override
            return _DiceLoss(class_weight=class_weight, **cfg)
        cfg['type'] = loss_type
        return build_loss(cfg)

    def set_class_weight_scale(self, scale: float):
        base_weight = getattr(self, 'loss_seg_base_class_weight', None)
        if base_weight is not None:
            device = self.cls_layer.weight.device
            scaled = base_weight.to(device) * scale
            self.loss_seg.class_weight = scaled
        if self.loss_aux is not None and hasattr(self.loss_aux, 'set_class_weight_scale'):
            self.loss_aux.set_class_weight_scale(scale)
        for aux in self.binary_aux:
            if hasattr(aux['loss'], 'set_class_weight_scale'):
                aux['loss'].set_class_weight_scale(scale)


class _DiceLoss(nn.Module):
    """简易多类 DiceLoss，实现于本地以避免额外依赖."""

    def __init__(self,
                 smooth=1e-5,
                 ignore_index=255,
                 reduction='mean',
                 loss_weight=1.0,
                 class_weight=None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.loss_weight = loss_weight
        if class_weight is not None:
            weight = torch.tensor(class_weight, dtype=torch.float32)
            self.register_buffer('base_class_weight', weight.clone())
            self.register_buffer('class_weight', weight.clone())
        else:
            self.base_class_weight = None
            self.class_weight = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 4:
            target = target.squeeze(1)
        num_classes = pred.shape[1]
        pred_prob = F.softmax(pred, dim=1)
        target = target.long()
        valid_mask = (target != self.ignore_index)
        mask = valid_mask.unsqueeze(1).float()
        target = target.clone()
        target[~valid_mask] = 0
        target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        target_onehot = target_onehot * mask
        pred_prob = pred_prob * mask

        intersection = (pred_prob * target_onehot).sum(dim=(0, 2, 3))
        pred_sum = pred_prob.sum(dim=(0, 2, 3))
        target_sum = target_onehot.sum(dim=(0, 2, 3))
        dice = (2 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        loss = 1 - dice
        if self.class_weight is not None:
            loss = loss * self.class_weight.to(loss.device)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            pass
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}')
        return loss * self.loss_weight

    def set_class_weight_scale(self, scale: float):
        base_weight = getattr(self, 'base_class_weight', None)
        class_weight = getattr(self, 'class_weight', None)
        if base_weight is not None and class_weight is not None:
            scaled = base_weight * scale
            class_weight.copy_(scaled)


class _BinarySegLoss(nn.Module):
    """针对单类别的 BCEWithLogitsLoss，可配合权重缩放."""

    def __init__(self, loss_weight=1.0, pos_weight=None):
        super().__init__()
        self.loss_weight = loss_weight
        if pos_weight is not None:
            self.register_buffer('pos_weight', torch.tensor(pos_weight, dtype=torch.float32))
        else:
            self.pos_weight = None

    def forward(self, logits, target):
        if logits.numel() == 0:
            return logits.new_tensor(0.0)
        pos_weight = self.pos_weight
        if pos_weight is not None:
            pos_weight = pos_weight.to(logits.device)
        loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
        return loss * self.loss_weight

    def set_class_weight_scale(self, scale: float):
        if self.pos_weight is not None:
            self.pos_weight.mul_(scale)
