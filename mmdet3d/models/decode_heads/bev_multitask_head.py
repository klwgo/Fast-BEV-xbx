# -*- coding: utf-8 -*-
"""Fast-BEV 多任务 Head：Drivable / Marking / Slot / Obstacle / Occlusion."""

import torch
import torch.nn as nn

from mmcv.cnn import kaiming_init
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


def _build_loss_module(cfg, default_type=None):
    """支持 mmcv 配置或 PyTorch 原生损失."""
    if cfg is None:
        return None
    loss_cfg = cfg.copy()
    loss_type = loss_cfg.pop('type', default_type)
    if loss_type in (None,):
        return None
    loss_weight = loss_cfg.pop('loss_weight', 1.0)
    if loss_type == 'BCEWithLogitsLoss':
        return _WeightedLoss(nn.BCEWithLogitsLoss(**loss_cfg), loss_weight)
    if loss_type == 'MSELoss':
        return _WeightedLoss(nn.MSELoss(**loss_cfg), loss_weight)
    if loss_type == 'CrossEntropyLoss':
        return _WeightedLoss(nn.CrossEntropyLoss(**loss_cfg), loss_weight)
    loss_cfg['type'] = loss_type
    criterion = build_loss(loss_cfg)
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
                     type='CrossEntropyLoss', ignore_index=255, loss_weight=1.0),
                 loss_marking=dict(
                     type='CrossEntropyLoss', ignore_index=255, loss_weight=1.0),
                 loss_boundary=dict(
                     type='BCEWithLogitsLoss', reduction='mean', loss_weight=1.0),
                 loss_slot=dict(
                     type='BCEWithLogitsLoss', reduction='mean', loss_weight=1.0),
                 loss_obstacle=dict(
                     type='CrossEntropyLoss', ignore_index=255, loss_weight=1.0),
                 loss_occlusion=dict(
                     type='BCEWithLogitsLoss', reduction='mean', loss_weight=1.0)):
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

        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, shared_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(shared_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_channels, shared_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(shared_channels),
            nn.ReLU(inplace=True),
        )

        if self.enable_drivable:
            self.drivable_head = nn.Sequential(
                nn.Conv2d(shared_channels, shared_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(shared_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_channels, drivable_classes, kernel_size=1))
            self.loss_drivable = _build_loss_module(loss_drivable)
        else:
            self.drivable_head = None
            self.loss_drivable = None

        if self.enable_marking:
            padding_v = (line_kernel // 2, 0)
            padding_h = (0, line_kernel // 2)
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
            self.marking_fuse = nn.Sequential(
                nn.Conv2d(shared_channels * 2, shared_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(shared_channels),
                nn.ReLU(inplace=True),
            )
            self.marking_cls = nn.Conv2d(shared_channels, marking_classes, kernel_size=1)
            self.boundary_head = nn.Conv2d(shared_channels, 1, kernel_size=1)
            self.loss_marking = _build_loss_module(loss_marking)
            self.loss_boundary = _build_loss_module(
                loss_boundary, default_type='BCEWithLogitsLoss')
        else:
            self.marking_vertical = None
            self.marking_horizontal = None
            self.marking_fuse = None
            self.marking_cls = None
            self.boundary_head = None
            self.loss_marking = None
            self.loss_boundary = None

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
            self.obstacle_head = nn.Sequential(
                nn.Conv2d(shared_channels, shared_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(shared_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_channels, obstacle_classes, kernel_size=1))
            self.loss_obstacle = _build_loss_module(loss_obstacle)
        else:
            self.obstacle_head = None
            self.loss_obstacle = None

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

    def forward(self, bev_feat):
        """返回启用分支的预测字典."""
        if isinstance(bev_feat, (list, tuple)):
            bev_feat = bev_feat[0]
        shared = self.shared(bev_feat)
        preds = {}
        if self.enable_drivable:
            preds['drivable'] = self.drivable_head(shared)
        if self.enable_marking:
            feat_v = self.marking_vertical(shared)
            feat_h = self.marking_horizontal(shared)
            marking_feat = self.marking_fuse(torch.cat([feat_v, feat_h], dim=1))
            preds['marking'] = self.marking_cls(marking_feat)
            preds['marking_boundary'] = self.boundary_head(marking_feat)
        if self.enable_slot:
            slot_feat = self.slot_decoder(shared)
            preds['slot'] = self.slot_head(slot_feat)
        if self.enable_obstacle:
            preds['obstacle'] = self.obstacle_head(shared)
        if self.enable_occlusion:
            preds['occlusion'] = self.occlusion_head(shared)
        return preds

    def loss(self, preds, targets):
        """多头联合损失."""
        losses = {}
        if self.loss_drivable is not None and 'gt_drivable_mask' in targets:
            gt = self._prepare_class_target(targets['gt_drivable_mask'])
            losses['loss_drivable'] = self.loss_drivable(preds['drivable'], gt)
        if self.loss_marking is not None and 'gt_marking_mask' in targets:
            gt_mark = self._prepare_class_target(targets['gt_marking_mask'])
            losses['loss_marking'] = self.loss_marking(preds['marking'], gt_mark)
        if self.loss_boundary is not None and 'gt_marking_boundary' in targets:
            boundary = self._prepare_binary_target(targets['gt_marking_boundary'])
            losses['loss_marking_boundary'] = self.loss_boundary(
                preds['marking_boundary'], boundary)
        if self.loss_slot is not None and 'gt_slot_masks' in targets:
            slot = self._prepare_slot_target(targets['gt_slot_masks'])
            losses['loss_slot'] = self.loss_slot(preds['slot'], slot)
        if self.loss_obstacle is not None and 'gt_obstacle_mask' in targets:
            obstacle = self._prepare_class_target(targets['gt_obstacle_mask'])
            losses['loss_obstacle'] = self.loss_obstacle(preds['obstacle'], obstacle)
        if self.loss_occlusion is not None and 'gt_occlusion_mask' in targets:
            occlusion = self._prepare_binary_target(targets['gt_occlusion_mask'])
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
