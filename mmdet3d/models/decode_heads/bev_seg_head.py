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
                     type='CrossEntropyLoss', loss_weight=1.0)):
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
        self.cls_layer = nn.Conv2d(mid_channels, num_classes, kernel_size=1)

        self.loss_seg = build_loss(loss_seg)

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
        x = self.cls_layer(x)
        return x

    def losses(self, seg_pred: torch.Tensor, seg_gt: torch.Tensor) -> Dict[str, torch.Tensor]:
        if seg_pred.shape[-2:] != seg_gt.shape[-2:]:
            seg_pred = F.interpolate(seg_pred, size=seg_gt.shape[-2:], mode='bilinear', align_corners=False)
        loss = self.loss_seg(seg_pred, seg_gt.squeeze(1)) if seg_gt.ndim == 4 else self.loss_seg(seg_pred, seg_gt)
        return {'loss_bev_seg': loss}
