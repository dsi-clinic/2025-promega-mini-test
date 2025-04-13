import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmengine.registry import MODELS

@MODELS.register_module()
class BCELoss(BaseModule):
    def __init__(self, loss_weight=1.0, use_sigmoid=True, reduction='mean', loss_name='loss_bce'):
        super().__init__()
        self.loss_weight = loss_weight
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_name = loss_name
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction) if use_sigmoid else nn.BCELoss(reduction=reduction)

    def forward(self, pred, target, **kwargs):
        # Assumes pred shape is (N, C, H, W) and target is (N, H, W)
        if pred.shape != target.shape:
            target = target.unsqueeze(1).float()
        loss = self.bce(pred, target)
        return self.loss_weight * loss
