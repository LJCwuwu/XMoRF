"""带 mask 的损失：标签 -1 的位置（padding / 未标注）不参与。"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_bce(logits: torch.Tensor, labels: torch.Tensor, pos_weight: float) -> torch.Tensor:
    m = labels >= 0
    y = labels[m].float()
    return F.binary_cross_entropy_with_logits(
        logits[m].float(), y, pos_weight=torch.tensor(pos_weight, device=logits.device))


def masked_focal(logits: torch.Tensor, labels: torch.Tensor, pos_weight: float,
                 gamma: float = 2.0) -> torch.Tensor:
    m = labels >= 0
    x, y = logits[m].float(), labels[m].float()
    p = torch.sigmoid(x)
    pt = torch.where(y == 1, p, 1 - p)
    w = torch.where(y == 1, torch.full_like(y, pos_weight), torch.ones_like(y))
    ce = F.binary_cross_entropy_with_logits(x, y, reduction="none")
    return (w * (1 - pt).pow(gamma) * ce).mean()
