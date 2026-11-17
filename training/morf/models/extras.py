"""Optional extensions to the global-summary architecture.

MultiScaleSelect combines depthwise-convolution windows of 5, 11, 21 and 41
residues. MultiScaleLocalAttention uses four local attention radii. TerminalPosition
encodes distance to sequence ends, and GatedFusion combines language-model
features at each residue. Zero-initialized output layers preserve the base
architecture at initialization. These extensions are disabled in the final
XMoRF configurations."""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _zero_(lin: nn.Linear) -> nn.Linear:
    nn.init.zeros_(lin.weight)
    if lin.bias is not None:
        nn.init.zeros_(lin.bias)
    return lin


class MultiScaleSelect(nn.Module):
    """h:[B,L,d]、mask:[B,L] → [B,L,d]（padding 处为 0）。"""

    def __init__(self, d: int, windows: Sequence[int] = (5, 11, 21, 41)):
        super().__init__()
        for w in windows:
            assert w % 2 == 1, f"窗口长度必须是奇数，收到 {w}"
        self.windows = list(windows)
        self.norm = nn.LayerNorm(d)
        self.convs = nn.ModuleList([nn.Conv1d(d, d, w, padding=w // 2, groups=d) for w in self.windows])
        self.att = nn.Linear(d, len(self.windows))
        self.out = nn.Sequential(nn.LayerNorm(d), _zero_(nn.Linear(d, d)))

    def weights(self, h: torch.Tensor) -> torch.Tensor:
        """每个残基在各个窗口上的权重 [B, L, S]（和为 1）。"""
        return torch.softmax(self.att(self.norm(h)), -1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask[..., None].to(h.dtype)
        x = (self.norm(h) * m).transpose(1, 2)                                  # [B, d, L]
        us = torch.stack([c(x) for c in self.convs], -1).transpose(1, 2)       # [B, L, d, S]
        a = self.weights(h)                                                     # [B, L, S]
        u = (us * a[:, :, None, :]).sum(-1)
        return self.out(u) * m


class MultiScaleLocalAttention(nn.Module):
    """h:[B,L,d]、mask:[B,L] → [B,L,d]（padding 处为 0）。第 k 个头只看 |i − j| ≤ radii[k] 的邻居（同一条蛋白里）。"""

    def __init__(self, d: int, radii: Sequence[int] = (2, 5, 10, 20)):
        super().__init__()
        self.radii = list(radii)
        self.H = len(self.radii)
        assert d % self.H == 0, f"d = {d} 要能被头数 {self.H} 整除"
        self.dh = d // self.H
        self.R = max(self.radii)
        self.norm = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = _zero_(nn.Linear(d, d))
        self.rel = nn.Parameter(torch.zeros(self.H, 2 * self.R + 1))            # 相对位置偏置
        off = torch.arange(-self.R, self.R + 1).abs()
        self.register_buffer("allow", torch.stack([off <= r for r in self.radii]), persistent=False)   # [H, W]

    def _windows(self, x: torch.Tensor) -> torch.Tensor:
        """x:[B,L,...] → [B,L,W,...]：每个位置 i 取 i − R … i + R（两端补 0）。"""
        B, L = x.shape[:2]
        rest = x.shape[2:]
        xp = F.pad(x.reshape(B, L, -1), (0, 0, self.R, self.R))                # [B, L+2R, C]
        w = xp.unfold(1, 2 * self.R + 1, 1)                                     # [B, L, C, W]
        return w.permute(0, 1, 3, 2).reshape(B, L, 2 * self.R + 1, *rest)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, L, d = h.shape
        m = mask.to(h.dtype)
        x = self.norm(h) * m[..., None]
        q, k, v = self.qkv(x).chunk(3, -1)
        q = q.view(B, L, self.H, self.dh)
        kw = self._windows(k.view(B, L, self.H, self.dh))                       # [B, L, W, H, dh]
        vw = self._windows(v.view(B, L, self.H, self.dh))
        valid = self._windows(m[..., None])[..., 0] > 0                          # [B, L, W]：邻居在蛋白里
        s = torch.einsum("blhd,blwhd->blhw", q, kw) / math.sqrt(self.dh) + self.rel[None, None]
        ok = valid[:, :, None, :] & self.allow[None, None]                       # [B, L, H, W]
        s = s.masked_fill(~ok, float("-inf"))
        a = torch.softmax(s, -1)
        a = torch.nan_to_num(a, nan=0.0)                                         # padding 处整行都不可见
        out = torch.einsum("blhw,blwhd->blhd", a, vw).reshape(B, L, d)
        return self.o(out) * m[..., None]


class TerminalPosition(nn.Module):
    """mask:[B,L] → [B,L,d]：每个残基的 5 个位置特征经两层小网络（最后一层初值为 0）。"""

    N_FEAT = 5

    def __init__(self, d: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(self.N_FEAT, hidden), nn.GELU(), _zero_(nn.Linear(hidden, d)))

    @staticmethod
    def features(mask: torch.Tensor) -> torch.Tensor:
        """[相对位置, log(1 + 到 N 端)/log(1001), log(1 + 到 C 端)/log(1001), exp(−到 N 端/10), exp(−到 C 端/10)]，padding 处为 0。"""
        m = mask.float()
        L = m.sum(1, keepdim=True).clamp(min=1)
        i = (torch.cumsum(m, 1) - 1).clamp(min=0)                                # 第几个残基（从 0 数）
        j = (L - 1 - i).clamp(min=0)                                            # 到 C 端还有几个
        scale = math.log(1001.0)
        f = torch.stack([(i + 0.5) / L, torch.log1p(i) / scale, torch.log1p(j) / scale,
                         torch.exp(-i / 10.0), torch.exp(-j / 10.0)], -1)
        return f * m[..., None]

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        f = self.features(mask).to(h.dtype)
        return self.net(f) * mask[..., None].to(h.dtype)


class GatedFusion(nn.Module):
    """zs：每个语言模型投影后的 [B,L,d]（M 个）→ [B,L,d]：逐残基 softmax 权重加权求和，再经线性层（初值为 0）。"""

    def __init__(self, d: int, n_models: int):
        super().__init__()
        self.gate = nn.Linear(d * n_models, n_models)
        self.out = _zero_(nn.Linear(d, d))

    def weights(self, zs) -> torch.Tensor:
        return torch.softmax(self.gate(torch.cat(zs, -1)), -1)                 # [B, L, M]

    def forward(self, zs) -> torch.Tensor:
        a = self.weights(zs)
        v = sum(a[..., i:i + 1] * z for i, z in enumerate(zs))
        return self.out(v)
