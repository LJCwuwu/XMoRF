"""Spectral summaries with learnable frequency gates.

LocalSpectrum evaluates windowed sinusoidal projections over several window
sizes; GlobalSpectrum summarizes the full sequence. Projection magnitudes are
normalized and gated before being passed to the prediction network. Frequency
grids and window sizes are configurable. The final XMoRF model uses global mode
with spec_freq_step=360, leaving only the zero-frequency component."""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _taper(W: int, kind: str) -> torch.Tensor:
    if kind == "rect":
        return torch.ones(W)
    # 两端非零的 Hann 窗（避免端点权重为 0 浪费窗口长度）
    return torch.hann_window(W + 2, periodic=False)[1:-1]


def _theta_grid(step_deg: float) -> torch.Tensor:
    n = int(round(180.0 / step_deg)) + 1
    return torch.linspace(0.0, math.pi, n)


class LocalSpectralMoment(nn.Module):
    """输入 x:[B,K,L]（padding 处已为 0），mask:[B,L] → 输出 [B, R*K*F, L]。"""

    def __init__(self, n_channels: int, windows: List[int], freq_step_deg: float = 10.0,
                 taper: str = "hann", learn_freq: bool = False):
        super().__init__()
        for W in windows:
            assert W % 2 == 1, f"窗口长度必须是奇数（居中对齐），收到 {W}"
        self.K, self.windows = n_channels, list(windows)
        theta = _theta_grid(freq_step_deg)
        if learn_freq:
            self.theta = nn.Parameter(theta)
        else:
            self.register_buffer("theta", theta)
        self.n_freq = theta.numel()
        for r, W in enumerate(self.windows):
            self.register_buffer(f"taper{r}", _taper(W, taper))
        self.gate_logit = nn.Parameter(torch.zeros(len(self.windows), self.n_freq))

    @property
    def out_dim(self) -> int:
        return len(self.windows) * self.K * self.n_freq

    def gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    def _kernels(self, W: int, taper: torch.Tensor) -> torch.Tensor:
        n = (torch.arange(W, device=taper.device, dtype=taper.dtype) - W // 2)
        ang = n[:, None] * self.theta.to(taper.dtype)[None, :]        # [W, F]
        cos = (torch.cos(ang) * taper[:, None]).t()                    # [F, W]
        sin = (torch.sin(ang) * taper[:, None]).t()
        return torch.cat([cos, sin], 0)                                # [2F, W]

    def forward(self, x: torch.Tensor, mask: torch.Tensor, return_maps: bool = False):
        B, K, L = x.shape
        m = mask[:, None, :].to(x.dtype)
        x = x * m
        g = self.gates()
        outs, maps = [], []
        for r, W in enumerate(self.windows):
            taper = getattr(self, f"taper{r}").to(x.dtype)
            ker = self._kernels(W, taper)                               # [2F, W]
            weight = ker.unsqueeze(1).repeat(K, 1, 1)                   # [K*2F, 1, W]
            y = F.conv1d(x, weight, padding=W // 2, groups=K)           # [B, K*2F, L]
            y = y.view(B, K, 2, self.n_freq, L)
            mag = torch.sqrt(y[:, :, 0] ** 2 + y[:, :, 1] ** 2 + 1e-8)  # [B, K, F, L]
            # 用“窗口内有效残基的权重和”归一化：序列两端和 padding 处不会被低估
            denom = F.conv1d(m, taper.view(1, 1, W), padding=W // 2)    # [B, 1, L]
            mag = mag / (denom.unsqueeze(2) + 1e-6)
            feat = torch.log1p(mag)
            if return_maps:
                maps.append(feat)
            feat = feat * g[r].view(1, 1, -1, 1)
            outs.append(feat.reshape(B, K * self.n_freq, L))
        out = torch.cat(outs, 1) * m
        return (out, maps) if return_maps else out


class GlobalSpectrum(nn.Module):
    """对照组：整条序列做一次 DFT（同一频率网格），再把谱广播给每个残基。
    它知道“这条蛋白有什么周期”，但不知道“周期出现在哪一段”。"""

    def __init__(self, n_channels: int, freq_step_deg: float = 10.0, learn_freq: bool = False):
        super().__init__()
        theta = _theta_grid(freq_step_deg)
        if learn_freq:
            self.theta = nn.Parameter(theta)
        else:
            self.register_buffer("theta", theta)
        self.K, self.n_freq = n_channels, theta.numel()
        self.gate_logit = nn.Parameter(torch.zeros(1, self.n_freq))
        self.windows = [0]

    @property
    def out_dim(self) -> int:
        return self.K * self.n_freq

    def gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, return_maps: bool = False):
        B, K, L = x.shape
        m = mask.to(x.dtype)
        x = x * m[:, None, :]
        n = torch.arange(L, device=x.device, dtype=x.dtype)
        ang = n[:, None] * self.theta.to(x.dtype)[None, :]               # [L, F]
        c, s = x @ torch.cos(ang), x @ torch.sin(ang)                    # [B, K, F]
        mag = torch.sqrt(c ** 2 + s ** 2 + 1e-8) / m.sum(-1).clamp(min=1)[:, None, None]
        feat = torch.log1p(mag) * self.gates()[0].view(1, 1, -1)
        out = feat.reshape(B, K * self.n_freq, 1).expand(B, K * self.n_freq, L) * m[:, None, :]
        return (out, [feat]) if return_maps else out


class GlobalPartition(nn.Module):
    """无序感知的全局上下文：一个可学的门 g ∈ (0, 1) 把整条蛋白软切成两部分，
    同样的 K 条性质信号在两部分上各取加权平均（取绝对值、对数压缩，和 GlobalSpectrum 只有 0° 时同一种摘要），
    再加上第一部分的占比；输出 [B, 2K + 1, L]（每个残基同一份，padding 处为 0）。
    g 处处相同时，两部分的平均都等于整条蛋白的平均。"""

    def __init__(self, n_channels: int, eps: float = 1e-6):
        super().__init__()
        self.K, self.eps = n_channels, eps

    @property
    def out_dim(self) -> int:
        return 2 * self.K + 1

    def summary(self, x: torch.Tensor, g: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x:[B,K,L]，g:[B,L]，mask:[B,L] → 蛋白级摘要 [B, 2K + 1]。"""
        m = mask.to(x.dtype)
        x = x * m[:, None, :]
        a = g.to(x.dtype) * m
        b = (1.0 - g.to(x.dtype)) * m
        sa, sb = a.sum(-1), b.sum(-1)                                     # [B]
        ma = (x * a[:, None, :]).sum(-1) / (sa[:, None] + self.eps)       # [B, K]
        mb = (x * b[:, None, :]).sum(-1) / (sb[:, None] + self.eps)
        frac = sa / m.sum(-1).clamp(min=1)
        return torch.cat([torch.log1p(ma.abs()), torch.log1p(mb.abs()), frac[:, None]], 1)

    def forward(self, x: torch.Tensor, g: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        feat = self.summary(x, g, mask)
        L = x.shape[-1]
        return feat[:, :, None].expand(-1, -1, L) * mask.to(x.dtype)[:, None, :]


class GlobalLocal(nn.Module):
    """全局 + 局部：同一组性质通道，既算整条蛋白上的谱（GlobalSpectrum），
    又算以每个残基为中心的几个长窗口内的谱（LocalSpectralMoment），两者拼在一起。
    只取 0°（spec_freq_step 很大）、窗口内等权（spec_taper = rect）时，就是“整条蛋白的平均 + 45、91 个残基窗口内的平均”，
    借鉴 TransFuse 在 10、45、90 个残基的窗口上汇总特征的做法（10 个残基以内的上下文窗口头已经覆盖）。"""

    def __init__(self, n_channels: int, windows: List[int], freq_step_deg: float = 10.0,
                 taper: str = "hann", learn_freq: bool = False):
        super().__init__()
        self.glob = GlobalSpectrum(n_channels, freq_step_deg, learn_freq)
        self.loc = LocalSpectralMoment(n_channels, windows, freq_step_deg, taper, learn_freq)
        self.K, self.n_freq = n_channels, self.glob.n_freq
        self.windows = [0] + list(windows)

    @property
    def out_dim(self) -> int:
        return self.glob.out_dim + self.loc.out_dim

    def gates(self) -> torch.Tensor:
        return torch.cat([self.glob.gates(), self.loc.gates()], 0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, return_maps: bool = False):
        if return_maps:
            a, ma = self.glob(x, mask, return_maps=True)
            b, mb = self.loc(x, mask, return_maps=True)
            return torch.cat([a, b], 1), list(ma) + list(mb)
        return torch.cat([self.glob(x, mask), self.loc(x, mask)], 1)
