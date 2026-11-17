"""Configurable MoRF prediction networks.

Language-model features feed spatial and/or spectral-summary branches, followed
by residue-level scoring. Masks are applied throughout the network to support
variable-length batches. The final XMoRF configurations use the global-summary
branch with a single zero-frequency component."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from .extras import GatedFusion, MultiScaleLocalAttention, MultiScaleSelect, TerminalPosition
from .spectral import GlobalLocal, GlobalPartition, GlobalSpectrum, LocalSpectralMoment

AA_EXTRA_DIM = 24  # 21 维 one-hot + 3 个氨基酸标度
# Optional extensions of the global-summary branch.
GLOBAL_PLUS = ("global_da", "global_msk", "global_lma", "global_tpe", "global_gpf")


class InputFusion(nn.Module):
    """每个语言模型：LayerNorm → Dropout → 线性投影到 d；再拼接混合。
    训练时以概率 p_mod 整块丢弃某个模型（至少保留一个），防止网络只依赖其中一个。"""

    def __init__(self, dims: Dict[str, int], d: int, p_in: float, p_mod: float, gated: bool = False):
        super().__init__()
        self.names = list(dims)
        self.p_mod = p_mod
        self.proj = nn.ModuleDict({
            n: nn.Sequential(nn.LayerNorm(dim), nn.Dropout(p_in), nn.Linear(dim, d))
            for n, dim in dims.items()})
        self.mix = nn.Sequential(nn.Linear(d * len(self.names), d), nn.GELU())
        # Residue-level gated fusion, initialized to preserve the base branch.
        self.gated = GatedFusion(d, len(self.names)) if gated else None

    def forward(self, feats: Dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        zs = [self.proj[n](feats[n].float()) for n in self.names]
        M = len(zs)
        if self.training and self.p_mod > 0 and M > 1:
            B = zs[0].shape[0]
            keep = (torch.rand(B, M, device=zs[0].device) > self.p_mod).float()
            dead = keep.sum(1) == 0
            if dead.any():
                keep[dead, torch.randint(0, M, (int(dead.sum()),), device=keep.device)] = 1.0
            # 反向缩放（inverted dropout）：训练和推理时输入的期望尺度一致
            zs = [z * keep[:, i, None, None] / (1.0 - self.p_mod) for i, z in enumerate(zs)]
        if self.gated is None:
            h = self.mix(torch.cat(zs, -1))
        else:
            h = self.mix[1](self.mix[0](torch.cat(zs, -1)) + self.gated(zs))
        return h * mask[..., None]


class ResDilatedBlock(nn.Module):
    def __init__(self, d: int, k: int, dilation: int, p: float):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.conv1 = nn.Conv1d(d, d, k, padding=dilation * (k // 2), dilation=dilation)
        self.conv2 = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        y = (self.norm(x) * m).transpose(1, 2)
        y = self.conv2(F.gelu(self.conv1(y))).transpose(1, 2)
        return (x + self.drop(y)) * m


class SpatialBranch(nn.Module):
    """膨胀残差卷积。dilations=[1,2,4,8,16,32]、k=3 时感受野 = 1+2*(1+2+...+32) = 127 残基，
    覆盖 MoRF 本身 + 两侧侧翼（TransFuse 用 90 残基窗口也是这个道理）。"""

    def __init__(self, d: int, k: int, dilations, p: float):
        super().__init__()
        self.blocks = nn.ModuleList([ResDilatedBlock(d, k, r, p) for r in dilations])
        self.rf = 1 + sum(r * (k - 1) for r in dilations)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask[..., None].to(h.dtype)
        for b in self.blocks:
            h = b(h, m)
        return h


class SpectralBranch(nn.Module):
    def __init__(self, d: int, cfg: Config):
        super().__init__()
        self.mode = cfg.spec_mode
        if self.mode not in ("local", "global", "global_local", "fixed_hm", "off") + GLOBAL_PLUS:
            raise ValueError(f"未知 spec_mode: {self.mode}")
        if self.mode == "off":
            # 对照组：没有任何谱特征，只剩残差 + 小卷积块（其余结构与 LSM 版完全相同）
            self.proj = self.spec = self.out = None
            self.smooth = ResDilatedBlock(d, 5, 1, cfg.dropout)
            return
        if self.mode == "fixed_hm":
            K = 3                       # 固定用 Eisenberg / Kyte-Doolittle / 电荷 三个标度
            self.proj = None
        else:
            K = cfg.spec_channels
            self.proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, K))
        if self.mode == "global" or self.mode in GLOBAL_PLUS:
            self.spec = GlobalSpectrum(K, cfg.spec_freq_step, cfg.spec_learn_freq)
        elif self.mode == "global_local":
            # Combine full-sequence and long-window summaries of the same channels.
            self.spec = GlobalLocal(K, cfg.spec_windows, cfg.spec_freq_step, cfg.spec_taper, cfg.spec_learn_freq)
        else:
            self.spec = LocalSpectralMoment(K, cfg.spec_windows, cfg.spec_freq_step,
                                            cfg.spec_taper, cfg.spec_learn_freq)
        n = self.spec.out_dim
        self.out = nn.Sequential(nn.LayerNorm(n), nn.Linear(n, d), nn.GELU(), nn.Dropout(cfg.dropout))
        self.smooth = ResDilatedBlock(d, 5, 1, cfg.dropout)
        if self.mode == "global_da":
            # A learned gate partitions the sequence into two weighted summaries.
            # 经 LayerNorm 和一个线性层（初值为 0、不带偏置）加到原来整条蛋白摘要的线性输出上，再过 GELU。
            # 这个线性层为 0 时，就是原来的全局头（global）。
            self.part_gate = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
            self.part = GlobalPartition(K)
            self.part_out = nn.Sequential(nn.LayerNorm(self.part.out_dim), nn.Linear(self.part.out_dim, d, bias=False))
            nn.init.zeros_(self.part_out[1].weight)
        # Optional summary extensions; global_gpf is implemented in InputFusion.
        self.extra = None
        if self.mode == "global_msk":
            self.extra = MultiScaleSelect(d)
        elif self.mode == "global_lma":
            self.extra = MultiScaleLocalAttention(d)
        elif self.mode == "global_tpe":
            self.extra = TerminalPosition(d)

    def gate(self, h: torch.Tensor) -> torch.Tensor:
        """global_da 的门：每个残基一个 (0, 1) 的分 [B, L]。"""
        return torch.sigmoid(self.part_gate(h.float())).squeeze(-1)

    def signals(self, h: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """返回送进谱分析的 K 条一维信号 [B, K, L]。"""
        p = scales.float() if self.proj is None else self.proj(h.float())
        return p.float().transpose(1, 2)

    def forward(self, h, scales, mask, return_maps: bool = False):
        if self.mode == "off":
            m = mask[..., None].to(h.dtype)
            s = self.smooth(h * m, m)
            return (s, [], None) if return_maps else s
        with torch.autocast(device_type=h.device.type, enabled=False):   # 谱计算固定用 fp32
            sig = self.signals(h, scales)
            res = self.spec(sig, mask.float(), return_maps=return_maps)
            if self.mode == "global_da":
                g = self.gate(h)
                part = self.part(sig, g, mask.float())
        s, maps = res if return_maps else (res, None)
        if self.mode == "global_da":
            a = self.out[1](self.out[0](s.transpose(1, 2).to(h.dtype)))
            b = self.part_out(part.transpose(1, 2).to(h.dtype))
            s = self.out[3](self.out[2](a + b))
            if return_maps:
                maps = list(maps) + [g]
        else:
            s = self.out(s.transpose(1, 2).to(h.dtype))
        m = mask[..., None].to(h.dtype)
        if self.extra is not None:
            s = s + self.extra(h, mask)
        s = self.smooth((h + s) * m, m)     # 残差：保留逐位点身份信息 + 局部谱上下文
        return (s, maps, sig) if return_maps else s


class WindowMLP(nn.Module):
    """旧基线的等价写法：固定窗口内的特征 → MLP。"""

    def __init__(self, d: int, W: int, p: float):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.conv = nn.Conv1d(d, d, W, padding=W // 2)
        self.drop = nn.Dropout(p)

    def forward(self, h, mask):
        m = mask[..., None].to(h.dtype)
        y = self.conv((self.norm(h) * m).transpose(1, 2)).transpose(1, 2)
        return self.drop(F.gelu(y)) * m


class MoRFNet(nn.Module):
    def __init__(self, cfg: Config, plm_dims: Dict[str, int]):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        dims = {n: plm_dims[n] for n in cfg.plms}
        if cfg.use_aa_scales:
            dims["aa"] = AA_EXTRA_DIM
        self.inp = InputFusion(dims, d, cfg.input_dropout, cfg.modality_dropout,
                               gated=cfg.spec_mode == "global_gpf" and cfg.head in ("spectral", "dual", "mlp_dual"))

        self.head_type = cfg.head
        if cfg.head not in ("window_mlp", "spatial", "spectral", "dual", "mlp_dual"):
            raise ValueError(f"未知 head: {cfg.head}")
        if cfg.head in ("spatial", "dual"):
            self.spatial = SpatialBranch(d, cfg.kernel_size, cfg.dilations, cfg.dropout)
        if cfg.head in ("spectral", "dual", "mlp_dual"):
            self.spectral = SpectralBranch(d, cfg)
        if cfg.head in ("window_mlp", "mlp_dual"):
            self.wmlp = WindowMLP(d, cfg.window_mlp_size, cfg.dropout)
        if cfg.head in ("dual", "mlp_dual"):
            self.fuse = nn.Linear(2 * d, d)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                 nn.Dropout(cfg.dropout), nn.Linear(d, 1))
        self.decoder = cfg.decoder
        if cfg.decoder not in ("sigmoid", "semicrf"):
            raise ValueError(f"未知 decoder: {cfg.decoder}")
        if cfg.decoder == "semicrf":
            from .semicrf import SemiCRF
            self.crf = SemiCRF(d, cfg.crf_max_span, cfg.crf_boundary, cfg.crf_len_prior)

    def trunk(self, batch: Dict[str, torch.Tensor], return_aux: bool = False):
        """主干 + 预测头：返回 (逐残基表示 z, 发射分 logits)。半马尔可夫 CRF 需要 z 来算边界分。"""
        out = self.forward(batch, return_aux=return_aux, return_z=True)
        return out

    def forward(self, batch: Dict[str, torch.Tensor], return_aux: bool = False,
                return_z: bool = False):
        mask = batch["mask"]
        feats = {n: batch[f"feat_{n}"] for n in self.inp.names}
        h = self.inp(feats, mask)
        aux: Dict[str, Optional[torch.Tensor]] = {}

        if self.head_type == "window_mlp":
            z = self.wmlp(h, mask)
        elif self.head_type == "spatial":
            z = self.spatial(h, mask)
        elif self.head_type == "spectral":
            z = self.spectral(h, batch["scales"], mask)
        else:
            # dual = 膨胀卷积 + LSM；mlp_dual = 窗口 MLP + LSM（把 LSM 当插件加到最常用的基线上）
            a = self.spatial(h, mask) if self.head_type == "dual" else self.wmlp(h, mask)
            b = self.spectral(h, batch["scales"], mask)
            ab = torch.cat([a, b], -1)
            if self.cfg.fusion == "gate":
                g = torch.sigmoid(self.fuse(ab))
                z = g * a + (1 - g) * b
                aux["fusion_gate"] = g          # 越接近 1 越依赖空域分支（mlp_dual 时是 MLP 分支）
            else:
                z = F.gelu(self.fuse(ab))
        logits = self.out(z).squeeze(-1)
        logits = logits.masked_fill(mask == 0, 0.0)
        if return_z:
            return (z, logits, aux) if return_aux else (z, logits)
        return (logits, aux) if return_aux else logits


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
