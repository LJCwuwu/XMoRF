"""统计检验：蛋白级 bootstrap 置信区间 + 配对比较 + DeLong。

为什么按“蛋白”重采样而不是按“残基”：
    同一条蛋白里相邻残基高度相关，按残基重采样会把置信区间算得过窄（假装样本很多）。
    整条蛋白一起抽，才是对“换一批蛋白结果会怎样”的诚实估计。

实现技巧：bootstrap = 给每条蛋白一个“被抽中次数”权重，排序只做一次，
每轮只算加权累加和 → 1000 轮也就几秒。
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import norm, rankdata


class _Sorted:
    """把 (y, p, protein_id) 按分数降序排好，供加权 AUC/AP 快速计算。"""

    def __init__(self, y: np.ndarray, p: np.ndarray, groups: np.ndarray):
        m = y >= 0
        y, p, g = y[m], p[m], groups[m]
        o = np.argsort(-p, kind="mergesort")
        ps = p[o]
        self.y = y[o].astype(np.float64)
        self.g = g[o]
        # 并列分数归为同一组（与 sklearn 的并列处理一致）
        self.tie = np.r_[0, np.cumsum(ps[1:] != ps[:-1])] if len(ps) else np.zeros(0, int)
        self.n_tie = int(self.tie[-1]) + 1 if len(ps) else 0

    def auc_ap(self, w_group: np.ndarray):
        w = w_group[self.g]
        WPg = np.bincount(self.tie, w * self.y, minlength=self.n_tie)
        WNg = np.bincount(self.tie, w * (1 - self.y), minlength=self.n_tie)
        Wp, Wn = WPg.sum(), WNg.sum()
        if Wp == 0 or Wn == 0:
            return np.nan, np.nan
        # 降序排列：正样本胜过“排在后面的负样本”，与自己并列的负样本算半个
        neg_after = Wn - np.cumsum(WNg)
        auc = float(np.sum(WPg * (neg_after + 0.5 * WNg)) / (Wp * Wn))
        prec = np.cumsum(WPg) / np.maximum(np.cumsum(WPg + WNg), 1e-12)
        ap = float(np.sum(WPg * prec) / Wp)
        return auc, ap


def _group_ids(lengths: Sequence[int]) -> np.ndarray:
    return np.repeat(np.arange(len(lengths)), lengths)


def bootstrap_ci(y: np.ndarray, p: np.ndarray, lengths: Sequence[int],
                 n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> Dict[str, List[float]]:
    g = _group_ids(lengths)
    S = _Sorted(y, p, g)
    rng = np.random.RandomState(seed)
    G = len(lengths)
    res = np.array([S.auc_ap(rng.multinomial(G, np.ones(G) / G).astype(np.float64))
                    for _ in range(n_boot)])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {"auc_ci": np.nanpercentile(res[:, 0], [lo, hi]).tolist(),
            "aupr_ci": np.nanpercentile(res[:, 1], [lo, hi]).tolist()}


def paired_bootstrap(y: np.ndarray, p1: np.ndarray, p2: np.ndarray, lengths: Sequence[int],
                     n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """模型1 - 模型2 的 ΔAUC / ΔAUPR，同一批 bootstrap 权重配对计算。"""
    g = _group_ids(lengths)
    S1, S2 = _Sorted(y, p1, g), _Sorted(y, p2, g)
    rng = np.random.RandomState(seed)
    G = len(lengths)
    d = []
    for _ in range(n_boot):
        w = rng.multinomial(G, np.ones(G) / G).astype(np.float64)
        a1, b1 = S1.auc_ap(w); a2, b2 = S2.auc_ap(w)
        d.append((a1 - a2, b1 - b2))
    d = np.array(d)
    full1, full2 = S1.auc_ap(np.ones(G)), S2.auc_ap(np.ones(G))
    out = {}
    for j, name in enumerate(["auc", "aupr"]):
        x = d[:, j]
        out[f"d_{name}"] = float(full1[j] - full2[j])
        out[f"d_{name}_ci"] = np.nanpercentile(x, [2.5, 97.5]).tolist()
        # 双侧 bootstrap p 值：Δ 的符号翻转比例
        out[f"p_{name}"] = float(min(1.0, 2 * min(np.mean(x <= 0), np.mean(x >= 0))))
    return out


def delong_test(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> Dict[str, float]:
    """DeLong 配对 AUC 检验（Sun & Xu 2014 快速算法）。
    注意：它假设残基独立，会偏乐观；论文里建议和 paired_bootstrap 一起报。"""
    m = y >= 0
    y, p1, p2 = y[m], p1[m], p2[m]
    pos, neg = y == 1, y == 0
    M, N = pos.sum(), neg.sum()
    aucs, v01, v10 = [], [], []
    for p in (p1, p2):
        tz = rankdata(p)
        tx, ty = rankdata(p[pos]), rankdata(p[neg])
        aucs.append((tz[pos].sum() - M * (M + 1) / 2) / (M * N))
        v01.append((tz[pos] - tx) / N)
        v10.append(1.0 - (tz[neg] - ty) / M)
    sx, sy = np.cov(np.vstack(v01)), np.cov(np.vstack(v10))
    cov = sx / M + sy / N
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(max(var, 1e-20))
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "z": float(z),
            "p": float(2 * norm.sf(abs(z)))}
