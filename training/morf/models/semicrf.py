"""半马尔可夫条件随机场（semi-Markov CRF）——把"逐残基打分 + 滑动平均"换成"直接对片段建模"。

【一句话】模型不再独立地问"这个残基是不是 MoRF"，而是问"[i, j) 这一整段是不是一个 MoRF"；
         片段长度和起止边界都是学出来的，推断用精确动态规划。

【为什么要做】MoRF 本来就是连续片段，不是独立残基。但现有所有预测器
    （MoRFpred → MoRFchibi 2.0 → MoRFs_TransFuse → AdapTrans → 本文 A4）
    都是"逐残基打分 + 事后滑动平均"。滑动平均是个拍脑袋的超参，而且完全没有长度先验。
    我们自己的实验也指向这里：平滑窗口值 0.011 AUC（比大部分结构改动都大），
    长 MoRF 全赢而短 MoRF 全平，AUPR 比 AUC 对方法差异敏感得多——这三件事都和"片段"有关。

【模型】一个分段 y = 一串首尾相接、互不重叠的片段，每段标 0 或 1：
      标 0（非 MoRF）的片段长度固定为 1，分数固定为 0（只有分差有意义）；
      标 1（MoRF）的片段长度 ℓ ∈ [1, max_span]，分数
          φ(i, j) = Σ_{l=i}^{j-1} e(l)      逐残基发射分（沿用原来的预测头输出）
                  + b_s(i) + b_e(j-1)       起点 / 终点边界分
                  + ψ(ℓ)                    可学习的长度先验
      再加两个标量转移 T01（非 MoRF → MoRF）、T11（MoRF 紧接 MoRF）。
      P(y | x) = exp(score(y)) / Z

【训练】负对数似然 = logZ − logZ_gold。
    logZ_gold 用同一套递推、但只允许和金标准一致的分段来算，所以：
      · 金标准里任意长的 MoRF（超过 max_span）会被自动拆成几段相邻片段，不会出现"无法到达"的无穷损失；
      · 标签为 -1（未标注）的位置既可以当 0 也可以当 1，等于对未知部分做边缘化——
        这正是 MoRF 标注稀疏、且我们在跨基准子集上做了局部掩蔽时需要的语义。

【推断】
    · 边际概率 P(残基 l 落在某个 MoRF 片段里) = ∂logZ / ∂e(l)。
      因为片段互不重叠，"覆盖 l 的片段个数"只能是 0 或 1，期望值就等于概率，所以用 autograd
      求一次梯度就拿到全部边际，不必再写一遍 backward 递推（少一处可能写错的地方）。
      这个边际天生是平滑的、带长度先验的，直接取代滑动平均。
    · Viterbi 给出最可能的那个分段，于是模型第一次能直接输出"片段列表"，可以算片段级指标。

【复杂度】O(L × max_span)。片段内求和用一个长度为 max_span 的滚动缓冲增量维护
    （不用前缀和相减——长序列上前缀和会很大，相减会损失精度）。
    DP 沿序列串行，但每步只有几个小算子，L=1024、max_span=48 时一次前向约几十毫秒。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

NEG = -1e4          # 代替 -inf：float32 下不会把 logaddexp 弄出 nan，又足够小


def _roll_in(buf: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
    """把 new 推进滚动缓冲的第 0 格，其余整体右移一格（丢掉最旧的一格）。"""
    return torch.cat([new.unsqueeze(1), buf[:, :-1]], 1)


class SemiCRF(nn.Module):
    """输入逐残基发射分 e 和表示 z，输出 NLL / 边际概率 / Viterbi 分段。"""

    def __init__(self, d: int, max_span: int = 48, use_boundary: bool = True,
                 use_len_prior: bool = True):
        super().__init__()
        if max_span < 1:
            raise ValueError(f"max_span 至少是 1，收到 {max_span}")
        self.max_span = int(max_span)
        self.bound = nn.Linear(d, 2) if use_boundary else None
        self.len_prior = nn.Parameter(torch.zeros(self.max_span)) if use_len_prior else None
        self.trans = nn.Parameter(torch.zeros(2))        # [T01, T11]

    # ---------------- 公共部分 ----------------

    def _terms(self, logits: torch.Tensor, z: torch.Tensor, mask: torch.Tensor):
        """发射分、起止边界分、长度先验。padding 位置一律置 0，不影响任何分数。"""
        m = mask.to(logits.dtype)
        e = logits.float() * m
        if self.bound is not None:
            b = self.bound(z).float()
            bs, be = b[..., 0] * m, b[..., 1] * m
        else:
            bs = be = torch.zeros_like(e)
        lp = (self.len_prior.float() if self.len_prior is not None
              else torch.zeros(self.max_span, device=e.device))
        return e, bs, be, lp

    @staticmethod
    def _gold_masks(labels: torch.Tensor, max_span: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """金标准约束。

        allow_zero[b, t]：位置 t 可以被判成"非 MoRF"（标签是 0 或 -1）。
        run1[b, t]：以位置 t 结尾、且标签都属于 {1, -1} 的最长连续段长度（截断到 max_span）；
                    于是"长度 ℓ、结束于 t 的 MoRF 片段"合法当且仅当 ℓ ≤ run1[b, t]。
        """
        allow_zero = (labels == 0) | (labels < 0)
        ones = ((labels == 1) | (labels < 0))
        B, L = labels.shape
        run1 = torch.zeros(B, L, dtype=torch.long, device=labels.device)
        prev = torch.zeros(B, dtype=torch.long, device=labels.device)
        for t in range(L):
            prev = torch.where(ones[:, t], prev + 1, torch.zeros_like(prev))
            run1[:, t] = prev
        return allow_zero, run1.clamp(max=max_span)

    def _dp(self, e: torch.Tensor, bs: torch.Tensor, be: torch.Tensor, lp: torch.Tensor,
            lengths: torch.Tensor, labels: Optional[torch.Tensor] = None,
            viterbi: bool = False):
        """沿序列做一次动态规划。

        labels 为 None  → 自由配分函数 logZ；
        labels 给定      → 只在和金标准一致的分段上求和，得到 logZ_gold。
        viterbi=True    → 用 max 代替 logsumexp，并返回回溯所需的中间量。
        """
        B, L = e.shape
        K, dev, dt = self.max_span, e.device, e.dtype
        T01, T11 = self.trans[0].float(), self.trans[1].float()
        allow_zero, run1 = (None, None) if labels is None else self._gold_masks(labels, K)
        ks = torch.arange(K, device=dev)                       # k = ℓ - 1

        A0 = torch.zeros(B, device=dev, dtype=dt)              # t = 0：空前缀，当作"以非 MoRF 结尾"
        A1 = torch.full((B,), NEG, device=dev, dtype=dt)
        out = torch.full((B,), NEG, device=dev, dtype=dt)      # 每条序列在 t = n 处取走的结果

        P = torch.logaddexp(A0 + T01, A1 + T11)                # "可以开一个新 MoRF 片段"的前缀分
        Pbuf = torch.full((B, K), NEG, device=dev, dtype=dt); Pbuf[:, 0] = P
        Sbuf = torch.zeros(B, K, device=dev, dtype=dt); Sbuf[:, 0] = e[:, 0]   # 片段内发射分之和
        Bbuf = torch.zeros(B, K, device=dev, dtype=dt); Bbuf[:, 0] = bs[:, 0]  # 片段起点的边界分

        A0_all, A1_all, back_k = [], [], []
        if viterbi:
            A0_all.append(A0.clone()); A1_all.append(A1.clone())

        for t in range(1, L + 1):
            cand = Pbuf + Sbuf + Bbuf + lp.unsqueeze(0) + be[:, t - 1].unsqueeze(1)   # [B, K]
            if labels is not None:
                cand = cand.masked_fill(ks.unsqueeze(0) >= run1[:, t - 1].unsqueeze(1), NEG)
            if viterbi:
                A1_new, bk = cand.max(1)
                back_k.append(bk)
            else:
                A1_new = torch.logsumexp(cand, 1)
            A0_new = torch.maximum(A0, A1) if viterbi else torch.logaddexp(A0, A1)
            if labels is not None:
                A0_new = torch.where(allow_zero[:, t - 1], A0_new, torch.full_like(A0_new, NEG))

            active = t <= lengths
            A0 = torch.where(active, A0_new, A0)
            A1 = torch.where(active, A1_new, A1)
            out = torch.where(t == lengths,
                              torch.maximum(A0, A1) if viterbi else torch.logaddexp(A0, A1), out)
            if viterbi:
                A0_all.append(A0.clone()); A1_all.append(A1.clone())

            if t < L:
                P = (torch.maximum(A0 + T01, A1 + T11) if viterbi
                     else torch.logaddexp(A0 + T01, A1 + T11))
                Pbuf = _roll_in(Pbuf, P)
                Bbuf = _roll_in(Bbuf, bs[:, t])
                Sbuf = _roll_in(Sbuf, torch.zeros(B, device=dev, dtype=dt)) + e[:, t].unsqueeze(1)
        if viterbi:
            return out, torch.stack(A0_all, 1), torch.stack(A1_all, 1), torch.stack(back_k, 1)
        return out

    # ---------------- 对外接口 ----------------

    def nll(self, logits: torch.Tensor, z: torch.Tensor, mask: torch.Tensor,
            labels: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """负对数似然，按残基数归一化（不同长度的蛋白权重才可比）。

        自由配分函数和金标准受限配分函数拼成一个两倍大的 batch 跑一次 DP：
        标签全是 -1 时"哪种分段都允许"，恰好就是自由情形，所以两者可以共用同一套递推，
        序列方向的 python 循环因此只跑一遍（这是整个模块最贵的地方）。
        """
        e, bs, be, lp = self._terms(logits, z, mask)
        B = e.shape[0]
        free_lab = torch.full_like(labels, -1)
        both = self._dp(torch.cat([e, e]), torch.cat([bs, bs]), torch.cat([be, be]), lp,
                        torch.cat([lengths, lengths]),
                        labels=torch.cat([free_lab, labels]))
        n = lengths.to(both.dtype).clamp(min=1)
        return ((both[:B] - both[B:]) / n).mean()

    def marginals(self, logits: torch.Tensor, z: torch.Tensor, mask: torch.Tensor,
                  lengths: torch.Tensor) -> torch.Tensor:
        """P(残基 l 落在某个 MoRF 片段里) = ∂logZ/∂e(l)。片段不重叠，所以期望计数就是概率。"""
        with torch.enable_grad():
            e, bs, be, lp = self._terms(logits, z, mask)
            e = e.detach().requires_grad_(True)
            logZ = self._dp(e, bs.detach(), be.detach(), lp.detach(), lengths)
            g, = torch.autograd.grad(logZ.sum(), e)
        return g.clamp(0.0, 1.0).detach()

    def decode(self, logits: torch.Tensor, z: torch.Tensor, mask: torch.Tensor,
               lengths: torch.Tensor) -> List[List[Tuple[int, int]]]:
        """Viterbi：每条序列返回 [(起, 止), ...]，左闭右开，按起点排好序。"""
        e, bs, be, lp = self._terms(logits, z, mask)
        with torch.no_grad():
            _, A0, A1, back_k = self._dp(e, bs, be, lp, lengths, viterbi=True)
        T01, T11 = float(self.trans[0]), float(self.trans[1])
        spans: List[List[Tuple[int, int]]] = []
        for b in range(e.shape[0]):
            n, cur, got = int(lengths[b]), int(lengths[b]), []
            state = 1 if float(A1[b, n]) > float(A0[b, n]) else 0
            while cur > 0:
                if state == 0:
                    nxt = cur - 1
                    state = 1 if float(A1[b, nxt]) > float(A0[b, nxt]) else 0
                    cur = nxt
                else:
                    ell = int(back_k[b, cur - 1]) + 1
                    i = max(0, cur - ell)
                    got.append((i, cur))
                    state = 1 if float(A1[b, i] + T11) > float(A0[b, i] + T01) else 0
                    cur = i
            got.reverse()
            spans.append(_merge_adjacent(got))
        return spans


def _merge_adjacent(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """相邻（首尾相接）的 MoRF 片段合并成一个——金标准里超过 max_span 的长片段就是这样被拆开的。"""
    out: List[Tuple[int, int]] = []
    for s, t in spans:
        if out and out[-1][1] == s:
            out[-1] = (out[-1][0], t)
        else:
            out.append((s, t))
    return out


def spans_to_labels(spans: List[Tuple[int, int]], n: int) -> torch.Tensor:
    y = torch.zeros(n)
    for s, t in spans:
        y[s:t] = 1
    return y


def brute_force_logz(e: torch.Tensor, bs: torch.Tensor, be: torch.Tensor, lp: torch.Tensor,
                     trans: torch.Tensor, n: int, max_span: int,
                     labels: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """自测专用：枚举 n 很小时的全部分段，直接验证 DP 的 logZ / 边际 / 最优分段。"""
    import itertools
    import math
    T01, T11 = float(trans[0]), float(trans[1])
    best, scores = (-math.inf, None), []

    def walk(pos: int, prev: int, acc: float, segs: List[Tuple[int, int]]):
        nonlocal best
        if pos == n:
            scores.append(acc)
            if acc > best[0]:
                best = (acc, list(segs))
            return
        if labels is None or labels[pos] == 0 or labels[pos] < 0:
            walk(pos + 1, 0, acc, segs)
        for ell in range(1, min(max_span, n - pos) + 1):
            j = pos + ell
            if labels is not None and any(not (labels[k] == 1 or labels[k] < 0) for k in range(pos, j)):
                break
            phi = float(e[pos:j].sum()) + float(bs[pos]) + float(be[j - 1]) + float(lp[ell - 1])
            walk(j, 1, acc + phi + (T01 if prev == 0 else T11), segs + [(pos, j)])

    walk(0, 0, 0.0, [])
    mx = max(scores)
    logz = mx + math.log(sum(math.exp(s - mx) for s in scores))

    marg = [0.0] * n

    def walk2(pos: int, prev: int, acc: float, segs: List[Tuple[int, int]]):
        if pos == n:
            w = math.exp(acc - logz)
            for s, t in segs:
                for k in range(s, t):
                    marg[k] += w
            return
        if labels is None or labels[pos] == 0 or labels[pos] < 0:
            walk2(pos + 1, 0, acc, segs)
        for ell in range(1, min(max_span, n - pos) + 1):
            j = pos + ell
            if labels is not None and any(not (labels[k] == 1 or labels[k] < 0) for k in range(pos, j)):
                break
            phi = float(e[pos:j].sum()) + float(bs[pos]) + float(be[j - 1]) + float(lp[ell - 1])
            walk2(j, 1, acc + phi + (T01 if prev == 0 else T11), segs + [(pos, j)])

    walk2(0, 0, 0.0, [])
    return {"logz": logz, "marginals": marg, "best_score": best[0],
            "best_spans": _merge_adjacent(best[1] or [])}
