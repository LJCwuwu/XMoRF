"""评估协议：按领域惯例拆子集。

EXP53 惯例（MoRFchibi / OPAL）：分 all / short / long 三组报告。
  short = 长度 ≤ 30 的 MoRF 片段；long = 长度 > 30 的片段；
  负样本始终是全部非 MoRF 残基，另一组 MoRF 不参与（置为 -1）。
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def morf_runs(labels: np.ndarray):
    """返回所有连续 MoRF 片段的 (起点, 终点不含)。"""
    runs, i, L = [], 0, len(labels)
    while i < L:
        if labels[i] == 1:
            j = i
            while j < L and labels[j] == 1:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def length_subset(labels: np.ndarray, keep_short: bool, cut: int = 30) -> np.ndarray:
    out = labels.copy()
    for s, e in morf_runs(labels):
        is_short = (e - s) <= cut
        if is_short != keep_short:
            out[s:e] = -1
    return out


def subsets_for(set_name: str) -> Dict[str, object]:
    """返回 {子集名: 标签变换函数或 None(=原样)}。"""
    # EXP53 和 MC2 的测试集都有大量长 MoRF（>30 残基），分 all / short / long 报告
    if set_name.upper().startswith("EXP53") or set_name.upper().startswith("MC2_T"):
        return {"all": None,
                "short": lambda lab: length_subset(lab, True),
                "long": lambda lab: length_subset(lab, False)}
    return {"all": None}
