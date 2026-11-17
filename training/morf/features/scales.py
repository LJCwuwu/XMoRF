"""氨基酸标度：给“经典疏水矩”对照（A6_fixed_hm）和动机分析用。

Eisenberg 共识疏水标度（1984）、Kyte-Doolittle 亲疏水标度（1982）、pH7 近似电荷。
索引顺序与 parse.AA20 一致；第 21 位（非标准氨基酸）置 0。
"""
from __future__ import annotations

import numpy as np

from ..data.parse import AA20

EISENBERG = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29, "Q": -0.85, "E": -0.74,
    "G": 0.48, "H": -0.40, "I": 1.38, "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19,
    "P": 0.12, "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08,
}
KYTE_DOOLITTLE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
CHARGE = {a: 0.0 for a in AA20}
CHARGE.update({"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1})

SCALES = {"eisenberg": EISENBERG, "kyte_doolittle": KYTE_DOOLITTLE, "charge": CHARGE}


def scale_table(normalize: bool = True) -> np.ndarray:
    """返回 [21, 3] 表：行=氨基酸（含未知），列=三个标度（各自 z-score）。"""
    tab = np.zeros((21, len(SCALES)), dtype=np.float32)
    for j, sc in enumerate(SCALES.values()):
        col = np.array([sc[a] for a in AA20], dtype=np.float32)
        if normalize:
            col = (col - col.mean()) / (col.std() + 1e-8)
        tab[:20, j] = col
    return tab


def per_residue_scales(tokens: np.ndarray) -> np.ndarray:
    return scale_table()[tokens]
