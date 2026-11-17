"""特征缓存：data/features_v2/<语言模型名>/<序列md5>.npy（float16）。

按序列内容做键，所以 TEST419 ⊂ TEST464 这种重复蛋白只存一份；
换数据集文件也不会读错缓存。
"""
from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from ..data.parse import Protein


class FeatureStore:
    def __init__(self, root: str):
        self.root = root

    def path(self, plm: str, key: str) -> str:
        return os.path.join(self.root, plm, f"{key}.npy")

    def has(self, plm: str, key: str) -> bool:
        return os.path.exists(self.path(plm, key))

    def save(self, plm: str, key: str, arr: np.ndarray) -> None:
        os.makedirs(os.path.join(self.root, plm), exist_ok=True)
        np.save(self.path(plm, key), arr.astype(np.float16))

    def load(self, plm: str, key: str, mmap: bool = False) -> np.ndarray:
        return np.load(self.path(plm, key), mmap_mode="r" if mmap else None)

    def check(self, prots: List[Protein], plms: List[str]) -> Dict[str, int]:
        """返回每个语言模型缺多少条，训练前用来提醒先跑 01_extract_features.py。"""
        miss = {}
        for m in plms:
            miss[m] = sum(not self.has(m, p.key) for p in prots)
        return miss
