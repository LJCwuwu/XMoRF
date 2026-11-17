"""蛋白级 Dataset + 变长 padding 的 collate。

一个样本 = 一整条蛋白（训练时超过 max_len 会随机裁剪，并以 50% 概率保证裁剪窗包含 MoRF）。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..features.scales import scale_table
from ..features.store import FeatureStore
from .parse import Protein

_SCALES = scale_table()  # [21, 3]


class ProteinDataset(Dataset):
    def __init__(self, prots: List[Protein], store: FeatureStore, plms: List[str],
                 use_aa_scales: bool = False, max_len: Optional[int] = None,
                 train: bool = False, preload: bool = True, seed: int = 0):
        self.prots, self.store, self.plms = prots, store, plms
        self.use_aa, self.max_len, self.train = use_aa_scales, max_len, train
        self.rng = np.random.RandomState(seed)
        self.cache = None
        if preload:  # 938 条蛋白 × 2304 维 fp16 ≈ 2.6 GB，AutoDL 内存放得下
            self.cache = [{m: store.load(m, p.key) for m in plms} for p in prots]

    def __len__(self) -> int:
        return len(self.prots)

    def _crop(self, p: Protein):
        L = len(p.seq)
        if not self.train or self.max_len is None or L <= self.max_len:
            return 0, L
        pos = np.where(p.labels == 1)[0]
        if len(pos) and self.rng.rand() < 0.5:
            c = int(self.rng.choice(pos))
            lo = max(0, c - self.max_len + 1)
            hi = min(c, L - self.max_len)
            s = int(self.rng.randint(lo, hi + 1))
        else:
            s = int(self.rng.randint(0, L - self.max_len + 1))
        return s, s + self.max_len

    def __getitem__(self, i: int) -> Dict:
        p = self.prots[i]
        s, e = self._crop(p)
        feats = self.cache[i] if self.cache is not None else {
            m: self.store.load(m, p.key, mmap=True) for m in self.plms}
        tok = p.tokens[s:e]
        item = {"pid": p.pid, "key": p.key, "start": s, "seq": p.seq[s:e],   # seq：微调语言模型时现场分词用
                "tokens": tok, "labels": p.labels[s:e].astype(np.int64),
                "scales": _SCALES[tok]}
        for m in self.plms:
            item[f"feat_{m}"] = np.asarray(feats[m][s:e], dtype=np.float16)
        if self.use_aa:
            oh = np.eye(21, dtype=np.float32)[tok]
            item["feat_aa"] = np.concatenate([oh, _SCALES[tok]], 1)
        return item


def collate(items: List[Dict]) -> Dict:
    B = len(items)
    L = max(len(it["tokens"]) for it in items)
    out: Dict = {"pid": [it["pid"] for it in items], "key": [it["key"] for it in items],
                 "start": [it["start"] for it in items], "seq": [it["seq"] for it in items],
                 "length": torch.tensor([len(it["tokens"]) for it in items])}
    out["mask"] = torch.zeros(B, L)
    out["labels"] = torch.full((B, L), -1, dtype=torch.long)
    out["tokens"] = torch.full((B, L), 20, dtype=torch.long)
    out["scales"] = torch.zeros(B, L, 3)
    feat_keys = [k for k in items[0] if k.startswith("feat_")]
    for k in feat_keys:
        dt = torch.float16 if items[0][k].dtype == np.float16 else torch.float32
        out[k] = torch.zeros(B, L, items[0][k].shape[1], dtype=dt)
    for b, it in enumerate(items):
        n = len(it["tokens"])
        out["mask"][b, :n] = 1
        out["labels"][b, :n] = torch.from_numpy(it["labels"])
        out["tokens"][b, :n] = torch.from_numpy(it["tokens"])
        out["scales"][b, :n] = torch.from_numpy(it["scales"])
        for k in feat_keys:
            out[k][b, :n] = torch.from_numpy(np.ascontiguousarray(it[k]))
    return out


def to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


class LengthBucketSampler(torch.utils.data.Sampler):
    """长度相近的蛋白放一个 batch，减少 padding 浪费；每个 epoch 打乱。"""

    def __init__(self, lengths: List[int], batch_size: int, seed: int = 0):
        self.lengths, self.bs = np.asarray(lengths), batch_size
        self.rng = np.random.RandomState(seed)

    def __iter__(self):
        idx = np.argsort(self.lengths + self.rng.rand(len(self.lengths)) * 50)
        batches = [idx[i: i + self.bs].tolist() for i in range(0, len(idx), self.bs)]
        self.rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return (len(self.lengths) + self.bs - 1) // self.bs
