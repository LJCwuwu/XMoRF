"""训练/验证划分 + 泄漏检查。

关键原则：
1. 验证集只能从 TRAIN 里切。TEST419 和 TESTNEW 都是 TEST464 的子集，
   拿 TEST419 选模型再报 TEST464 = 数据泄漏（有些旧论文就是这么做的）。
2. 如果装了 mmseqs，按 30% 序列一致性聚类后“整簇”划分，避免同源蛋白同时出现在训练和验证里；
   没装 mmseqs 时，至少把完全相同的序列分到同一边（TRAIN 421 条里只有 375 条不重复序列）。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Dict, List, Tuple

import numpy as np

from .parse import Protein


def _mmseqs_clusters(prots: List[Protein], min_id: float = 0.3) -> List[int]:
    exe = shutil.which("mmseqs")
    if exe is None:
        return []
    with tempfile.TemporaryDirectory() as td:
        fa = os.path.join(td, "in.fa")
        with open(fa, "w") as f:
            for i, p in enumerate(prots):
                f.write(f">{i}\n{p.seq}\n")
        cmd = [exe, "easy-cluster", fa, os.path.join(td, "res"), os.path.join(td, "tmp"),
               "--min-seq-id", str(min_id), "-c", "0.5", "--cov-mode", "0", "-v", "1"]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:  # noqa
            print(f"[split] mmseqs 运行失败（{e}），退回按蛋白随机划分")
            return []
        rep2cid: Dict[str, int] = {}
        cid = [0] * len(prots)
        with open(os.path.join(td, "res_cluster.tsv")) as f:
            for ln in f:
                rep, mem = ln.split()
                cid[int(mem)] = rep2cid.setdefault(rep, len(rep2cid))
        # mmseqs 多线程时输出顺序会变 → 按“簇里第一条蛋白在文件中的位置”重新编号，
        # 让同一份数据在不同机器（不同核数）上得到完全相同的划分
        first: Dict[int, int] = {}
        return [first.setdefault(c, len(first)) for c in cid]


def _groups(prots: List[Protein], homology: bool) -> Tuple[List[int], str]:
    """每条蛋白的分组号：有 mmseqs 用 30% 同源簇，没有就把完全相同的序列归为一组。"""
    clusters = _mmseqs_clusters(prots) if homology else []
    if clusters:
        return clusters, f"mmseqs 30% 同源聚类（{len(set(clusters))} 簇）"
    # 没有 mmseqs 时至少保证：完全相同的序列（TRAIN 里有 37 组、83 条，标注不同）一定分在同一边，
    # 否则同一条蛋白一份在训练、一份在验证，验证分数会虚高，早停和阈值都会被带偏。
    seq2id: Dict[str, int] = {}
    clusters = [seq2id.setdefault(p.seq, len(seq2id)) for p in prots]
    return clusters, f"相同序列分组（{len(seq2id)} 组；装 mmseqs 可按同源家族分）"


def kfold_assign(prots: List[Protein], k: int, seed: int, homology: bool = True) -> List[int]:
    """把 TRAIN 按“整个同源簇”分到 k 折，每折残基数尽量相等。返回每条蛋白的折号。

    做法：簇随机打乱后按大小从大到小，依次放进当前残基最少的那一折（贪心装箱）。
    同一个簇（含完全相同的序列）永远在同一折 → 任何一折做验证时，训练集里都没有它的同源蛋白。
    """
    if k < 2:
        raise ValueError("k 折交叉验证需要 k >= 2")
    clusters, how = _groups(prots, homology)
    size: Dict[int, int] = {}
    for p, c in zip(prots, clusters):
        size[c] = size.get(c, 0) + len(p.seq)
    if len(size) < k:
        raise ValueError(f"只有 {len(size)} 个簇，不够分成 {k} 折")
    uniq = np.array(sorted(size))
    np.random.RandomState(seed).shuffle(uniq)
    order = sorted(uniq.tolist(), key=lambda c: -size[c])   # 稳定排序：同样大小的簇保持随机顺序
    load = np.zeros(k)
    fold_of: Dict[int, int] = {}
    for c in order:
        f = int(np.argmin(load))
        fold_of[c] = f
        load[f] += size[c]
    folds = [fold_of[c] for c in clusters]
    for f in range(k):
        inside = {p.seq for p, g in zip(prots, folds) if g == f}
        outside = {p.seq for p, g in zip(prots, folds) if g != f}
        assert not inside & outside, "划分错误：有序列同时出现在两折里"
    n_prot = np.bincount(folds, minlength=k)
    n_pos = np.zeros(k, int)
    for p, g in zip(prots, folds):
        n_pos[g] += int((p.labels == 1).sum())
    share = load / load.sum()
    print(f"[split] {k} 折交叉验证，{how}；每折蛋白数 {n_prot.tolist()}，"
          f"残基占比 {[f'{s:.0%}' for s in share]}，MoRF 残基数 {n_pos.tolist()}")
    return folds


def train_val_split(prots: List[Protein], val_frac: float, seed: int,
                    homology: bool = True) -> Tuple[List[Protein], List[Protein]]:
    rng = np.random.RandomState(seed)
    clusters, how = _groups(prots, homology)
    if not how.startswith("mmseqs"):
        how = how.replace("相同序列分组", "相同序列分组后随机")
    uniq = np.array(sorted(set(clusters)))
    rng.shuffle(uniq)
    size: Dict[int, int] = {}
    for p, c in zip(prots, clusters):
        size[c] = size.get(c, 0) + len(p.seq)
    target = val_frac * sum(size.values())
    val_c, acc = set(), 0
    for c in uniq:
        if acc >= target:
            break
        val_c.add(c)
        acc += size[c]
    va = [p for p, c in zip(prots, clusters) if c in val_c]
    tr = [p for p, c in zip(prots, clusters) if c not in val_c]
    shared = {p.seq for p in tr} & {p.seq for p in va}
    assert not shared, "划分错误：有序列同时出现在训练和验证里"
    print(f"[split] {how}: 训练 {len(tr)} 条 / 验证 {len(va)} 条（按残基约 {acc / sum(size.values()):.0%}）")
    return tr, va


def leakage_report(sets: Dict[str, List[Protein]], train_name: str = "TRAIN") -> None:
    """报告训练集与各测试集之间完全相同的序列。"""
    if train_name not in sets:
        return
    tr = {p.seq for p in sets[train_name]}
    for name, prots in sets.items():
        if name == train_name:
            continue
        dup = sum(p.seq in tr for p in prots)
        flag = "⚠️ 存在完全相同序列" if dup else "OK"
        print(f"[leak] {train_name} vs {name}: {dup}/{len(prots)} 条完全相同  {flag}")
