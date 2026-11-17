"""读取原始 MoRF 基准集。

支持的格式（自动识别）：
  三行格式（最常见）：
      >P12345
      MSEQ...
      000111...
  也支持序列/标签折行（只要标签行全是标签字符且总长与序列一致）。

标签映射：'1' → 1（MoRF），'0' → 0（非 MoRF），其他字符 → -1（忽略，不参与训练和评估）。
特例：EXP53（MoRFchibi 整理）用 '1' 标短 MoRF（≤30 残基）、'2' 标长 MoRF（>30 残基），
      两者都是 MoRF，所以 '2' → 1。（已核对：'1' 片段长 9~30，'2' 片段长 31~151，合计 2432 个
      MoRF 残基，与文献一致。）短/长子集由 protocols.py 按片段长度重新划分，结果与 1/2 完全一致。
如果你的文件用别的字符（比如 'M'/'N'），改 LABEL_MAP 即可；00_inspect_data.py 会先把字符统计打印出来。
"""
from __future__ import annotations

import glob
import hashlib
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(AA20)}   # 非标准氨基酸 → 20
LABEL_MAP = {"1": 1, "0": 0, "2": 1}
LABEL_CHARS = set("0123456789-xXmMnN.?*")         # 判断“这一行像标签行”用


@dataclass
class Protein:
    pid: str
    seq: str
    labels: np.ndarray            # int8, 1/0/-1（训练用的标签）
    source: str = ""
    labels_eval: Optional[np.ndarray] = None   # 评估用的标签（MC2：只用高质量标注）；None = 和 labels 相同
    group: str = ""                            # 来源分组（MC2：CV1~CV4，交叉验证按它分折）

    def eval_view(self) -> "Protein":
        """验证/测试时用的版本：标签换成评估标签（没有单独的评估标签就原样返回）。"""
        if self.labels_eval is None:
            return self
        return Protein(self.pid, self.seq, self.labels_eval, self.source, None, self.group)

    @property
    def key(self) -> str:
        """按序列内容取哈希做缓存键：同一序列在不同集合里只提一次特征。"""
        return seq_key(self.seq)

    @property
    def tokens(self) -> np.ndarray:
        return np.array([AA_TO_IDX.get(a, 20) for a in self.seq], dtype=np.int64)


def seq_key(seq: str) -> str:
    return hashlib.md5(seq.encode()).hexdigest()


def _is_label_line(s: str) -> bool:
    return len(s) > 0 and set(s) <= LABEL_CHARS


def read_benchmark(path: str, label_map: Optional[Dict[str, int]] = None,
                   verbose: bool = True) -> List[Protein]:
    label_map = label_map or LABEL_MAP
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    blocks, cur = [], None
    for ln in lines:
        if ln.startswith(">"):
            if cur:
                blocks.append(cur)
            cur = [ln[1:].split()[0] if ln[1:].split() else f"prot{len(blocks)}", []]
        elif cur is not None:
            cur[1].append(ln)
    if cur:
        blocks.append(cur)

    prots, unknown = [], Counter()
    for pid, body in blocks:
        # 从后往前收集标签行，剩下的是序列行
        lab_lines: List[str] = []
        seq_lines = list(body)
        while seq_lines and _is_label_line(seq_lines[-1]):
            lab_lines.insert(0, seq_lines.pop())
            if len("".join(lab_lines)) >= len("".join(seq_lines)):
                break
        seq = re.sub(r"\s", "", "".join(seq_lines)).upper()
        lab = "".join(lab_lines)
        if not seq:
            raise ValueError(f"{path}: 蛋白 {pid} 没有序列行")
        if len(lab) != len(seq):
            raise ValueError(
                f"{path}: 蛋白 {pid} 序列长 {len(seq)}，标签长 {len(lab)}，对不上。"
                f" 请用 scripts/00_inspect_data.py 看一下文件格式。")
        arr = np.full(len(seq), -1, dtype=np.int8)
        for i, c in enumerate(lab):
            if c in label_map:
                arr[i] = label_map[c]
            else:
                unknown[c] += 1
        prots.append(Protein(pid=pid, seq=seq, labels=arr, source=os.path.basename(path)))

    if verbose and unknown:
        print(f"[parse] {os.path.basename(path)}: 以下标签字符被当成“忽略(-1)”: {dict(unknown)}")
    return prots


# 文件名关键字 → 集合名（按顺序匹配，先匹配到先用）
_DISCOVERY_RULES = [
    ("TEST464", ["464"]),
    ("TRAIN", ["train", "421"]),
    ("EXP53", ["exp53", "exp_53", "test53", "test_53"]),
    ("TEST419", ["419", "test_dataset1"]),      # 该仓库里实际只有 418 条（少 1 条），只用于子集检查
    ("TESTNEW", ["testnew", "test_new", "test2012"]),
]


def _skip(rel: str) -> bool:
    """跳过隐藏目录（如 .ipynb_checkpoints）和 Jupyter 自动备份文件——
    有的仓库里备份文件名和正式文件几乎一样，内容却不同，会被误认成测试集。"""
    parts = rel.replace("\\", "/").split("/")
    return any(x.startswith(".") for x in parts) or "checkpoint" in parts[-1].lower() \
        or parts[-1].lower().endswith((".xlsx", ".xls", ".py", ".pyc", ".zip", ".gz"))


def discover_files(raw_dir: str) -> Dict[str, str]:
    """按文件名自动识别五个基准集。识别不了时请在配置 data_files 里手动指定。"""
    found: Dict[str, str] = {}
    files = sorted(p for p in glob.glob(os.path.join(raw_dir, "**", "*"), recursive=True)
                   if os.path.isfile(p) and not _skip(os.path.relpath(p, raw_dir)))
    for p in files:
        name = os.path.basename(p).lower()
        for set_name, keys in _DISCOVERY_RULES:
            if set_name in found:
                continue
            if any(k in name for k in keys):
                found[set_name] = p
                break
    return found


AF_LABEL_MAP = {"1": 1, "0": 0}           # MC2 的 .af：'1' MoRF，'0' 非 MoRF，'-' 及其他 → -1（不参与训练和评估）


def read_af(path: str, train_labels: str = "mixed", verbose: bool = True) -> List[Protein]:
    """读 MoRFchibi 2.0（MC2）的 annotated-fasta 格式。

    训练文件（CV1~CV4.af）每条 4 行：>编号 / 序列 / 高质量标签（HQ）/ 高+低质量混合标签；
    测试文件（Test-*.af）每条 3 行：>编号 / 序列 / 标签。
    train_labels="mixed"：训练用混合标签（数据作者的做法），"hq"：训练也只用高质量标签。
    评估（验证折、测试集）一律用高质量标签 → 存在 labels_eval 里，见 Protein.eval_view()。
    """
    if train_labels not in ("mixed", "hq"):
        raise ValueError(f"train_labels 只能是 mixed / hq，收到 {train_labels}")
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    heads = [i for i, ln in enumerate(lines) if ln.startswith(">")]
    group = os.path.splitext(os.path.basename(path))[0]

    def to_arr(lab: str, n: int, pid: str) -> np.ndarray:
        if len(lab) != n:
            raise ValueError(f"{path}: 蛋白 {pid} 序列长 {n}，标签长 {len(lab)}，对不上")
        return np.array([AF_LABEL_MAP.get(c, -1) for c in lab], dtype=np.int8)

    prots = []
    for k, i in enumerate(heads):
        j = heads[k + 1] if k + 1 < len(heads) else len(lines)
        block = lines[i:j]
        pid = block[0][1:].split()[0] if block[0][1:].split() else f"prot{k}"
        if len(block) not in (3, 4):
            raise ValueError(f"{path}: 蛋白 {pid} 有 {len(block)} 行，.af 应该是 3 行或 4 行")
        seq = block[1].upper()
        hq = to_arr(block[2], len(seq), pid)
        if len(block) == 4:
            mixed = to_arr(block[3], len(seq), pid)
            train = mixed if train_labels == "mixed" else hq
            prots.append(Protein(pid, seq, train, os.path.basename(path), hq, group))
        else:
            prots.append(Protein(pid, seq, hq, os.path.basename(path), None, group))
    if verbose:
        print(f"[parse] {os.path.basename(path)}: {len(prots)} 条（.af 格式，训练标签 = {train_labels}）")
    return prots


def discover_mc2(raw_dir: str) -> Dict[str, List[str]]:
    """识别 MoRFchibi 2.0 数据目录（AdapTrans 仓库的 data/raw_seq2/）：
    MC2_TRAIN = CV1~CV4.af 合在一起（按文件记分组，交叉验证用作者分好的折）；
    MC2_T143 / MC2_T143U / MC2_T128 / MC2_T106 = 四个测试集。"""
    out: Dict[str, List[str]] = {}
    if not os.path.isdir(raw_dir):
        return out
    names = {f.lower(): f for f in os.listdir(raw_dir)}
    cv = [os.path.join(raw_dir, names[n]) for n in sorted(names) if re.fullmatch(r"cv\d+\.af", n)]
    if cv:
        out["MC2_TRAIN"] = cv
    for set_name, fn in [("MC2_T143", "test-143.af"), ("MC2_T143U", "test-143_unmasked.af"),
                         ("MC2_T128", "test-128_af2.af"), ("MC2_T106", "test-106_success.af")]:
        if fn in names:
            out[set_name] = [os.path.join(raw_dir, names[fn])]
    return out


def read_any(paths, train_labels: str = "mixed", verbose: bool = True) -> List[Protein]:
    """一个或多个文件：.af 用 read_af，其他用 read_benchmark。"""
    paths = [paths] if isinstance(paths, str) else list(paths)
    out: List[Protein] = []
    for p in paths:
        out += read_af(p, train_labels, verbose) if p.lower().endswith(".af") else read_benchmark(p, verbose=verbose)
    return out


def load_sets(raw_dir: str, data_files: Optional[Dict[str, str]] = None,
              wanted: Optional[List[str]] = None, verbose: bool = True,
              train_labels: str = "mixed") -> Dict[str, List[Protein]]:
    files: Dict[str, object] = dict(discover_files(raw_dir)) if os.path.isdir(raw_dir) else {}
    files.update(discover_mc2(raw_dir))
    files.update(data_files or {})
    wanted = wanted or list(files)
    out = {}
    for name in wanted:
        if name not in files:
            raise FileNotFoundError(
                f"找不到集合 {name}。自动识别到: {files}。"
                f" 请用 --files {name}=路径 手动指定（正式实验建议三个集合都手动指定）。")
        out[name] = read_any(files[name], train_labels, verbose=verbose)
        if verbose:
            p = out[name]
            n = sum(len(x.seq) for x in p)
            pos = sum(int((x.labels == 1).sum()) for x in p)
            lab = sum(int((x.labels >= 0).sum()) for x in p)
            print(f"[data] {name:8s} {len(p):4d} 条蛋白  {n:7d} 残基（有标签 {lab}）  MoRF 占有标签残基 {pos / max(lab, 1):.2%}"
                  f"  ← {files[name]}")
    return out
