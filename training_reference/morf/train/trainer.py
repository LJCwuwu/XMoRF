"""训练一次 = 一个配置 × 一个种子。产出写进 out_dir/runs/<tag>_s<seed>/：
    config.json         本次配置
    model.pt            验证集 AUC 最好的权重
    history.csv         每个 epoch 的 loss / 验证 AUC / AUPR
    metrics.json        各测试集指标（阈值来自验证集）
    preds_<集合>.npz    逐残基预测（后面做集成、DeLong、画图都靠它）
"""
from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
import random
import time
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import Config
from ..data.dataset import LengthBucketSampler, ProteinDataset, collate, to_device
from ..data.parse import Protein, load_sets, read_any
from ..data.splits import kfold_assign, leakage_report, train_val_split
from ..eval.metrics import all_metrics, best_mcc_threshold
from ..eval.protocols import subsets_for
from ..eval.stats import bootstrap_ci
from ..features.plm import PLM_DIMS
from ..features.store import FeatureStore
from ..models.net import MoRFNet, count_params
from .losses import masked_bce, masked_focal


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _cv_split(cfg: Config, prots: List[Protein]):
    """K 折交叉验证的划分同样只做一次并存盘（序列哈希 → 折号），所有模型、种子共用同一份。"""
    k = int(cfg.cv_folds)
    if not 0 <= cfg.fold < k:
        raise ValueError(f"fold={cfg.fold} 超出范围（共 {k} 折）")
    if cfg.cv_predefined:
        # 数据自带的分折（MC2：CV1~CV4，作者按 40% 序列一致性划分）→ 直接用，不重新聚类
        groups = sorted({p.group for p in prots})
        if len(groups) != k:
            raise ValueError(f"数据自带 {len(groups)} 个分组 {groups}，和 cv_folds={k} 不一致")
        fold_of = {}
        for p in prots:
            f = groups.index(p.group)
            if fold_of.setdefault(p.key, f) != f:
                raise ValueError(f"序列 {p.pid} 同时出现在两个预定义分组里，不能按组做交叉验证")
        va = [p for p in prots if fold_of[p.key] == cfg.fold]
        tr = [p for p in prots if fold_of[p.key] != cfg.fold]
        print(f"[split] 预定义 {k} 折（{groups}）第 {cfg.fold} 折 = {groups[cfg.fold]}: 训练 {len(tr)} / 验证 {len(va)}")
        return tr, va
    path = os.path.join(cfg.out_dir, f"cv{k}_seed{cfg.split_seed}.json")
    if os.path.exists(path):
        with open(path) as f:
            fold_of = json.load(f)["fold_of_key"]
        miss = [p.pid for p in prots if p.key not in fold_of]
        if miss:
            raise RuntimeError(f"{path} 里没有这些蛋白（数据换过了？）: {miss[:5]}")
    else:
        folds = kfold_assign(prots, k, cfg.split_seed)
        fold_of = {p.key: f for p, f in zip(prots, folds)}
        os.makedirs(cfg.out_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"k": k, "fold_of_key": fold_of,
                       "pids_by_fold": {str(i): [p.pid for p in prots if fold_of[p.key] == i]
                                        for i in range(k)}}, f, indent=1)
    va = [p for p in prots if fold_of[p.key] == cfg.fold]
    tr = [p for p in prots if fold_of[p.key] != cfg.fold]
    print(f"[split] {k} 折交叉验证（{path}）第 {cfg.fold} 折: 训练 {len(tr)} / 验证 {len(va)}")
    return tr, va


def _fixed_split(cfg: Config, prots: List[Protein]):
    """验证集划分只做一次并存盘（按序列哈希）。之后所有种子、所有消融、换机器重跑都读同一份，
    保证集成和配对检验比较的是同一批蛋白（mmseqs 装没装也不影响）。"""
    if int(cfg.cv_folds) > 1:
        return _cv_split(cfg, prots)
    path = os.path.join(cfg.out_dir, f"split_val{cfg.val_frac}_seed{cfg.split_seed}.json")
    if os.path.exists(path):
        with open(path) as f:
            val_keys = set(json.load(f)["val_keys"])
        va = [p for p in prots if p.key in val_keys]
        tr = [p for p in prots if p.key not in val_keys]
        print(f"[split] 复用已保存的划分 {path}: 训练 {len(tr)} / 验证 {len(va)}")
        return tr, va
    tr, va = train_val_split(prots, cfg.val_frac, cfg.split_seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"val_keys": [p.key for p in va], "val_pids": [p.pid for p in va]}, f, indent=1)
    return tr, va


def _select_split(cfg: Config, prots: List[Protein], tr: List[Protein], held: List[Protein]):
    """cv_select="next"：第 fold 折只做最后的评估；下一折（fold+1）做选择（早停、平滑窗口、阈值）；
    训练只用其余的折。这样评估折完全不参与任何选择。返回 (训练, 选择折, 评估折)。"""
    k = int(cfg.cv_folds)
    if cfg.cv_select != "next" or k < 3:
        raise ValueError(f"cv_select={cfg.cv_select!r} 需要等于 'next'，且至少 3 折（现在 {k} 折）")
    _, sel = _cv_split(dataclasses.replace(cfg, fold=(cfg.fold + 1) % k), prots)
    sel_keys = {p.key for p in sel}
    return [p for p in tr if p.key not in sel_keys], sel, held


def _subsample(cfg: Config, tr: List[Protein]) -> List[Protein]:
    """小样本实验：训练蛋白按蛋白随机只留 train_frac。同一个 train_subset_seed 下先把蛋白打乱一次、
    再取前 n 条，所以同一种子的 10% ⊂ 20% ⊂ 50%（嵌套子集）。选择折、评估折不受影响。"""
    f = float(cfg.train_frac)
    if f >= 1.0:
        return tr
    if f <= 0.0:
        raise ValueError(f"train_frac={f} 应在 (0, 1] 之间")
    order = np.random.RandomState(int(cfg.train_subset_seed)).permutation(len(tr))
    keep = set(order[:max(1, int(round(f * len(tr))))].tolist())
    sub = [p for i, p in enumerate(tr) if i in keep]
    if not any((p.labels == 1).any() for p in sub):
        raise ValueError(f"train_frac={f} 抽到的 {len(sub)} 条蛋白里没有 MoRF 残基")
    return sub


def mask_flanks(p: Protein, k: int) -> Protein:
    """第 61 步：离最近的 MoRF 残基（标签 1）不超过 k 个位置的非 MoRF 残基（标签 0）改成 -1，不进损失。
    MoRF 残基、更远的非 MoRF 残基、本来就是 -1 的位置都不变；k <= 0 或这条蛋白没有 MoRF 时原样返回。"""
    k = int(k)
    lab = np.asarray(p.labels)
    pos = np.flatnonzero(lab == 1)
    if k <= 0 or pos.size == 0:
        return p
    n = len(lab)
    idx = np.arange(n)
    j = np.searchsorted(pos, idx)
    far = n + k + 1
    left = np.where(j > 0, idx - pos[np.maximum(j - 1, 0)], far)
    right = np.where(j < len(pos), pos[np.minimum(j, len(pos) - 1)] - idx, far)
    new = lab.copy()
    new[(lab == 0) & (np.minimum(left, right) <= k)] = -1
    return dataclasses.replace(p, labels=new.astype(lab.dtype))


def _adv_prepare(batch: Dict, keys: List[str], need_grad: bool) -> None:
    """嵌入转成 fp32（模型里本来也会转）；对抗扰动要对嵌入求梯度。"""
    for k in keys:
        batch[k] = batch[k].float().requires_grad_(need_grad)


@torch.no_grad()
def _adv_perturb(batch: Dict, keys: List[str], mode: str, eps: float) -> Dict[str, torch.Tensor]:
    """返回加了扰动的嵌入 {键: x + r}。
    方向：fgm = 原损失对嵌入的梯度（让损失上升最快的方向）；rand = 随机方向（对照，大小完全相同）。
    大小：每个残基的扰动长度 = eps × 这个残基嵌入自身的长度；padding 位置不扰动。"""
    m = batch["mask"][..., None].float()
    out = {}
    for k in keys:
        x = batch[k].detach()
        if mode == "fgm":
            g = batch[k].grad
            d = torch.zeros_like(x) if g is None else torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            d = torch.randn_like(x)
        d = d * m
        out[k] = x + d * (eps * x.norm(dim=-1, keepdim=True) / d.norm(dim=-1, keepdim=True).clamp_min(1e-12))
    return out


def _adv_backward(model, batch: Dict, keys: List[str], cfg: Config, loss_fn, use_amp: bool,
                  device: torch.device, scaler) -> float:
    """在嵌入上加扰动，再算一遍损失并反传（梯度累加到原损失的梯度上），返回扰动后的损失。"""
    pert = _adv_perturb(batch, keys, cfg.adv, cfg.adv_eps)
    with torch.autocast(device_type=device.type, enabled=use_amp):
        loss_a = loss_fn(model({**batch, **pert}), batch["labels"])
    w = cfg.adv_weight * loss_a
    (scaler.scale(w) if scaler is not None else w).backward()
    return float(loss_a.detach().item())


class _GradReverse(torch.autograd.Function):
    """梯度反转层：前向原样输出；反向把梯度乘以 −coeff（主干因此学着"骗过"判别器）。"""

    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.coeff * g, None


def _cdan_discriminator(d: int, hidden: int) -> torch.nn.Module:
    """CDAN 的判别器：输入 = 残基表示 f（d 维）和预测 g = [1−p, p] 的外积（2d 维），输出"来自源域"的打分。"""
    return torch.nn.Sequential(torch.nn.Linear(2 * d, hidden), torch.nn.ReLU(), torch.nn.Dropout(0.5),
                               torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Dropout(0.5),
                               torch.nn.Linear(hidden, 1))


def _cdan_loss(disc, zs, ls, ms, zt, lt, mt, coeff: float) -> torch.Tensor:
    """条件域对抗损失（Long 等 2018，PRIDA 用的就是它）：判别器看 f ⊗ g，分辨残基来自源域还是目标域；
    经过梯度反转层，主干被推向"源域、目标域的 (表示, 预测) 分不出来"。
    g 不回传梯度（和 CDAN 原始代码一致）；两个域各占一半权重（残基数不同也公平）；padding 不参与。"""
    def outer(z, logit, m):
        p = torch.sigmoid(logit.float()).detach()
        g = torch.stack([1 - p, p], -1)                                    # [B, L, 2]
        return (z.float().unsqueeze(-1) * g.unsqueeze(-2)).flatten(-2)[m > 0]   # [残基数, 2d]
    hs, ht = outer(zs, ls, ms), outer(zt, lt, mt)
    d = disc(_GradReverse.apply(torch.cat([hs, ht]), coeff)).squeeze(-1).float()
    dev = d.device
    y = torch.cat([torch.ones(len(hs), device=dev), torch.zeros(len(ht), device=dev)])
    w = torch.cat([torch.full((len(hs),), 0.5 / max(len(hs), 1), device=dev),
                   torch.full((len(ht),), 0.5 / max(len(ht), 1), device=dev)])
    return (torch.nn.functional.binary_cross_entropy_with_logits(d, y, reduction="none") * w).sum()


def _cycle(loader):
    while True:
        for b in loader:
            yield b


def _epoch_batches(loader, n_steps: int):
    """一轮取 n_steps 个 batch：n_steps = len(loader) 时和直接遍历 loader 一模一样（只打乱一次）；
    更大时把训练子集循环使用（每过一遍重新打乱），直到凑够 n_steps 步。"""
    done = 0
    while done < n_steps:
        for b in loader:
            yield b
            done += 1
            if done >= n_steps:
                return


def _load_state(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:                                   # 旧版 PyTorch 没有 weights_only
        return torch.load(path, map_location="cpu")


def _param_groups(model: torch.nn.Module, wd: float):
    """偏置、LayerNorm、频率门控、角频率不做权重衰减（标准做法；
    否则门控会被衰减钉在 σ(0)=0.5，失去解释价值）。"""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "gate_logit" in name or name.endswith("theta"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def _make_scaler(enabled: bool):
    """兼容新旧版 PyTorch 的 GradScaler 写法。"""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def predict(model: MoRFNet, ds: ProteinDataset, device: torch.device, amp: bool) -> Dict:
    """全长推理（模型是全卷积的，不需要切窗口），返回按蛋白顺序拼接的结果。"""
    model.eval()
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)
    probs, labels, lengths, pids, emits = [], [], [], [], []
    crf = getattr(model, "crf", None)
    for batch in loader:
        batch = to_device(batch, device)
        # 推理一律 fp32：避免半精度造成大量并列分数，影响 AUC 和阈值
        if crf is not None:
            z, logit = model.trunk(batch)
            p = crf.marginals(logit, z, batch["mask"], batch["length"])[0]
            # 同一个模型的两种读数都存下来：CRF 边际 vs 直接对发射分取 sigmoid。
            # 这样"结构化解码到底有没有用"不用重训就能比。
            emits.append(torch.sigmoid(logit[0, :int(batch["length"][0])].float()).cpu().numpy())
        else:
            logit = model(batch)
            p = torch.sigmoid(logit[0].float())
        n = int(batch["length"][0])
        probs.append(p[:n].float().cpu().numpy())
        labels.append(batch["labels"][0, :n].cpu().numpy())
        lengths.append(n); pids.append(batch["pid"][0])
    out = {"prob": np.concatenate(probs), "label": np.concatenate(labels),
           "length": np.array(lengths), "pid": np.array(pids)}
    if emits:
        out["prob_emit"] = np.concatenate(emits)
    return out


def smooth_probs(prob: np.ndarray, lengths, w: int) -> np.ndarray:
    """逐蛋白做长度为 w 的滑动平均（两端用边缘值补齐），不跨蛋白。"""
    w = int(w)
    if w <= 1:
        return prob
    out, o, k = np.empty_like(prob), 0, np.ones(w) / w
    for n in lengths:
        n = int(n)
        x = np.pad(prob[o:o + n], (w // 2, w // 2), mode="edge")
        out[o:o + n] = np.convolve(x, k, mode="valid")
        o += n
    return out


def pick_smooth_window(cfg: Config, pv: Dict):
    """只用验证集选平滑窗口；候选里包含 1（不平滑），所以平滑没用时会自动选 1。"""
    if str(cfg.smooth) != "auto":
        return int(cfg.smooth), {}
    scores = {}
    for w in sorted(set(int(x) for x in cfg.smooth_windows) | {1}):
        scores[w] = all_metrics(pv["label"], smooth_probs(pv["prob"], pv["length"], w), None)["auc"]
    best_w = max(scores, key=lambda k: (scores[k], -k))   # 并列时取更小的窗口
    return best_w, scores


def _lr_lambda(total: int, warm: int):
    def f(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        prog = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return f


def evaluate_sets(pred_by_set: Dict[str, Dict], thr: float, n_boot: int, seed: int) -> Dict:
    res = {}
    for name, pr in pred_by_set.items():
        for sub, fn in subsets_for(name).items():
            lab = pr["label"]
            if fn is not None:   # 子集变换要逐蛋白做（片段不能跨蛋白）
                parts, o = [], 0
                for n in pr["length"]:
                    parts.append(fn(lab[o:o + n])); o += n
                lab = np.concatenate(parts)
            key = name if sub == "all" else f"{name}_{sub}"
            if not ((lab == 1).any() and (lab == 0).any()):
                continue                     # 子集里没有正样本或没有负样本（比如没有长 MoRF），AUC 无定义
            m = all_metrics(lab, pr["prob"], thr)
            if n_boot > 0:
                m.update(bootstrap_ci(lab, pr["prob"], pr["length"], n_boot=n_boot, seed=seed))
            res[key] = m
    return res


def run_name(cfg: Config) -> str:
    """单次划分：<tag>_s<seed>；交叉验证：<tag>_f<fold>_s<seed>。"""
    if int(cfg.cv_folds) > 1:
        return f"{cfg.tag}_f{cfg.fold}_s{cfg.seed}"
    return f"{cfg.tag}_s{cfg.seed}"


def run_training(cfg: Config, verbose: bool = True) -> Dict:
    t0 = time.time()
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    run_dir = os.path.join(cfg.out_dir, "runs", run_name(cfg))
    os.makedirs(run_dir, exist_ok=True)
    cfg.save(os.path.join(run_dir, "config.json"))

    # ---- 数据 ----
    sets = load_sets(cfg.raw_dir, cfg.data_files, [cfg.train_set] + cfg.eval_sets, verbose=verbose,
                     train_labels=cfg.train_labels)
    if verbose:
        leakage_report(sets, cfg.train_set)
    tr_p, va_p = _fixed_split(cfg, sets[cfg.train_set])
    held_p, n_pool = None, len(tr_p)
    if cfg.cv_select:
        tr_p, va_p, held_p = _select_split(cfg, sets[cfg.train_set], tr_p, va_p)
        held_p, n_pool = [p.eval_view() for p in held_p], len(tr_p)
    tr_p = _subsample(cfg, tr_p)
    if verbose and (held_p is not None or len(tr_p) < n_pool):
        print(f"[subset] 训练 {len(tr_p)}/{n_pool} 条（train_frac={cfg.train_frac}，子集种子 {cfg.train_subset_seed}）"
              + (f"；选择折 {len(va_p)} 条，评估折 {len(held_p)} 条（只在最后预测）" if held_p is not None else ""))
    # 验证和测试一律用评估标签（MC2 = 高质量标注）；训练用 train_labels 指定的那套
    va_p = [p.eval_view() for p in va_p]
    for n in cfg.eval_sets:
        sets[n] = [p.eval_view() for p in sets[n]]
    va_orig = None
    if int(cfg.flank_mask) > 0:
        # 第 61 步：训练折、验证折里，核心两侧 flank_mask 个残基以内的非 MoRF 残基不算负样本
        # （早停、选平滑窗口也按这套标签）；preds_VAL.npz 仍存原始标签
        va_orig = va_p
        tr_p = [mask_flanks(p, cfg.flank_mask) for p in tr_p]
        va_p = [mask_flanks(p, cfg.flank_mask) for p in va_p]
        if verbose:
            n0 = sum(int((p.labels == 0).sum()) for p in tr_p)
            n1 = sum(int((p.labels == -1).sum()) for p in tr_p)
            print(f"[flank] 核心两侧 {cfg.flank_mask} 个残基以内的非 MoRF 残基不算负样本：训练折剩 {n0} 个负样本，"
                  f"不进损失的 {n1} 个")
    store = FeatureStore(cfg.feat_dir)
    allp = [p for v in sets.values() for p in v]
    ft = cfg.finetune                                  # 被微调的语言模型现场计算，不读提前存好的特征
    store_plms = [m for m in cfg.plms if m != ft]
    miss = store.check(allp, store_plms)
    if any(miss.values()):
        raise RuntimeError(f"特征缺失 {miss}，请先运行 scripts/01_extract_features.py")
    plm_dims = {m: int(store.load(m, allp[0].key, mmap=True).shape[1]) for m in store_plms}
    max_len = cfg.ft_max_len if ft else cfg.max_len

    ds_tr = ProteinDataset(tr_p, store, store_plms, cfg.use_aa_scales, max_len, train=True, seed=cfg.seed)
    ds_va = ProteinDataset(va_p, store, store_plms, cfg.use_aa_scales)
    ds_te = {n: ProteinDataset(sets[n], store, store_plms, cfg.use_aa_scales) for n in cfg.eval_sets}
    lens = [min(len(p.seq), max_len) for p in tr_p]
    loader = DataLoader(ds_tr, batch_sampler=LengthBucketSampler(lens, cfg.batch_size, cfg.seed),
                        collate_fn=collate, num_workers=0)

    n_pos = sum(int((p.labels == 1).sum()) for p in tr_p)
    n_neg = sum(int((p.labels == 0).sum()) for p in tr_p)
    pos_w = cfg.pos_weight if cfg.pos_weight > 0 else math.sqrt(n_neg / max(n_pos, 1))

    # ---- 模型 ----
    if ft:
        from ..models.finetune import FTMoRFNet, trainable_state
        model = FTMoRFNet(cfg, plm_dims).to(device)
        plm_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("plm.")]
        groups = _param_groups(model.head, cfg.weight_decay) + [
            {"params": plm_params, "lr": cfg.lr_plm, "weight_decay": 0.0}]
        snapshot = lambda: trainable_state(model)          # noqa: E731  只存 LoRA + 预测头
        ft_note = (f"  微调 {ft}：LoRA r={cfg.lora_r} 加在 {model.plm.n_lora} 个线性层上，"
                   f"LoRA 参数 {sum(p.numel() for p in plm_params) / 1e6:.2f}M，训练裁剪 {max_len}")
    else:
        model = MoRFNet(cfg, plm_dims).to(device)
        groups = _param_groups(model, cfg.weight_decay)
        snapshot = lambda: {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}  # noqa: E731
        ft_note = ""
    if cfg.init_from:
        # 整体微调：结构和来源模型完全相同，全部权重都从来源模型出发，全部参与训练
        if ft or cfg.decoder == "semicrf":
            raise ValueError("init_from 只用于冻结特征 + sigmoid 解码器的 MoRFNet")
        model.load_state_dict(_load_state(cfg.init_from))
        ft_note += f"  从 {cfg.init_from} 出发整体微调"
    cdan = cfg.cdan_target
    if cdan:
        # 目标域只用序列，标签全部换成 -1（裁剪、采样都看不到标签）；判别器不存进 model.pt
        #   heldout = 评估折；files = cdan_files 里的蛋白，去掉评估折、选择折那两组
        if cdan not in ("heldout", "files") or held_p is None or cfg.adv or ft or cfg.decoder == "semicrf":
            raise ValueError("cdan_target 只支持 'heldout' / 'files'（需要 cv_select='next'），且不能和 adv、微调、CRF 同时用")
        if cdan == "files":
            skip = {p.group for p in held_p} | {p.group for p in va_p}
            src_t = [p for p in read_any(cfg.cdan_files, cfg.train_labels, verbose=False) if p.group not in skip]
            if not src_t or not skip:
                raise ValueError(f"cdan_files 去掉 {sorted(skip)} 两组之后没有蛋白了（或数据没有分组）")
            miss_t = store.check(src_t, store_plms)
            if any(miss_t.values()):
                raise RuntimeError(f"CDAN 目标域的特征缺失 {miss_t}")
        else:
            src_t = held_p
        tgt_p = [Protein(p.pid, p.seq, np.full(len(p.seq), -1, dtype=np.int8), p.source, None, p.group) for p in src_t]
        ds_tg = ProteinDataset(tgt_p, store, store_plms, cfg.use_aa_scales, max_len, train=True, seed=cfg.seed + 1)
        tg_iter = _cycle(DataLoader(ds_tg, batch_sampler=LengthBucketSampler(
            [min(len(p.seq), max_len) for p in tgt_p], cfg.batch_size, cfg.seed + 1), collate_fn=collate, num_workers=0))
        disc = _cdan_discriminator(cfg.d_model, cfg.cdan_hidden).to(device)
        groups = groups + [{"params": list(disc.parameters()), "weight_decay": cfg.weight_decay}]
        ft_note += (f"  CDAN：目标域 = {'评估折' if cdan == 'heldout' else '指定文件里的'} {len(tgt_p)} 条（无标签），"
                    f"权重 {cfg.cdan_weight}")
    if verbose:
        print(f"[model] {cfg.tag}: head={cfg.head} spec={cfg.spec_mode} plms={cfg.plms} "
              f"参数量 {count_params(model) / 1e6:.2f}M  pos_weight={pos_w:.2f}  device={device}" + ft_note)
    opt = torch.optim.AdamW(groups, lr=cfg.lr)
    # 每一轮的更新步数：默认 = 子集遍历一遍；equal_steps = 补到和 100% 训练池一样多（各比例的更新步数、早停粒度都相同）
    steps_ep = max(len(loader), math.ceil(n_pool / cfg.batch_size)) if cfg.equal_steps else len(loader)
    total = cfg.epochs * steps_ep
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(total, int(cfg.warmup_frac * total)))
    # 微调时 ProtT5 自己以 bf16 运行，预测头用 fp32，不再套 fp16 自动混合精度（T5 用 fp16 容易溢出）
    use_amp = cfg.amp and device.type == "cuda" and not ft
    scaler = _make_scaler(use_amp) if device.type == "cuda" else None
    loss_fn = (lambda lo, la: masked_focal(lo, la, pos_w, cfg.focal_gamma)) if cfg.loss == "focal" \
        else (lambda lo, la: masked_bce(lo, la, pos_w))
    crf = cfg.decoder == "semicrf"
    adv, adv_keys = cfg.adv, [f"feat_{m}" for m in store_plms]
    if adv and (adv not in ("fgm", "rand") or crf or ft or cfg.adv_eps <= 0):
        raise ValueError(f"adv={adv!r}：只支持 fgm / rand，adv_eps 要 > 0，且只用于 sigmoid 解码器、冻结特征")

    # ---- 训练循环（早停看验证集 AUC）----
    best, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(1, cfg.epochs + 1):
        model.train()
        tot, nb, t_ep, tot_a, tot_d = 0.0, 0, time.time(), 0.0, 0.0
        if cdan:
            disc.train()
        for batch in _epoch_batches(loader, steps_ep):
            batch = to_device(batch, device)
            if adv:
                _adv_prepare(batch, adv_keys, adv == "fgm")
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model.trunk(batch) if crf else (model(batch, return_z=True) if cdan else model(batch))
            if crf:
                # 片段级似然 + 一点辅助 BCE：后者稳住发射分的标定，也顺带处理类别不平衡
                z, logits = out
                loss = model.crf.nll(logits, z, batch["mask"], batch["labels"], batch["length"].to(device))
                if cfg.crf_aux_bce > 0:
                    loss = loss + cfg.crf_aux_bce * loss_fn(logits, batch["labels"])
            else:
                logits = out[1] if cdan else out
                loss = loss_fn(logits, batch["labels"])
            if cdan:
                # 同时取一批目标域（无标签）蛋白；梯度反转系数按训练进度从 0 升到 1（CDAN / DANN 的标准做法）
                bt = to_device(next(tg_iter), device)
                coeff = 2.0 / (1.0 + math.exp(-10.0 * sched.last_epoch / max(total, 1))) - 1.0
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    zt, lt = model(bt, return_z=True)
                    l_dom = _cdan_loss(disc, out[0], logits, batch["mask"], zt, lt, bt["mask"], coeff)
                loss = loss + cfg.cdan_weight * l_dom
                tot_d += l_dom.detach().item()
            opt.zero_grad(set_to_none=True)
            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                if adv:
                    tot_a += _adv_backward(model, batch, adv_keys, cfg, loss_fn, use_amp, device, scaler)
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                if adv:
                    tot_a += _adv_backward(model, batch, adv_keys, cfg, loss_fn, use_amp, device, None)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
            sched.step()
            tot += loss.detach().item(); nb += 1
        pv = predict(model, ds_va, device, use_amp)
        vm = all_metrics(pv["label"], pv["prob"], None)
        hist.append({"epoch": ep, "loss": tot / max(nb, 1), "val_auc": vm["auc"], "val_aupr": vm["aupr"]})
        if adv:
            hist[-1]["loss_adv"] = tot_a / max(nb, 1)
        if cdan:
            hist[-1]["loss_dom"] = tot_d / max(nb, 1)
        if verbose:
            print(f"  ep{ep:3d}  loss {tot / max(nb, 1):.4f}  val AUC {vm['auc']:.4f}  AUPR {vm['aupr']:.4f}"
                  + (f"  扰动后 loss {tot_a / max(nb, 1):.4f}" if adv else "")
                  + (f"  域判别 loss {tot_d / max(nb, 1):.4f}" if cdan else "")
                  + f"  （本轮含验证 {time.time() - t_ep:.0f} 秒）")
        if vm["auc"] > best + 1e-4:
            best, bad = vm["auc"], 0
            best_state = snapshot()
        else:
            bad += 1
            if bad >= cfg.patience:
                if verbose:
                    print(f"  早停：验证 AUC {cfg.patience} 个 epoch 没提升")
                break

    model.load_state_dict(best_state, strict=not ft)     # 微调版只存了可训练部分，原始权重本来就在模型里
    torch.save(best_state, os.path.join(run_dir, "model.pt"))
    with open(os.path.join(run_dir, "history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0])); w.writeheader(); w.writerows(hist)

    # ---- 阈值只在验证集上定 ----
    pv = predict(model, ds_va, device, use_amp)
    w, w_scores = pick_smooth_window(cfg, pv)
    pv["prob_raw"] = pv["prob"]
    pv["prob"] = smooth_probs(pv["prob"], pv["length"], w)
    thr = best_mcc_threshold(pv["label"], pv["prob"])
    pv_save = pv
    if va_orig is not None:
        lab_orig = np.concatenate([np.asarray(p.labels) for p in va_orig])
        if len(lab_orig) != len(pv["label"]):
            raise RuntimeError("验证折的原始标签和预测长度对不上")
        pv_save = dict(pv)
        pv_save["label"] = lab_orig.astype(pv["label"].dtype)
    np.savez_compressed(os.path.join(run_dir, "preds_VAL.npz"), **pv_save)
    vmet = all_metrics(pv["label"], pv["prob"], thr)
    if verbose:
        extra = (f"（验证集选定平滑窗口 w={w}；不平滑时 AUC {w_scores[1]:.4f}）" if w_scores
                 else f"（固定平滑窗口 w={w}）")
        print(f"[val]  AUC {vmet['auc']:.4f}  AUPR {vmet['aupr']:.4f}  MCC {vmet['mcc']:.4f}  {extra}")
    preds = {}
    for n, ds in ds_te.items():
        preds[n] = predict(model, ds, device, use_amp)
        preds[n]["prob_raw"] = preds[n]["prob"]
        preds[n]["prob"] = smooth_probs(preds[n]["prob"], preds[n]["length"], w)
        np.savez_compressed(os.path.join(run_dir, f"preds_{n}.npz"), **preds[n])
    metrics = {"val": vmet, "threshold": thr, "smooth_window": w,
               "val_auc_by_window": {str(k): v for k, v in w_scores.items()},
               "test": evaluate_sets(preds, thr, cfg.n_boot, cfg.seed),
               "n_params": count_params(model), "best_val_auc": best,
               "minutes": (time.time() - t0) / 60}
    if va_orig is not None:
        mo = all_metrics(pv_save["label"], pv["prob"], None)
        metrics.update(flank_mask=int(cfg.flank_mask), val_orig_labels={k: mo[k] for k in ("auc", "aupr")})
    if held_p is not None or len(tr_p) < n_pool:
        metrics.update(n_train=len(tr_p), n_train_pool=n_pool)
    if cfg.equal_steps:
        metrics.update(steps_per_epoch=steps_ep, epochs_run=len(hist))
    if held_p is not None:
        # 评估折：训练、早停、平滑窗口、阈值都没用过它；只在这里预测一次（平滑窗口用选择折上选的 w）
        ph = predict(model, ProteinDataset(held_p, store, store_plms, cfg.use_aa_scales), device, use_amp)
        ph["prob_raw"] = ph["prob"]
        ph["prob"] = smooth_probs(ph["prob"], ph["length"], w)
        np.savez_compressed(os.path.join(run_dir, "preds_HELDOUT.npz"), **ph)
        metrics["held_out"] = {k: all_metrics(ph["label"], ph["prob"], None)[k] for k in ("auc", "aupr")}
        metrics["minutes"] = (time.time() - t0) / 60
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    if verbose and cfg.blind_test:
        print("[test] 盲测模式：测试集预测已存盘，开发阶段不显示")
    elif verbose:
        for k, m in metrics["test"].items():
            ci = m.get("auc_ci")
            ci_s = f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
            print(f"[test] {k:14s} AUC {m['auc']:.4f}{ci_s}  AUPR {m['aupr']:.4f}  MCC {m['mcc']:.4f}"
                  f"  FPR@TPR0.3 {m['fpr@tpr0.3']:.4f}")
    if verbose:
        mem = (f"，显存峰值 {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB"
               if device.type == "cuda" else "")
        print(f"[done] {run_dir}  用时 {metrics['minutes']:.1f} 分钟{mem}")
    return {"run_dir": run_dir, "metrics": metrics}
