"""嵌套模型：区域网络 r + 核心网络 q。核心是区域里的一部分：P(核心) = P(区域) × P(核心 | 区域) = r·q。

两个网络都是 MoRFNet（各用一份配置）。r 直接用区域标注训练（MC2-ext 的整段结合区域）；
q 只通过乘积 r·q 用经典的核心标注训练（TRAIN421）。一个模型给两路输出：区域 = r，核心 = r·q（核心永远不超过区域）。

损失：一个 batch 内，两部分各自按有效残基取平均，再相加。
  MC2-ext 蛋白：r 上的加权 BCE；正样本权重 √(负/正)，按 MC2-ext 训练蛋白的训练标签数。
  经典蛋白：p = r·q 上的加权 BCE；正样本权重 √(负/正)，按经典训练蛋白数（全部非核心都算负）：
    核心残基                          −w·log(r·q)                      → r、q 都更新；
    离最近核心 > k 个残基的非核心       −log(1 − r·q)                    → r、q 都更新（当背景）；
    离最近核心 ≤ k 个残基的非核心       −log(1 − r·q)，r 不回传梯度        → 只更新 q
      （这些残基可能和核心在同一段结合区域里：它们只说明“不是核心”，不说明“不是区域”）。
    k = 0：全部非核心都更新 r、q（对照：不截断梯度）。距离计算与 mask_flanks 使用同一算法。
  数值上：log(r·q) = logσ(a) + logσ(b)；log(1 − r·q) = logsumexp(−a, −b, −a−b) − softplus(−a) − softplus(−b)
  （a、b 是两个网络的对数几率；两个式子对任何 a、b 都稳定）。
早停：每轮在验证折上算 MC2-ext 蛋白上 r 的合并 AUC（评估标签）和经典蛋白上 r·q 的合并 AUC（原始标签），取平均。
平滑窗口：区域输出在 MC2-ext 验证蛋白上选（r 那份配置的候选窗口），核心输出在经典验证蛋白上选（q 那份配置的候选窗口）；
  都是“合并 AUC 最高、并列取小的”（和 trainer.pick_smooth_window 一样）。
分折：MC2-ext 用 r 那份配置的分法（作者的组），经典用 q 那份配置的分法（q 的 out_dir 里的分折文件）；两份配置的 fold 必须相同。
其余（AdamW、学习率、warmup + 余弦、AMP、梯度裁剪、训练裁剪长度、按长度分桶的 batch）和 trainer.run_training 同一套写法。
产出（run_dir）：config_r.json、config_q.json、nested_meta.json、history.csv、model.pt、
  preds_VAL_mc2.npz、preds_VAL_cl.npz（两套验证蛋白上的两路输出，平滑和不平滑的都存）、metrics.json（最后写，表示做完）。
"""
from __future__ import annotations

import contextlib
import csv
import dataclasses
import io
import json
import math
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..config import Config
from ..data.dataset import LengthBucketSampler, ProteinDataset, collate, to_device
from ..data.parse import Protein
from ..eval.metrics import all_metrics
from ..features.store import FeatureStore
from ..models.net import MoRFNet, count_params
from .trainer import (_cv_split, _load_state, _lr_lambda, _make_scaler, _param_groups, mask_flanks, pick_device,
                      set_seed, smooth_probs)

SOURCES = ("mc2", "cl")
NEAR = 2            # 经典训练标签里“离最近核心 ≤ k 个残基的非核心”的记号：损失里只更新 q
# r、q 两份配置里必须相同的设置（同一个训练循环、同一份输入特征）
SHARED = ("plms", "use_aa_scales", "feat_dir", "epochs", "batch_size", "max_len", "lr", "weight_decay", "warmup_frac",
          "grad_clip", "patience", "amp", "device", "loss", "pos_weight", "smooth")
VAL_KEYS = ("pid", "label", "length", "prob", "prob_raw", "region", "region_raw", "core", "core_raw", "q_raw")


# ------------------------------------------------------------------ 配置、标签
def check_cfgs(cfg_r: Config, cfg_q: Config, shared=None) -> None:
    """两份配置能不能一起训练；不能就抛 ValueError（说明原因）。"""
    shared = SHARED if shared is None else shared
    diff = [k for k in shared if getattr(cfg_r, k) != getattr(cfg_q, k)]
    if diff:
        raise ValueError(f"r、q 两份配置的这些设置不一样，没法放在一个训练循环里：{diff}")
    for nm, c in (("r", cfg_r), ("q", cfg_q)):
        if (c.finetune or c.decoder != "sigmoid" or c.adv or c.cdan_target or c.init_from or c.cv_select
                or int(c.flank_mask) or float(c.train_frac) < 1.0 or c.equal_steps):
            raise ValueError(f"{nm} 的配置开了微调 / CRF / 对抗 / CDAN / init_from / cv_select / flank_mask / 训练子集之一；"
                             "嵌套训练只支持冻结特征 + sigmoid 解码器")
        if c.loss != "bce" or float(c.pos_weight) > 0:
            raise ValueError(f"{nm} 的损失不是“BCE + 自动 √(负/正)”（loss={c.loss}，pos_weight={c.pos_weight}）")
        if int(c.cv_folds) < 2:
            raise ValueError(f"{nm} 的配置不是交叉验证（cv_folds={c.cv_folds}）")
    if cfg_r.fold != cfg_q.fold or cfg_r.seed != cfg_q.seed:
        raise ValueError(f"r、q 的折、种子要一样（r：{cfg_r.fold}/{cfg_r.seed}，q：{cfg_q.fold}/{cfg_q.seed}）")


def mark_near(p: Protein, k: int) -> Protein:
    """经典训练蛋白：离最近核心残基不超过 k 个位置的非核心残基标成 NEAR（2），其余不变。
    被 mask_flanks(k) 屏蔽掉的残基，正好就是这里标成 2 的。k ≤ 0 或没有核心时原样返回。"""
    k = int(k)
    if k <= 0:
        return p
    lab = np.asarray(p.labels)
    near = (lab == 0) & (np.asarray(mask_flanks(p, k).labels) == -1)
    if not near.any():
        return p
    new = lab.copy()
    new[near] = NEAR
    return dataclasses.replace(p, labels=new.astype(lab.dtype))


def pos_weights(train: List[Tuple[Protein, int]]) -> Dict[str, float]:
    """[(蛋白, 来源编号)] → 每个来源的 √(负/正)（训练标签里的 1 算正，0 和 NEAR 算负）。"""
    out = {}
    for k, s in enumerate(SOURCES):
        labs = [np.asarray(p.labels) for p, kk in train if kk == k]
        npos = sum(int((x == 1).sum()) for x in labs)
        nneg = sum(int(((x == 0) | (x == NEAR)).sum()) for x in labs)
        out[s] = math.sqrt(nneg / max(npos, 1))
    return out


# ------------------------------------------------------------------ 模型、损失
class NestedNet(nn.Module):
    """r（区域网络）、q（核心网络）两个 MoRFNet；forward 返回两个对数几率 (a, b)，都是 [B, L]（padding 处为 0）。"""

    def __init__(self, cfg_r: Config, cfg_q: Config, plm_dims: Dict[str, int]):
        super().__init__()
        self.r = MoRFNet(cfg_r, plm_dims)
        self.q = MoRFNet(cfg_q, plm_dims)

    def forward(self, batch):
        return self.r(batch), self.q(batch)


def log_rq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """log(σ(a)·σ(b))。"""
    return F.logsigmoid(a) + F.logsigmoid(b)


def log_1m_rq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """log(1 − σ(a)·σ(b)) = log(e^−a + e^−b + e^−a−b) − log(1 + e^−a) − log(1 + e^−b)。"""
    return torch.logsumexp(torch.stack([-a, -b, -a - b]), 0) - F.softplus(-a) - F.softplus(-b)


def nested_loss(a: torch.Tensor, b: torch.Tensor, labels: torch.Tensor, source: torch.Tensor, pos_w: Dict[str, float]):
    """一个 batch 的损失。a、b：r、q 的对数几率 [B, L]；labels：[B, L]，取值 1 / 0 / NEAR / -1（-1 不进损失）；
    source：[B]，0 = MC2-ext、1 = 经典。返回 (损失, {来源: 这一部分})；某个来源在这个 batch 里没有有效标签就不算它
    （两个来源都没有 → 损失为 None）。"""
    loss, parts = None, {}
    if a.dtype != torch.float64:                    # 半精度（AMP）一律先转 fp32 再算损失（和 masked_bce 一样）
        a, b = a.float(), b.float()
    for k, s in enumerate(SOURCES):
        rows = source == k
        if not bool(rows.any()):
            continue
        lab = labels[rows]
        m = lab >= 0
        if not bool(m.any()):
            continue
        y, ar = lab[m], a[rows][m]
        if s == "mc2":
            if bool((y == NEAR).any()):
                raise ValueError("MC2-ext 的训练标签里不该有 NEAR")
            part = F.binary_cross_entropy_with_logits(ar, y.to(ar.dtype), pos_weight=torch.tensor(pos_w[s], device=a.device, dtype=ar.dtype))
        else:
            br = b[rows][m]
            a_neg = torch.where(y == NEAR, ar.detach(), ar)          # 离核心近的非核心：r 不回传梯度
            per = torch.where(y == 1, -pos_w[s] * log_rq(ar, br), -log_1m_rq(a_neg, br))
            part = per.mean()
        parts[s] = part
        loss = part if loss is None else loss + part
    return loss, parts


# ------------------------------------------------------------------ 数据
_FEAT: Dict = {}


def _feats(store: FeatureStore, plms, p: Protein):
    """同一条序列的特征整个进程只读一次（两套数据、各次训练之间共用）。"""
    if p.key not in _FEAT:
        _FEAT[p.key] = {m: store.load(m, p.key) for m in plms}
    return _FEAT[p.key]


def make_ds(prots, store: FeatureStore, cfg: Config, train: bool = False, seed: int = 0) -> ProteinDataset:
    ds = ProteinDataset(prots, store, cfg.plms, cfg.use_aa_scales, cfg.max_len if train else None,
                        train=train, preload=False, seed=seed)
    ds.cache = [_feats(store, cfg.plms, p) for p in prots]
    return ds


class SourceDataset(Dataset):
    def __init__(self, base: ProteinDataset, sources: List[int]):
        self.base, self.sources = base, sources

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        it = self.base[i]
        it["source"] = self.sources[i]
        return it


def collate_src(items):
    out = collate(items)
    out["source"] = torch.tensor([it["source"] for it in items])
    return out


def split_fold(cfg_r: Config, cfg_q: Config, data: Dict) -> Dict:
    """{"mc2": (训练, 验证), "cl": (训练, 验证)}：MC2-ext 按 cfg_r 分，经典按 cfg_q 分（同一折）。"""
    with contextlib.redirect_stdout(io.StringIO()):
        return {"mc2": _cv_split(cfg_r, data["mc2"]), "cl": _cv_split(cfg_q, data["cl"])}


# ------------------------------------------------------------------ 预测、平滑窗口
@torch.no_grad()
def predict_nested(model: NestedNet, ds: ProteinDataset, device) -> Dict[str, np.ndarray]:
    """全长推理（batch = 1，fp32，和 trainer.predict 一样）：r、q、r·q（float64）、标签、长度、pid。"""
    model.eval()
    rs, qs, labs, lens, pids = [], [], [], [], []
    for batch in DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate):
        batch = to_device(batch, device)
        a, b = model(batch)
        n = int(batch["length"][0])
        rs.append(torch.sigmoid(a[0, :n].float()).cpu().numpy())
        qs.append(torch.sigmoid(b[0, :n].float()).cpu().numpy())
        labs.append(batch["labels"][0, :n].cpu().numpy())
        lens.append(n)
        pids.append(batch["pid"][0])
    r = np.concatenate(rs).astype(np.float64)
    q = np.concatenate(qs).astype(np.float64)
    return {"r": r, "q": q, "rq": r * q, "label": np.concatenate(labs).astype(np.int64),
            "length": np.array(lens, dtype=np.int64), "pid": np.array(pids)}


def _auc(y, p) -> float:
    y = np.asarray(y)
    m = y >= 0
    if not ((y[m] == 1).any() and (y[m] == 0).any()):
        return float("nan")
    return float(all_metrics(y, p, None)["auc"])


def pick_window(cfg: Config, y, prob, lengths) -> Tuple[int, Dict[int, float]]:
    """候选窗口（含 1 = 不平滑）里取合并 AUC 最高的，并列取小的（和 trainer.pick_smooth_window 一样）。"""
    if str(cfg.smooth) != "auto":
        return int(cfg.smooth), {}
    scores = {w: _auc(y, smooth_probs(prob, lengths, w)) for w in sorted(set(int(x) for x in cfg.smooth_windows) | {1})}
    return max(scores, key=lambda k: (scores[k] if scores[k] == scores[k] else -1.0, -k)), scores


def val_columns(pv: Dict, src: str, w_reg: int, w_core: int) -> Dict[str, np.ndarray]:
    """存盘的列：region / core（平滑后）和 *_raw（不平滑）；prob / prob_raw = 这套数据的“主输出”
    （MC2-ext → 区域，经典 → 核心）。"""
    L = pv["length"]
    region, core = smooth_probs(pv["r"], L, w_reg), smooth_probs(pv["rq"], L, w_core)
    main, main_raw = (region, pv["r"]) if src == "mc2" else (core, pv["rq"])
    return {"pid": pv["pid"], "label": pv["label"], "length": L, "prob": main, "prob_raw": main_raw,
            "region": region, "region_raw": pv["r"], "core": core, "core_raw": pv["rq"], "q_raw": pv["q"]}


# ------------------------------------------------------------------ 训练
def train_nested(cfg_r: Config, cfg_q: Config, k_stop: int, data: Dict, run_dir: str, meta: Dict,
                 verbose: bool = True, shared=None) -> Dict:
    """data = {"mc2": [MC2-ext 蛋白（labels = 训练标签，labels_eval = 评估标签）], "cl": [TRAIN421 蛋白]}；
    cfg_r.fold（= cfg_q.fold）决定哪一折做验证。k_stop：经典里离核心 ≤ k_stop 的非核心只更新 q（0 = 都更新 r、q）。"""
    check_cfgs(cfg_r, cfg_q, shared)
    t0 = time.time()
    set_seed(cfg_r.seed)
    device = pick_device(cfg_r.device)
    os.makedirs(run_dir, exist_ok=True)
    done = os.path.join(run_dir, "metrics.json")
    if os.path.exists(done):                     # 重跑：先删完成标记，中途断了不会被当成“已完成”
        os.remove(done)
    cfg_r.save(os.path.join(run_dir, "config_r.json"))
    cfg_q.save(os.path.join(run_dir, "config_q.json"))
    with open(os.path.join(run_dir, "nested_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    sp = split_fold(cfg_r, cfg_q, data)
    tr_m, va_m = sp["mc2"]
    tr_c, va_c = sp["cl"]
    tr = [(p, 0) for p in tr_m] + [(mark_near(p, k_stop), 1) for p in tr_c]
    va = {"mc2": [p.eval_view() for p in va_m], "cl": [p.eval_view() for p in va_c]}
    if not (tr_m and tr_c and va["mc2"] and va["cl"]):
        raise RuntimeError(f"第 {cfg_r.fold} 折缺数据：MC2-ext 训练 {len(tr_m)} / 验证 {len(va_m)}，"
                           f"经典训练 {len(tr_c)} / 验证 {len(va_c)}")
    store = FeatureStore(cfg_r.feat_dir)
    allp = [p for p, _ in tr] + va["mc2"] + va["cl"]
    miss = store.check(allp, cfg_r.plms)
    if any(miss.values()):
        raise RuntimeError(f"特征缺失 {miss}")
    plm_dims = {m: int(store.load(m, allp[0].key, mmap=True).shape[1]) for m in cfg_r.plms}
    pos_w = pos_weights(tr)
    n_near = sum(int((np.asarray(p.labels) == NEAR).sum()) for p, k in tr if k == 1)
    ds_tr = SourceDataset(make_ds([p for p, _ in tr], store, cfg_r, train=True, seed=cfg_r.seed), [k for _, k in tr])
    lens = [min(len(p.seq), cfg_r.max_len) for p, _ in tr]
    loader = DataLoader(ds_tr, batch_sampler=LengthBucketSampler(lens, cfg_r.batch_size, cfg_r.seed),
                        collate_fn=collate_src, num_workers=0)
    ds_va = {s: make_ds(va[s], store, cfg_r) for s in SOURCES}

    model = NestedNet(cfg_r, cfg_q, plm_dims).to(device)
    groups = _param_groups(model.r, cfg_r.weight_decay) + _param_groups(model.q, cfg_q.weight_decay)
    opt = torch.optim.AdamW(groups, lr=cfg_r.lr)
    total = cfg_r.epochs * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(total, int(cfg_r.warmup_frac * total)))
    use_amp = cfg_r.amp and device.type == "cuda"
    scaler = _make_scaler(use_amp) if device.type == "cuda" else None
    if verbose:
        print(f"[nested] {os.path.basename(run_dir)}  第 {cfg_r.fold} 折  k = {k_stop}  训练 MC2-ext {len(tr_m)} 条 + "
              f"经典 {len(tr_c)} 条（离核心近、只更新 q 的非核心 {n_near:,} 个）  验证 {len(va_m)} + {len(va_c)} 条  "
              f"参数 r {count_params(model.r) / 1e6:.3f}M、q {count_params(model.q) / 1e6:.3f}M  pos_weight "
              + "，".join(f"{s}={w:.2f}" for s, w in pos_w.items()) + f"  device={device}")

    def val_preds() -> Dict[str, Dict]:
        return {s: predict_nested(model, ds_va[s], device) for s in SOURCES}

    best, best_state, best_ep, bad, hist = -1.0, None, 0, 0, []
    for ep in range(1, cfg_r.epochs + 1):
        model.train()
        t_ep = time.time()
        tot = {s: [0.0, 0] for s in SOURCES}
        for batch in loader:
            batch = to_device(batch, device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                a, b = model(batch)
            loss, parts = nested_loss(a, b, batch["labels"], batch["source"], pos_w)
            if loss is None:
                continue
            for s, v in parts.items():
                tot[s][0] += float(v.detach())
                tot[s][1] += 1
            opt.zero_grad(set_to_none=True)
            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_r.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_r.grad_clip)
                opt.step()
            sched.step()
        pv = val_preds()
        sc = {"region_mc2": _auc(pv["mc2"]["label"], pv["mc2"]["r"]), "core_cl": _auc(pv["cl"]["label"], pv["cl"]["rq"]),
              "r_cl": _auc(pv["cl"]["label"], pv["cl"]["r"])}
        v = float(np.nanmean([sc["region_mc2"], sc["core_cl"]]))
        hist.append({"epoch": ep, **{f"loss_{s}": t / max(n, 1) for s, (t, n) in tot.items()},
                     **{f"val_{k}": x for k, x in sc.items()}, "val": v, "seconds": time.time() - t_ep})
        if verbose:
            print(f"  ep{ep:3d}  loss mc2 {hist[-1]['loss_mc2']:.4f} cl {hist[-1]['loss_cl']:.4f}  验证 AUC 区域(MC2-ext) "
                  f"{sc['region_mc2']:.4f}  核心(经典) {sc['core_cl']:.4f}（只用 r {sc['r_cl']:.4f}）  平均 {v:.4f}")
        improved = (not math.isnan(v)) and v > best + 1e-4
        if improved or best_state is None:
            if improved:
                best, best_ep = v, ep
            bad = 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg_r.patience:
                if verbose:
                    print(f"  早停：{cfg_r.patience} 个 epoch 没提升")
                break
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(run_dir, "model.pt"))
    with open(os.path.join(run_dir, "history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0]))
        w.writeheader()
        w.writerows(hist)

    pv = val_preds()
    w_reg, s_reg = pick_window(cfg_r, pv["mc2"]["label"], pv["mc2"]["r"], pv["mc2"]["length"])
    w_core, s_core = pick_window(cfg_q, pv["cl"]["label"], pv["cl"]["rq"], pv["cl"]["length"])
    for s in SOURCES:
        np.savez_compressed(os.path.join(run_dir, f"preds_VAL_{s}.npz"), **val_columns(pv[s], s, w_reg, w_core))
    metrics = {"k_stop": int(k_stop), "best_val": best, "best_epoch": best_ep, "epochs_run": len(hist),
               "val_auc": {"region_mc2": s_reg.get(w_reg, _auc(pv["mc2"]["label"], pv["mc2"]["r"])),
                           "core_cl": s_core.get(w_core, _auc(pv["cl"]["label"], pv["cl"]["rq"])),
                           "r_cl": _auc(pv["cl"]["label"], pv["cl"]["r"])},
               "smooth_window": {"region": w_reg, "core": w_core},
               "val_auc_by_window": {"region": {str(k): x for k, x in s_reg.items()},
                                     "core": {str(k): x for k, x in s_core.items()}},
               "pos_weight": pos_w, "n_near": n_near,
               "n_params": {"r": count_params(model.r), "q": count_params(model.q)},
               "n_train": {"mc2": len(tr_m), "cl": len(tr_c)}, "n_val": {"mc2": len(va_m), "cl": len(va_c)},
               "minutes": (time.time() - t0) / 60}
    with open(done, "w") as f:
        json.dump(metrics, f, indent=2)
    if verbose:
        print(f"[nested] 验证折 AUC 区域 {metrics['val_auc']['region_mc2']:.4f}（平滑 {w_reg}）/ 核心 "
              f"{metrics['val_auc']['core_cl']:.4f}（平滑 {w_core}）  第 {best_ep} 轮最好  用时 {metrics['minutes']:.1f} 分钟")
    return metrics


def load_nested(run_dir: str, plm_dims: Dict[str, int], device) -> Tuple[NestedNet, Config, Config]:
    """Load a nested checkpoint and return the evaluation model and both configurations."""
    cr = Config.load(os.path.join(run_dir, "config_r.json"))
    cq = Config.load(os.path.join(run_dir, "config_q.json"))
    model = NestedNet(cr, cq, plm_dims)
    model.load_state_dict(_load_state(os.path.join(run_dir, "model.pt")))
    return model.to(device).eval(), cr, cq
