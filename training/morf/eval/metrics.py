"""逐残基评估指标。

除了 AUC，还报：
  - AUPR：正样本只占 2%，AUPR 比 AUC 更能区分好坏；
  - FPR@TPR=0.2/0.3/0.4：TransFuse、OPAL 都报这个，必须有才能和它们直接对比；
  - TPR@FPR=0.05/0.10：低假阳性区的召回；
  - MCC/F1/Sn/Sp/Precision：阈值一律在【验证集】上按 MCC 最大选定，再原样用到测试集。
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def _valid(y: np.ndarray, p: np.ndarray):
    m = y >= 0
    return y[m].astype(np.int64), p[m].astype(np.float64)


def best_mcc_threshold(y: np.ndarray, p: np.ndarray) -> float:
    y, p = _valid(y, p)
    fpr, tpr, thr = roc_curve(y, p, drop_intermediate=False)
    P, N = y.sum(), len(y) - y.sum()
    tp, fp = tpr * P, fpr * N
    fn, tn = P - tp, N - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.where(den > 0, (tp * tn - fp * fn) / np.maximum(den, 1e-12), 0.0)
    i = int(np.argmax(mcc[1:]) + 1)          # 跳过 thr[0]=inf
    return float(thr[i])


def threshold_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, float]:
    y, p = _valid(y, p)
    pred = p >= thr
    tp = float(np.sum(pred & (y == 1))); fp = float(np.sum(pred & (y == 0)))
    fn = float(np.sum(~pred & (y == 1))); tn = float(np.sum(~pred & (y == 0)))
    sn = tp / max(tp + fn, 1); sp = tn / max(tn + fp, 1); pre = tp / max(tp + fp, 1)
    f1 = 2 * pre * sn / max(pre + sn, 1e-12)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den > 0 else 0.0
    return {"thr": thr, "mcc": mcc, "f1": f1, "sn": sn, "sp": sp, "precision": pre}


def ranking_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y, p = _valid(y, p)
    out = {"auc": float(roc_auc_score(y, p)), "aupr": float(average_precision_score(y, p)),
           "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum())}
    fpr, tpr, _ = roc_curve(y, p, drop_intermediate=False)
    for t in (0.2, 0.3, 0.4):
        out[f"fpr@tpr{t:.1f}"] = float(fpr[np.searchsorted(tpr, t, side="left")])
    for f in (0.05, 0.10):
        out[f"tpr@fpr{f:.2f}"] = float(tpr[np.searchsorted(fpr, f, side="right") - 1])
    return out


def all_metrics(y: np.ndarray, p: np.ndarray, thr: Optional[float]) -> Dict[str, float]:
    out = ranking_metrics(y, p)
    if thr is not None:
        out.update(threshold_metrics(y, p, thr))
    return out
