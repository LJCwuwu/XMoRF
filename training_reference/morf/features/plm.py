"""蛋白语言模型逐残基嵌入提取。

支持：
  esm2    : ESM-2 650M                                                     → 1280 维
            两种后端，同一套权重、同一个输出（最后一层，经过最终 LayerNorm）：
              fair-esm    : 本地 TORCH_HOME 里已有 esm2_t33_650M_UR50D.pt 时用
              huggingface : 没有本地权重时用，走 HF 镜像下载 facebook/esm2_t33_650M_UR50D
  prott5  : ProtT5-XL-U50 半精度编码器（HuggingFace Rostlab）                → 1024 维
  protbert: ProtBERT（HuggingFace Rostlab，TransFuse 用的就是它）           → 1024 维
  dummy   : 不需要 GPU 的假特征，只给本地自测用                              →   64 维

模型来源可以用环境变量改成本地文件夹（比如在电脑上下载好再上传）：
  MORF_ESM2_HF / MORF_PROTT5_HF / MORF_PROTBERT_HF = 本地模型目录

长序列：切成重叠片段分别跑，再用三角权重做 overlap-add 拼回，
片段边缘（上下文不完整、表示质量差）权重低，片段中心权重高。
"""
from __future__ import annotations

import os
import re
from typing import List, Tuple

import numpy as np

PLM_DIMS = {"esm2": 1280, "esm2_3b": 2560, "prott5": 1024, "prostt5": 1024,
            "protbert": 1024, "dummy": 64}
ESM2_FAIR_CKPT = "esm2_t33_650M_UR50D.pt"


def _chunks(L: int, win: int, stride: int) -> List[tuple]:
    if L <= win:
        return [(0, L)]
    starts = list(range(0, L - win + 1, stride))
    if starts[-1] + win < L:
        starts.append(L - win)
    return [(s, s + win) for s in starts]


def overlap_add(L: int, win: int, stride: int, run_fn) -> np.ndarray:
    """run_fn(start, end) -> [end-start, D] 数组。"""
    acc, wsum = None, np.zeros(L, dtype=np.float64)
    for s, e in _chunks(L, win, stride):
        emb = run_fn(s, e).astype(np.float64)
        n = e - s
        w = np.minimum(np.arange(1, n + 1), np.arange(n, 0, -1)).astype(np.float64)
        if s == 0:            # 序列真实起点不降权
            w[: n // 2] = w.max()
        if e == L:            # 序列真实终点不降权
            w[n // 2:] = w.max()
        if acc is None:
            acc = np.zeros((L, emb.shape[1]), dtype=np.float64)
        acc[s:e] += emb * w[:, None]
        wsum[s:e] += w
    return (acc / wsum[:, None]).astype(np.float32)


class BaseExtractor:
    name = "base"
    win, stride = 1000, 750
    layers = None          # None = 只取最后一层；否则是层号列表，一次前向同时取出

    def __call__(self, seq: str):
        """返回 [L, dim]；指定了 layers 时返回 {特征名: [L, dim]}（名字形如 prott5@18）。

        为什么要能取中间层：掩码语言模型的最后一层是为"还原被遮住的氨基酸"服务的，
        对残基级下游任务往往不是最好的一层；中间层通常更通用。
        取多层只多花显存、不多花前向时间，所以一次提完、分目录存，之后换层不用重提。
        """
        out = overlap_add(len(seq), self.win, self.stride, lambda s, e: self._embed(seq[s:e]))
        if not self.layers:
            return out
        d = out.shape[1] // len(self.layers)
        return {f"{self.name}@{k}": np.ascontiguousarray(out[:, i * d:(i + 1) * d])
                for i, k in enumerate(self.layers)}

    def _embed(self, seq: str) -> np.ndarray:
        raise NotImplementedError

    def source(self) -> Tuple[str, str]:
        """(后端类型, 模型名)：写进特征目录的 _meta.json，防止一个目录里混用两种来源。"""
        return ("none", self.name)


# ---------------------------------------------------------------- ESM-2（fair-esm）
def fair_esm_ready() -> bool:
    """fair-esm 装了，并且本地已经有 650M 权重（不需要联网）时返回 True。"""
    try:
        import esm  # noqa: F401
        import torch
    except Exception:  # noqa
        return False
    return os.path.exists(os.path.join(torch.hub.get_dir(), "checkpoints", ESM2_FAIR_CKPT))


class ESM2Extractor(BaseExtractor):
    name = "esm2"
    win, stride = 1022, 766

    def __init__(self, device: str = "cuda", layer: int = 33, layers=None):
        import esm
        import torch
        self.layers = list(layers) if layers else None
        self.torch = torch
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.model = self.model.eval().to(device)
        if device.startswith("cuda"):
            self.model = self.model.half()
        self.bc = self.alphabet.get_batch_converter()
        self.device, self.layer = device, layer

    def source(self):
        return ("fair-esm", "esm2_t33_650M_UR50D")

    def _embed(self, seq: str) -> np.ndarray:
        torch = self.torch
        _, _, toks = self.bc([("p", seq)])
        assert toks.shape[1] == len(seq) + 2, f"ESM-2 token 数 {toks.shape[1]} ≠ 序列长 {len(seq)}+2"
        want = self.layers or [self.layer]
        with torch.no_grad():
            out = self.model(toks.to(self.device), repr_layers=want)
        reps = [out["representations"][k][0, 1: len(seq) + 1] for k in want]
        return torch.cat(reps, -1).float().cpu().numpy()


# ---------------------------------------------------------------- HuggingFace 系列
class _HFExtractor(BaseExtractor):
    default_id = ""
    env_var = ""

    def __init__(self, device: str = "cuda", layers=None):
        import torch
        self.torch, self.device = torch, device
        self.layers = list(layers) if layers else None
        self.hf_id = os.environ.get(self.env_var, "") or self.default_id
        self.tok, self.model = self._load()
        self.model = self.model.eval().to(device)
        # GPU 用半精度省显存；CPU 上半精度很多算子不支持，转回 float32
        self.model = self.model.half() if device.startswith("cuda") else self.model.float()

    def source(self):
        return ("huggingface", os.path.basename(self.hf_id.rstrip("/")))

    def _prep(self, seq: str) -> str:
        return " ".join(re.sub(r"[UZOB]", "X", seq))

    def _embed(self, seq: str) -> np.ndarray:
        torch = self.torch
        enc = self.tok([self._prep(seq)], add_special_tokens=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=bool(self.layers))
        if not self.layers:
            return self._strip(out.last_hidden_state[0], len(seq)).float().cpu().numpy()
        # hidden_states[0] 是词嵌入层，hidden_states[k] 是第 k 层的输出
        hs = [self._strip(out.hidden_states[k][0], len(seq)) for k in self.layers]
        return torch.cat(hs, -1).float().cpu().numpy()


class ESM2HFExtractor(_HFExtractor):
    """ESM-2 650M 的 HuggingFace 版：权重与 fair-esm 相同，last_hidden_state
    同样是最后一层经过 emb_layer_norm_after 的输出，即 fair-esm 的 representations[33]。"""
    name = "esm2"
    default_id = "facebook/esm2_t33_650M_UR50D"
    env_var = "MORF_ESM2_HF"
    win, stride = 1022, 766

    def _load(self):
        from transformers import AutoTokenizer, EsmModel
        tok = AutoTokenizer.from_pretrained(self.hf_id)
        model = EsmModel.from_pretrained(self.hf_id, add_pooling_layer=False)
        return tok, model

    def _prep(self, seq: str) -> str:     # ESM 分词器直接吃原始序列，不加空格
        return seq

    def _strip(self, h, L):               # 开头 <cls>，末尾 <eos>
        assert h.shape[0] == L + 2, f"ESM-2(HF) token 数 {h.shape[0]} ≠ 序列长 {L}+2，分词没对齐"
        return h[1: L + 1]


class ESM2_3BHFExtractor(ESM2HFExtractor):
    """ESM-2 3B（2560 维）。AdapTrans 用的就是这个规模（ESMFold 里的语言模型）。
    可选：需要“同等模型规模下公平比较”时再提，下载约 11 GB，4090 上半精度推理没问题。"""
    name = "esm2_3b"
    default_id = "facebook/esm2_t36_3B_UR50D"
    env_var = "MORF_ESM2_3B_HF"


class ProtT5Extractor(_HFExtractor):
    name = "prott5"
    default_id = "Rostlab/prot_t5_xl_half_uniref50-enc"
    env_var = "MORF_PROTT5_HF"

    def _load(self):
        from transformers import T5EncoderModel, T5Tokenizer
        tok = T5Tokenizer.from_pretrained(self.hf_id, do_lower_case=False, legacy=True)
        model = T5EncoderModel.from_pretrained(self.hf_id)
        return tok, model

    def _strip(self, h, L):          # T5：末尾一个 </s>
        assert h.shape[0] == L + 1, f"ProtT5 token 数 {h.shape[0]} ≠ 序列长 {L}+1，分词没对齐"
        return h[:L]


class ProstT5Extractor(_HFExtractor):
    """ProstT5：ProtT5 在"氨基酸 ↔ 3Di 结构字母"上继续训练过，嵌入里带结构信息。

    对 MoRF 这个任务是对症的——MoRF 的定义就是"单独时无序、结合时折叠"，
    而三个对手（MoRFchibi 2.0 / TransFuse / AdapTrans）都没有用带结构信息的表示。
    用法按官方模型卡：氨基酸序列要加 <AA2fold> 前缀，输出里第 0 位是前缀、第 1..L 位才是残基。
    """
    name = "prostt5"
    default_id = "Rostlab/ProstT5"
    env_var = "MORF_PROSTT5_HF"

    def _load(self):
        from transformers import T5EncoderModel, T5Tokenizer
        tok = T5Tokenizer.from_pretrained(self.hf_id, do_lower_case=False, legacy=True)
        model = T5EncoderModel.from_pretrained(self.hf_id)
        return tok, model

    def _prep(self, seq: str) -> str:
        return "<AA2fold> " + " ".join(re.sub(r"[UZOB]", "X", seq.upper()))

    def _strip(self, h, L):          # 开头 <AA2fold>，末尾 </s>
        assert h.shape[0] == L + 2, f"ProstT5 token 数 {h.shape[0]} ≠ 序列长 {L}+2，分词没对齐"
        return h[1: L + 1]


class ProtBertExtractor(_HFExtractor):
    name = "protbert"
    default_id = "Rostlab/prot_bert"
    env_var = "MORF_PROTBERT_HF"

    def _load(self):
        from transformers import BertModel, BertTokenizer
        tok = BertTokenizer.from_pretrained(self.hf_id, do_lower_case=False)
        model = BertModel.from_pretrained(self.hf_id)
        return tok, model

    def _strip(self, h, L):          # BERT：开头 [CLS]，末尾 [SEP]
        assert h.shape[0] == L + 2, f"ProtBERT token 数 {h.shape[0]} ≠ 序列长 {L}+2，分词没对齐"
        return h[1: L + 1]


class DummyExtractor(BaseExtractor):
    """自测用：每种氨基酸一个固定随机向量 + 局部平滑 + 噪声，模拟“带上下文的嵌入”。"""
    name = "dummy"

    def __init__(self, device: str = "cpu", dim: int = 64, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.table = rng.randn(21, dim).astype(np.float32)
        self.rng = np.random.RandomState(seed + 1)

    def source(self):
        return ("dummy", "dummy")

    def __call__(self, seq: str) -> np.ndarray:
        from ..data.parse import AA_TO_IDX
        x = self.table[[AA_TO_IDX.get(a, 20) for a in seq]]
        k = np.ones(5) / 5
        sm = np.stack([np.convolve(x[:, j], k, mode="same") for j in range(x.shape[1])], 1)
        return (0.7 * x + 0.3 * sm + 0.1 * self.rng.randn(*x.shape)).astype(np.float32)


EXTRACTORS = {"esm2_3b": ESM2_3BHFExtractor, "prott5": ProtT5Extractor,
              "prostt5": ProstT5Extractor, "protbert": ProtBertExtractor,
              "dummy": DummyExtractor}


def make_extractor(name: str, device: str = "cuda", esm_backend: str = "auto",
                   layers=None) -> BaseExtractor:
    """esm_backend: auto（有本地 fair-esm 权重就用，否则走 HuggingFace）/ fair / hf
    layers: 层号列表；给了就一次前向同时取这几层，分别存成 <名字>@<层号>。"""
    if name == "esm2":
        if esm_backend == "auto":
            esm_backend = "fair" if fair_esm_ready() else "hf"
        print(f"[extract] ESM-2 后端: {'fair-esm（本地权重）' if esm_backend == 'fair' else 'HuggingFace'}")
        return (ESM2Extractor(device, layers=layers) if esm_backend == "fair"
                else ESM2HFExtractor(device, layers=layers))
    if name not in EXTRACTORS:
        raise KeyError(f"未知语言模型 {name}，可选: esm2, {', '.join(EXTRACTORS)}")
    if name == "dummy":
        return EXTRACTORS[name](device=device)
    return EXTRACTORS[name](device=device, layers=layers)
