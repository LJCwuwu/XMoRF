"""全局配置：一个 dataclass 管所有超参，外加消融预设。

用法：
    cfg = Config()                          # 默认 = 完整模型
    cfg = Config.from_preset("A2_esm_t5_mlp")
    cfg.update({"lr": 3e-4})                # 命令行 --set lr=3e-4 也走这里
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class Config:
    # ---------------- 路径 ----------------
    raw_dir: str = "data/MoRFs_AdapTrans/data/raw_seqs1"   # 原始基准集目录（github.com/zhang403/MoRFs_AdapTrans 的布局）
    data_files: Dict[str, str] = field(default_factory=dict)  # 手动指定 {集合名: 文件路径}，空则自动识别
    feat_dir: str = "data/features_v2"                 # 新特征缓存（每个语言模型一个子目录）
    out_dir: str = "outputs_v2"

    # ---------------- 特征 ----------------
    plms: List[str] = field(default_factory=lambda: ["esm2", "prott5"])
    use_aa_scales: bool = False   # 额外输入：20 维 one-hot + 3 个氨基酸标度

    # ---------------- 模型结构 ----------------
    # head: window_mlp = 旧基线（语言模型特征 + 窗口 MLP）
    #       spatial    = 只有空域膨胀卷积分支
    #       spectral   = 只有谱分支
    #       dual       = 空域 + 谱分支（完整模型）
    #       mlp_dual   = 窗口 MLP + 谱分支（LSM 作为插件加到 MLP 基线上）
    head: str = "dual"
    fusion: str = "gate"          # gate（逐通道门控） / concat
    d_model: int = 128
    window_mlp_size: int = 31

    # 空域分支
    kernel_size: int = 3
    dilations: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])

    # 谱分支（LSM，Learnable Spectral Moment）
    # spec_mode: local    = 局部多分辨率谱（本文方法）
    #            global   = 整条序列一次 FFT 再广播（对照：证明“局部”必要）
    #            fixed_hm = 固定氨基酸标度的经典疏水矩（对照：证明“可学习”必要）
    #            off      = 去掉谱特征，只留残差 + 小卷积块（对照：证明 A4 的提升来自 LSM，而不是这个小卷积块）
    spec_mode: str = "local"
    spec_channels: int = 16                    # 可学习“性质通道”数 K
    spec_windows: List[int] = field(default_factory=lambda: [11, 17, 25])  # 多分辨率窗口（奇数）
    spec_freq_step: float = 10.0               # 角频率网格步长（度），0°~180°
    spec_taper: str = "hann"                   # hann / rect
    spec_learn_freq: bool = False              # 角频率是否可学习

    # ---------------- 输出端（解码器）----------------
    # decoder: sigmoid  = 逐残基独立打分（之前所有实验的做法），预测后做滑动平均
    #          semicrf  = 半马尔可夫 CRF：直接对"片段"建模（长度先验 + 起止边界 + 精确 DP），
    #                     逐残基概率改用 CRF 的边际概率，天然平滑，不再依赖滑动平均。
    decoder: str = "sigmoid"
    crf_max_span: int = 48        # MoRF 片段的最大长度；金标准里更长的片段会被拆成相邻几段
    crf_boundary: bool = True     # 片段起点 / 终点各一个可学习的边界分
    crf_len_prior: bool = True    # 可学习的长度先验 ψ(ℓ)
    crf_aux_bce: float = 0.3      # 辅助 BCE 的权重（0 = 纯 CRF）：稳住发射分的标定，并处理类别不平衡

    # ---------------- 正则 ----------------
    dropout: float = 0.3
    input_dropout: float = 0.2
    modality_dropout: float = 0.1              # 训练时随机整块丢掉某个语言模型

    # ---------------- 训练 ----------------
    epochs: int = 40
    batch_size: int = 8
    max_len: int = 1024                        # 训练时随机裁剪长度（推理用全长）
    lr: float = 5e-4
    weight_decay: float = 0.05
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    loss: str = "bce"                          # bce / focal
    pos_weight: float = -1.0                   # -1 = 自动取 sqrt(负/正)
    focal_gamma: float = 2.0
    patience: int = 8
    val_frac: float = 0.15                     # 从 TRAIN 内部切出验证集（绝不用测试集）
    split_seed: int = 0                        # 验证集划分种子（多种子实验时固定）
    # 交叉验证：cv_folds>1 时，TRAIN 按同源家族分成 K 折，第 fold 折做验证、其余训练（见 scripts/09_cv.py）
    # cv_folds=0 = 上面那种单次划分（val_frac）
    cv_folds: int = 0
    fold: int = 0
    cv_predefined: bool = False                # True：用数据自带的分组当折（MC2 的 CV1~CV4，作者按 40% 一致性分好）
    train_labels: str = "mixed"                # MC2 训练标签：mixed = 高+低质量混合（作者做法），hq = 只用高质量
    # 小样本 / 对抗训练实验（第 34 步）。下面 6 项的默认值都等于"不启用"，和之前所有实验完全一样。
    # cv_select="next"：早停、平滑窗口、阈值改在下一折（fold+1）上选，训练只用其余的折；
    #                   第 fold 折只在训练结束后预测一次（preds_HELDOUT.npz），不参与任何选择
    cv_select: str = ""
    train_frac: float = 1.0                    # 训练蛋白只留这一比例（按蛋白随机抽；选择折、评估折不变）
    train_subset_seed: int = 0                 # 抽子集的种子：同一种子先打乱一次再取前 n 条，所以 10% ⊂ 20% ⊂ 50%
    adv: str = ""                              # 嵌入扰动：fgm = 对抗方向（损失对嵌入的梯度），rand = 随机方向（对照）
    adv_eps: float = 0.0                       # 扰动大小：每个残基的嵌入最多改动自身长度的 adv_eps 倍
    adv_weight: float = 1.0                    # 总损失 = 原损失 + adv_weight × 扰动后输入的损失
    # 条件域对抗（CDAN，PRIDA 用的方法；第 36 步）。下面 4 项的默认值都等于"不启用"。
    # cdan_target="heldout"：目标域 = 评估折的蛋白（需要 cv_select="next"），只用序列、不用标签
    # cdan_target="files"（第 40 步）：目标域 = cdan_files 里的蛋白，去掉评估折、选择折那两组，只用序列、不用标签
    cdan_target: str = ""
    cdan_weight: float = 1.0                   # 总损失 = 分类损失 + cdan_weight × 域对抗损失
    cdan_hidden: int = 256                     # 判别器隐藏层宽度
    cdan_files: List[str] = field(default_factory=list)
    # 小样本迁移（第 40 步）。下面 2 项的默认值都等于"不启用"。
    init_from: str = ""                        # 从这个 model.pt 出发整体微调；空 = 随机初始化
    equal_steps: bool = False                  # True：训练比例 <100% 时，每一轮的更新步数补到和 100% 一样多（子集循环使用）
    # 核心两侧不算负样本（第 61 步）。默认 0 = 不启用，和之前所有实验完全一样。
    # k > 0：训练折和验证折里，离最近的 MoRF 残基不超过 k 个位置的非 MoRF 残基改成 -1
    #        （不进损失，也不参与早停和选平滑窗口）；preds_VAL.npz 仍存原始标签，折外预测照常拼
    flank_mask: int = 0
    seed: int = 0                              # 模型初始化/训练种子
    amp: bool = True
    device: str = "auto"

    # ---------------- 评估 ----------------
    train_set: str = "TRAIN"
    eval_sets: List[str] = field(default_factory=lambda: ["TEST464", "EXP53"])
    n_boot: int = 1000
    # 预测平滑：MoRF 是连续片段，对逐残基分数做滑动平均通常有帮助（AdapTrans 的代码里固定用 5 点平均）。
    # auto = 只在【验证集】上从候选窗口里选（含 1 = 不平滑），选定后原样用到测试集；也可填固定整数
    smooth: str = "auto"
    smooth_windows: List[int] = field(default_factory=lambda: [1, 3, 5, 7, 9, 11, 15, 21])
    # 盲测：测试集照常预测、存盘，但开发阶段一律不打印测试集指标，所有选择只看验证集/折外预测；
    # 最终只揭晓一次（MC2 基准默认开启）
    blind_test: bool = False
    tag: str = "full"

    # ---------------- 微调语言模型（路线 2）----------------
    # finetune="prott5"：ProtT5 不再用提前算好的冻结特征，而是在训练里现场计算，并用 LoRA 微调
    #   （原始权重全部冻结，只在注意力的 q/k/v/o 上加低秩适配器，可训练参数只多几百万）。
    # 默认 "" = 不微调，和之前所有实验完全一样。
    finetune: str = ""
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_targets: List[str] = field(default_factory=lambda: ["q", "k", "v", "o"])
    lr_plm: float = 1e-4                       # LoRA 参数的学习率（预测头仍用 lr）
    ft_max_len: int = 512                      # 微调时训练裁剪长度（推理仍用全长，长序列分段拼接）
    ft_chunk: int = 1000                       # 推理时长序列的分段长度和步长（与提特征时一致）
    ft_stride: int = 750
    grad_ckpt: bool = True                     # 梯度检查点：省显存，多花约三成时间
    ft_model: str = ""                         # 模型目录；空 = 环境变量 MORF_PROTT5_HF 或默认的 HF 名字
    ft_dummy: bool = False                     # 自测用：随机初始化的极小 T5，不下载任何东西

    # ---------- 工具方法 ----------
    def update(self, d: Dict[str, Any]) -> "Config":
        for k, v in d.items():
            if not hasattr(self, k):
                raise KeyError(f"未知配置项: {k}")
            setattr(self, k, copy.deepcopy(v))
        return self

    @classmethod
    def from_preset(cls, name: str) -> "Config":
        if name not in PRESETS:
            raise KeyError(f"未知预设 {name}，可选: {list(PRESETS)}")
        cfg = cls().update(PRESETS[name])
        cfg.tag = name
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, encoding="utf-8") as f:
            return cls().update(json.load(f))


# 基准：--bench mc2 一次性切换数据目录、训练集、测试集、输出目录（其他超参不变）
BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "classic": {},
    "mc2": dict(raw_dir="data/MoRFs_AdapTrans/data/raw_seq2", train_set="MC2_TRAIN",
                eval_sets=["MC2_T143", "MC2_T143U", "MC2_T128", "MC2_T106"],
                out_dir="outputs_mc2", cv_predefined=True, blind_test=True,
                smooth_windows=[1, 3, 5, 7, 9, 11, 15, 21, 31, 45]),   # MC2 有长 MoRF，平滑候选加到 45（仍只在验证折上选）
}


def parse_overrides(pairs: List[str]) -> Dict[str, Any]:
    """把 ["lr=3e-4", "plms=esm2,prott5", "amp=false"] 变成字典。"""
    out: Dict[str, Any] = {}
    default = Config()
    for p in pairs or []:
        k, v = p.split("=", 1)
        ref = getattr(default, k)
        if isinstance(ref, bool):
            out[k] = v.lower() in ("1", "true", "yes")
        elif isinstance(ref, int):
            out[k] = int(v)
        elif isinstance(ref, float):
            out[k] = float(v)
        elif isinstance(ref, list):
            items = [x for x in v.split(",") if x]
            if ref and isinstance(ref[0], int):
                items = [int(x) for x in items]
            out[k] = items
        else:
            out[k] = v
    return out


# 消融预设：每一行回答审稿人的一个问题
PRESETS: Dict[str, Dict[str, Any]] = {
    # Q1：旧基线能复现吗？（应接近你之前的 0.81）
    "A1_esm_mlp": dict(plms=["esm2"], head="window_mlp"),
    # Q2：“0.81 是纯序列天花板”成立吗？加第二个语言模型看能否突破
    "A2_esm_t5_mlp": dict(plms=["esm2", "prott5"], head="window_mlp"),
    # Q3：各分支单独贡献
    "A3_spatial_only": dict(head="spatial"),
    "A4_spectral_only": dict(head="spectral"),
    # 完整模型
    "A5_full": dict(head="dual"),
    # Q4：“可学习”有必要吗？换成经典固定疏水矩
    "A6_fixed_hm": dict(head="dual", spec_mode="fixed_hm"),
    # Q5：“局部”有必要吗？换成整条序列的全局 FFT
    "A7_global_fft": dict(head="dual", spec_mode="global"),
    # Q6：“多分辨率”有必要吗？只留一个窗口
    "A8_single_res": dict(head="dual", spec_windows=[17]),
    # Q7：每个语言模型单独的贡献
    "A9_esm_full": dict(plms=["esm2"], head="dual"),
    "A10_t5_full": dict(plms=["prott5"], head="dual"),
    # 可选：角频率可学习（看网络会不会自己学到 ~100°）
    "A11_learn_freq": dict(head="dual", spec_learn_freq=True),
    # 可选：融合方式
    "A12_concat_fusion": dict(head="dual", fusion="concat"),
    # 可选：和 AdapTrans 同等语言模型规模（ESM-2 3B）下的公平比较；需先提 esm2_3b 特征
    "A13_esm3b_t5_full": dict(plms=["esm2_3b", "prott5"], head="dual"),
    # 3B 下的两个对照：MLP 头（同 A2）、只有空域分支（同 A3）→ 和 A2/A3/A5 组成 2×3 设计
    "A15_esm3b_t5_mlp": dict(plms=["esm2_3b", "prott5"], head="window_mlp"),
    "A16_esm3b_t5_spatial": dict(plms=["esm2_3b", "prott5"], head="spatial"),
    # Q8：LSM 当“插件”加到最常用的 语言模型 + 窗口 MLP 基线上，有没有提升？（和 A2 只差 LSM）
    "A17_mlp_lsm": dict(head="mlp_dual"),
    # 路线 2：A4 的结构不变，只把冻结的 ProtT5 特征换成 LoRA 微调的 ProtT5（和 A4 只差“微调”这一处）
    "F1_lora_t5": dict(head="spectral", finetune="prott5", batch_size=4, epochs=10, patience=3),
    # 路线 3（片段结构化输出）：主干完全不动，只把输出端从“逐残基 sigmoid”换成半马尔可夫 CRF。
    #   C1 = 在 A4 上换解码器（和 A4 只差“解码器”这一处）；
    #   C2 = 在“ProtT5 + 窗口 MLP”这个极简基线上换解码器（检验 CRF 的收益是否依赖那个谱头）。
    "C1_semicrf": dict(head="spectral", decoder="semicrf"),
    "C2_semicrf_t5_mlp": dict(head="window_mlp", plms=["prott5"], decoder="semicrf"),
    # 可选：不做预测平滑，看平滑本身贡献多少
    "A14_no_smooth": dict(head="dual", smooth="1"),
}


def config_matches(old: Dict[str, Any], new: Dict[str, Any]) -> bool:
    """旧 run 的 config.json 和现在的配置是否相同（--skip_existing 复用旧结果时用）。
    后来新增的配置项（比如微调相关的）在旧文件里没有：按默认值补齐再比——
    新增项的默认值都等于“不启用”，所以旧 run 仍然能被正确地认出来、复用。"""
    base = json.loads(json.dumps(Config().to_dict()))
    old = json.loads(json.dumps(old))
    new = json.loads(json.dumps(new))
    return all(old.get(k, base.get(k)) == v for k, v in new.items())
