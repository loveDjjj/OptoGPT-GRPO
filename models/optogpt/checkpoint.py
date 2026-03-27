"""OptoGPT 基座模型加载与基础能力封装。"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from core.transformer import make_model_I


def resolve_device(device_arg: str = "auto", local_rank: int | None = None) -> torch.device:
    """解析运行设备。

    约定：
    - 多卡场景下优先绑定 `local_rank` 对应的 GPU；
    - 单卡场景下支持 `auto/cpu/cuda:0` 等显式写法。
    """

    if local_rank is not None and torch.cuda.is_available():
        return torch.device(f"cuda:{int(local_rank)}")
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_optogpt_checkpoint(checkpoint_path: str | Path, device: torch.device) -> dict:
    """加载 OptoGPT checkpoint，并兼容不同版本的 torch.load 参数。"""

    checkpoint_path = str(checkpoint_path)
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


class OptoGPTModel(nn.Module):
    """OptoGPT 基座模型统一封装。

    这层只负责四件事：
    1. 加载 checkpoint；
    2. 管理词表、输入光谱维度与最大长度；
    3. 提供生成与 teacher forcing 打分所需的基础方法；
    4. 在多卡场景下包装 DDP。

    这里不再引入任何 RL/Reward 语义，后续所有评测和训练都基于这层。
    """

    def __init__(self, checkpoint_path: str | Path, device: str | torch.device = "auto") -> None:
        super().__init__()
        self.device = resolve_device(device) if not isinstance(device, torch.device) else device
        self.checkpoint_path = str(checkpoint_path)
        self.checkpoint = load_optogpt_checkpoint(self.checkpoint_path, self.device)
        self.ckpt_args = self.checkpoint["configs"]

        self.raw_model = make_model_I(
            self.ckpt_args.spec_dim,
            self.ckpt_args.struc_dim,
            self.ckpt_args.layers,
            self.ckpt_args.d_model,
            self.ckpt_args.d_ff,
            self.ckpt_args.head_num,
            self.ckpt_args.dropout,
        ).to(self.device)
        self.raw_model.load_state_dict(self.checkpoint["model_state_dict"])
        self.raw_model.eval()

        # `self.model` 在单卡时指向原始模型，在多卡时会被 DDP 包装。
        self.model: nn.Module = self.raw_model

        self.spec_type = getattr(self.ckpt_args, "spec_type", "R_T")
        self.spec_dim = int(self.ckpt_args.spec_dim)
        self.max_len = int(self.ckpt_args.max_len)
        self.struc_word_dict = dict(self.ckpt_args.struc_word_dict)
        self.struc_index_dict = dict(self.ckpt_args.struc_index_dict)
        self.vocab_size = len(self.struc_word_dict)
        self.pad_token = "PAD" if "PAD" in self.struc_word_dict else "EOS"
        self.eos_token = "EOS"
        self.bos_token = "BOS"

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.model(src, tgt, src_mask, tgt_mask)

    @property
    def generator(self):
        """生成头始终从原始模型读取，保证保存/导出与 DDP 解耦。"""

        return self.raw_model.generator

    def configure_distributed(self, local_rank: int | None = None) -> None:
        """在需要时启用 DDP。

        当前项目模型规模不大，最合适的并行方式是“每卡一份完整模型”的数据并行，
        而不是模型并行。这里默认关闭 `broadcast_buffers`，减少无意义同步。
        """

        if isinstance(self.model, DDP):
            return
        if self.device.type == "cuda":
            self.model = DDP(
                self.raw_model,
                device_ids=[int(local_rank)] if local_rank is not None else None,
                output_device=int(local_rank) if local_rank is not None else None,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        else:
            self.model = DDP(
                self.raw_model,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )

    def trainable_parameters(self):
        """返回真实可训练参数。

        注意这里始终从 `raw_model` 取参数，避免外层 DDP 包装影响优化器构建。
        """

        return self.raw_model.parameters()

    def export_checkpoint(self, path: str | Path, extra_state: Optional[dict] = None) -> None:
        """按当前项目统一格式导出 checkpoint。"""

        payload = {
            "model_state_dict": self.raw_model.state_dict(),
            "configs": self.ckpt_args,
            "source_checkpoint": self.checkpoint_path,
        }
        if extra_state:
            payload.update(extra_state)

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output_path)

    def adapt_target_spectrum(self, spectrum: Sequence[float]) -> np.ndarray:
        """把运行时光谱适配到 checkpoint 期望的输入维度。"""

        spectrum = np.asarray(spectrum, dtype=np.float32).reshape(-1)
        if spectrum.size == self.spec_dim:
            return spectrum

        if spectrum.size != 142:
            raise ValueError(f"光谱长度 {spectrum.size} 与 checkpoint 输入维度 {self.spec_dim} 不匹配。")

        half = spectrum.size // 2
        if self.spec_type == "R":
            spectrum = spectrum[:half]
        elif self.spec_type == "T":
            spectrum = spectrum[half:]
        elif self.spec_type == "R_T":
            pass
        else:
            raise ValueError(f"不支持的 spec_type: {self.spec_type}")

        if spectrum.size != self.spec_dim:
            raise ValueError(f"适配后的光谱长度 {spectrum.size} 与 checkpoint 输入维度 {self.spec_dim} 不匹配。")
        return spectrum.astype(np.float32)

    def target_to_tensor(self, spectrum: Sequence[float]) -> torch.Tensor:
        """把单条目标光谱转成模型输入张量。"""

        adapted = self.adapt_target_spectrum(spectrum)
        return torch.from_numpy(adapted).to(device=self.device, dtype=torch.float32).view(1, 1, -1)

    def target_to_tensor_batch(self, spectrum: Sequence[float], batch_size: int) -> torch.Tensor:
        """把同一条光谱复制成 batch。"""

        src = self.target_to_tensor(spectrum)
        if batch_size <= 1:
            return src
        return src.expand(batch_size, -1, -1).contiguous()

    def targets_to_tensor_batch(self, spectra: Sequence[Sequence[float]]) -> torch.Tensor:
        """把多条不同光谱堆叠成一个 batch。"""

        if len(spectra) == 0:
            return torch.empty((0, 1, self.spec_dim), dtype=torch.float32, device=self.device)
        adapted = [self.adapt_target_spectrum(spectrum) for spectrum in spectra]
        src = torch.from_numpy(np.asarray(adapted, dtype=np.float32)).to(device=self.device, dtype=torch.float32)
        return src.unsqueeze(1)

    def prompt_ids(self, start_symbol: str = "BOS", start_mat: Optional[str] = None) -> List[int]:
        """构造解码前缀。"""

        ids = [int(self.struc_word_dict[start_symbol])]
        if start_mat is not None:
            ids.append(int(self.struc_word_dict[start_mat]))
        return ids

    def token_id_to_str(self, token_id: int) -> str:
        return str(self.struc_index_dict[int(token_id)])

    def token_str_to_id(self, token: str) -> int:
        if token not in self.struc_word_dict:
            raise KeyError(f"结构 token 不在词表中: {token}")
        return int(self.struc_word_dict[token])

    def structure_tokens_to_ids(self, structure_tokens: Sequence[str], append_eos: bool = True) -> List[int]:
        """把结构 token 序列编码成词表 id。"""

        token_ids: List[int] = []
        for token in structure_tokens:
            normalized = str(token).strip()
            if not normalized:
                continue
            token_ids.append(self.token_str_to_id(normalized))
        if append_eos:
            token_ids.append(self.token_str_to_id(self.eos_token))
        return token_ids

    def token_ids_to_structure_tokens(self, token_ids: Sequence[int], stop_at_eos: bool = True) -> List[str]:
        """把词表 id 序列还原成结构 token 序列。"""

        structure_tokens: List[str] = []
        for token_id in token_ids:
            token = self.token_id_to_str(int(token_id))
            if token == self.eos_token and stop_at_eos:
                break
            if token in {self.bos_token, self.pad_token, "UNK"}:
                continue
            structure_tokens.append(token)
        return structure_tokens
