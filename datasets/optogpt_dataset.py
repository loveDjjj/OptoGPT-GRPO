"""OptoGPT 光谱-结构成对数据集。"""

from __future__ import annotations

import pickle as pkl
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from torch.utils.data import Dataset


def _load_array_or_pickle(path: str | Path) -> Any:
    """读取 `.pkl` / `.npy` 数据。

    这里保留两种格式，是为了后续可以把超大的 pkl 平滑迁移到更适合多卡读取的
    分片/数组格式，而不需要再改训练和评测主流程。
    """

    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".npy":
        return np.load(file_path, mmap_mode="r")
    if suffix == ".pkl":
        with file_path.open("rb") as handle:
            return pkl.load(handle)
    raise ValueError(f"不支持的数据格式: {file_path}")


class OptoGPTPairDataset(Dataset):
    """面向 OptoGPT 的光谱-结构配对数据集。"""

    def __init__(
        self,
        spectrum_path: str | Path,
        structure_path: str | Path,
        start_index: int = 0,
        stop_index: int | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.spectrum_path = Path(spectrum_path)
        self.structure_path = Path(structure_path)
        self._spectra = _load_array_or_pickle(self.spectrum_path)
        self._structures = _load_array_or_pickle(self.structure_path)

        total_count = len(self._spectra)
        if total_count != len(self._structures):
            raise ValueError("光谱数据与结构数据的样本数不一致。")

        resolved_start = max(0, int(start_index))
        resolved_stop = total_count if stop_index is None else min(total_count, int(stop_index))
        indices = list(range(resolved_start, resolved_stop))
        if max_samples is not None:
            indices = indices[: int(max_samples)]

        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_index = int(self.indices[index])
        spectrum = np.asarray(self._spectra[sample_index], dtype=np.float32)
        structure_tokens = [str(token) for token in self._structures[sample_index]]
        return {
            "sample_index": sample_index,
            "spectrum": spectrum,
            "structure_tokens": structure_tokens,
        }

    @property
    def spectrum_dim(self) -> int:
        if len(self.indices) == 0:
            return 0
        first = np.asarray(self._spectra[self.indices[0]], dtype=np.float32).reshape(-1)
        return int(first.size)

    @property
    def raw_structure_store(self) -> Sequence[Any]:
        return self._structures
