from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from pretrain.dataset.hf_dataset import load_split_records


class SpectralRecordDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[int(index)]


def load_rl_split_records(
    dataset_dir: str,
    split_name: str,
    max_samples: int | None = None,
) -> SpectralRecordDataset:
    records = load_split_records(dataset_dir, split_name)
    if max_samples is not None:
        records = records[: int(max_samples)]
    return SpectralRecordDataset(records)


def rl_batch_collator(samples: list[dict[str, Any]]) -> dict[str, Any]:
    spectra = torch.from_numpy(np.asarray([sample["spectrum_rt"] for sample in samples], dtype=np.float32))
    return {
        "spectra": spectra,
        "records": samples,
    }
