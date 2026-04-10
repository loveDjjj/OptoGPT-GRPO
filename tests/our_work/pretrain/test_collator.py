import warnings

import numpy as np
import torch

from our_work.pretrain.dataset.collator import SpectralCausalCollator
from our_work.pretrain.dataset.tokenizer import SpectralStructureTokenizer


def test_collator_masks_prefix_positions_with_ignore_index():
    tokenizer = SpectralStructureTokenizer(
        tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]", "Ge_10", "SiO2_20"]
    )
    collator = SpectralCausalCollator(tokenizer=tokenizer, prefix_length=3)
    batch = collator(
        [
            {"spectrum_rt": [0.1] * 2048, "structure_tokens": ["Ge_10", "SiO2_20"]},
            {"spectrum_rt": [0.2] * 2048, "structure_tokens": ["Ge_10"]},
        ]
    )
    assert batch["spectra"].shape == (2, 2048)
    assert batch["input_ids"].shape[0] == 2
    assert torch.all(batch["labels"][:, :3] == -100)


def test_collator_accepts_numpy_spectra_without_warning():
    tokenizer = SpectralStructureTokenizer(
        tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]", "Ge_10"]
    )
    collator = SpectralCausalCollator(tokenizer=tokenizer, prefix_length=2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        batch = collator(
            [
                {
                    "spectrum_rt": np.full(2048, 0.1, dtype=np.float32),
                    "structure_tokens": ["Ge_10"],
                }
            ]
        )
    assert batch["spectra"].shape == (1, 2048)
    assert not caught
