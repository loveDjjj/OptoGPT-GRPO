from __future__ import annotations

import numpy as np
import torch

from pretrain.dataset.tokenizer import SpectralStructureTokenizer


class SpectralCausalCollator:
    def __init__(
        self,
        tokenizer: SpectralStructureTokenizer,
        prefix_length: int,
        ignore_index: int = -100,
    ) -> None:
        self.tokenizer = tokenizer
        self.prefix_length = prefix_length
        self.ignore_index = ignore_index

    def __call__(self, samples: list[dict]) -> dict[str, torch.Tensor]:
        encoded = [self.tokenizer.encode(sample["structure_tokens"]) for sample in samples]
        max_token_length = max(len(ids) for ids in encoded)

        input_ids = torch.full(
            (len(samples), max_token_length),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        token_attention = torch.zeros((len(samples), max_token_length), dtype=torch.long)

        for row_index, token_ids in enumerate(encoded):
            token_count = len(token_ids)
            input_ids[row_index, :token_count] = torch.tensor(token_ids, dtype=torch.long)
            token_attention[row_index, :token_count] = 1

        labels = torch.full(
            (len(samples), self.prefix_length + max_token_length),
            self.ignore_index,
            dtype=torch.long,
        )
        # Prefix slots are always masked because they come from the projected spectrum.
        labels[:, self.prefix_length:] = input_ids
        # The BOS target is not predicted; the first supervised position starts at token 1.
        labels[:, self.prefix_length] = self.ignore_index

        attention_mask = torch.cat(
            [torch.ones((len(samples), self.prefix_length), dtype=torch.long), token_attention],
            dim=1,
        )

        return {
            "spectra": torch.from_numpy(
                np.asarray([sample["spectrum_rt"] for sample in samples], dtype=np.float32)
            ),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
