from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from our_work.pretrain.model.generation import generate_structure_tokens


@torch.inference_mode()
def run_eval_sample(model, tokenizer, spectra: torch.Tensor, max_new_tokens: int = 10) -> list[list[str]]:
    return generate_structure_tokens(
        model=model,
        tokenizer=tokenizer,
        spectra=spectra,
        max_new_tokens=max_new_tokens,
    )
