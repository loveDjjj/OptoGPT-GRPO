from __future__ import annotations

import torch


@torch.inference_mode()
def generate_structure_tokens(
    model,
    tokenizer,
    spectra: torch.Tensor,
    max_new_tokens: int,
) -> list[list[str]]:
    batch_size = spectra.size(0)
    input_ids = torch.full(
        (batch_size, 1),
        tokenizer.bos_token_id,
        dtype=torch.long,
        device=spectra.device,
    )

    for _ in range(max_new_tokens):
        attention_mask = torch.ones(
            (batch_size, model.config.prefix_length + input_ids.size(1)),
            dtype=torch.long,
            device=spectra.device,
        )
        outputs = model(
            spectra=spectra,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        next_ids = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_ids], dim=1)

    return [tokenizer.decode(row.tolist()) for row in input_ids]


def score_structure_tokens(
    model,
    spectra: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    outputs = model(
        spectra=spectra,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    log_probs = outputs.logits.log_softmax(dim=-1)
    return log_probs[:, :-1].gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
