from __future__ import annotations

import torch


def _suppress_non_structural_tokens(logits: torch.Tensor, tokenizer) -> torch.Tensor:
    blocked_ids = {
        int(tokenizer.pad_token_id),
        int(tokenizer.bos_token_id),
        int(tokenizer.unk_token_id),
    }
    filtered = logits.clone()
    filtered[..., sorted(blocked_ids)] = float("-inf")
    return filtered


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
        next_token_logits = _suppress_non_structural_tokens(outputs.logits[:, -1, :], tokenizer)
        next_ids = next_token_logits.argmax(dim=-1, keepdim=True)
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
    token_start = int(model.config.prefix_length)
    token_stop = token_start + input_ids.size(1) - 1
    return log_probs[:, token_start:token_stop, :].gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
