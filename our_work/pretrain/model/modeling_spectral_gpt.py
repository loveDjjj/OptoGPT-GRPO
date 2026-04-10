from __future__ import annotations

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from our_work.pretrain.model.configuration_spectral_gpt import SpectralGPTConfig
from our_work.pretrain.model.projector import SpectrumProjector


class SpectralGPTForCausalLM(PreTrainedModel):
    config_class = SpectralGPTConfig

    def __init__(self, config: SpectralGPTConfig) -> None:
        super().__init__(config)
        backbone_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_positions=config.n_positions,
            n_embd=config.n_embd,
            n_layer=config.n_layer,
            n_head=config.n_head,
            embd_pdrop=config.embd_pdrop,
            resid_pdrop=config.resid_pdrop,
            attn_pdrop=config.attn_pdrop,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
            pad_token_id=config.pad_token_id,
        )
        self.backbone = GPT2Model(backbone_config)
        self.projector = SpectrumProjector(
            spectrum_dim=config.spectrum_dim,
            prefix_length=config.prefix_length,
            hidden_size=config.n_embd,
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        spectra: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        kwargs.pop("num_items_in_batch", None)
        prefix_embeds = self.projector(spectra)
        token_embeds = self.backbone.wte(input_ids)
        # The decoder always sees prefix embeddings first, followed by structure tokens.
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )
        logits = self.lm_head(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
