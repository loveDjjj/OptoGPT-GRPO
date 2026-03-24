"""Policy wrapper around the legacy OptoGPT checkpoint.

The raw checkpoint only exposes the original Transformer module and vocabulary
objects. This wrapper turns it into a policy object that the GRPO trainer can
use for:

- batched autoregressive sampling,
- checkpoint export,
- log-probability recomputation for already sampled token sequences.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from core.transformer import make_model_I, subsequent_mask


@dataclass
class DecodeConfig:
    """Sampling hyper-parameters shared across rollout generation."""

    decode: str = "top-k"
    top_k: int = 10
    top_p: float = 0.8
    temperature: float = 1.0
    start_symbol: str = "BOS"
    start_mat: Optional[str] = None
    max_len: Optional[int] = None
    batch_size: Optional[int] = None


@dataclass
class RolloutSample:
    """One decoded candidate structure together with token-level statistics."""

    target_index: Optional[int]
    candidate_index: int
    prompt_ids: List[int]
    token_ids: List[int]
    tokens: List[str]
    structure_tokens: List[str]
    raw_logprobs: List[float]
    sample_logprobs: List[float]
    raw_entropies: List[float]
    sample_entropies: List[float]
    sequence_raw_logprob: float
    sequence_sample_logprob: float
    terminated_by_eos: bool
    max_len_reached: bool
    decode: str


def resolve_device(device_arg: str = "auto") -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_checkpoint(checkpoint_path: str, device: torch.device) -> dict:
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


class OptoGPTPolicy(nn.Module):
    """Thin RL-friendly wrapper around the original OptoGPT model."""

    def __init__(self, checkpoint_path: str, device: str | torch.device = "auto") -> None:
        super().__init__()
        self.device = resolve_device(device) if not isinstance(device, torch.device) else device
        self.checkpoint_path = checkpoint_path
        self.checkpoint = load_checkpoint(checkpoint_path, self.device)
        self.ckpt_args = self.checkpoint["configs"]

        self.model = make_model_I(
            self.ckpt_args.spec_dim,
            self.ckpt_args.struc_dim,
            self.ckpt_args.layers,
            self.ckpt_args.d_model,
            self.ckpt_args.d_ff,
            self.ckpt_args.head_num,
            self.ckpt_args.dropout,
        ).to(self.device)
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.eval()

        self.spec_type = getattr(self.ckpt_args, "spec_type", "R_T")
        self.spec_dim = int(self.ckpt_args.spec_dim)
        self.max_len = int(self.ckpt_args.max_len)
        self.struc_word_dict = dict(self.ckpt_args.struc_word_dict)
        self.struc_index_dict = dict(self.ckpt_args.struc_index_dict)
        self.vocab_size = len(self.struc_word_dict)

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.model(src, tgt, src_mask, tgt_mask)

    @property
    def generator(self):
        return self.model.generator

    def make_reference_model(self) -> nn.Module:
        """Freeze a snapshot of the current policy for KL regularization."""

        reference = copy.deepcopy(self.model).to(self.device)
        reference.eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        return reference

    def export_checkpoint(self, path: str | Path, extra_state: Optional[dict] = None) -> None:
        """Save the current policy state using a lightweight checkpoint schema."""

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "configs": self.ckpt_args,
            "source_checkpoint": self.checkpoint_path,
        }
        if extra_state:
            checkpoint.update(extra_state)

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, output_path)

    def adapt_target_spectrum(self, spectrum: Sequence[float]) -> np.ndarray:
        """Match the runtime target layout to the checkpoint's expected input."""

        spectrum = np.asarray(spectrum, dtype=np.float32).reshape(-1)
        if spectrum.size == self.spec_dim:
            return spectrum

        if spectrum.size != 142:
            raise ValueError(f"Spectrum length {spectrum.size} does not match checkpoint input dim {self.spec_dim}")

        half = spectrum.size // 2
        if self.spec_type == "R":
            spectrum = spectrum[:half]
        elif self.spec_type == "T":
            spectrum = spectrum[half:]
        elif self.spec_type == "R_T":
            pass
        else:
            raise ValueError(f"Unsupported checkpoint spec_type: {self.spec_type}")

        if spectrum.size != self.spec_dim:
            raise ValueError(
                f"Adapted spectrum length {spectrum.size} does not match checkpoint input dim {self.spec_dim}"
            )
        return spectrum.astype(np.float32)

    def target_to_tensor(self, spectrum: Sequence[float]) -> torch.Tensor:
        adapted = self.adapt_target_spectrum(spectrum)
        return torch.from_numpy(adapted).to(device=self.device, dtype=torch.float32).view(1, 1, -1)

    def target_to_tensor_batch(self, spectrum: Sequence[float], batch_size: int) -> torch.Tensor:
        """Repeat one target spectrum into a batch for batched decoding/scoring."""

        src = self.target_to_tensor(spectrum)
        if batch_size <= 1:
            return src
        return src.expand(batch_size, -1, -1).contiguous()

    def prompt_ids(self, start_symbol: str = "BOS", start_mat: Optional[str] = None) -> List[int]:
        """Build the decoder prompt used to start autoregressive generation."""

        ids = [int(self.struc_word_dict[start_symbol])]
        if start_mat is not None:
            ids.append(int(self.struc_word_dict[start_mat]))
        return ids

    def token_id_to_str(self, token_id: int) -> str:
        return self.struc_index_dict[int(token_id)]

    def _filtered_distribution(
        self,
        raw_log_prob: torch.Tensor,
        decode_config: DecodeConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply greedy / top-k / top-p filtering to a single-step distribution."""

        if decode_config.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if decode_config.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < decode_config.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")

        if decode_config.decode not in {"greedy", "top-k", "sample"}:
            raise ValueError(f"Unsupported decode mode: {decode_config.decode}")

        tempered_log_prob = raw_log_prob / decode_config.temperature
        prob = torch.softmax(tempered_log_prob, dim=-1)

        if decode_config.decode == "greedy":
            greedy_idx = int(torch.argmax(prob).item())
            filtered = torch.zeros_like(prob)
            filtered[greedy_idx] = 1.0
            return prob, filtered

        filtered = prob.clone()
        if decode_config.decode == "top-k":
            if decode_config.top_k > 0 and decode_config.top_k < filtered.numel():
                topk_values, _ = torch.topk(filtered, decode_config.top_k, dim=-1)
                kth = topk_values[-1]
                filtered = torch.where(filtered < kth, torch.zeros_like(filtered), filtered)

            if decode_config.top_p < 1.0:
                sorted_prob, sorted_idx = torch.sort(filtered, descending=True, dim=-1)
                cumulative = torch.cumsum(sorted_prob, dim=-1)
                remove_mask = cumulative > decode_config.top_p
                remove_mask[1:] = remove_mask[:-1].clone()
                remove_mask[0] = False
                sorted_prob = torch.where(remove_mask, torch.zeros_like(sorted_prob), sorted_prob)
                filtered.zero_()
                filtered.scatter_(0, sorted_idx, sorted_prob)

        filtered_sum = filtered.sum()
        if float(filtered_sum.item()) <= 0:
            filtered = prob
            filtered_sum = filtered.sum()
        filtered = filtered / filtered_sum
        return prob, filtered

    def _filtered_distribution_batch(
        self,
        raw_log_prob: torch.Tensor,
        decode_config: DecodeConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched version of ``_filtered_distribution`` for faster sampling."""

        if raw_log_prob.dim() != 2:
            raise ValueError(f"raw_log_prob must be 2D [batch, vocab], got shape {tuple(raw_log_prob.shape)}")
        if decode_config.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if decode_config.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < decode_config.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if decode_config.decode not in {"greedy", "top-k", "sample"}:
            raise ValueError(f"Unsupported decode mode: {decode_config.decode}")

        tempered_log_prob = raw_log_prob / decode_config.temperature
        prob = torch.softmax(tempered_log_prob, dim=-1)

        if decode_config.decode == "greedy":
            greedy_idx = torch.argmax(prob, dim=-1, keepdim=True)
            filtered = torch.zeros_like(prob)
            filtered.scatter_(1, greedy_idx, 1.0)
            return prob, filtered

        filtered = prob.clone()
        vocab_size = filtered.size(-1)
        if decode_config.decode == "top-k":
            if decode_config.top_k > 0 and decode_config.top_k < vocab_size:
                topk_values = torch.topk(filtered, decode_config.top_k, dim=-1).values
                kth = topk_values[:, -1:].expand_as(filtered)
                filtered = torch.where(filtered < kth, torch.zeros_like(filtered), filtered)

            if decode_config.top_p < 1.0:
                sorted_prob, sorted_idx = torch.sort(filtered, descending=True, dim=-1)
                cumulative = torch.cumsum(sorted_prob, dim=-1)
                remove_mask = cumulative > decode_config.top_p
                remove_mask[:, 1:] = remove_mask[:, :-1].clone()
                remove_mask[:, 0] = False
                sorted_prob = torch.where(remove_mask, torch.zeros_like(sorted_prob), sorted_prob)
                filtered.zero_()
                filtered.scatter_(1, sorted_idx, sorted_prob)

        filtered_sum = filtered.sum(dim=-1, keepdim=True)
        empty_mask = filtered_sum <= 0
        if empty_mask.any():
            filtered = torch.where(empty_mask, prob, filtered)
            filtered_sum = filtered.sum(dim=-1, keepdim=True)
        filtered = filtered / filtered_sum.clamp_min(1e-12)
        return prob, filtered

    def _decode_step(
        self,
        src: torch.Tensor,
        ys: torch.Tensor,
        decode_config: DecodeConfig,
        rng: Optional[torch.Generator] = None,
    ) -> dict:
        trg_mask = Variable(subsequent_mask(ys.size(1)).type_as(src.data)).to(self.device)
        out = self.model(src, ys, None, trg_mask)
        raw_log_prob = self.generator(out[:, -1])[0]
        raw_prob, sample_prob = self._filtered_distribution(raw_log_prob, decode_config)

        if decode_config.decode == "greedy":
            next_id = int(torch.argmax(sample_prob).item())
        else:
            next_id = int(torch.multinomial(sample_prob, num_samples=1, generator=rng).item())

        raw_logprob = float(raw_log_prob[next_id].item())
        sample_logprob = float(torch.log(sample_prob[next_id].clamp_min(1e-12)).item())
        raw_entropy = float((-(raw_prob * torch.log(raw_prob.clamp_min(1e-12))).sum()).item())
        sample_entropy = float((-(sample_prob * torch.log(sample_prob.clamp_min(1e-12))).sum()).item())

        return {
            "next_id": next_id,
            "raw_logprob": raw_logprob,
            "sample_logprob": sample_logprob,
            "raw_entropy": raw_entropy,
            "sample_entropy": sample_entropy,
        }

    def sample(
        self,
        target_spectrum: Sequence[float],
        decode_config: DecodeConfig,
        target_index: Optional[int] = None,
        candidate_index: int = 0,
        rng: Optional[torch.Generator] = None,
    ) -> RolloutSample:
        return self._sample_batch(
            target_spectrum=target_spectrum,
            num_samples=1,
            decode_config=decode_config,
            target_index=target_index,
            candidate_offset=candidate_index,
            rng=rng,
        )[0]

    def _sample_batch(
        self,
        target_spectrum: Sequence[float],
        num_samples: int,
        decode_config: DecodeConfig,
        target_index: Optional[int] = None,
        candidate_offset: int = 0,
        rng: Optional[torch.Generator] = None,
    ) -> List[RolloutSample]:
        """Decode a batch of candidates for one target in parallel."""

        if num_samples <= 0:
            return []

        src = self.target_to_tensor_batch(target_spectrum, batch_size=num_samples)
        prompt_ids = self.prompt_ids(
            start_symbol=decode_config.start_symbol,
            start_mat=decode_config.start_mat,
        )
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        ys = prompt_tensor.expand(num_samples, -1).clone()
        generation_limit = int(decode_config.max_len or self.max_len)
        pad_id = int(self.struc_word_dict.get("PAD", self.struc_word_dict.get("EOS", 0)))

        generated_ids: List[List[int]] = [[] for _ in range(num_samples)]
        generated_tokens: List[List[str]] = [[] for _ in range(num_samples)]
        structure_tokens: List[List[str]] = [[] for _ in range(num_samples)]
        raw_logprobs: List[List[float]] = [[] for _ in range(num_samples)]
        sample_logprobs: List[List[float]] = [[] for _ in range(num_samples)]
        raw_entropies: List[List[float]] = [[] for _ in range(num_samples)]
        sample_entropies: List[List[float]] = [[] for _ in range(num_samples)]
        terminated_by_eos = [False for _ in range(num_samples)]
        active_mask = torch.ones(num_samples, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            # All candidates in this batch share the same target spectrum. We
            # keep one decoding loop over sequence length, but every step is
            # evaluated in parallel across the whole candidate batch.
            while ys.size(1) < generation_limit and bool(active_mask.any().item()):
                trg_mask = Variable(subsequent_mask(ys.size(1)).type_as(src.data)).to(self.device)
                out = self.model(src, ys, None, trg_mask)
                raw_log_prob = self.generator(out[:, -1])
                raw_prob, sample_prob = self._filtered_distribution_batch(raw_log_prob, decode_config)

                if decode_config.decode == "greedy":
                    next_ids = torch.argmax(sample_prob, dim=-1)
                else:
                    next_ids = torch.multinomial(sample_prob, num_samples=1, generator=rng).squeeze(-1)

                step_raw_logprob = raw_log_prob.gather(1, next_ids.unsqueeze(-1)).squeeze(-1)
                step_sample_logprob = torch.log(
                    sample_prob.gather(1, next_ids.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
                )
                step_raw_entropy = -(raw_prob * torch.log(raw_prob.clamp_min(1e-12))).sum(dim=-1)
                step_sample_entropy = -(sample_prob * torch.log(sample_prob.clamp_min(1e-12))).sum(dim=-1)

                next_ids_to_append = torch.where(active_mask, next_ids, torch.full_like(next_ids, pad_id))
                ys = torch.cat([ys, next_ids_to_append.unsqueeze(1)], dim=1)

                active_indices = torch.nonzero(active_mask, as_tuple=False).view(-1).tolist()
                finished_indices: List[int] = []
                for idx in active_indices:
                    next_id = int(next_ids[idx].item())
                    token = self.token_id_to_str(next_id)
                    generated_ids[idx].append(next_id)
                    generated_tokens[idx].append(token)
                    raw_logprobs[idx].append(float(step_raw_logprob[idx].item()))
                    sample_logprobs[idx].append(float(step_sample_logprob[idx].item()))
                    raw_entropies[idx].append(float(step_raw_entropy[idx].item()))
                    sample_entropies[idx].append(float(step_sample_entropy[idx].item()))
                    if token == "EOS":
                        terminated_by_eos[idx] = True
                        finished_indices.append(idx)
                    elif token not in {"BOS", "PAD", "UNK"}:
                        structure_tokens[idx].append(token)

                if finished_indices:
                    active_mask[torch.tensor(finished_indices, dtype=torch.long, device=self.device)] = False

        samples: List[RolloutSample] = []
        for sample_idx in range(num_samples):
            samples.append(
                RolloutSample(
                    target_index=target_index,
                    candidate_index=candidate_offset + sample_idx,
                    prompt_ids=list(prompt_ids),
                    token_ids=generated_ids[sample_idx],
                    tokens=generated_tokens[sample_idx],
                    structure_tokens=structure_tokens[sample_idx],
                    raw_logprobs=raw_logprobs[sample_idx],
                    sample_logprobs=sample_logprobs[sample_idx],
                    raw_entropies=raw_entropies[sample_idx],
                    sample_entropies=sample_entropies[sample_idx],
                    sequence_raw_logprob=float(sum(raw_logprobs[sample_idx])),
                    sequence_sample_logprob=float(sum(sample_logprobs[sample_idx])),
                    terminated_by_eos=terminated_by_eos[sample_idx],
                    max_len_reached=not terminated_by_eos[sample_idx],
                    decode=decode_config.decode,
                )
            )
        return samples

    def sample_group(
        self,
        target_spectrum: Sequence[float],
        num_samples: int,
        decode_config: DecodeConfig,
        target_index: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[RolloutSample]:
        """Sample a rollout group, optionally chunked to control GPU memory."""

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device.type)
            generator.manual_seed(int(seed))
        effective_batch_size = int(decode_config.batch_size or num_samples)
        effective_batch_size = max(1, effective_batch_size)

        samples: List[RolloutSample] = []
        for start in range(0, num_samples, effective_batch_size):
            current_batch = min(effective_batch_size, num_samples - start)
            samples.extend(
                self._sample_batch(
                    target_spectrum=target_spectrum,
                    num_samples=current_batch,
                    decode_config=decode_config,
                    target_index=target_index,
                    candidate_offset=start,
                    rng=generator,
                )
            )
        return samples

    def evaluate_sequence_logprobs(
        self,
        target_spectrum: Sequence[float],
        token_ids: Sequence[int],
        start_symbol: str = "BOS",
        start_mat: Optional[str] = None,
    ) -> List[float]:
        return self.sequence_logprobs_tensor(
            target_spectrum=target_spectrum,
            token_ids=token_ids,
            start_symbol=start_symbol,
            start_mat=start_mat,
            model=None,
            require_grad=False,
        ).detach().cpu().tolist()

    def sequence_logprobs_batch_tensor(
        self,
        target_spectrum: Sequence[float],
        token_id_groups: Sequence[Sequence[int]],
        start_symbol: str = "BOS",
        start_mat: Optional[str] = None,
        model: Optional[nn.Module] = None,
        require_grad: bool = False,
        batch_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recompute token log-probabilities for a batch of sampled sequences.

        This is used during GRPO updates where the trainer needs:
        - old log-probs from sampling time,
        - current log-probs from the updated policy,
        - reference log-probs from the frozen anchor model.
        """

        sequences = [[int(token_id) for token_id in token_ids] for token_ids in token_id_groups]
        batch_count = len(sequences)
        if batch_count == 0:
            empty = torch.empty((0, 0), dtype=torch.float32, device=self.device)
            return empty, empty.to(dtype=torch.bool)

        prompt_ids = self.prompt_ids(start_symbol=start_symbol, start_mat=start_mat)
        prompt_len = len(prompt_ids)
        max_target_len = max((len(sequence) for sequence in sequences), default=0)
        if max_target_len == 0:
            empty = torch.empty((batch_count, 0), dtype=torch.float32, device=self.device)
            return empty, torch.zeros((batch_count, 0), dtype=torch.bool, device=self.device)

        max_input_len = prompt_len + max_target_len - 1
        pad_id = int(self.struc_word_dict.get("PAD", self.struc_word_dict.get("EOS", 0)))
        effective_batch_size = int(batch_size or batch_count)
        effective_batch_size = max(1, effective_batch_size)

        all_logprobs: List[torch.Tensor] = []
        all_masks: List[torch.Tensor] = []
        model = self.model if model is None else model
        generator = model.generator if hasattr(model, "generator") else self.generator
        grad_context = torch.enable_grad() if require_grad else torch.no_grad()

        with grad_context:
            for start in range(0, batch_count, effective_batch_size):
                batch_sequences = sequences[start : start + effective_batch_size]
                current_batch = len(batch_sequences)
                src = self.target_to_tensor_batch(target_spectrum, batch_size=current_batch)
                tgt_input = torch.full(
                    (current_batch, max_input_len),
                    pad_id,
                    dtype=torch.long,
                    device=self.device,
                )
                target_ids = torch.full(
                    (current_batch, max_target_len),
                    pad_id,
                    dtype=torch.long,
                    device=self.device,
                )
                token_mask = torch.zeros(
                    (current_batch, max_target_len),
                    dtype=torch.bool,
                    device=self.device,
                )

                for row_idx, token_id_list in enumerate(batch_sequences):
                    prefix_ids = prompt_ids + token_id_list[:-1]
                    prefix_tensor = torch.tensor(prefix_ids, dtype=torch.long, device=self.device)
                    tgt_input[row_idx, : prefix_tensor.numel()] = prefix_tensor
                    if token_id_list:
                        target_tensor = torch.tensor(token_id_list, dtype=torch.long, device=self.device)
                        target_ids[row_idx, : target_tensor.numel()] = target_tensor
                        token_mask[row_idx, : target_tensor.numel()] = True

                trg_mask = Variable(subsequent_mask(max_input_len).type_as(src.data)).to(self.device)
                out = model(src, tgt_input, None, trg_mask)
                raw_log_probs = generator(out)
                gather_positions = slice(prompt_len - 1, prompt_len - 1 + max_target_len)
                relevant_log_probs = raw_log_probs[:, gather_positions, :]
                gathered = relevant_log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
                all_logprobs.append(gathered)
                all_masks.append(token_mask)

        return torch.cat(all_logprobs, dim=0), torch.cat(all_masks, dim=0)

    def sequence_logprobs_tensor(
        self,
        target_spectrum: Sequence[float],
        token_ids: Sequence[int],
        start_symbol: str = "BOS",
        start_mat: Optional[str] = None,
        model: Optional[nn.Module] = None,
        require_grad: bool = False,
    ) -> torch.Tensor:
        logprobs, token_mask = self.sequence_logprobs_batch_tensor(
            target_spectrum=target_spectrum,
            token_id_groups=[token_ids],
            start_symbol=start_symbol,
            start_mat=start_mat,
            model=model,
            require_grad=require_grad,
            batch_size=1,
        )
        if logprobs.numel() == 0:
            return torch.empty(0, dtype=torch.float32, device=self.device)
        return logprobs[0, token_mask[0]]
