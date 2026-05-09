from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

from _shared.utils.dist import DistributedContext, reduce_tensor
from pretrain.dataset.tokenizer import SpectralStructureTokenizer
from pretrain.model.modeling_spectral_gpt import SpectralGPTForCausalLM
from rl.dataset import SpectralRecordDataset, rl_batch_collator
from rl.monitoring import RLVisualizationMonitor
from rl.objective import group_relative_advantages, grpo_clipped_surrogate
from rl.policy import RolloutConfig, batch_sequence_logprobs, sample_structure_rollouts
from rl.reward import compute_rollout_rewards


@dataclass
class RLComponents:
    model: torch.nn.Module
    raw_model: SpectralGPTForCausalLM
    tokenizer: SpectralStructureTokenizer


class SpectralGRPOTrainer:
    def __init__(
        self,
        *,
        components: RLComponents,
        config: Mapping[str, Any],
        run_dir: str | Path,
        dist_ctx: DistributedContext,
        resume_checkpoint: str | Path | None = None,
    ) -> None:
        self.model = components.model
        self.raw_model = components.raw_model
        self.tokenizer = components.tokenizer
        self.config = config
        self.run_dir = Path(run_dir)
        self.dist_ctx = dist_ctx

        data_cfg = config["data"]
        training_cfg = config["training"]
        rollout_cfg = config["rollout"]
        reward_cfg = config["reward"]

        self.epochs = int(training_cfg.get("epochs", 1))
        self.per_device_batch_size = int(training_cfg["per_device_batch_size"])
        self.gradient_accumulation_steps = max(1, int(training_cfg.get("gradient_accumulation_steps", 1)))
        self.learning_rate = float(training_cfg.get("learning_rate", 1.0e-5))
        self.weight_decay = float(training_cfg.get("weight_decay", 0.0))
        self.grad_clip_norm = float(training_cfg.get("grad_clip_norm", 1.0))
        self.log_steps = max(1, int(training_cfg.get("log_steps", 10)))
        self.eval_steps = int(training_cfg.get("eval_steps", 0))
        self.save_steps = int(training_cfg.get("save_steps", 0))
        self.num_workers = int(data_cfg.get("num_workers", 0))
        self.pin_memory = bool(data_cfg.get("pin_memory", False))
        self.prefetch_factor = data_cfg.get("prefetch_factor")
        self.persistent_workers = bool(data_cfg.get("persistent_workers", self.num_workers > 0))
        self.bf16 = bool(training_cfg.get("bf16", False))

        self.group_size = int(rollout_cfg.get("group_size", 4))
        self.rollout_config = RolloutConfig(
            decode=str(rollout_cfg.get("decode", "sample")),
            temperature=float(rollout_cfg.get("temperature", 1.0)),
            top_k=int(rollout_cfg.get("top_k", 0)),
            top_p=float(rollout_cfg.get("top_p", 1.0)),
            max_new_tokens=int(rollout_cfg.get("max_new_tokens", 12)),
            batch_size=int(rollout_cfg.get("batch_size", self.per_device_batch_size * self.group_size)),
        )
        self.score_batch_size = int(config.get("scoring", {}).get("batch_size", self.rollout_config.batch_size))
        self.clip_epsilon = float(training_cfg.get("clip_epsilon", 0.2))
        self.advantage_mode = str(training_cfg.get("advantage_mode", "zscore"))
        self.advantage_eps = float(training_cfg.get("advantage_eps", 1.0e-6))
        self.invalid_structure_penalty = float(reward_cfg.get("invalid_structure_penalty", 1.0))

        self.reward_tmm_cfg = reward_cfg["tmm"]
        self.reward_metric = str(reward_cfg.get("spectrum_metric", "rt_rmse"))
        self.resume_checkpoint = None if resume_checkpoint is None else Path(resume_checkpoint)
        self.global_step = 0
        self.resume_epoch = 0
        self.resume_batch_index = 0

        self.optimizer = torch.optim.AdamW(
            self.raw_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.lr_scheduler_type = str(training_cfg.get("lr_scheduler_type", "linear")).strip().lower()
        self.warmup_ratio = float(training_cfg.get("warmup_ratio", 0.0))
        self.warmup_steps_cfg = training_cfg.get("warmup_steps")
        self.total_training_steps = 1
        self.warmup_steps = 0
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self._lr_lambda)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = self.run_dir / "metrics"
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        monitoring_cfg = config.get("monitoring", {})
        self.monitoring = RLVisualizationMonitor(
            output_dir=self.run_dir,
            is_main=self.dist_ctx.is_main,
            enable_tensorboard=monitoring_cfg.get("tensorboard", True),
            enable_jsonl=monitoring_cfg.get("jsonl", True),
            enable_csv=monitoring_cfg.get("csv", True),
            save_plots=monitoring_cfg.get("save_plots", True),
            plot_every_eval=monitoring_cfg.get("plot_every_eval", True),
            flush_secs=int(monitoring_cfg.get("flush_secs", 10)),
        )
        if self.resume_checkpoint is not None:
            self._load_checkpoint_state(self.resume_checkpoint)

    def _set_policy_eval(self) -> None:
        # Rollout generation and policy rescoring must disable dropout so PPO ratios
        # compare the same policy under identical forward semantics.
        self.raw_model.eval()
        self.model.eval()

    def _set_policy_train(self) -> None:
        self.raw_model.train()
        self.model.train()

    def _make_dataloader(self, dataset: SpectralRecordDataset, *, shuffle: bool) -> tuple[DataLoader, DistributedSampler | None]:
        sampler = None
        if self.dist_ctx.enabled:
            sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=False)
        kwargs = {
            "dataset": dataset,
            "batch_size": self.per_device_batch_size,
            "shuffle": shuffle if sampler is None else False,
            "sampler": sampler,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers if self.num_workers > 0 else False,
            "collate_fn": rl_batch_collator,
        }
        if self.num_workers > 0 and self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(self.prefetch_factor)
        return DataLoader(**kwargs), sampler

    def _reward_kwargs(self) -> dict[str, Any]:
        return {
            "database_path": self.reward_tmm_cfg["database_path"],
            "wavelength_range_um": tuple(self.reward_tmm_cfg["wavelength_range_um"]),
            "num_points": int(self.reward_tmm_cfg["num_points"]),
            "incident_angle": float(self.reward_tmm_cfg.get("incident_angle", 0.0)),
            "polarization": int(self.reward_tmm_cfg.get("polarization", 0)),
            "tolerance": float(self.reward_tmm_cfg.get("tolerance", 1.0e-3)),
            "complex_dtype": str(self.reward_tmm_cfg.get("complex_dtype", "complex128")),
            "batch_size": int(self.reward_tmm_cfg.get("batch_size", self.score_batch_size)),
            "invalid_structure_penalty": self.invalid_structure_penalty,
            "spectrum_metric": self.reward_metric,
            "device": self._resolve_reward_device(self.reward_tmm_cfg.get("device")),
        }

    def _resolve_reward_device(self, requested_device: str | None) -> str:
        if requested_device is None:
            requested = "auto"
        else:
            requested = str(requested_device).strip().lower()
        if requested in {"", "auto"}:
            if self.dist_ctx.device.type == "cuda":
                return f"cuda:{self.dist_ctx.device.index}"
            return self.dist_ctx.device.type
        if requested == "cuda" and self.dist_ctx.enabled:
            return f"cuda:{self.dist_ctx.local_rank}"
        return str(requested_device)

    def _lr_lambda(self, current_step: int) -> float:
        total_steps = max(1, int(self.total_training_steps))
        warmup_steps = max(0, min(int(self.warmup_steps), total_steps))
        step = int(current_step)
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        if self.lr_scheduler_type == "constant":
            return 1.0

        decay_steps = max(1, total_steps - warmup_steps)
        progress = float(step - warmup_steps) / float(decay_steps)
        progress = max(0.0, min(1.0, progress))
        if self.lr_scheduler_type == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if self.lr_scheduler_type == "linear":
            return max(0.0, 1.0 - progress)
        raise ValueError(f"unsupported lr_scheduler_type: {self.lr_scheduler_type}")

    def _save_checkpoint(self, step: int) -> None:
        if not self.dist_ctx.is_main:
            return
        checkpoint_dir = self.checkpoints_dir / f"checkpoint-{step}"
        self.raw_model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        torch.save(self.optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        torch.save(self.scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
        (checkpoint_dir / "trainer_state.json").write_text(
            json.dumps(
                {
                    "global_step": int(self.global_step),
                    "resume_epoch": int(self.resume_epoch),
                    "resume_batch_index": int(self.resume_batch_index),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _load_checkpoint_state(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        optimizer_path = checkpoint_dir / "optimizer.pt"
        scheduler_path = checkpoint_dir / "scheduler.pt"
        trainer_state_path = checkpoint_dir / "trainer_state.json"

        if optimizer_path.exists():
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.dist_ctx.device))
        if scheduler_path.exists():
            self.scheduler.load_state_dict(torch.load(scheduler_path, map_location=self.dist_ctx.device))
        if trainer_state_path.exists():
            payload = json.loads(trainer_state_path.read_text(encoding="utf-8"))
            self.global_step = int(payload.get("global_step", 0))
            self.resume_epoch = int(payload.get("resume_epoch", 0))
            self.resume_batch_index = int(payload.get("resume_batch_index", 0))

    def _evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        previous_training = bool(self.raw_model.training)
        self._set_policy_eval()
        total_reward = 0.0
        total_count = 0
        for batch in dataloader:
            spectra = batch["spectra"].to(self.dist_ctx.device)
            records = batch["records"]
            rollout_samples = sample_structure_rollouts(
                self.model,
                self.tokenizer,
                spectra,
                [record["sample_id"] for record in records],
                group_size=1,
                config=RolloutConfig(
                    decode="greedy",
                    temperature=1.0,
                    top_k=0,
                    top_p=1.0,
                    max_new_tokens=self.rollout_config.max_new_tokens,
                    batch_size=self.rollout_config.batch_size,
                ),
            )
            rewards = compute_rollout_rewards(
                structure_token_groups=[sample.structure_tokens for sample in rollout_samples],
                target_spectra=[record["spectrum_rt"] for record in records],
                **self._reward_kwargs(),
            )["rewards"].to(dtype=torch.float32, device=self.dist_ctx.device)
            total_reward += float(rewards.sum().item())
            total_count += int(rewards.numel())
        reward_tensor = torch.tensor([total_reward, total_count], dtype=torch.float32, device=self.dist_ctx.device)
        reduced = reduce_tensor(reward_tensor, op="sum")
        mean_reward = float(reduced[0].item() / max(1.0, reduced[1].item()))
        if previous_training:
            self._set_policy_train()
        else:
            self._set_policy_eval()
        return {"mean_eval_reward": mean_reward}

    def train(self, train_dataset: SpectralRecordDataset, eval_dataset: SpectralRecordDataset | None = None) -> None:
        self._set_policy_eval()
        train_loader, train_sampler = self._make_dataloader(train_dataset, shuffle=True)
        eval_loader = None
        if eval_dataset is not None and len(eval_dataset) > 0:
            eval_loader, _ = self._make_dataloader(eval_dataset, shuffle=False)
        update_steps_per_epoch = max(1, math.ceil(len(train_loader) / self.gradient_accumulation_steps))
        self.total_training_steps = max(1, self.epochs * update_steps_per_epoch)
        if self.warmup_steps_cfg is not None:
            self.warmup_steps = int(self.warmup_steps_cfg)
        else:
            self.warmup_steps = int(round(self.total_training_steps * self.warmup_ratio))

        progress = tqdm(
            range(self.epochs * len(train_loader)),
            disable=not self.dist_ctx.is_main,
            dynamic_ncols=True,
            desc="grpo",
        )
        self.optimizer.zero_grad(set_to_none=True)
        pending_optimizer_step = False
        for epoch in range(self.resume_epoch, self.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            for batch_index, batch in enumerate(train_loader):
                if epoch == self.resume_epoch and batch_index < self.resume_batch_index:
                    progress.update(1)
                    continue
                spectra = batch["spectra"].to(self.dist_ctx.device)
                records = batch["records"]
                rollout_samples = sample_structure_rollouts(
                    self.model,
                    self.tokenizer,
                    spectra,
                    [record["sample_id"] for record in records],
                    group_size=self.group_size,
                    config=self.rollout_config,
                )
                expanded_spectra = spectra.repeat_interleave(self.group_size, dim=0)
                reward_outputs = compute_rollout_rewards(
                    structure_token_groups=[sample.structure_tokens for sample in rollout_samples],
                    target_spectra=[record["spectrum_rt"] for record in records for _ in range(self.group_size)],
                    **self._reward_kwargs(),
                )
                rewards = reward_outputs["rewards"].to(device=self.dist_ctx.device, dtype=torch.float32)
                advantages = group_relative_advantages(
                    rewards,
                    target_count=len(records),
                    group_size=self.group_size,
                    mode=self.advantage_mode,
                    eps=self.advantage_eps,
                )
                current_logprobs, _ = batch_sequence_logprobs(
                    self.model,
                    self.tokenizer,
                    expanded_spectra,
                    [sample.token_ids for sample in rollout_samples],
                    batch_size=self.score_batch_size,
                    rollout_config=self.rollout_config,
                )
                old_logprobs = torch.tensor(
                    [sample.sequence_logprob for sample in rollout_samples],
                    dtype=torch.float32,
                    device=self.dist_ctx.device,
                )

                autocast_enabled = self.bf16 and self.dist_ctx.device.type == "cuda"
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                    objective_outputs = grpo_clipped_surrogate(
                        current_logprob=current_logprobs,
                        old_logprob=old_logprobs,
                        advantage=advantages,
                        clip_epsilon=self.clip_epsilon,
                    )
                    loss = -objective_outputs["surrogate"].mean() / self.gradient_accumulation_steps

                loss.backward()
                pending_optimizer_step = True
                if (batch_index + 1) % self.gradient_accumulation_steps == 0:
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), self.grad_clip_norm).item())
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    pending_optimizer_step = False
                    self.global_step += 1
                    current_lr = float(self.optimizer.param_groups[0]["lr"])
                    mean_ratio = float(objective_outputs["ratio"].mean().item())
                    clip_fraction = float(objective_outputs["clip_mask"].float().mean().item())
                    mean_approx_kl = float(objective_outputs["approx_kl"].mean().item())
                    if batch_index + 1 < len(train_loader):
                        self.resume_epoch = epoch
                        self.resume_batch_index = batch_index + 1
                    else:
                        self.resume_epoch = epoch + 1
                        self.resume_batch_index = 0
                    if hasattr(progress, "set_postfix"):
                        progress.set_postfix(
                            {
                                "loss": float(loss.item() * self.gradient_accumulation_steps),
                                "reward": float(rewards.mean().item()),
                                "valid": float(reward_outputs["ok_mask"].float().mean().item()),
                            },
                            refresh=False,
                        )
                    progress.update(1)

                    if self.dist_ctx.is_main and self.global_step % self.log_steps == 0:
                        self.monitoring.log_train(
                            {
                                "step": self.global_step,
                                "epoch": float(epoch + 1),
                                "loss": float(loss.item() * self.gradient_accumulation_steps),
                                "mean_reward": float(rewards.mean().item()),
                                "valid_ratio": float(reward_outputs["ok_mask"].float().mean().item()),
                                "learning_rate": current_lr,
                                "grad_norm": grad_norm,
                                "mean_ratio": mean_ratio,
                                "clip_fraction": clip_fraction,
                                "mean_approx_kl": mean_approx_kl,
                            }
                        )

                    if self.save_steps > 0 and self.global_step % self.save_steps == 0:
                        self._save_checkpoint(self.global_step)
                    if eval_loader is not None and self.eval_steps > 0 and self.global_step % self.eval_steps == 0:
                        metrics = self._evaluate(eval_loader)
                        if self.dist_ctx.is_main:
                            self.monitoring.log_eval({"step": self.global_step, **metrics})

        if pending_optimizer_step:
            torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), self.grad_clip_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
            self.resume_epoch = self.epochs
            self.resume_batch_index = 0
        self.monitoring.close()
        self._save_checkpoint(self.global_step)
