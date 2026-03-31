"""基于光谱 reward 的 GRPO 训练器。"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets import build_distributed_sampler, optogpt_batch_collator
from evaluators import SpectrumEvaluator
from losses import evaluate_generated_structures, grpo_clipped_surrogate, group_relative_advantages, masked_sequence_logprob
from models.optogpt import (
    build_decode_config,
    export_optogpt_checkpoint,
    generate_structures_for_targets,
    sequence_logprobs_multi_target_batch_tensor,
    validate_policy_config,
)
from utils.dist import DistributedContext, barrier, reduce_tensor
from utils.logging import write_summary_csv
from utils.plotting import save_grpo_epoch_summary_plot


class GRPOTrainer:
    """按同一 rollout policy 做采样、打分与 clipped update 的训练器。"""

    def __init__(
        self,
        model,
        config: Mapping[str, Any],
        run_dir: str | Path,
        dist_ctx: DistributedContext,
    ) -> None:
        self.model = model
        self.config = config
        self.run_dir = Path(run_dir)
        self.dist_ctx = dist_ctx

        data_cfg = config["data"]
        training_cfg = config["training"]
        tmm_cfg = config["tmm"]
        logging_cfg = config.get("logging", {})

        self.epochs = int(training_cfg["epochs"])
        self.batch_size = int(training_cfg["batch_size"])
        self.learning_rate = float(training_cfg["learning_rate"])
        self.weight_decay = float(training_cfg.get("weight_decay", 0.0))
        self.grad_clip_norm = float(training_cfg.get("grad_clip_norm", 1.0))
        self.grad_accum_steps = max(1, int(training_cfg.get("grad_accum_steps", 1)))
        self.log_interval = max(1, int(training_cfg.get("log_interval", 10)))
        self.eval_every_epochs = max(1, int(training_cfg.get("eval_every_epochs", 1)))
        self.group_size = int(training_cfg.get("group_size", training_cfg.get("num_samples_per_target", 4)))
        self.clip_epsilon = float(training_cfg.get("clip_epsilon", 0.2))
        self.advantage_mode = str(training_cfg.get("advantage_mode", "zscore")).strip().lower()
        self.advantage_eps = float(training_cfg.get("advantage_eps", 1e-6))
        self.normalize_logprob_by_length = bool(training_cfg.get("normalize_logprob_by_length", True))
        self.policy_forward_mode = str(training_cfg.get("policy_forward_mode", "eval")).strip().lower()
        self.scoring_batch_size = max(1, int(training_cfg.get("scoring_batch_size", self.batch_size * self.group_size)))
        self.num_workers = int(data_cfg.get("num_workers", 0))
        self.pin_memory = bool(data_cfg.get("pin_memory", False))
        self.prefetch_factor = int(data_cfg.get("prefetch_factor", 2))
        self.save_best = bool(training_cfg.get("save_best", True))
        self.save_final = bool(training_cfg.get("save_final", True))
        self.save_epoch_plots = bool(training_cfg.get("save_epoch_plots", True))
        self.console_log = bool(logging_cfg.get("console_log", True))
        self.show_progress_bar = bool(logging_cfg.get("show_progress_bar", True))

        sampling_cfg = config["sampling"]
        rollout_sampling_cfg = sampling_cfg.get("rollout", sampling_cfg.get("train"))
        if rollout_sampling_cfg is None:
            raise KeyError("GRPO 训练缺少 sampling.rollout 配置。")
        self.rollout_decode_config = build_decode_config(rollout_sampling_cfg, default_max_len=self.model.max_len)
        validate_policy_config(self.rollout_decode_config)

        if self.group_size < 2:
            raise ValueError(f"training.group_size 至少需要 2，当前为 {self.group_size}")
        if self.policy_forward_mode not in {"train", "eval"}:
            raise ValueError(
                "training.policy_forward_mode 只支持 'train' 或 'eval'，"
                f"当前为 {self.policy_forward_mode!r}"
            )
        if self.rollout_decode_config.decode == "greedy":
            raise ValueError("GRPO 训练的 sampling.rollout.decode 不能为 'greedy'。")

        self.tmm_kwargs = {
            "wavelength_range_um": tmm_cfg["wavelength_range_um"],
            "num_points": int(tmm_cfg["num_points"]),
            "incident_angle": float(tmm_cfg["incident_angle"]),
            "polarization": int(tmm_cfg["polarization"]),
            "metric": str(config["losses"]["spectrum_metric"]),
            "invalid_structure_penalty": float(config["losses"]["invalid_structure_penalty"]),
            "nonphysical_spectrum_penalty": float(
                config["losses"].get("nonphysical_spectrum_penalty", config["losses"]["invalid_structure_penalty"])
            ),
            "physical_tolerance": float(config["losses"].get("physical_tolerance", 0.01)),
            "database_path": str(config["paths"]["materials_dir"]),
            "material_aliases": tmm_cfg.get("material_aliases", {}),
            "return_spectra": False,
            "pad_to_max_layers": bool(tmm_cfg.get("pad_to_max_layers", True)),
            "bucket_by_layer_count": bool(tmm_cfg.get("bucket_by_layer_count", True)),
            "fixed_max_layers": tmm_cfg.get("fixed_max_layers"),
            "pad_material": str(tmm_cfg.get("pad_material", "Air")),
            "batch_size": int(tmm_cfg.get("batch_size", self.scoring_batch_size)),
            "tmm_debug": bool(tmm_cfg.get("debug", False)),
        }

        self.optimizer = torch.optim.AdamW(
            self.model.trainable_parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.metrics_dir = self.run_dir / "metrics"
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.plots_dir = self.run_dir / "plots"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.evaluator = SpectrumEvaluator(
            model=self.model,
            config=config,
            run_dir=self.run_dir,
            dist_ctx=dist_ctx,
        )

    def _log(self, message: str) -> None:
        if self.console_log and self.dist_ctx.is_main:
            print(message)

    def _reduce_mean(self, value: float) -> float:
        tensor = torch.tensor([float(value)], dtype=torch.float64, device=self.model.device)
        reduced = reduce_tensor(tensor, op="mean")
        return float(reduced.item())

    def _reduce_sum(self, value: float) -> float:
        tensor = torch.tensor([float(value)], dtype=torch.float64, device=self.model.device)
        reduced = reduce_tensor(tensor, op="sum")
        return float(reduced.item())

    def _reduce_mean_tensor(self, value: torch.Tensor) -> float:
        reduced = reduce_tensor(value.detach().to(dtype=torch.float64).reshape(1), op="mean")
        return float(reduced.item())

    def _reduce_sum_tensor(self, value: torch.Tensor) -> float:
        reduced = reduce_tensor(value.detach().to(dtype=torch.float64).reshape(1), op="sum")
        return float(reduced.item())

    def _make_progress(self, iterable, total: int, desc: str):
        """仅在主进程显示按 batch 数量统计的 tqdm 进度条。"""

        if not (self.console_log and self.show_progress_bar and self.dist_ctx.is_main):
            return iterable
        return tqdm(
            iterable,
            total=total,
            desc=desc,
            dynamic_ncols=True,
            leave=True,
            unit="batch",
        )

    def _update_progress_stage(self, progress, stage: str, **metrics: str) -> None:
        """在现有 batch 进度条上补充当前子阶段，便于定位卡顿位置。"""

        if not hasattr(progress, "set_postfix"):
            return
        payload = {"stage": stage}
        payload.update(metrics)
        progress.set_postfix(payload, refresh=False)

    def _zero_stats(self) -> dict[str, float | int | torch.Tensor]:
        zero = torch.zeros((), dtype=torch.float32, device=self.model.device)
        return {
            "sample_count": 0,
            "objective_mean": zero,
            "objective_sum": zero,
            "spectrum_mean": zero,
            "spectrum_sum": zero,
            "sequence_mean": zero,
            "sequence_sum": zero,
            "reward_mean": zero,
            "reward_sum": zero,
            "ratio_mean": zero,
            "ratio_sum": zero,
            "kl_mean": zero,
            "kl_sum": zero,
            "clip_fraction_mean": zero,
            "clip_fraction_sum": zero,
            "valid_ratio": 0.0,
            "valid_count": 0.0,
            "r_mean": 0.0,
            "r_sum": 0.0,
            "t_mean": 0.0,
            "t_sum": 0.0,
        }

    def _expand_target_spectra(self, spectra: Sequence[Sequence[float]]) -> Sequence[Sequence[float]]:
        """把 `[B, spec_dim]` 目标光谱扩成与 rollout 样本顺序一致的 `[B*K, spec_dim]`。"""

        if torch.is_tensor(spectra):
            if spectra.dim() == 1:
                spectra = spectra.view(1, -1)
            return spectra.repeat_interleave(self.group_size, dim=0)

        expanded_spectra: list[Sequence[float]] = []
        for target_spectrum in spectra:
            for _ in range(self.group_size):
                expanded_spectra.append(target_spectrum)
        return expanded_spectra

    def _rollout_policy_logprob_tensors(self, generated) -> tuple[torch.Tensor, torch.Tensor]:
        """把 rollout 期间记录的 old logprobs 整理成批量张量。"""

        sample_count = len(generated)
        max_len = max((len(item.policy_logprobs) for item in generated), default=0)
        if sample_count == 0 or max_len == 0:
            empty = torch.empty((sample_count, 0), dtype=torch.float32, device=self.model.device)
            return empty, torch.zeros((sample_count, 0), dtype=torch.bool, device=self.model.device)

        policy_logprobs = torch.zeros((sample_count, max_len), dtype=torch.float32, device=self.model.device)
        token_mask = torch.zeros((sample_count, max_len), dtype=torch.bool, device=self.model.device)
        for sample_idx, item in enumerate(generated):
            token_count = len(item.policy_logprobs)
            if token_count <= 0:
                continue
            row = torch.as_tensor(item.policy_logprobs, dtype=torch.float32, device=self.model.device)
            policy_logprobs[sample_idx, :token_count] = row
            token_mask[sample_idx, :token_count] = True
        return policy_logprobs, token_mask

    def _train_batch(
        self,
        spectra: Sequence[Sequence[float]],
        sample_indices: Sequence[int],
        sync_gradients: bool,
        progress=None,
    ) -> dict[str, float | int | torch.Tensor]:
        """执行一个 GRPO 训练 batch。"""

        target_count = len(sample_indices)
        if target_count == 0:
            return self._zero_stats()

        previous_training = bool(self.model.model.training)
        if self.policy_forward_mode == "eval":
            self.model.model.eval()
        else:
            self.model.model.train()

        try:
            self._update_progress_stage(progress, "generate")
            generated = generate_structures_for_targets(
                model=self.model,
                target_spectra=spectra,
                decode_config=self.rollout_decode_config,
                num_samples_per_target=self.group_size,
                target_indices=list(sample_indices),
                seeds=[int(self.config["experiment"]["seed"]) + int(sample_index) for sample_index in sample_indices],
            )
            if not generated:
                return self._zero_stats()

            expanded_spectra = self._expand_target_spectra(spectra)
            token_id_groups = [item.token_ids for item in generated]
            old_logprobs, old_token_mask = self._rollout_policy_logprob_tensors(generated)
            if old_logprobs.numel() == 0:
                return self._zero_stats()

            self._update_progress_stage(progress, "tmm")
            spectrum_aux = evaluate_generated_structures(
                structure_token_groups=[item.structure_tokens for item in generated],
                target_spectra=expanded_spectra,
                return_item_results=False,
                return_aux_arrays=True,
                **self.tmm_kwargs,
            )

            spectrum_loss_tensor = torch.as_tensor(
                spectrum_aux["spectrum_losses"],
                dtype=old_logprobs.dtype,
                device=self.model.device,
            )
            batch_sample_count = int(spectrum_loss_tensor.numel())
            if batch_sample_count == 0:
                return self._zero_stats()

            reward_tensor = -spectrum_loss_tensor
            advantage_tensor = group_relative_advantages(
                reward_tensor,
                target_count=target_count,
                group_size=self.group_size,
                mode=self.advantage_mode,
                eps=self.advantage_eps,
            )

            valid_count = float(np.sum(spectrum_aux["ok_mask"].astype(np.float32)))
            valid_ratio = valid_count / float(batch_sample_count) if batch_sample_count > 0 else 0.0
            r_rmse = np.asarray(spectrum_aux["r_rmse"], dtype=np.float32)
            t_rmse = np.asarray(spectrum_aux["t_rmse"], dtype=np.float32)

            sync_context = nullcontext()
            if not sync_gradients and hasattr(self.model.model, "no_sync"):
                sync_context = self.model.model.no_sync()

            with sync_context:
                self._update_progress_stage(progress, "score")
                current_logprobs, token_mask = sequence_logprobs_multi_target_batch_tensor(
                    model=self.model,
                    target_spectra=expanded_spectra,
                    token_id_groups=token_id_groups,
                    start_symbol=self.rollout_decode_config.start_symbol,
                    start_mat=self.rollout_decode_config.start_mat,
                    decode_config=self.rollout_decode_config,
                    require_grad=True,
                    batch_size=self.scoring_batch_size,
                )
                if current_logprobs.shape != old_logprobs.shape:
                    raise ValueError(
                        "rollout old logprob 与当前重算 logprob 形状不一致，"
                        f"当前为 {tuple(old_logprobs.shape)} vs {tuple(current_logprobs.shape)}"
                    )
                if not torch.equal(token_mask, old_token_mask):
                    raise ValueError("rollout token_mask 与当前 teacher forcing token_mask 不一致。")

                old_sequence_logprob = masked_sequence_logprob(
                    old_logprobs,
                    old_token_mask,
                    normalize_by_length=self.normalize_logprob_by_length,
                ).detach()
                current_sequence_logprob = masked_sequence_logprob(
                    current_logprobs,
                    token_mask,
                    normalize_by_length=self.normalize_logprob_by_length,
                )
                grpo_stats = grpo_clipped_surrogate(
                    current_sequence_logprob,
                    old_sequence_logprob,
                    advantage_tensor.detach(),
                    clip_epsilon=self.clip_epsilon,
                )
                objective = grpo_stats["surrogate"].mean()
                policy_loss = -objective
                self._update_progress_stage(progress, "backward")
                (policy_loss / float(self.grad_accum_steps)).backward()
        finally:
            if previous_training:
                self.model.model.train()
            else:
                self.model.model.eval()

        sequence_loss_tensor = -current_sequence_logprob.detach()
        objective_detached = objective.detach()
        ratio_mean = grpo_stats["ratio"].mean().detach()
        approx_kl_mean = grpo_stats["approx_kl"].mean().detach()
        clip_fraction_mean = grpo_stats["clip_mask"].to(dtype=torch.float32).mean().detach()
        reward_mean = reward_tensor.mean().detach()

        return {
            "sample_count": batch_sample_count,
            "objective_mean": objective_detached,
            "objective_sum": objective_detached * float(batch_sample_count),
            "spectrum_mean": spectrum_loss_tensor.mean().detach(),
            "spectrum_sum": spectrum_loss_tensor.sum().detach(),
            "sequence_mean": sequence_loss_tensor.mean(),
            "sequence_sum": sequence_loss_tensor.sum(),
            "reward_mean": reward_mean,
            "reward_sum": reward_tensor.sum().detach(),
            "ratio_mean": ratio_mean,
            "ratio_sum": ratio_mean * float(batch_sample_count),
            "kl_mean": approx_kl_mean,
            "kl_sum": approx_kl_mean * float(batch_sample_count),
            "clip_fraction_mean": clip_fraction_mean,
            "clip_fraction_sum": clip_fraction_mean * float(batch_sample_count),
            "valid_ratio": float(valid_ratio),
            "valid_count": float(valid_count),
            "r_mean": float(r_rmse.mean()) if r_rmse.size > 0 else 0.0,
            "r_sum": float(r_rmse.sum()),
            "t_mean": float(t_rmse.mean()) if t_rmse.size > 0 else 0.0,
            "t_sum": float(t_rmse.sum()),
        }

    def train(self, train_dataset, val_dataset=None) -> list[dict]:
        sampler = build_distributed_sampler(
            dataset=train_dataset,
            shuffle=True,
            seed=int(self.config["experiment"]["seed"]),
            drop_last=False,
        )
        dataloader_kwargs = {
            "dataset": train_dataset,
            "batch_size": self.batch_size,
            "shuffle": False if sampler is not None else True,
            "sampler": sampler,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": bool(self.num_workers > 0),
            "collate_fn": optogpt_batch_collator,
        }
        if self.num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = self.prefetch_factor
        dataloader = DataLoader(**dataloader_kwargs)

        train_rows: list[dict] = []
        best_val_spectrum = float("inf")
        global_step = 0

        for epoch in range(1, self.epochs + 1):
            if sampler is not None:
                sampler.set_epoch(epoch)
            self.model.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            epoch_objective_sum = torch.zeros((), dtype=torch.float32, device=self.model.device)
            epoch_spectrum_sum = torch.zeros((), dtype=torch.float32, device=self.model.device)
            epoch_sequence_sum = torch.zeros((), dtype=torch.float32, device=self.model.device)
            epoch_reward_sum = torch.zeros((), dtype=torch.float32, device=self.model.device)
            epoch_ratio_sum = torch.zeros((), dtype=torch.float32, device=self.model.device)
            epoch_kl_sum = torch.zeros((), dtype=torch.float32, device=self.model.device)
            epoch_clip_fraction_sum = torch.zeros((), dtype=torch.float32, device=self.model.device)
            epoch_r_rmse_sum = 0.0
            epoch_t_rmse_sum = 0.0
            epoch_valid_count = 0.0
            epoch_sample_count = 0
            accum_counter = 0

            progress = self._make_progress(
                dataloader,
                total=len(dataloader),
                desc=f"train epoch {epoch}/{self.epochs}",
            )
            for batch in progress:
                global_step += 1
                accum_counter += 1
                batch_stats = self._train_batch(
                    spectra=batch["spectra"],
                    sample_indices=batch["sample_indices"].tolist(),
                    sync_gradients=(accum_counter >= self.grad_accum_steps),
                    progress=progress,
                )
                epoch_objective_sum = epoch_objective_sum + batch_stats["objective_sum"]
                epoch_spectrum_sum = epoch_spectrum_sum + batch_stats["spectrum_sum"]
                epoch_sequence_sum = epoch_sequence_sum + batch_stats["sequence_sum"]
                epoch_reward_sum = epoch_reward_sum + batch_stats["reward_sum"]
                epoch_ratio_sum = epoch_ratio_sum + batch_stats["ratio_sum"]
                epoch_kl_sum = epoch_kl_sum + batch_stats["kl_sum"]
                epoch_clip_fraction_sum = epoch_clip_fraction_sum + batch_stats["clip_fraction_sum"]
                epoch_r_rmse_sum += float(batch_stats["r_sum"])
                epoch_t_rmse_sum += float(batch_stats["t_sum"])
                epoch_valid_count += float(batch_stats["valid_count"])
                epoch_sample_count += int(batch_stats["sample_count"])

                if accum_counter >= self.grad_accum_steps:
                    torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    accum_counter = 0

                if global_step == 1 or global_step % self.log_interval == 0:
                    reduced_objective = self._reduce_mean_tensor(batch_stats["objective_mean"])
                    reduced_spectrum = self._reduce_mean_tensor(batch_stats["spectrum_mean"])
                    reduced_sequence = self._reduce_mean_tensor(batch_stats["sequence_mean"])
                    reduced_reward = self._reduce_mean_tensor(batch_stats["reward_mean"])
                    reduced_ratio = self._reduce_mean_tensor(batch_stats["ratio_mean"])
                    reduced_kl = self._reduce_mean_tensor(batch_stats["kl_mean"])
                    reduced_clip = self._reduce_mean_tensor(batch_stats["clip_fraction_mean"])
                    reduced_r = self._reduce_mean(float(batch_stats["r_mean"]))
                    reduced_t = self._reduce_mean(float(batch_stats["t_mean"]))
                    reduced_valid = self._reduce_mean(float(batch_stats["valid_ratio"]))
                    if hasattr(progress, "set_postfix"):
                        progress.set_postfix(
                            objective=f"{reduced_objective:.4f}",
                            spectrum=f"{reduced_spectrum:.4f}",
                            seq=f"{reduced_sequence:.4f}",
                            reward=f"{reduced_reward:.4f}",
                            ratio=f"{reduced_ratio:.4f}",
                            clip=f"{reduced_clip:.3f}",
                            valid=f"{reduced_valid:.3f}",
                        )
                    self._log(
                        f"[train] epoch={epoch}/{self.epochs} step={global_step} "
                        f"objective={reduced_objective:.6f} "
                        f"spectrum={reduced_spectrum:.6f} "
                        f"sequence={reduced_sequence:.6f} "
                        f"reward={reduced_reward:.6f} "
                        f"ratio={reduced_ratio:.6f} "
                        f"kl={reduced_kl:.6f} "
                        f"clip={reduced_clip:.6f} "
                        f"r={reduced_r:.6f} "
                        f"t={reduced_t:.6f} "
                        f"valid={reduced_valid:.4f}"
                    )

            if hasattr(progress, "close"):
                progress.close()

            if accum_counter > 0:
                torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters(), self.grad_clip_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            if epoch_sample_count > 0:
                total_sample_count = self._reduce_sum(float(epoch_sample_count))
                mean_objective = self._reduce_sum_tensor(epoch_objective_sum) / total_sample_count
                mean_spectrum = self._reduce_sum_tensor(epoch_spectrum_sum) / total_sample_count
                mean_sequence = self._reduce_sum_tensor(epoch_sequence_sum) / total_sample_count
                mean_reward = self._reduce_sum_tensor(epoch_reward_sum) / total_sample_count
                mean_ratio = self._reduce_sum_tensor(epoch_ratio_sum) / total_sample_count
                mean_kl = self._reduce_sum_tensor(epoch_kl_sum) / total_sample_count
                mean_clip_fraction = self._reduce_sum_tensor(epoch_clip_fraction_sum) / total_sample_count
                mean_r_rmse = self._reduce_sum(epoch_r_rmse_sum) / total_sample_count
                mean_t_rmse = self._reduce_sum(epoch_t_rmse_sum) / total_sample_count
                mean_valid = self._reduce_sum(epoch_valid_count) / total_sample_count
            else:
                mean_objective = 0.0
                mean_spectrum = 0.0
                mean_sequence = 0.0
                mean_reward = 0.0
                mean_ratio = 0.0
                mean_kl = 0.0
                mean_clip_fraction = 0.0
                mean_r_rmse = 0.0
                mean_t_rmse = 0.0
                mean_valid = 0.0
            epoch_row = {
                "epoch": int(epoch),
                "global_step": int(global_step),
                "mean_objective": float(mean_objective),
                "mean_train_spectrum_loss": float(mean_spectrum),
                "mean_train_sequence_loss": float(mean_sequence),
                "mean_train_reward": float(mean_reward),
                "mean_train_ratio": float(mean_ratio),
                "mean_train_approx_kl": float(mean_kl),
                "mean_train_clip_fraction": float(mean_clip_fraction),
                "mean_train_r_rmse": float(mean_r_rmse),
                "mean_train_t_rmse": float(mean_t_rmse),
                "mean_train_valid_ratio": float(mean_valid),
                "val_sequence_loss": float("nan"),
                "val_spectrum_loss": float("nan"),
                "val_r_rmse": float("nan"),
                "val_t_rmse": float("nan"),
            }

            if val_dataset is not None and epoch % self.eval_every_epochs == 0:
                self.model.model.eval()
                val_row = self.evaluator.evaluate(val_dataset, split_name="val")
                if self.dist_ctx.is_main and val_row is not None:
                    epoch_row["val_sequence_loss"] = float(val_row["mean_sequence_loss"])
                    epoch_row["val_spectrum_loss"] = float(val_row["mean_spectrum_loss"])
                    epoch_row["val_r_rmse"] = float(val_row["mean_r_rmse"])
                    epoch_row["val_t_rmse"] = float(val_row["mean_t_rmse"])
                    if self.save_best and float(val_row["mean_spectrum_loss"]) < best_val_spectrum:
                        best_val_spectrum = float(val_row["mean_spectrum_loss"])
                        export_optogpt_checkpoint(
                            self.model,
                            self.checkpoints_dir / "best.pt",
                            {
                                "epoch": int(epoch),
                                "global_step": int(global_step),
                                "mean_spectrum_loss": float(best_val_spectrum),
                            },
                        )
                barrier()

            train_rows.append(epoch_row)
            if self.dist_ctx.is_main:
                write_summary_csv(self.metrics_dir / "train_metrics.csv", train_rows)

        if self.save_final and self.dist_ctx.is_main:
            export_optogpt_checkpoint(
                self.model,
                self.checkpoints_dir / "final.pt",
                {
                    "epoch": int(self.epochs),
                    "global_step": int(global_step),
                },
            )
        barrier()
        if self.save_epoch_plots and self.dist_ctx.is_main:
            try:
                save_grpo_epoch_summary_plot(
                    path=self.plots_dir / "epoch_summary.png",
                    rows=train_rows,
                )
            except ImportError as exc:
                self._log(f"[plot] skip epoch summary plot because {exc}")
        barrier()
        return train_rows
