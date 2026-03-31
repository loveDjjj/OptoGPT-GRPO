"""基于光谱损失的微调训练器。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from contextlib import nullcontext
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets import build_distributed_sampler, optogpt_batch_collator
from evaluators import SpectrumEvaluator
from losses import evaluate_generated_structures
from models.optogpt import build_decode_config, export_optogpt_checkpoint, generate_structures_for_targets, sequence_logprobs_multi_target_batch_tensor
from utils.dist import DistributedContext, barrier, reduce_tensor
from utils.logging import write_summary_csv
from utils.plotting import save_sft_epoch_summary_plot


class SpectralSFTTrainer:
    """只基于生成结构的光谱损失做更新的训练器。"""

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
        self.center_spectrum_loss = bool(training_cfg.get("center_spectrum_loss", True))
        self.normalize_logprob_by_length = bool(training_cfg.get("normalize_logprob_by_length", True))
        self.train_num_samples_per_target = int(training_cfg.get("num_samples_per_target", 1))
        self.policy_forward_mode = str(training_cfg.get("policy_forward_mode", "eval")).strip().lower()
        self.num_workers = int(data_cfg.get("num_workers", 0))
        self.pin_memory = bool(data_cfg.get("pin_memory", False))
        self.prefetch_factor = int(data_cfg.get("prefetch_factor", 2))
        self.save_best = bool(training_cfg.get("save_best", True))
        self.save_final = bool(training_cfg.get("save_final", True))
        self.save_epoch_plots = bool(training_cfg.get("save_epoch_plots", True))
        self.console_log = bool(logging_cfg.get("console_log", True))
        self.show_progress_bar = bool(logging_cfg.get("show_progress_bar", True))

        self.train_decode_config = build_decode_config(config["sampling"]["train"], default_max_len=self.model.max_len)
        if self.policy_forward_mode not in {"train", "eval"}:
            raise ValueError(
                "training.policy_forward_mode 只支持 'train' 或 'eval'，"
                f"当前为 {self.policy_forward_mode!r}"
            )
        self.tmm_kwargs = {
            "wavelength_range_um": tmm_cfg["wavelength_range_um"],
            "num_points": int(tmm_cfg["num_points"]),
            "incident_angle": float(tmm_cfg["incident_angle"]),
            "polarization": int(tmm_cfg["polarization"]),
            "metric": str(config["losses"]["spectrum_metric"]),
            "invalid_structure_penalty": float(config["losses"]["invalid_structure_penalty"]),
            "nonphysical_spectrum_penalty": float(config["losses"].get("nonphysical_spectrum_penalty", config["losses"]["invalid_structure_penalty"])),
            "physical_tolerance": float(config["losses"].get("physical_tolerance", 0.01)),
            "database_path": str(config["paths"]["materials_dir"]),
            "material_aliases": tmm_cfg.get("material_aliases", {}),
            "return_spectra": False,
            "pad_to_max_layers": bool(tmm_cfg.get("pad_to_max_layers", True)),
            "bucket_by_layer_count": bool(tmm_cfg.get("bucket_by_layer_count", True)),
            "fixed_max_layers": tmm_cfg.get("fixed_max_layers"),
            "pad_material": str(tmm_cfg.get("pad_material", "Air")),
            "batch_size": int(tmm_cfg.get("batch_size", self.batch_size)),
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

    def _train_batch(
        self,
        spectra: Sequence[Sequence[float]],
        sample_indices: Sequence[int],
        sync_gradients: bool,
        progress=None,
    ) -> dict[str, float | int | torch.Tensor]:
        """执行一个训练 batch。

        这里采用的是“光谱风险最小化”式目标：
        - 先根据当前模型生成结构；
        - 再用 TMM 计算这些生成结构的光谱损失；
        - 最后用去中心化后的光谱损失作为权重，调整这些生成序列的对数概率。

        这不是 PPO/GRPO，也不引入参考模型和裁剪项；训练目标完全来自光谱误差。
        """
        # 这里把 rollout 和 score 两次前向统一到同一个策略模式下。
        # 当 policy_forward_mode='eval' 时，只是关闭 dropout 等训练态随机层；
        # scoring 仍会通过 require_grad=True 保留 autograd，并正常反传参数。
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
                decode_config=self.train_decode_config,
                num_samples_per_target=self.train_num_samples_per_target,
                target_indices=list(sample_indices),
                seeds=[int(self.config["experiment"]["seed"]) + int(sample_index) for sample_index in sample_indices],
            )
            if torch.is_tensor(spectra):
                expanded_spectra = spectra.repeat((self.train_num_samples_per_target, 1))
            else:
                expanded_spectra = []
                for _ in range(self.train_num_samples_per_target):
                    expanded_spectra.extend(list(spectra))
            # 训练阶段只需要光谱损失和有效样本掩码，不需要逐样本 dict。
            # 这里直接走批量数组快路，减少 Python 对象构造和二次遍历。
            self._update_progress_stage(progress, "tmm")
            spectrum_aux = evaluate_generated_structures(
                structure_token_groups=[item.structure_tokens for item in generated],
                target_spectra=expanded_spectra,
                return_item_results=False,
                return_aux_arrays=True,
                **self.tmm_kwargs,
            )
            token_id_groups = [item.token_ids for item in generated]
            sync_context = nullcontext()
            if not sync_gradients and hasattr(self.model.model, "no_sync"):
                sync_context = self.model.model.no_sync()

            with sync_context:
                self._update_progress_stage(progress, "score")
                logprobs, token_mask = sequence_logprobs_multi_target_batch_tensor(
                    model=self.model,
                    target_spectra=expanded_spectra,
                    token_id_groups=token_id_groups,
                    start_symbol=self.train_decode_config.start_symbol,
                    start_mat=self.train_decode_config.start_mat,
                    require_grad=True,
                    batch_size=self.batch_size,
                )
                if logprobs.numel() == 0:
                    zero = torch.zeros((), dtype=torch.float32, device=self.model.device)
                    return {
                        "sample_count": 0,
                        "objective_mean": zero,
                        "objective_sum": zero,
                        "spectrum_mean": zero,
                        "spectrum_sum": zero,
                        "sequence_mean": zero,
                        "sequence_sum": zero,
                        "valid_ratio": 0.0,
                        "valid_count": 0.0,
                        "r_mean": 0.0,
                        "r_sum": 0.0,
                        "t_mean": 0.0,
                        "t_sum": 0.0,
                    }

                token_mask_float = token_mask.to(dtype=logprobs.dtype)
                lengths = token_mask.sum(dim=-1).clamp_min(1).to(dtype=logprobs.dtype)
                sequence_logprob = (logprobs * token_mask_float).sum(dim=-1)
                if self.normalize_logprob_by_length:
                    sequence_logprob = sequence_logprob / lengths
                sequence_loss_tensor = -sequence_logprob.detach()

                spectrum_loss_tensor = torch.as_tensor(
                    spectrum_aux["spectrum_losses"],
                    dtype=logprobs.dtype,
                    device=self.model.device,
                )
                batch_sample_count = int(spectrum_loss_tensor.numel())
                valid_count = float(np.sum(spectrum_aux["ok_mask"].astype(np.float32)))
                valid_ratio = valid_count / float(batch_sample_count) if batch_sample_count > 0 else 0.0
                r_rmse = np.asarray(spectrum_aux["r_rmse"], dtype=np.float32)
                t_rmse = np.asarray(spectrum_aux["t_rmse"], dtype=np.float32)

                weight_tensor = spectrum_loss_tensor
                if self.center_spectrum_loss and spectrum_loss_tensor.numel() > 1:
                    weight_tensor = spectrum_loss_tensor - spectrum_loss_tensor.mean()

                # 注意：这里优化的是 E[loss * logprob]。
                # 开启中心化后，低于 batch 均值的样本会得到负权重，从而被提升概率；
                # 高于 batch 均值的样本得到正权重，会被压低概率。
                # 这条语义依赖 center_spectrum_loss=True，因此保持显式注释。
                objective = torch.mean(weight_tensor.detach() * sequence_logprob)
                self._update_progress_stage(progress, "backward")
                objective.backward()
        finally:
            if previous_training:
                self.model.model.train()
            else:
                self.model.model.eval()

        objective_detached = objective.detach()
        return {
            "sample_count": batch_sample_count,
            "objective_mean": objective_detached,
            "objective_sum": objective_detached * float(batch_sample_count),
            "spectrum_mean": spectrum_loss_tensor.mean().detach(),
            "spectrum_sum": spectrum_loss_tensor.sum().detach(),
            "sequence_mean": sequence_loss_tensor.mean(),
            "sequence_sum": sequence_loss_tensor.sum(),
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
                    reduced_r = self._reduce_mean(float(batch_stats["r_mean"]))
                    reduced_t = self._reduce_mean(float(batch_stats["t_mean"]))
                    reduced_valid = self._reduce_mean(float(batch_stats["valid_ratio"]))
                    if hasattr(progress, "set_postfix"):
                        progress.set_postfix(
                            objective=f"{reduced_objective:.4f}",
                            spectrum=f"{reduced_spectrum:.4f}",
                            seq=f"{reduced_sequence:.4f}",
                            r=f"{reduced_r:.4f}",
                            t=f"{reduced_t:.4f}",
                            valid=f"{reduced_valid:.3f}",
                        )
                    self._log(
                        f"[train] epoch={epoch}/{self.epochs} step={global_step} "
                        f"objective={reduced_objective:.6f} "
                        f"spectrum={reduced_spectrum:.6f} "
                        f"sequence={reduced_sequence:.6f} "
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
                mean_r_rmse = self._reduce_sum(epoch_r_rmse_sum) / total_sample_count
                mean_t_rmse = self._reduce_sum(epoch_t_rmse_sum) / total_sample_count
                mean_valid = self._reduce_sum(epoch_valid_count) / total_sample_count
            else:
                mean_objective = 0.0
                mean_spectrum = 0.0
                mean_sequence = 0.0
                mean_r_rmse = 0.0
                mean_t_rmse = 0.0
                mean_valid = 0.0
            epoch_row = {
                "epoch": int(epoch),
                "global_step": int(global_step),
                "mean_objective": float(mean_objective),
                "mean_train_spectrum_loss": float(mean_spectrum),
                "mean_train_sequence_loss": float(mean_sequence),
                "mean_train_r_rmse": float(mean_r_rmse),
                "mean_train_t_rmse": float(mean_t_rmse),
                "mean_train_valid_ratio": float(mean_valid),
                "val_sequence_loss": float("nan"),
                "val_spectrum_loss": float("nan"),
                "val_r_rmse": float("nan"),
                "val_t_rmse": float("nan"),
            }

            if val_dataset is not None and epoch % self.eval_every_epochs == 0:
                self.model.raw_model.eval()
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
                save_sft_epoch_summary_plot(
                    path=self.plots_dir / "epoch_summary.png",
                    rows=train_rows,
                )
            except ImportError as exc:
                self._log(f"[plot] skip epoch summary plot because {exc}")
        barrier()
        return train_rows
