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
        self.num_workers = int(data_cfg.get("num_workers", 0))
        self.pin_memory = bool(data_cfg.get("pin_memory", False))
        self.prefetch_factor = int(data_cfg.get("prefetch_factor", 2))
        self.save_best = bool(training_cfg.get("save_best", True))
        self.save_final = bool(training_cfg.get("save_final", True))
        self.console_log = bool(logging_cfg.get("console_log", True))
        self.show_progress_bar = bool(logging_cfg.get("show_progress_bar", True))

        self.train_decode_config = build_decode_config(config["sampling"]["train"], default_max_len=self.model.max_len)
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
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

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

    def _reduce_mean_tensor(self, value: torch.Tensor) -> float:
        reduced = reduce_tensor(value.detach().to(dtype=torch.float64).reshape(1), op="mean")
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

    def _train_batch(
        self,
        spectra: Sequence[Sequence[float]],
        sample_indices: Sequence[int],
        sync_gradients: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """执行一个训练 batch。

        这里采用的是“光谱风险最小化”式目标：
        - 先根据当前模型生成结构；
        - 再用 TMM 计算这些生成结构的光谱损失；
        - 最后用去中心化后的光谱损失作为权重，调整这些生成序列的对数概率。

        这不是 PPO/GRPO，也不引入参考模型和裁剪项；训练目标完全来自光谱误差。
        """

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
                return zero, zero, 0.0

            token_mask_float = token_mask.to(dtype=logprobs.dtype)
            lengths = token_mask.sum(dim=-1).clamp_min(1).to(dtype=logprobs.dtype)
            sequence_logprob = (logprobs * token_mask_float).sum(dim=-1)
            if self.normalize_logprob_by_length:
                sequence_logprob = sequence_logprob / lengths

            spectrum_loss_tensor = torch.as_tensor(
                spectrum_aux["spectrum_losses"],
                dtype=logprobs.dtype,
                device=self.model.device,
            )
            valid_ratio = float(np.mean(spectrum_aux["ok_mask"].astype(np.float32)))

            weight_tensor = spectrum_loss_tensor
            if self.center_spectrum_loss and spectrum_loss_tensor.numel() > 1:
                weight_tensor = spectrum_loss_tensor - spectrum_loss_tensor.mean()

            # 注意：这里优化的是 E[loss * logprob]。
            # 开启中心化后，低于 batch 均值的样本会得到负权重，从而被提升概率；
            # 高于 batch 均值的样本得到正权重，会被压低概率。
            # 这条语义依赖 center_spectrum_loss=True，因此保持显式注释。
            objective = torch.mean(weight_tensor.detach() * sequence_logprob)
            objective.backward()

        return (
            objective.detach(),
            spectrum_loss_tensor.mean().detach(),
            float(valid_ratio),
        )

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
            self.model.raw_model.train()
            self.optimizer.zero_grad(set_to_none=True)

            epoch_objective_sum = torch.zeros((), dtype=torch.float32, device=self.model.device)
            epoch_spectrum_sum = torch.zeros((), dtype=torch.float32, device=self.model.device)
            epoch_valid_ratio_sum = 0.0
            epoch_batch_count = 0
            accum_counter = 0

            progress = self._make_progress(
                dataloader,
                total=len(dataloader),
                desc=f"train epoch {epoch}/{self.epochs}",
            )
            for batch in progress:
                global_step += 1
                accum_counter += 1
                batch_objective_tensor, batch_spectrum_loss_tensor, batch_valid_ratio = self._train_batch(
                    spectra=batch["spectra"],
                    sample_indices=batch["sample_indices"].tolist(),
                    sync_gradients=(accum_counter >= self.grad_accum_steps),
                )
                epoch_objective_sum = epoch_objective_sum + batch_objective_tensor
                epoch_spectrum_sum = epoch_spectrum_sum + batch_spectrum_loss_tensor
                epoch_valid_ratio_sum += float(batch_valid_ratio)
                epoch_batch_count += 1

                if accum_counter >= self.grad_accum_steps:
                    torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    accum_counter = 0

                if global_step == 1 or global_step % self.log_interval == 0:
                    reduced_objective = self._reduce_mean_tensor(batch_objective_tensor)
                    reduced_spectrum = self._reduce_mean_tensor(batch_spectrum_loss_tensor)
                    reduced_valid = self._reduce_mean(batch_valid_ratio)
                    if hasattr(progress, "set_postfix"):
                        progress.set_postfix(
                            objective=f"{reduced_objective:.4f}",
                            spectrum=f"{reduced_spectrum:.4f}",
                            valid=f"{reduced_valid:.3f}",
                        )
                    self._log(
                        f"[train] epoch={epoch}/{self.epochs} step={global_step} "
                        f"objective={reduced_objective:.6f} "
                        f"spectrum={reduced_spectrum:.6f} "
                        f"valid={reduced_valid:.4f}"
                    )

            if hasattr(progress, "close"):
                progress.close()

            if accum_counter > 0:
                torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters(), self.grad_clip_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            if epoch_batch_count > 0:
                mean_objective = self._reduce_mean_tensor(epoch_objective_sum / float(epoch_batch_count))
                mean_spectrum = self._reduce_mean_tensor(epoch_spectrum_sum / float(epoch_batch_count))
                mean_valid = self._reduce_mean(epoch_valid_ratio_sum / float(epoch_batch_count))
            else:
                mean_objective = 0.0
                mean_spectrum = 0.0
                mean_valid = 0.0
            epoch_row = {
                "epoch": int(epoch),
                "global_step": int(global_step),
                "mean_objective": float(mean_objective),
                "mean_train_spectrum_loss": float(mean_spectrum),
                "mean_train_valid_ratio": float(mean_valid),
            }

            if val_dataset is not None and epoch % self.eval_every_epochs == 0:
                self.model.raw_model.eval()
                val_row = self.evaluator.evaluate(val_dataset, split_name="val")
                if self.dist_ctx.is_main and val_row is not None:
                    epoch_row["val_sequence_loss"] = float(val_row["mean_sequence_loss"])
                    epoch_row["val_spectrum_loss"] = float(val_row["mean_spectrum_loss"])
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
        return train_rows
