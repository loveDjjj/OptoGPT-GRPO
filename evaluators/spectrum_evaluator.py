"""光谱评测器。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets import build_distributed_sampler, optogpt_batch_collator
from losses import evaluate_generated_structures, masked_mean_negative_logprob
from models.optogpt import build_decode_config, generate_structures_for_targets, sequence_logprobs_multi_target_batch_tensor
from utils.dist import DistributedContext, barrier
from utils.logging import append_jsonl, write_summary_csv
from utils.plotting import save_eval_distribution_summary, save_spectrum_comparison_plot
from .metrics import (
    DistributionPlotAccumulator,
    MetricAccumulator,
    reduce_distribution_plot_accumulator,
    reduce_metric_accumulator,
)


class SpectrumEvaluator:
    """对给定 checkpoint 执行光谱评测。"""

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
        evaluation_cfg = config["evaluation"]
        sampling_cfg = config["sampling"]["eval"]
        tmm_cfg = config["tmm"]
        logging_cfg = config.get("logging", {})

        self.batch_size = int(evaluation_cfg["batch_size"])
        self.scoring_batch_size = int(evaluation_cfg.get("scoring_batch_size", self.batch_size))
        self.num_workers = int(data_cfg.get("num_workers", 0))
        self.pin_memory = bool(data_cfg.get("pin_memory", False))
        self.prefetch_factor = int(data_cfg.get("prefetch_factor", 2))
        self.save_samples = bool(evaluation_cfg.get("save_samples", False))
        self.save_plots = bool(evaluation_cfg.get("save_plots", False))
        self.plot_max_samples = max(0, int(evaluation_cfg.get("plot_max_samples", 0)))
        self.save_distribution_plots = bool(evaluation_cfg.get("save_distribution_plots", True))
        distribution_cfg = evaluation_cfg.get("distribution_plots", {})
        self.rt_rmse_bins = max(1, int(distribution_cfg.get("rt_rmse_bins", 100)))
        self.rt_rmse_max = float(distribution_cfg.get("rt_rmse_max", 1.0))
        self.sequence_loss_bins = max(1, int(distribution_cfg.get("sequence_loss_bins", 100)))
        self.sequence_loss_max = float(distribution_cfg.get("sequence_loss_max", 10.0))
        self.length_max = int(distribution_cfg.get("length_max", tmm_cfg.get("fixed_max_layers", 20) or 20))
        self.decode_config = build_decode_config(sampling_cfg, default_max_len=self.model.max_len)
        self.metric = str(tmm_cfg.get("metric", config["losses"]["spectrum_metric"]))
        self.materials_dir = str(config["paths"]["materials_dir"])
        self.console_log = bool(logging_cfg.get("console_log", True))
        self.show_progress_bar = bool(logging_cfg.get("show_progress_bar", True))
        # 验证会在训练过程中反复调用，同一个样本的真值结构 id 没必要每个 epoch 重算。
        self._token_id_cache: dict[int, list[int]] = {}
        self._target_length_cache: dict[int, int] = {}

        self.tmm_kwargs = {
            "wavelength_range_um": tmm_cfg["wavelength_range_um"],
            "num_points": int(tmm_cfg["num_points"]),
            "incident_angle": float(tmm_cfg["incident_angle"]),
            "polarization": int(tmm_cfg["polarization"]),
            "metric": str(config["losses"]["spectrum_metric"]),
            "invalid_structure_penalty": float(config["losses"]["invalid_structure_penalty"]),
            "nonphysical_spectrum_penalty": float(config["losses"].get("nonphysical_spectrum_penalty", config["losses"]["invalid_structure_penalty"])),
            "physical_tolerance": float(config["losses"].get("physical_tolerance", 0.01)),
            "database_path": self.materials_dir,
            "material_aliases": tmm_cfg.get("material_aliases", {}),
            # 若要画样本图或统计图，就必须保留生成结构对应的光谱曲线。
            "return_spectra": bool(
                evaluation_cfg.get("save_predicted_spectra", False)
                or self.save_plots
                or self.save_distribution_plots
            ),
            "pad_to_max_layers": bool(tmm_cfg.get("pad_to_max_layers", True)),
            "bucket_by_layer_count": bool(tmm_cfg.get("bucket_by_layer_count", True)),
            "fixed_max_layers": tmm_cfg.get("fixed_max_layers"),
            "pad_material": str(tmm_cfg.get("pad_material", "Air")),
            "batch_size": int(tmm_cfg.get("batch_size", self.batch_size)),
            "tmm_debug": bool(tmm_cfg.get("debug", False)),
        }

        self.samples_dir = self.run_dir / "samples"
        self.metrics_dir = self.run_dir / "metrics"
        self.plots_dir = self.run_dir / "plots"
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def _log(self, message: str) -> None:
        if self.console_log and self.dist_ctx.is_main:
            print(message)

    def _sample_output_path(self, split_name: str) -> Path:
        return self.samples_dir / f"{split_name}.rank{self.dist_ctx.rank:02d}.jsonl"

    def _summary_output_path(self, split_name: str) -> Path:
        return self.metrics_dir / f"{split_name}_summary.csv"

    def _plot_output_dir(self, split_name: str) -> Path:
        return self.plots_dir / split_name / f"rank{self.dist_ctx.rank:02d}"

    def _distribution_plot_output_path(self, split_name: str) -> Path:
        return self.plots_dir / "summary" / f"{split_name}_distribution.png"

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
        """在现有 batch 进度条上补充当前子阶段，方便观察卡顿点。"""

        if not hasattr(progress, "set_postfix"):
            return
        payload = {"stage": stage}
        payload.update(metrics)
        progress.set_postfix(payload, refresh=False)

    def _cached_token_id_groups(
        self,
        sample_indices: Sequence[int],
        structure_tokens: Sequence[Sequence[str]],
    ) -> tuple[list[list[int]], np.ndarray]:
        """缓存真值结构的 token id，避免训练中每轮验证都重新编码。"""

        token_id_groups: list[list[int]] = []
        target_lengths = np.empty((len(sample_indices),), dtype=np.int64)
        for row_idx, (sample_index, tokens) in enumerate(zip(sample_indices, structure_tokens)):
            cache_key = int(sample_index)
            token_ids = self._token_id_cache.get(cache_key)
            if token_ids is None:
                token_ids = list(self.model.structure_tokens_to_ids(tokens))
                self._token_id_cache[cache_key] = token_ids
                self._target_length_cache[cache_key] = len(tokens)
            token_id_groups.append(token_ids)
            target_lengths[row_idx] = self._target_length_cache[cache_key]
        return token_id_groups, target_lengths

    def evaluate(self, dataset, split_name: str) -> dict | None:
        """执行一次完整评测。"""

        sampler = build_distributed_sampler(
            dataset=dataset,
            shuffle=False,
            seed=int(self.config["experiment"]["seed"]),
            drop_last=False,
        )
        dataloader_kwargs = {
            "dataset": dataset,
            "batch_size": self.batch_size,
            "shuffle": False if sampler is not None else False,
            "sampler": sampler,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": bool(self.num_workers > 0),
            "collate_fn": optogpt_batch_collator,
        }
        if self.num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = self.prefetch_factor
        dataloader = DataLoader(**dataloader_kwargs)

        self.model.model.eval()
        metric_accumulator = MetricAccumulator()
        distribution_accumulator = (
            DistributionPlotAccumulator(
                rt_rmse_bins=self.rt_rmse_bins,
                rt_rmse_max=self.rt_rmse_max,
                sequence_loss_bins=self.sequence_loss_bins,
                sequence_loss_max=self.sequence_loss_max,
                length_max=self.length_max,
            )
            if self.save_distribution_plots
            else None
        )
        sample_output_path = self._sample_output_path(split_name)
        plot_output_dir = self._plot_output_dir(split_name)
        plots_saved = 0
        if sample_output_path.exists():
            sample_output_path.unlink()
        if self.save_plots:
            plot_output_dir.mkdir(parents=True, exist_ok=True)

        progress = self._make_progress(
            dataloader,
            total=len(dataloader),
            desc=f"eval {split_name}",
        )
        needs_item_results = bool(self.save_samples or self.save_plots)
        with torch.inference_mode():
            for batch in progress:
                spectra = batch["spectra"]
                structure_tokens = batch["structure_tokens"]
                sample_indices = batch["sample_indices"].tolist()
                token_id_groups, target_lengths = self._cached_token_id_groups(sample_indices, structure_tokens)

                self._update_progress_stage(progress, "score")
                logprobs, token_mask = sequence_logprobs_multi_target_batch_tensor(
                    model=self.model,
                    target_spectra=spectra,
                    token_id_groups=token_id_groups,
                    start_symbol=self.decode_config.start_symbol,
                    start_mat=self.decode_config.start_mat,
                    require_grad=False,
                    batch_size=self.scoring_batch_size,
                )
                sequence_losses_tensor = masked_mean_negative_logprob(
                    token_logprobs=logprobs,
                    token_mask=token_mask,
                    normalize_by_length=True,
                )

                self._update_progress_stage(progress, "generate")
                generated = generate_structures_for_targets(
                    model=self.model,
                    target_spectra=spectra,
                    decode_config=self.decode_config,
                    num_samples_per_target=1,
                    target_indices=sample_indices,
                    seeds=[int(self.config["experiment"]["seed"]) + int(sample_index) for sample_index in sample_indices],
                )
                self._update_progress_stage(progress, "tmm")
                spectrum_eval_output = evaluate_generated_structures(
                    structure_token_groups=[item.structure_tokens for item in generated],
                    target_spectra=spectra,
                    return_aux_arrays=True,
                    return_item_results=needs_item_results,
                    **self.tmm_kwargs,
                )
                if needs_item_results:
                    spectrum_results, spectrum_aux = spectrum_eval_output
                else:
                    spectrum_results = None
                    spectrum_aux = spectrum_eval_output

                sequence_losses = sequence_losses_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
                spectrum_losses = spectrum_aux["spectrum_losses"]
                ok_mask = spectrum_aux["ok_mask"]
                r_rmse = spectrum_aux["r_rmse"]
                t_rmse = spectrum_aux["t_rmse"]
                generated_lengths = np.asarray([len(item.structure_tokens) for item in generated], dtype=np.int64)
                metric_accumulator.update_batch(
                    sequence_losses=sequence_losses,
                    spectrum_losses=spectrum_losses,
                    ok_mask=ok_mask,
                    r_rmse=r_rmse,
                    t_rmse=t_rmse,
                )

                if distribution_accumulator is not None:
                    distribution_accumulator.update_batch(
                        r_rmse=r_rmse,
                        t_rmse=t_rmse,
                        sequence_loss=sequence_losses,
                        generated_length=generated_lengths,
                        target_length=target_lengths,
                    )

                if needs_item_results and spectrum_results is not None:
                    self._update_progress_stage(progress, "write")
                    for sample_index, gt_tokens, target_spectrum, sequence_loss, generated_item, spectrum_result in zip(
                        sample_indices,
                        structure_tokens,
                        spectra,
                        sequence_losses.tolist(),
                        generated,
                        spectrum_results,
                    ):
                        if self.save_samples:
                            append_jsonl(
                                sample_output_path,
                                {
                                    "sample_index": int(sample_index),
                                    "ground_truth_structure": list(gt_tokens),
                                    "generated": asdict(generated_item),
                                    "sequence_loss": float(sequence_loss),
                                    "spectrum_loss": float(spectrum_result["spectrum_loss"]),
                                    "r_rmse": float(spectrum_result.get("r_rmse", spectrum_result["spectrum_loss"])),
                                    "t_rmse": float(spectrum_result.get("t_rmse", spectrum_result["spectrum_loss"])),
                                    "status": str(spectrum_result["status"]),
                                },
                            )
                        if self.save_plots and plots_saved < self.plot_max_samples:
                            predicted_spectrum = spectrum_result.get("predicted_spectrum")
                            wavelengths_um = spectrum_result.get("wavelengths_um")
                            if predicted_spectrum is not None and wavelengths_um is not None:
                                save_spectrum_comparison_plot(
                                    path=plot_output_dir / f"sample_{int(sample_index):08d}.png",
                                    target_spectrum=target_spectrum,
                                    predicted_spectrum=predicted_spectrum,
                                    wavelengths_um=wavelengths_um,
                                    title=f"{split_name} sample={int(sample_index)}",
                                    spectrum_loss=float(spectrum_result["spectrum_loss"]),
                                    status=str(spectrum_result["status"]),
                                )
                                plots_saved += 1

                if hasattr(progress, "set_postfix"):
                    progress.set_postfix(
                        seq=f"{metric_accumulator.sequence_loss_sum / max(metric_accumulator.sample_count, 1):.4f}",
                        spec=f"{metric_accumulator.spectrum_loss_sum / max(metric_accumulator.sample_count, 1):.4f}",
                        r=f"{metric_accumulator.r_rmse_sum / max(metric_accumulator.sample_count, 1):.4f}",
                        t=f"{metric_accumulator.t_rmse_sum / max(metric_accumulator.sample_count, 1):.4f}",
                        samples=int(metric_accumulator.sample_count),
                    )

        if hasattr(progress, "close"):
            progress.close()

        merged_metrics = reduce_metric_accumulator(metric_accumulator, device=self.model.device)
        merged_distribution = (
            reduce_distribution_plot_accumulator(distribution_accumulator, device=self.model.device)
            if distribution_accumulator is not None
            else None
        )
        barrier()
        if not self.dist_ctx.is_main:
            return None

        summary_row = merged_metrics.to_summary_row(
            split=split_name,
            checkpoint_path=self.model.checkpoint_path,
        )
        write_summary_csv(self._summary_output_path(split_name), [summary_row])
        if merged_distribution is not None:
            save_eval_distribution_summary(
                path=self._distribution_plot_output_path(split_name),
                split_name=split_name,
                r_rmse_hist=merged_distribution.r_rmse_hist,
                t_rmse_hist=merged_distribution.t_rmse_hist,
                sequence_loss_hist=merged_distribution.sequence_loss_hist,
                length_heatmap=merged_distribution.length_heatmap,
                rt_rmse_max=self.rt_rmse_max,
                sequence_loss_max=self.sequence_loss_max,
                length_max=self.length_max,
            )
        self._log(
            f"[eval:{split_name}] samples={summary_row['sample_count']} "
            f"seq={summary_row['mean_sequence_loss']:.6f} "
            f"spec={summary_row['mean_spectrum_loss']:.6f} "
            f"r={summary_row['mean_r_rmse']:.6f} "
            f"t={summary_row['mean_t_rmse']:.6f}"
        )
        return summary_row
