"""光谱评测器。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from torch.utils.data import DataLoader

from datasets import build_distributed_sampler, optogpt_batch_collator
from losses import evaluate_generated_structures, masked_mean_negative_logprob
from models.optogpt import build_decode_config, generate_structures_for_targets, sequence_logprobs_multi_target_batch_tensor
from utils.dist import DistributedContext, barrier
from utils.logging import append_jsonl, write_summary_csv
from .metrics import MetricAccumulator, reduce_metric_accumulator


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
        self.decode_config = build_decode_config(sampling_cfg, default_max_len=self.model.max_len)
        self.metric = str(tmm_cfg.get("metric", config["losses"]["spectrum_metric"]))
        self.materials_dir = str(config["paths"]["materials_dir"])
        self.console_log = bool(logging_cfg.get("console_log", True))

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
            "return_spectra": bool(evaluation_cfg.get("save_predicted_spectra", False)),
            "pad_to_max_layers": bool(tmm_cfg.get("pad_to_max_layers", True)),
            "bucket_by_layer_count": bool(tmm_cfg.get("bucket_by_layer_count", True)),
            "pad_material": str(tmm_cfg.get("pad_material", "Air")),
            "batch_size": int(tmm_cfg.get("batch_size", self.batch_size)),
            "tmm_debug": bool(tmm_cfg.get("debug", False)),
        }

        self.samples_dir = self.run_dir / "samples"
        self.metrics_dir = self.run_dir / "metrics"
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def _log(self, message: str) -> None:
        if self.console_log and self.dist_ctx.is_main:
            print(message)

    def _sample_output_path(self, split_name: str) -> Path:
        return self.samples_dir / f"{split_name}.rank{self.dist_ctx.rank:02d}.jsonl"

    def _summary_output_path(self, split_name: str) -> Path:
        return self.metrics_dir / f"{split_name}_summary.csv"

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

        self.model.raw_model.eval()
        metric_accumulator = MetricAccumulator()
        sample_output_path = self._sample_output_path(split_name)
        if sample_output_path.exists():
            sample_output_path.unlink()

        for batch in dataloader:
            spectra = batch["spectra"]
            structure_tokens = batch["structure_tokens"]
            sample_indices = batch["sample_indices"].tolist()
            token_id_groups = [self.model.structure_tokens_to_ids(tokens) for tokens in structure_tokens]

            logprobs, token_mask = sequence_logprobs_multi_target_batch_tensor(
                model=self.model,
                target_spectra=spectra,
                token_id_groups=token_id_groups,
                start_symbol=self.decode_config.start_symbol,
                start_mat=self.decode_config.start_mat,
                require_grad=False,
                batch_size=self.scoring_batch_size,
            )
            sequence_losses = masked_mean_negative_logprob(
                token_logprobs=logprobs,
                token_mask=token_mask,
                normalize_by_length=True,
            ).detach().cpu().tolist()

            generated = generate_structures_for_targets(
                model=self.model,
                target_spectra=spectra,
                decode_config=self.decode_config,
                num_samples_per_target=1,
                target_indices=sample_indices,
                seeds=[int(self.config["experiment"]["seed"]) + int(sample_index) for sample_index in sample_indices],
            )
            spectrum_results = evaluate_generated_structures(
                structure_token_groups=[item.structure_tokens for item in generated],
                target_spectra=spectra,
                **self.tmm_kwargs,
            )

            for sample_index, gt_tokens, sequence_loss, generated_item, spectrum_result in zip(
                sample_indices,
                structure_tokens,
                sequence_losses,
                generated,
                spectrum_results,
            ):
                metric_accumulator.update(
                    sequence_loss=float(sequence_loss),
                    spectrum_loss=float(spectrum_result["spectrum_loss"]),
                    status=str(spectrum_result["status"]),
                )
                if self.save_samples:
                    append_jsonl(
                        sample_output_path,
                        {
                            "sample_index": int(sample_index),
                            "ground_truth_structure": list(gt_tokens),
                            "generated": asdict(generated_item),
                            "sequence_loss": float(sequence_loss),
                            "spectrum_loss": float(spectrum_result["spectrum_loss"]),
                            "status": str(spectrum_result["status"]),
                        },
                    )

        merged_metrics = reduce_metric_accumulator(metric_accumulator, device=self.model.device)
        barrier()
        if not self.dist_ctx.is_main:
            return None

        summary_row = merged_metrics.to_summary_row(
            split=split_name,
            checkpoint_path=self.model.checkpoint_path,
        )
        write_summary_csv(self._summary_output_path(split_name), [summary_row])
        self._log(
            f"[eval:{split_name}] samples={summary_row['sample_count']} "
            f"seq={summary_row['mean_sequence_loss']:.6f} "
            f"spec={summary_row['mean_spectrum_loss']:.6f}"
        )
        return summary_row
