"""Dataset-level GRPO trainer for OptoGPT.

This module owns the end-to-end RL loop after the target dataset has already
been created:

- sample structures for each target,
- score them with the TMM reward,
- compute per-target group-relative advantages,
- update one shared policy,
- periodically evaluate on a fixed validation target set.

The trainer is intentionally separated from ``run_grpo.py`` so orchestration and
optimization logic do not get mixed together.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None

from policy.optogpt_policy import OptoGPTPolicy
from rewards.tmm_reward import evaluate_structures_with_tmm
from rollouts.optogpt_sampler import sample_unique_rollout_groups
from utils.logging import write_summary_csv
from utils.plotting import save_before_after_plot, save_metric_curve


RolloutLogger = Optional[Callable[[Dict[str, Any]], None]]


class _ProgressHandle:
    """Minimal wrapper that uses tqdm when available and degrades gracefully."""

    def __init__(self, total: int, desc: str, enabled: bool, leave: bool = True) -> None:
        self.enabled = bool(enabled and _tqdm is not None)
        self.desc = desc
        self.total = int(total)
        self.current = 0
        self._bar = _tqdm(total=total, desc=desc, leave=leave, dynamic_ncols=True) if self.enabled else None

    def update(self, n: int = 1) -> None:
        self.current += int(n)
        if self._bar is not None:
            self._bar.update(n)

    def set_postfix(self, values: Mapping[str, Any]) -> None:
        if self._bar is not None:
            self._bar.set_postfix(dict(values), refresh=False)

    def write(self, message: str) -> None:
        if self._bar is not None:
            self._bar.write(message)
        else:
            print(message)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


class GRPOTrainer:
    """Standard multi-target GRPO trainer.

    One trainer instance owns exactly one policy. That policy is optimized
    continuously across the whole target dataset, which is the standard RL
    setting for this project.
    """

    def __init__(
        self,
        policy: OptoGPTPolicy,
        config: Mapping[str, Any],
        run_dir: str | Path | None = None,
    ) -> None:
        self.policy = policy
        self.config = config
        self.device = policy.device
        self.run_dir = Path(run_dir) if run_dir is not None else None

        experiment_cfg = config["experiment"]
        grpo_cfg = config["grpo"]
        sampling_cfg = config["sampling"]
        tmm_cfg = config["tmm"]
        plotting_cfg = config.get("plotting", {})
        checkpoint_cfg = config.get("checkpoints", {})
        evaluation_cfg = config.get("evaluation", {})
        logging_cfg = config.get("logging", {})

        self.base_seed = int(experiment_cfg["seed"])
        self.steps = int(grpo_cfg["steps"])
        self.target_batch_size = int(grpo_cfg["target_batch_size"])
        self.group_size = int(grpo_cfg.get("group_size", sampling_cfg.get("unique_candidates", 64)))
        self.eval_interval = int(grpo_cfg.get("eval_interval", 100))
        self.learning_rate = float(grpo_cfg["learning_rate"])
        self.clip_eps = float(grpo_cfg["clip_eps"])
        self.kl_beta = float(grpo_cfg["kl_beta"])
        self.max_grad_norm = float(grpo_cfg.get("max_grad_norm", 1.0))
        self.logprob_batch_size = int(grpo_cfg.get("logprob_batch_size", self.group_size))

        self.tmm_batch_size = int(tmm_cfg.get("batch_size", sampling_cfg.get("unique_candidates", 64)))
        self.tmm_pad_to_max_layers = bool(tmm_cfg.get("pad_to_max_layers", True))
        self.tmm_pad_material = str(tmm_cfg.get("pad_material", "Air"))
        self.tmm_debug = bool(tmm_cfg.get("debug", False))

        self.eval_target_batch_size = int(evaluation_cfg.get("target_batch_size", self.target_batch_size))
        self.train_compare_count = int(evaluation_cfg.get("train_compare_count", 0))
        self.train_compare_seed = int(evaluation_cfg.get("train_compare_seed", self.base_seed + 999))
        self.save_eval_plots = bool(plotting_cfg.get("save_eval_plots", True))
        self.save_train_plots = bool(plotting_cfg.get("save_train_plots", True))
        self.save_eval_curve = bool(plotting_cfg.get("save_eval_curve", True))
        self.eval_plot_limit = int(plotting_cfg.get("eval_plot_limit", 16))
        self.train_plot_limit = int(plotting_cfg.get("train_plot_limit", 16))
        self.save_best_checkpoint = bool(checkpoint_cfg.get("save_best", True))
        self.save_final_checkpoint = bool(checkpoint_cfg.get("save_final", True))

        self.train_metrics_filename = str(logging_cfg.get("train_metrics_filename", "train_metrics.csv"))
        self.eval_metrics_filename = str(logging_cfg.get("eval_metrics_filename", "eval_metrics.csv"))
        self.before_eval_filename = str(logging_cfg.get("before_eval_filename", "before_eval.csv"))
        self.after_eval_filename = str(logging_cfg.get("after_eval_filename", "after_eval.csv"))
        self.console_log = bool(logging_cfg.get("console_log", True))
        self.progress_bar = bool(logging_cfg.get("progress_bar", True))
        self.eval_progress_bar = bool(logging_cfg.get("eval_progress_bar", True))
        self.log_interval = max(1, int(logging_cfg.get("log_interval", 10)))
        self.before_train_compare_filename = str(
            logging_cfg.get("before_train_compare_filename", "before_train_compare.csv")
        )
        self.after_train_compare_filename = str(
            logging_cfg.get("after_train_compare_filename", "after_train_compare.csv")
        )

        self.reference_model = self.policy.make_reference_model()
        self.optimizer = torch.optim.Adam(self.policy.model.parameters(), lr=self.learning_rate)

        self.plots_dir = self.run_dir / "plots" if self.run_dir is not None else None
        self.checkpoints_dir = self.run_dir / "checkpoints" if self.run_dir is not None else None
        self.metrics_dir = self.run_dir / "metrics" if self.run_dir is not None else None
        for directory in (self.plots_dir, self.checkpoints_dir, self.metrics_dir):
            if directory is not None:
                directory.mkdir(parents=True, exist_ok=True)

    def _make_progress(
        self,
        total: int,
        desc: str,
        enabled: bool,
        leave: bool = True,
    ) -> _ProgressHandle:
        return _ProgressHandle(total=total, desc=desc, enabled=enabled, leave=leave)

    def _log(self, message: str, progress: Optional[_ProgressHandle] = None) -> None:
        if not self.console_log:
            return
        if progress is not None:
            progress.write(message)
        else:
            print(message)

    @staticmethod
    def _fmt_metric(value: float) -> str:
        value = float(value)
        if not np.isfinite(value):
            return "nan"
        return f"{value:.4f}"

    @staticmethod
    def _serialize_record_for_log(record: Mapping[str, Any]) -> Dict[str, Any]:
        serialized = dict(record)
        for key in ("wavelengths_um", "reflection", "transmission", "predicted_spectrum", "absorption"):
            if key in serialized and hasattr(serialized[key], "tolist"):
                serialized[key] = serialized[key].tolist()
        return serialized

    @staticmethod
    def _group_advantages(rewards: Sequence[float]) -> np.ndarray:
        """Normalize rewards within one target group only.

        Different targets can have different difficulty levels, so GRPO
        advantages are computed independently per target rather than across the
        whole mixed batch.
        """

        rewards_np = np.asarray(rewards, dtype=np.float32)
        mean = float(rewards_np.mean())
        std = float(rewards_np.std())
        if std < 1e-8:
            return rewards_np - mean
        return (rewards_np - mean) / (std + 1e-8)

    def _train_sample_seed(self, target_id: int, step: int) -> int:
        return self.base_seed + 100000 + step * 1000 + int(target_id)

    def _eval_sample_seed(self, target_id: int) -> int:
        return self.base_seed + 200000 + int(target_id)

    def _train_compare_sample_seed(self, target_id: int) -> int:
        return self.train_compare_seed + 300000 + int(target_id)

    def _records_to_old_logprob_tensor(
        self,
        records: Sequence[Mapping[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not records:
            empty = torch.empty((0, 0), dtype=torch.float32, device=self.device)
            return empty, empty.to(dtype=torch.bool)

        max_len = max(len(record["raw_logprobs"]) for record in records)
        if max_len == 0:
            empty = torch.empty((len(records), 0), dtype=torch.float32, device=self.device)
            return empty, torch.zeros((len(records), 0), dtype=torch.bool, device=self.device)

        old_logprobs = torch.zeros((len(records), max_len), dtype=torch.float32, device=self.device)
        token_mask = torch.zeros((len(records), max_len), dtype=torch.bool, device=self.device)
        for row_idx, record in enumerate(records):
            values = [float(value) for value in record["raw_logprobs"]]
            if not values:
                continue
            row_tensor = torch.tensor(values, dtype=old_logprobs.dtype, device=self.device)
            old_logprobs[row_idx, : row_tensor.numel()] = row_tensor
            token_mask[row_idx, : row_tensor.numel()] = True
        return old_logprobs, token_mask

    def _target_loss(
        self,
        target: Mapping[str, Any],
        records: Sequence[Mapping[str, Any]],
    ) -> tuple[Optional[torch.Tensor], Dict[str, float]]:
        """Build one GRPO loss term for one target and its sampled group."""

        if not records:
            return None, {"updated": 0.0, "policy_loss": 0.0, "kl_loss": 0.0, "mean_ratio": 1.0}

        usable_records = list(records[: min(self.group_size, len(records))])
        rewards = [float(record["reward"]) for record in usable_records]
        advantages = self._group_advantages(rewards)
        effective_pairs = [
            (record, float(advantage))
            for record, advantage in zip(usable_records, advantages)
            if len(record["token_ids"]) > 0
        ]
        if not effective_pairs:
            return None, {"updated": 0.0, "policy_loss": 0.0, "kl_loss": 0.0, "mean_ratio": 1.0}

        effective_records = [record for record, _ in effective_pairs]
        advantage_tensor = torch.tensor(
            [advantage for _, advantage in effective_pairs],
            dtype=torch.float32,
            device=self.device,
        )
        old_logprobs, old_mask = self._records_to_old_logprob_tensor(effective_records)

        current_logprobs, current_mask = self.policy.sequence_logprobs_batch_tensor(
            target_spectrum=target["spectrum"],
            token_id_groups=[record["token_ids"] for record in effective_records],
            start_symbol=str(self.config["sampling"].get("start_symbol", "BOS")),
            start_mat=self.config["sampling"].get("start_mat"),
            model=self.policy.model,
            require_grad=True,
            batch_size=self.logprob_batch_size,
        )
        reference_logprobs, reference_mask = self.policy.sequence_logprobs_batch_tensor(
            target_spectrum=target["spectrum"],
            token_id_groups=[record["token_ids"] for record in effective_records],
            start_symbol=str(self.config["sampling"].get("start_symbol", "BOS")),
            start_mat=self.config["sampling"].get("start_mat"),
            model=self.reference_model,
            require_grad=False,
            batch_size=self.logprob_batch_size,
        )

        token_mask = old_mask & current_mask & reference_mask
        valid_lengths = token_mask.sum(dim=-1)
        valid_sample_mask = valid_lengths > 0
        if not bool(valid_sample_mask.any().item()):
            return None, {"updated": 0.0, "policy_loss": 0.0, "kl_loss": 0.0, "mean_ratio": 1.0}

        # GRPO keeps the original PPO-style clipped ratio objective, while the
        # KL term anchors the policy to the frozen reference model.
        ratio = torch.exp(current_logprobs - old_logprobs)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        advantage_matrix = advantage_tensor.unsqueeze(-1)
        token_mask_float = token_mask.to(dtype=current_logprobs.dtype)
        valid_lengths_float = valid_lengths.clamp_min(1).to(dtype=current_logprobs.dtype)

        policy_gain = torch.min(ratio * advantage_matrix, clipped_ratio * advantage_matrix)
        policy_loss_per_sample = -(policy_gain * token_mask_float).sum(dim=-1) / valid_lengths_float

        log_ratio_ref = reference_logprobs - current_logprobs
        kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0
        kl_loss_per_sample = (kl * token_mask_float).sum(dim=-1) / valid_lengths_float
        mean_ratio_per_sample = (ratio * token_mask_float).sum(dim=-1) / valid_lengths_float

        policy_loss = policy_loss_per_sample[valid_sample_mask].mean()
        kl_loss = kl_loss_per_sample[valid_sample_mask].mean()
        total_loss = policy_loss + self.kl_beta * kl_loss
        if not torch.isfinite(total_loss):
            return None, {"updated": 0.0, "policy_loss": 0.0, "kl_loss": 0.0, "mean_ratio": 1.0}

        return total_loss, {
            "updated": float(valid_sample_mask.sum().item()),
            "policy_loss": float(policy_loss.detach().cpu().item()),
            "kl_loss": float(kl_loss.detach().cpu().item()),
            "mean_ratio": float(mean_ratio_per_sample[valid_sample_mask].mean().detach().cpu().item()),
        }

    def _flatten_target_batches_for_joint_loss(
        self,
        target_batches: Sequence[Mapping[str, Any]],
    ) -> tuple[List[Mapping[str, Any]], List[Sequence[float]], torch.Tensor, List[tuple[int, int]]]:
        """Prepare a mixed-target batch while keeping per-target grouping metadata."""

        flat_records: List[Mapping[str, Any]] = []
        flat_target_spectra: List[Sequence[float]] = []
        flat_advantages: List[float] = []
        group_slices: List[tuple[int, int]] = []

        for target_batch in target_batches:
            records = target_batch["records"]
            if not records:
                continue

            usable_records = list(records[: min(self.group_size, len(records))])
            rewards = [float(record["reward"]) for record in usable_records]
            advantages = self._group_advantages(rewards)
            effective_pairs = [
                (record, float(advantage))
                for record, advantage in zip(usable_records, advantages)
                if len(record["token_ids"]) > 0
            ]
            if not effective_pairs:
                continue

            start = len(flat_records)
            for record, advantage in effective_pairs:
                flat_records.append(record)
                flat_target_spectra.append(target_batch["target"]["spectrum"])
                flat_advantages.append(advantage)
            stop = len(flat_records)
            group_slices.append((start, stop))

        advantage_tensor = torch.tensor(flat_advantages, dtype=torch.float32, device=self.device)
        return flat_records, flat_target_spectra, advantage_tensor, group_slices

    def _update_from_target_batches(
        self,
        target_batches: Sequence[Mapping[str, Any]],
    ) -> Dict[str, float]:
        """Aggregate per-target losses and apply one optimizer step."""

        self.policy.model.train()
        self.optimizer.zero_grad()

        flat_records, flat_target_spectra, advantage_tensor, group_slices = self._flatten_target_batches_for_joint_loss(
            target_batches
        )
        if not flat_records:
            self.policy.model.eval()
            return {"updated": 0.0, "loss": 0.0, "policy_loss": 0.0, "kl_loss": 0.0, "mean_ratio": 1.0}

        old_logprobs, old_mask = self._records_to_old_logprob_tensor(flat_records)
        token_id_groups = [record["token_ids"] for record in flat_records]
        current_logprobs, current_mask = self.policy.sequence_logprobs_multi_target_batch_tensor(
            target_spectra=flat_target_spectra,
            token_id_groups=token_id_groups,
            start_symbol=str(self.config["sampling"].get("start_symbol", "BOS")),
            start_mat=self.config["sampling"].get("start_mat"),
            model=self.policy.model,
            require_grad=True,
            batch_size=self.logprob_batch_size,
        )
        reference_logprobs, reference_mask = self.policy.sequence_logprobs_multi_target_batch_tensor(
            target_spectra=flat_target_spectra,
            token_id_groups=token_id_groups,
            start_symbol=str(self.config["sampling"].get("start_symbol", "BOS")),
            start_mat=self.config["sampling"].get("start_mat"),
            model=self.reference_model,
            require_grad=False,
            batch_size=self.logprob_batch_size,
        )

        token_mask = old_mask & current_mask & reference_mask
        valid_lengths = token_mask.sum(dim=-1)
        valid_sample_mask = valid_lengths > 0
        if not bool(valid_sample_mask.any().item()):
            self.policy.model.eval()
            return {"updated": 0.0, "loss": 0.0, "policy_loss": 0.0, "kl_loss": 0.0, "mean_ratio": 1.0}

        ratio = torch.exp(current_logprobs - old_logprobs)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        advantage_matrix = advantage_tensor.unsqueeze(-1)
        token_mask_float = token_mask.to(dtype=current_logprobs.dtype)
        valid_lengths_float = valid_lengths.clamp_min(1).to(dtype=current_logprobs.dtype)

        policy_gain = torch.min(ratio * advantage_matrix, clipped_ratio * advantage_matrix)
        policy_loss_per_sample = -(policy_gain * token_mask_float).sum(dim=-1) / valid_lengths_float

        log_ratio_ref = reference_logprobs - current_logprobs
        kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0
        kl_loss_per_sample = (kl * token_mask_float).sum(dim=-1) / valid_lengths_float
        mean_ratio_per_sample = (ratio * token_mask_float).sum(dim=-1) / valid_lengths_float

        target_total_losses: List[torch.Tensor] = []
        updated_samples = 0.0
        policy_losses: List[float] = []
        kl_losses: List[float] = []
        mean_ratios: List[float] = []

        for start, stop in group_slices:
            group_valid_mask = valid_sample_mask[start:stop]
            if not bool(group_valid_mask.any().item()):
                continue
            group_policy_loss = policy_loss_per_sample[start:stop][group_valid_mask].mean()
            group_kl_loss = kl_loss_per_sample[start:stop][group_valid_mask].mean()
            group_mean_ratio = mean_ratio_per_sample[start:stop][group_valid_mask].mean()
            target_total_losses.append(group_policy_loss + self.kl_beta * group_kl_loss)
            updated_samples += float(group_valid_mask.sum().item())
            policy_losses.append(float(group_policy_loss.detach().cpu().item()))
            kl_losses.append(float(group_kl_loss.detach().cpu().item()))
            mean_ratios.append(float(group_mean_ratio.detach().cpu().item()))

        if not target_total_losses:
            self.policy.model.eval()
            return {"updated": 0.0, "loss": 0.0, "policy_loss": 0.0, "kl_loss": 0.0, "mean_ratio": 1.0}

        total_loss = torch.stack(target_total_losses).mean()
        if not torch.isfinite(total_loss):
            self.policy.model.eval()
            return {"updated": 0.0, "loss": 0.0, "policy_loss": 0.0, "kl_loss": 0.0, "mean_ratio": 1.0}

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.policy.model.eval()

        return {
            "updated": float(updated_samples),
            "loss": float(total_loss.detach().cpu().item()),
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "kl_loss": float(np.mean(kl_losses)) if kl_losses else 0.0,
            "mean_ratio": float(np.mean(mean_ratios)) if mean_ratios else 1.0,
        }

    @staticmethod
    def _best_record(records: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
        if not records:
            return None
        return dict(min(records, key=lambda item: float(item["error"])))

    @staticmethod
    def _chunk_targets(targets: Sequence[Mapping[str, Any]], batch_size: int) -> List[List[Mapping[str, Any]]]:
        effective_batch_size = max(1, int(batch_size))
        return [list(targets[start : start + effective_batch_size]) for start in range(0, len(targets), effective_batch_size)]

    def _evaluate_target_groups(
        self,
        targets: Sequence[Mapping[str, Any]],
        step: int,
        split: str,
        phase: str,
        rollout_logger: RolloutLogger = None,
    ) -> List[Dict[str, Any]]:
        """Sample, deduplicate, evaluate, and log one chunk of targets.

        The pipeline for each target is:
        oversample -> exact dedup -> keep the configured unique subset -> TMM
        evaluation -> merge reward fields back into rollout records.
        """

        sampled_groups: List[Dict[str, Any]] = []
        result_map: Dict[int, Dict[str, Any]] = {}
        target_ids = [int(target["target_id"]) for target in targets]
        sample_seeds: List[int] = []
        for target_id in target_ids:
            if split == "eval":
                sample_seed = self._eval_sample_seed(target_id)
            elif split == "train_compare":
                sample_seed = self._train_compare_sample_seed(target_id)
            else:
                sample_seed = self._train_sample_seed(target_id, step)
            sample_seeds.append(sample_seed)

        sample_results = sample_unique_rollout_groups(
            policy=self.policy,
            target_spectra=[target["spectrum"] for target in targets],
            sampling_config=self.config["sampling"],
            target_indices=target_ids,
            seeds=sample_seeds,
        )

        for target, sample_result in zip(targets, sample_results):
            target_id = int(target["target_id"])
            sampled_groups.append({"target": target, "sample_result": sample_result})
            result_map[target_id] = {
                "target": target,
                "records": [],
                "sampled_count": int(sample_result.sampled_count),
                "unique_count": int(sample_result.unique_count),
                "selected_count": int(sample_result.selected_count),
                "duplicate_count": int(sample_result.duplicate_count),
                "duplicate_ratio": 0.0
                if sample_result.sampled_count == 0
                else float(sample_result.duplicate_count / sample_result.sampled_count),
            }

        flat_entries: List[Dict[str, Any]] = []
        target_spectra: List[Sequence[float]] = []
        for group in sampled_groups:
            target = group["target"]
            sample_result = group["sample_result"]
            for sample in sample_result.selected_samples:
                flat_entries.append(
                    {
                        "target": target,
                        "sample": sample,
                        "sample_result": sample_result,
                    }
                )
                target_spectra.append(target["spectrum"])

        # TMM is invoked once on the flattened unique-structure list to keep the
        # reward path batched and avoid per-target Python overhead.
        if flat_entries:
            reward_results = evaluate_structures_with_tmm(
                structure_token_groups=[entry["sample"].structure_tokens for entry in flat_entries],
                target_spectra=target_spectra,
                wavelength_range_um=self.config["tmm"]["wavelength_range_um"],
                num_points=int(self.config["tmm"]["num_points"]),
                incident_angle=float(self.config["tmm"]["incident_angle"]),
                polarization=int(self.config["tmm"]["polarization"]),
                metric=str(self.config["reward"]["metric"]),
                invalid_structure_penalty=float(self.config["reward"]["invalid_structure_penalty"]),
                nonphysical_spectrum_penalty=float(
                    self.config["reward"].get(
                        "nonphysical_spectrum_penalty",
                        self.config["reward"]["invalid_structure_penalty"],
                    )
                ),
                physical_tolerance=float(self.config["reward"].get("physical_tolerance", 0.01)),
                database_path=self.config["paths"]["nk_dir"],
                material_aliases=self.config["tmm"].get("material_aliases", {}),
                return_spectra=True,
                pad_to_max_layers=self.tmm_pad_to_max_layers,
                pad_material=self.tmm_pad_material,
                batch_size=self.tmm_batch_size,
                tmm_debug=self.tmm_debug,
            )
        else:
            reward_results = []

        for entry, reward in zip(flat_entries, reward_results):
            target = entry["target"]
            sample = entry["sample"]
            sample_result = entry["sample_result"]
            target_id = int(target["target_id"])
            record = asdict(sample)
            record.update(
                {
                    "target_id": target_id,
                    "split": split,
                    "phase": phase,
                    "step": int(step),
                    "family": str(target["family"]),
                    "left_nm": float(target["left_nm"]),
                    "right_nm": float(target["right_nm"]),
                    "width_nm": float(target["width_nm"]),
                    "error": float(reward["error"]),
                    "reward": float(reward["reward"]),
                    "status": str(reward["status"]),
                    "layer_count": int(reward.get("layer_count", len(sample.structure_tokens))),
                    "padded_layer_count": int(reward.get("padded_layer_count", len(sample.structure_tokens))),
                    "sampled_count": int(sample_result.sampled_count),
                    "unique_count": int(sample_result.unique_count),
                    "selected_count": int(sample_result.selected_count),
                    "duplicate_count": int(sample_result.duplicate_count),
                    "duplicate_ratio": 0.0
                    if sample_result.sampled_count == 0
                    else float(sample_result.duplicate_count / sample_result.sampled_count),
                }
            )
            for key in ("wavelengths_um", "reflection", "transmission", "predicted_spectrum"):
                if key in reward:
                    record[key] = reward[key]
            if "reflection" in record and "transmission" in record:
                record["absorption"] = 1.0 - np.asarray(record["reflection"]) - np.asarray(record["transmission"])
            result_map[target_id]["records"].append(record)
            if rollout_logger is not None:
                rollout_logger(self._serialize_record_for_log(record))

        return [result_map[int(target["target_id"])] for target in targets]

    def _evaluate_split(
        self,
        targets: Sequence[Mapping[str, Any]],
        step: int,
        split: str,
        phase: str,
        rollout_logger: RolloutLogger = None,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Evaluate a whole split by processing targets in manageable chunks."""

        all_results: List[Dict[str, Any]] = []
        target_chunks = self._chunk_targets(
            targets=targets,
            batch_size=self.eval_target_batch_size if split == "eval" else self.target_batch_size,
        )
        progress = self._make_progress(
            total=len(targets),
            desc=progress_desc or f"{split}:{phase}",
            enabled=show_progress and self.eval_progress_bar and len(targets) > 0,
            leave=False,
        )
        for target_chunk in target_chunks:
            all_results.extend(
                self._evaluate_target_groups(
                    targets=target_chunk,
                    step=step,
                    split=split,
                    phase=phase,
                    rollout_logger=rollout_logger,
                )
            )
            progress.update(len(target_chunk))
        progress.close()
        return all_results

    def _build_eval_rows(
        self,
        results: Sequence[Mapping[str, Any]],
        prefix: str,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for result in results:
            target = result["target"]
            best_record = self._best_record(result["records"])
            rows.append(
                {
                    "target_id": int(target["target_id"]),
                    "family": str(target["family"]),
                    "left_nm": float(target["left_nm"]),
                    "right_nm": float(target["right_nm"]),
                    "width_nm": float(target["width_nm"]),
                    f"{prefix}_error": None if best_record is None else float(best_record["error"]),
                    f"{prefix}_structure": "" if best_record is None else "|".join(best_record["structure_tokens"]),
                    f"{prefix}_unique_count": int(result["unique_count"]),
                    f"{prefix}_valid_candidates": int(sum(1 for item in result["records"] if item["status"] == "ok")),
                }
            )
        return rows

    def _build_eval_metric_row(
        self,
        results: Sequence[Mapping[str, Any]],
        step: int,
        split: str,
        phase: str,
    ) -> Dict[str, Any]:
        best_errors = [
            float(best_record["error"])
            for result in results
            for best_record in [self._best_record(result["records"])]
            if best_record is not None
        ]
        sampled_counts = [int(result["sampled_count"]) for result in results]
        unique_counts = [int(result["unique_count"]) for result in results]
        selected_counts = [int(result["selected_count"]) for result in results]
        return {
            "step": int(step),
            "split": split,
            "phase": phase,
            "target_count": int(len(results)),
            "mean_error": float(np.mean(best_errors)) if best_errors else float("nan"),
            "median_error": float(np.median(best_errors)) if best_errors else float("nan"),
            "min_error": float(np.min(best_errors)) if best_errors else float("nan"),
            "max_error": float(np.max(best_errors)) if best_errors else float("nan"),
            "mean_sampled_count": float(np.mean(sampled_counts)) if sampled_counts else 0.0,
            "mean_unique_count": float(np.mean(unique_counts)) if unique_counts else 0.0,
            "mean_selected_count": float(np.mean(selected_counts)) if selected_counts else 0.0,
        }

    def _comparison_plot_path(self, target_id: int) -> Optional[Path]:
        if self.plots_dir is None:
            return None
        return self.plots_dir / f"eval_target_{target_id:03d}_before_after.png"

    def _split_comparison_plot_path(self, split_name: str, target_id: int) -> Optional[Path]:
        if self.plots_dir is None:
            return None
        return self.plots_dir / f"{split_name}_target_{target_id:03d}_before_after.png"

    def _metric_curve_path(self, name: str) -> Optional[Path]:
        if self.plots_dir is None:
            return None
        return self.plots_dir / f"{name}.png"

    def _checkpoint_path(self, name: str) -> Optional[Path]:
        if self.checkpoints_dir is None:
            return None
        return self.checkpoints_dir / f"{name}.pt"

    def _metrics_path(self, filename: str) -> Optional[Path]:
        if self.metrics_dir is None:
            return None
        return self.metrics_dir / filename

    def _save_policy_checkpoint(
        self,
        path: Optional[Path],
        step: int,
        mean_error: float,
        split: str,
    ) -> Optional[str]:
        if path is None:
            return None
        self.policy.export_checkpoint(
            path,
            extra_state={
                "step": int(step),
                "split": str(split),
                "mean_error": float(mean_error),
            },
        )
        return str(path)

    @staticmethod
    def _merge_before_after_rows(
        before_rows: Sequence[Mapping[str, Any]],
        after_rows: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        after_map = {int(row["target_id"]): dict(row) for row in after_rows}
        merged_rows: List[Dict[str, Any]] = []
        for before_row in before_rows:
            target_id = int(before_row["target_id"])
            after_row = after_map.get(target_id, {})
            before_error = before_row.get("before_error")
            after_error = after_row.get("after_error")
            improvement = None
            if before_error is not None and after_error is not None:
                improvement = float(before_error) - float(after_error)
            merged_rows.append(
                {
                    **dict(before_row),
                    **dict(after_row),
                    "improvement": improvement,
                }
            )
        return merged_rows

    def _select_train_compare_targets(
        self,
        train_targets: Sequence[Mapping[str, Any]],
    ) -> List[Mapping[str, Any]]:
        """Pick a deterministic training-target subset for before/after plotting."""

        if self.train_compare_count <= 0 or not train_targets:
            return []
        effective_count = min(self.train_compare_count, len(train_targets))
        rng = np.random.default_rng(self.train_compare_seed)
        indices = rng.choice(len(train_targets), size=effective_count, replace=False)
        return [train_targets[int(index)] for index in indices]

    def train(
        self,
        train_targets: Sequence[Mapping[str, Any]],
        eval_targets: Sequence[Mapping[str, Any]],
        rollout_logger: RolloutLogger = None,
    ) -> List[Dict[str, Any]]:
        """Run full dataset-level GRPO and return before/after summary rows."""

        rng = np.random.default_rng(self.base_seed)
        train_compare_targets = self._select_train_compare_targets(train_targets)
        train_progress = self._make_progress(
            total=self.steps,
            desc="GRPO Train",
            enabled=self.progress_bar and self.steps > 0,
            leave=True,
        )

        self._log(
            (
                f"[startup] train_targets={len(train_targets)} eval_targets={len(eval_targets)} "
                f"train_compare={len(train_compare_targets)} steps={self.steps} "
                f"target_batch={self.target_batch_size} group_size={self.group_size} "
                f"oversample={self.config['sampling']['oversample_count']}"
            ),
            progress=train_progress,
        )

        # Always benchmark the frozen starting checkpoint on a fixed eval set so
        # before/after comparisons are directly comparable.
        self._log("[eval-before] running initial evaluation on eval split", progress=train_progress)
        before_results = self._evaluate_split(
            targets=eval_targets,
            step=0,
            split="eval",
            phase="before",
            rollout_logger=rollout_logger,
            show_progress=True,
            progress_desc="Eval before",
        )
        before_rows = self._build_eval_rows(before_results, prefix="before")
        before_metric_row = self._build_eval_metric_row(before_results, step=0, split="eval", phase="before")
        self._log(
            (
                f"[eval-before] mean={self._fmt_metric(before_metric_row['mean_error'])} "
                f"median={self._fmt_metric(before_metric_row['median_error'])} "
                f"min={self._fmt_metric(before_metric_row['min_error'])} "
                f"max={self._fmt_metric(before_metric_row['max_error'])}"
            ),
            progress=train_progress,
        )
        eval_metric_rows: List[Dict[str, Any]] = [before_metric_row]
        train_metric_rows: List[Dict[str, Any]] = []

        before_train_compare_results: List[Dict[str, Any]] = []
        before_train_compare_rows: List[Dict[str, Any]] = []
        if train_compare_targets:
            self._log("[train-compare-before] running initial evaluation on train subset", progress=train_progress)
            before_train_compare_results = self._evaluate_split(
                targets=train_compare_targets,
                step=0,
                split="train_compare",
                phase="before",
                rollout_logger=rollout_logger,
                show_progress=True,
                progress_desc="Train subset before",
            )
            before_train_compare_rows = self._build_eval_rows(before_train_compare_results, prefix="before")

        best_eval_mean = float(before_metric_row["mean_error"])
        if self.save_best_checkpoint:
            self._save_policy_checkpoint(
                path=self._checkpoint_path("best_eval"),
                step=0,
                mean_error=best_eval_mean,
                split="eval",
            )

        for step in range(1, self.steps + 1):
            # Sample a minibatch of targets, then compute one shared optimizer
            # update from the union of their per-target GRPO losses.
            effective_batch_size = min(self.target_batch_size, len(train_targets))
            replace = len(train_targets) < effective_batch_size
            selected_indices = rng.choice(len(train_targets), size=effective_batch_size, replace=replace)
            selected_targets = [train_targets[int(index)] for index in selected_indices]
            train_results = self._evaluate_target_groups(
                targets=selected_targets,
                step=step,
                split="train",
                phase="train",
                rollout_logger=rollout_logger,
            )
            update_stats = self._update_from_target_batches(train_results)

            train_best_errors = [
                float(best_record["error"])
                for result in train_results
                for best_record in [self._best_record(result["records"])]
                if best_record is not None
            ]
            train_metric_rows.append(
                {
                    "step": int(step),
                    "target_batch_size": int(len(selected_targets)),
                    "mean_train_error": float(np.mean(train_best_errors)) if train_best_errors else float("nan"),
                    "median_train_error": float(np.median(train_best_errors)) if train_best_errors else float("nan"),
                    "mean_sampled_count": float(np.mean([result["sampled_count"] for result in train_results]))
                    if train_results
                    else 0.0,
                    "mean_unique_count": float(np.mean([result["unique_count"] for result in train_results]))
                    if train_results
                    else 0.0,
                    "mean_selected_count": float(np.mean([result["selected_count"] for result in train_results]))
                    if train_results
                    else 0.0,
                    "updated": float(update_stats["updated"]),
                    "loss": float(update_stats["loss"]),
                    "policy_loss": float(update_stats["policy_loss"]),
                    "kl_loss": float(update_stats["kl_loss"]),
                    "mean_ratio": float(update_stats["mean_ratio"]),
                }
            )
            current_train_row = train_metric_rows[-1]
            train_progress.update(1)
            train_progress.set_postfix(
                {
                    "loss": self._fmt_metric(current_train_row["loss"]),
                    "train_err": self._fmt_metric(current_train_row["mean_train_error"]),
                    "unique": self._fmt_metric(current_train_row["mean_unique_count"]),
                }
            )
            if step == 1 or step % self.log_interval == 0 or step == self.steps:
                self._log(
                    (
                        f"[train] step={step}/{self.steps} "
                        f"loss={self._fmt_metric(current_train_row['loss'])} "
                        f"train_mean={self._fmt_metric(current_train_row['mean_train_error'])} "
                        f"unique_mean={self._fmt_metric(current_train_row['mean_unique_count'])} "
                        f"updated={current_train_row['updated']:.0f}"
                    ),
                    progress=train_progress,
                )

            if self.eval_interval > 0 and (step % self.eval_interval == 0 or step == self.steps):
                self._log(f"[eval] step={step} running evaluation on eval split", progress=train_progress)
                eval_results = self._evaluate_split(
                    targets=eval_targets,
                    step=step,
                    split="eval",
                    phase="eval",
                    rollout_logger=rollout_logger,
                    show_progress=True,
                    progress_desc=f"Eval @{step}",
                )
                eval_metric_row = self._build_eval_metric_row(eval_results, step=step, split="eval", phase="eval")
                eval_metric_rows.append(eval_metric_row)
                current_eval_mean = float(eval_metric_row["mean_error"])
                self._log(
                    (
                        f"[eval] step={step} mean={self._fmt_metric(eval_metric_row['mean_error'])} "
                        f"median={self._fmt_metric(eval_metric_row['median_error'])} "
                        f"min={self._fmt_metric(eval_metric_row['min_error'])} "
                        f"max={self._fmt_metric(eval_metric_row['max_error'])}"
                    ),
                    progress=train_progress,
                )
                if current_eval_mean < best_eval_mean:
                    best_eval_mean = current_eval_mean
                    self._log(
                        f"[checkpoint] new best eval mean error={self._fmt_metric(current_eval_mean)} at step={step}",
                        progress=train_progress,
                    )
                    if self.save_best_checkpoint:
                        self._save_policy_checkpoint(
                            path=self._checkpoint_path("best_eval"),
                            step=step,
                            mean_error=current_eval_mean,
                            split="eval",
                        )

        # Final evaluation uses the same fixed eval set as the initial baseline.
        self._log("[eval-after] running final evaluation on eval split", progress=train_progress)
        after_results = self._evaluate_split(
            targets=eval_targets,
            step=self.steps,
            split="eval",
            phase="after",
            rollout_logger=rollout_logger,
            show_progress=True,
            progress_desc="Eval after",
        )
        after_rows = self._build_eval_rows(after_results, prefix="after")
        after_eval_metric_row = self._build_eval_metric_row(after_results, step=self.steps, split="eval", phase="after")
        eval_metric_rows.append(after_eval_metric_row)
        self._log(
            (
                f"[eval-after] mean={self._fmt_metric(after_eval_metric_row['mean_error'])} "
                f"median={self._fmt_metric(after_eval_metric_row['median_error'])} "
                f"min={self._fmt_metric(after_eval_metric_row['min_error'])} "
                f"max={self._fmt_metric(after_eval_metric_row['max_error'])}"
            ),
            progress=train_progress,
        )

        after_train_compare_results: List[Dict[str, Any]] = []
        after_train_compare_rows: List[Dict[str, Any]] = []
        if train_compare_targets:
            self._log("[train-compare-after] running final evaluation on train subset", progress=train_progress)
            after_train_compare_results = self._evaluate_split(
                targets=train_compare_targets,
                step=self.steps,
                split="train_compare",
                phase="after",
                rollout_logger=rollout_logger,
                show_progress=True,
                progress_desc="Train subset after",
            )
            after_train_compare_rows = self._build_eval_rows(after_train_compare_results, prefix="after")

        if self.save_final_checkpoint:
            final_mean_error = float(eval_metric_rows[-1]["mean_error"])
            self._save_policy_checkpoint(
                path=self._checkpoint_path("final"),
                step=self.steps,
                mean_error=final_mean_error,
                split="eval",
            )

        summary_rows = self._merge_before_after_rows(before_rows, after_rows)

        before_path = self._metrics_path(self.before_eval_filename)
        if before_path is not None:
            write_summary_csv(before_path, before_rows)
        after_path = self._metrics_path(self.after_eval_filename)
        if after_path is not None:
            write_summary_csv(after_path, after_rows)
        before_train_compare_path = self._metrics_path(self.before_train_compare_filename)
        if before_train_compare_path is not None and before_train_compare_rows:
            write_summary_csv(before_train_compare_path, before_train_compare_rows)
        after_train_compare_path = self._metrics_path(self.after_train_compare_filename)
        if after_train_compare_path is not None and after_train_compare_rows:
            write_summary_csv(after_train_compare_path, after_train_compare_rows)
        train_metrics_path = self._metrics_path(self.train_metrics_filename)
        if train_metrics_path is not None:
            write_summary_csv(train_metrics_path, train_metric_rows)
        eval_metrics_path = self._metrics_path(self.eval_metrics_filename)
        if eval_metrics_path is not None:
            write_summary_csv(eval_metrics_path, eval_metric_rows)

        if self.save_eval_plots:
            before_map = {int(result["target"]["target_id"]): result for result in before_results}
            after_map = {int(result["target"]["target_id"]): result for result in after_results}
            for target in eval_targets[: self.eval_plot_limit]:
                target_id = int(target["target_id"])
                before_best = self._best_record(before_map[target_id]["records"]) if target_id in before_map else None
                after_best = self._best_record(after_map[target_id]["records"]) if target_id in after_map else None
                if before_best is None or after_best is None:
                    continue
                plot_path = self._comparison_plot_path(target_id)
                if plot_path is not None:
                    save_before_after_plot(
                        path=plot_path,
                        target=target,
                        before_record=before_best,
                        after_record=after_best,
                    )

        if self.save_train_plots and train_compare_targets:
            before_train_map = {
                int(result["target"]["target_id"]): result for result in before_train_compare_results
            }
            after_train_map = {
                int(result["target"]["target_id"]): result for result in after_train_compare_results
            }
            for target in train_compare_targets[: self.train_plot_limit]:
                target_id = int(target["target_id"])
                before_best = (
                    self._best_record(before_train_map[target_id]["records"])
                    if target_id in before_train_map
                    else None
                )
                after_best = (
                    self._best_record(after_train_map[target_id]["records"])
                    if target_id in after_train_map
                    else None
                )
                if before_best is None or after_best is None:
                    continue
                plot_path = self._split_comparison_plot_path("train", target_id)
                if plot_path is not None:
                    save_before_after_plot(
                        path=plot_path,
                        target=target,
                        before_record=before_best,
                        after_record=after_best,
                    )

        if self.save_eval_curve:
            curve_path = self._metric_curve_path("eval_mean_error")
            if curve_path is not None:
                save_metric_curve(
                    path=curve_path,
                    rows=eval_metric_rows,
                    x_key="step",
                    y_key="mean_error",
                    title="Evaluation Mean Absorption RMSE",
                    y_label="Mean Absorption RMSE",
                )

        train_progress.close()
        return summary_rows
