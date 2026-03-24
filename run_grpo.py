"""Entry point for dataset-level GRPO fine-tuning on synthetic target spectra.

The runtime flow is intentionally simple:

1. Load one yaml config.
2. Generate a train/eval target dataset in memory.
3. Load the OptoGPT checkpoint once as the initial policy.
4. Train a single policy across the whole target dataset.
5. Persist metrics, rollouts, plots, and checkpoints under one run directory.

This file stays lightweight on purpose so orchestration logic is easy to audit.
"""

from __future__ import annotations

import argparse

from data.spectrum_generator import GateDatasetConfig, generate_gate_target_batch
from policy.optogpt_policy import OptoGPTPolicy
from trainers.grpo_trainer import GRPOTrainer
from utils.config import dump_yaml_config, load_yaml_config
from utils.logging import append_jsonl, make_run_dir, write_summary_csv


def _save_targets(path, targets, split: str) -> None:
    """Persist generated target spectra so each run is reproducible."""

    for target in targets:
        append_jsonl(
            path,
            {
                "split": split,
                "target_id": int(target["target_id"]),
                "family": str(target["family"]),
                "left_nm": float(target["left_nm"]),
                "right_nm": float(target["right_nm"]),
                "width_nm": float(target["width_nm"]),
                "edge_smooth_nm": float(target["edge_smooth_nm"]),
                "wavelengths_um": target["wavelengths_um"].tolist(),
                "reflection": target["reflection"].tolist(),
                "transmission": target["transmission"].tolist(),
                "absorption": target["absorption"].tolist(),
                "spectrum": target["spectrum"].tolist(),
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset-level GRPO entrypoint for OptoGPT.")
    parser.add_argument("--config", default="configs/grpo_base.yaml", help="Path to the yaml config.")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    run_dir = make_run_dir(config["paths"]["output_dir"], config["experiment"]["name"])
    dump_yaml_config(run_dir / "config.snapshot.yaml", config)

    target_cfg = config["target"]
    target_mode = str(target_cfg.get("mode", "gate_dataset")).lower()
    if target_mode != "gate_dataset":
        raise ValueError(f"Unsupported target.mode: {target_mode}. Only 'gate_dataset' is supported.")

    # Train/eval targets share the same target family definition and only differ
    # in sample count / random seed.
    common_target_kwargs = {
        "wavelength_min_um": float(target_cfg["wavelength_min_um"]),
        "wavelength_max_um": float(target_cfg["wavelength_max_um"]),
        "num_points": int(target_cfg["num_points"]),
        "width_min_nm": float(target_cfg["width_min_nm"]),
        "width_max_nm": float(target_cfg["width_max_nm"]),
        "edge_smooth_nm": float(target_cfg["edge_smooth_nm"]),
        "families": tuple(target_cfg["families"]),
    }
    train_targets = generate_gate_target_batch(
        GateDatasetConfig(
            count=int(target_cfg["train_count"]),
            seed=int(target_cfg.get("train_seed", config["experiment"]["seed"])),
            **common_target_kwargs,
        )
    )
    eval_targets = generate_gate_target_batch(
        GateDatasetConfig(
            count=int(target_cfg["eval_count"]),
            seed=int(target_cfg.get("eval_seed", config["experiment"]["seed"]) + 1),
            **common_target_kwargs,
        )
    )

    if config["logging"]["save_targets"]:
        _save_targets(run_dir / config["logging"]["train_targets_filename"], train_targets, split="train")
        _save_targets(run_dir / config["logging"]["eval_targets_filename"], eval_targets, split="eval")

    rollouts_path = run_dir / config["logging"]["rollouts_filename"]
    rollout_logger = None
    if config["logging"]["save_rollouts"]:
        rollout_logger = lambda record: append_jsonl(rollouts_path, record)

    # Dataset-level RL loads one policy and keeps updating it across many
    # targets. This is the main difference from the earlier single-target
    # optimization experiments.
    policy = OptoGPTPolicy(config["paths"]["optogpt_checkpoint"], device=config.get("device", "auto"))
    trainer = GRPOTrainer(policy=policy, config=config, run_dir=run_dir)
    summary_rows = trainer.train(
        train_targets=train_targets,
        eval_targets=eval_targets,
        rollout_logger=rollout_logger,
    )
    write_summary_csv(run_dir / config["logging"]["summary_filename"], summary_rows)

    print("GRPO run completed.")
    print(f"Run directory: {run_dir}")
    print(f"Prepared train targets: {len(train_targets)}")
    print(f"Prepared eval targets: {len(eval_targets)}")
    print("Saved targets, rollout logs, metrics, checkpoints, and summary metrics.")


if __name__ == "__main__":
    main()
