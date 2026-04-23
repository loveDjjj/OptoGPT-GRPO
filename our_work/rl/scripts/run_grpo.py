from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from our_work._shared.io.config import load_yaml_config, resolve_repo_path
from our_work._shared.utils.seed import set_global_seed
from our_work.pretrain.dataset.tokenizer import SpectralStructureTokenizer
from our_work.pretrain.model.modeling_spectral_gpt import SpectralGPTForCausalLM
from our_work.rl.dataset import load_rl_split_records
from our_work.rl.trainer import RLComponents, SpectralGRPOTrainer
from utils.dist import barrier, cleanup_distributed, init_distributed


_GENERATED_RUN_ARTIFACTS = (
    "metrics",
    "plots",
    "tensorboard",
    "checkpoints",
    "config.snapshot.yaml",
)


def resolve_checkpoint_dir(checkpoint_dir: str | Path) -> Path:
    resolved = resolve_repo_path(checkpoint_dir, project_root=PROJECT_ROOT)
    if not (resolved / "config.json").exists():
        checkpoint_dirs = [child for child in resolved.iterdir() if child.is_dir() and child.name.startswith("checkpoint-")]
        if not checkpoint_dirs:
            raise FileNotFoundError(f"No checkpoint directory found under: {resolved}")
        checkpoint_dirs.sort(key=lambda child: int(child.name.split("-")[-1]))
        resolved = checkpoint_dirs[-1]
    return resolved


def load_rl_components(checkpoint_dir: str | Path, device: torch.device) -> RLComponents:
    resolved = resolve_checkpoint_dir(checkpoint_dir)
    raw_model = SpectralGPTForCausalLM.from_pretrained(resolved)
    raw_model.to(device)
    tokenizer = SpectralStructureTokenizer.from_pretrained(resolved)
    model: torch.nn.Module = raw_model
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        if device.type == "cuda":
            model = DDP(raw_model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)
        else:
            model = DDP(raw_model, find_unused_parameters=False)
    return RLComponents(model=model, raw_model=raw_model, tokenizer=tokenizer)


def _current_run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _clear_generated_run_outputs(run_dir: Path) -> None:
    for artifact_name in _GENERATED_RUN_ARTIFACTS:
        artifact_path = run_dir / artifact_name
        if artifact_path.is_dir():
            shutil.rmtree(artifact_path)
        elif artifact_path.exists():
            artifact_path.unlink()


def _create_timestamped_run_dir(base_output_dir: Path) -> Path:
    base_output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _current_run_timestamp()
    candidate = base_output_dir / timestamp
    suffix = 1
    while candidate.exists():
        candidate = base_output_dir / f"{timestamp}-{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _prepare_run_dir_local(
    *,
    base_output_dir: Path,
    overwrite_output_dir: bool,
    resume_checkpoint: Path | None,
) -> Path:
    if resume_checkpoint is not None:
        run_dir = resume_checkpoint.parent.parent
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    if overwrite_output_dir:
        base_output_dir.mkdir(parents=True, exist_ok=True)
        _clear_generated_run_outputs(base_output_dir)
        return base_output_dir

    return _create_timestamped_run_dir(base_output_dir)


def prepare_run_dir(
    config: dict,
    *,
    dist_ctx,
    resume_checkpoint: Path | None,
) -> Path:
    training_cfg = config["training"]
    base_output_dir = Path(training_cfg["output_dir"])
    overwrite_output_dir = bool(training_cfg.get("overwrite_output_dir", False))

    if not dist_ctx.enabled:
        return _prepare_run_dir_local(
            base_output_dir=base_output_dir,
            overwrite_output_dir=overwrite_output_dir,
            resume_checkpoint=resume_checkpoint,
        )

    payload = [None]
    if dist_ctx.is_main:
        payload[0] = str(
            _prepare_run_dir_local(
                base_output_dir=base_output_dir,
                overwrite_output_dir=overwrite_output_dir,
                resume_checkpoint=resume_checkpoint,
            )
        )
    dist.broadcast_object_list(payload, src=0)
    run_dir = Path(payload[0])
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run lightweight GRPO for our_work spectral model.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    config_path = resolve_repo_path(args.config, project_root=PROJECT_ROOT)
    config = load_yaml_config(config_path, project_root=PROJECT_ROOT, resolve_relative_paths=True)

    local_rank = int(os.environ.get("LOCAL_RANK", "0")) if int(os.environ.get("WORLD_SIZE", "1")) > 1 else 0
    requested_device = str(config.get("device", "auto"))
    if requested_device == "auto":
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    elif requested_device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if int(os.environ.get("WORLD_SIZE", "1")) > 1 else requested_device)
    else:
        device = torch.device(requested_device)

    distributed_cfg = config.get("distributed", {})
    dist_ctx = init_distributed(
        device=device,
        timeout_minutes=int(distributed_cfg.get("timeout_minutes", 30)),
        backend=distributed_cfg.get("backend"),
    )
    set_global_seed(int(config.get("seed", 42)), rank_offset=dist_ctx.rank)
    resume_checkpoint = config["training"].get("resume_from_checkpoint")
    resolved_resume_checkpoint = resolve_checkpoint_dir(resume_checkpoint) if resume_checkpoint else None
    run_dir = prepare_run_dir(config, dist_ctx=dist_ctx, resume_checkpoint=resolved_resume_checkpoint)
    components = load_rl_components(resolved_resume_checkpoint or config["model"]["checkpoint_dir"], device=dist_ctx.device)

    train_dataset = load_rl_split_records(
        config["data"]["dataset_dir"],
        config["data"].get("train_split", "train"),
        config["data"].get("max_train_samples"),
    )
    eval_dataset = load_rl_split_records(
        config["data"]["dataset_dir"],
        config["data"].get("eval_split", "val"),
        config["data"].get("max_eval_samples"),
    )

    trainer = SpectralGRPOTrainer(
        components=components,
        config=config,
        run_dir=run_dir,
        dist_ctx=dist_ctx,
        resume_checkpoint=resolved_resume_checkpoint,
    )
    trainer.train(train_dataset=train_dataset, eval_dataset=eval_dataset)

    barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
