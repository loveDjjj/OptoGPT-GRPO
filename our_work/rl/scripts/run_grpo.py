from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from our_work._shared.io.config import load_yaml_config, resolve_repo_path
from our_work.pretrain.dataset.tokenizer import SpectralStructureTokenizer
from our_work.pretrain.model.modeling_spectral_gpt import SpectralGPTForCausalLM
from our_work.rl.dataset import load_rl_split_records
from our_work.rl.trainer import RLComponents, SpectralGRPOTrainer
from utils.dist import barrier, cleanup_distributed, init_distributed


def load_rl_components(checkpoint_dir: str | Path, device: torch.device) -> RLComponents:
    resolved = resolve_repo_path(checkpoint_dir, project_root=PROJECT_ROOT)
    if not (resolved / "config.json").exists():
        checkpoint_dirs = sorted(child for child in resolved.iterdir() if child.is_dir() and child.name.startswith("checkpoint-"))
        if not checkpoint_dirs:
            raise FileNotFoundError(f"No checkpoint directory found under: {resolved}")
        resolved = checkpoint_dirs[-1]
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight GRPO for our_work spectral model.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

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

    dist_ctx = init_distributed(device=device, timeout_minutes=int(config.get("distributed", {}).get("timeout_minutes", 30)))
    components = load_rl_components(config["model"]["checkpoint_dir"], device=dist_ctx.device)

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
        run_dir=config["training"]["output_dir"],
        dist_ctx=dist_ctx,
    )
    trainer.train(train_dataset=train_dataset, eval_dataset=eval_dataset)

    barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
