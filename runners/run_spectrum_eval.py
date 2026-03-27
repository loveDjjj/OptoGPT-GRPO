"""光谱评测入口。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch.distributed as dist

from datasets import build_split_datasets
from evaluators import SpectrumEvaluator
from models.optogpt import OptoGPTModel, resolve_device
from utils.config import dump_yaml_config, load_yaml_config
from utils.dist import barrier, cleanup_distributed, init_distributed
from utils.logging import make_run_dir, write_summary_csv
from utils.seed import set_global_seed


def _shared_run_dir(config, dist_ctx) -> Path:
    if not dist_ctx.enabled:
        return make_run_dir(config["paths"]["output_dir"], config["experiment"]["name"])

    payload = [None]
    if dist_ctx.is_main:
        payload[0] = str(make_run_dir(config["paths"]["output_dir"], config["experiment"]["name"]))
    dist.broadcast_object_list(payload, src=0)
    run_dir = Path(payload[0])
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="OptoGPT 光谱评测入口。")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径。")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    local_rank = int(os.environ.get("LOCAL_RANK", "0")) if int(os.environ.get("WORLD_SIZE", "1")) > 1 else None
    device = resolve_device(config.get("device", "auto"), local_rank=local_rank)
    dist_ctx = init_distributed(device=device, timeout_minutes=int(config.get("distributed", {}).get("timeout_minutes", 30)))
    set_global_seed(int(config["experiment"]["seed"]), rank_offset=dist_ctx.rank)

    run_dir = _shared_run_dir(config, dist_ctx)
    if dist_ctx.is_main:
        dump_yaml_config(run_dir / "config.snapshot.yaml", config)

    datasets = build_split_datasets(config)
    model = OptoGPTModel(config["paths"]["checkpoint"], device=device)
    if dist_ctx.enabled:
        model.configure_distributed(local_rank=dist_ctx.local_rank)

    evaluator = SpectrumEvaluator(
        model=model,
        config=config,
        run_dir=run_dir,
        dist_ctx=dist_ctx,
    )

    summary_rows = []
    for split_name in config["evaluation"]["splits"]:
        dataset = datasets.get(split_name)
        if dataset is None:
            continue
        row = evaluator.evaluate(dataset, split_name=split_name)
        if row is not None:
            summary_rows.append(row)

    if dist_ctx.is_main and summary_rows:
        write_summary_csv(run_dir / "metrics" / "all_summaries.csv", summary_rows)

    barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
