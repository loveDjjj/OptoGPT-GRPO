from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

import torch
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import transformers.trainer as trainer_module
from transformers import Trainer, TrainingArguments

from _shared.io.config import load_yaml_config, resolve_repo_path
from pretrain.dataset.collator import SpectralCausalCollator
from pretrain.dataset.hf_dataset import load_split_records
from pretrain.dataset.tokenizer import SpectralStructureTokenizer
from pretrain.monitoring import PretrainVisualizationCallback
from pretrain.model.configuration_spectral_gpt import SpectralGPTConfig
from pretrain.model.modeling_spectral_gpt import SpectralGPTForCausalLM
from pretrain.trainer.metrics import compute_token_accuracy, preprocess_logits_for_metrics


def _ensure_trainer_dataset_namespace() -> None:
    datasets_module = getattr(trainer_module, "datasets", None)
    if datasets_module is not None and hasattr(datasets_module, "Dataset"):
        return

    class _PlaceholderDataset:
        pass

    # The repo already has a top-level `datasets/` package, which shadows the HF package name.
    # Trainer only needs a `Dataset` type for an isinstance check in our list-backed smoke runs.
    trainer_module.datasets = types.SimpleNamespace(Dataset=_PlaceholderDataset)


def _distributed_training_requested() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return True
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _sanitize_single_process_distributed_env() -> None:
    if _distributed_training_requested():
        return
    for key in ("LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        os.environ.pop(key, None)


def build_trainer_components(model_config: dict, token_list: list[str]) -> dict:
    tokenizer = SpectralStructureTokenizer(tokens=token_list)
    config = SpectralGPTConfig(**model_config)
    model = SpectralGPTForCausalLM(config)
    collator = SpectralCausalCollator(
        tokenizer=tokenizer,
        prefix_length=config.prefix_length,
    )
    return {
        "tokenizer": tokenizer,
        "config": config,
        "model": model,
        "collator": collator,
    }


def build_trainer(
    model,
    train_dataset,
    eval_dataset,
    collator,
    output_dir: str,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    max_steps: int | None = None,
    num_train_epochs: float = 1.0,
    learning_rate: float = 5.0e-4,
    lr_scheduler_type: str = "linear",
    warmup_ratio: float = 0.0,
    logging_steps: int = 10,
    eval_steps: int = 50,
    save_steps: int = 50,
    gradient_accumulation_steps: int = 1,
    dataloader_num_workers: int = 0,
    dataloader_prefetch_factor: int | None = None,
    dataloader_pin_memory: bool = False,
    dataloader_persistent_workers: bool = False,
    bf16: bool = False,
    tf32: bool = False,
    ddp_find_unused_parameters: bool | None = None,
    ddp_backend: str | None = None,
    save_total_limit: int | None = None,
    max_grad_norm: float = 1.0,
    callbacks: list | None = None,
) -> Trainer:
    _ensure_trainer_dataset_namespace()
    distributed_requested = _distributed_training_requested()
    if not distributed_requested:
        _sanitize_single_process_distributed_env()
        ddp_find_unused_parameters = None
        ddp_backend = None
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps if max_steps is not None else -1,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_prefetch_factor=dataloader_prefetch_factor,
        dataloader_pin_memory=dataloader_pin_memory,
        dataloader_persistent_workers=dataloader_persistent_workers,
        bf16=bf16,
        tf32=tf32,
        ddp_find_unused_parameters=ddp_find_unused_parameters,
        ddp_backend=ddp_backend,
        save_total_limit=save_total_limit,
        max_grad_norm=max_grad_norm,
        report_to=[],
        remove_unused_columns=False,
    )
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_token_accuracy,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=callbacks,
    )


def _load_token_list(vocab_path: str | Path) -> list[str]:
    vocab_path = Path(vocab_path)
    payload = json.loads(vocab_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "tokens" in payload:
        return list(payload["tokens"])
    raise ValueError(f"Unsupported vocab format: {vocab_path}")


def resolve_checkpoint_dir(path: str | Path) -> Path:
    path = resolve_repo_path(path, project_root=PROJECT_ROOT)
    if (path / "config.json").exists():
        return path
    checkpoint_dirs = [child for child in path.iterdir() if child.is_dir() and child.name.startswith("checkpoint-")]
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint directory found under: {path}")
    return max(checkpoint_dirs, key=lambda child: int(child.name.split("-")[-1]))


def validate_record_spectrum_dim(records: list[dict], *, split_name: str, spectrum_dim: int) -> None:
    if not records:
        return

    first_record = records[0]
    actual_dim = len(first_record.get("spectrum_rt", []))
    if actual_dim != int(spectrum_dim):
        raise ValueError(
            f"{split_name} split spectrum_dim mismatch: model expects {int(spectrum_dim)} values "
            f"but dataset rows contain {int(actual_dim)}. "
            "Keep model.spectrum_dim aligned with 2 * data_gen.tmm.num_points."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run small-scale spectral pretraining.")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--init-checkpoint-dir", default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--vocab-path", default=None)
    parser.add_argument("--train-split", default=None)
    parser.add_argument("--eval-split", default=None)
    args = parser.parse_args()

    model_config_path = resolve_repo_path(args.model_config, project_root=PROJECT_ROOT)
    train_config_path = resolve_repo_path(args.train_config, project_root=PROJECT_ROOT)
    model_yaml = load_yaml_config(model_config_path, project_root=PROJECT_ROOT, resolve_relative_paths=True)
    train_yaml = load_yaml_config(train_config_path, project_root=PROJECT_ROOT, resolve_relative_paths=True)

    dataset_dir = (
        str(resolve_repo_path(args.dataset_dir, project_root=PROJECT_ROOT))
        if args.dataset_dir
        else str(train_yaml["data"]["dataset_dir"])
    )
    vocab_path = (
        str(resolve_repo_path(args.vocab_path, project_root=PROJECT_ROOT))
        if args.vocab_path
        else str(train_yaml["data"]["vocab_path"])
    )
    train_split = args.train_split or str(train_yaml["data"].get("train_split", "train"))
    eval_split = args.eval_split or str(train_yaml["data"].get("eval_split", "val"))
    token_list = _load_token_list(vocab_path)

    init_checkpoint_raw = args.init_checkpoint_dir or train_yaml["training"].get("init_checkpoint_dir")
    init_checkpoint_dir = resolve_checkpoint_dir(init_checkpoint_raw) if init_checkpoint_raw else None
    if init_checkpoint_dir is not None:
        model = SpectralGPTForCausalLM.from_pretrained(init_checkpoint_dir)
        if int(getattr(model.config, "vocab_size", len(token_list))) != int(len(token_list)):
            raise ValueError(
                f"vocab size mismatch: checkpoint expects {int(model.config.vocab_size)} "
                f"but vocab file provides {int(len(token_list))} tokens"
            )
        config = model.config
        tokenizer = SpectralStructureTokenizer(tokens=token_list)
        collator = SpectralCausalCollator(
            tokenizer=tokenizer,
            prefix_length=int(config.prefix_length),
        )
        components = {
            "tokenizer": tokenizer,
            "config": config,
            "model": model,
            "collator": collator,
        }
    else:
        components = build_trainer_components(
            model_config={
                **model_yaml["model"],
                "vocab_size": len(token_list),
                "pad_token_id": 0,
                "bos_token_id": 1,
                "eos_token_id": 2,
            },
            token_list=token_list,
        )

    train_dataset = load_split_records(dataset_dir, train_split)
    eval_dataset = load_split_records(dataset_dir, eval_split)
    validate_record_spectrum_dim(
        train_dataset,
        split_name=train_split,
        spectrum_dim=int(components["config"].spectrum_dim),
    )
    validate_record_spectrum_dim(
        eval_dataset,
        split_name=eval_split,
        spectrum_dim=int(components["config"].spectrum_dim),
    )
    trainer = build_trainer(
        model=components["model"],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        collator=components["collator"],
        output_dir=train_yaml["training"]["output_dir"],
        per_device_train_batch_size=train_yaml["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=train_yaml["training"]["per_device_eval_batch_size"],
        max_steps=train_yaml["training"].get("max_steps"),
        num_train_epochs=train_yaml["training"].get("num_train_epochs", 1.0),
        learning_rate=train_yaml["training"].get("learning_rate", 5.0e-4),
        lr_scheduler_type=str(train_yaml["training"].get("lr_scheduler_type", "linear")),
        warmup_ratio=float(train_yaml["training"].get("warmup_ratio", 0.0)),
        logging_steps=train_yaml["training"].get("logging_steps", 10),
        eval_steps=train_yaml["training"].get("eval_steps", 50),
        save_steps=train_yaml["training"].get("save_steps", 50),
        gradient_accumulation_steps=train_yaml["training"].get("gradient_accumulation_steps", 1),
        dataloader_num_workers=train_yaml["data"].get("num_workers", 0),
        dataloader_prefetch_factor=train_yaml["data"].get("prefetch_factor"),
        dataloader_pin_memory=train_yaml["data"].get("pin_memory", False),
        dataloader_persistent_workers=train_yaml["data"].get("persistent_workers", train_yaml["data"].get("num_workers", 0) > 0),
        bf16=train_yaml["training"].get("bf16", False),
        tf32=train_yaml["training"].get("tf32", False),
        ddp_find_unused_parameters=train_yaml.get("distributed", {}).get("ddp_find_unused_parameters"),
        ddp_backend=train_yaml.get("distributed", {}).get("backend"),
        save_total_limit=train_yaml["training"].get("save_total_limit"),
        max_grad_norm=float(train_yaml["training"].get("max_grad_norm", 1.0)),
        callbacks=[
            PretrainVisualizationCallback(
                output_dir=train_yaml["training"]["output_dir"],
                enable_tensorboard=train_yaml.get("monitoring", {}).get("tensorboard", True),
                enable_jsonl=train_yaml.get("monitoring", {}).get("jsonl", True),
                enable_csv=train_yaml.get("monitoring", {}).get("csv", True),
                save_plots=train_yaml.get("monitoring", {}).get("save_plots", True),
                plot_every_eval=train_yaml.get("monitoring", {}).get("plot_every_eval", True),
                flush_secs=int(train_yaml.get("monitoring", {}).get("flush_secs", 10)),
            )
        ],
    )
    resume_from_checkpoint = train_yaml["training"].get("resume_from_checkpoint")
    if isinstance(resume_from_checkpoint, str) and resume_from_checkpoint.strip():
        resume_from_checkpoint = str(resolve_checkpoint_dir(resume_from_checkpoint))
    elif bool(resume_from_checkpoint):
        resume_from_checkpoint = True
    else:
        resume_from_checkpoint = None
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


if __name__ == "__main__":
    main()
