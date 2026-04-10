from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import transformers.trainer as trainer_module
from transformers import Trainer, TrainingArguments

from our_work._shared.io.config import load_yaml_config
from our_work.pretrain.dataset.collator import SpectralCausalCollator
from our_work.pretrain.dataset.hf_dataset import load_split_records
from our_work.pretrain.dataset.tokenizer import SpectralStructureTokenizer
from our_work.pretrain.model.configuration_spectral_gpt import SpectralGPTConfig
from our_work.pretrain.model.modeling_spectral_gpt import SpectralGPTForCausalLM
from our_work.pretrain.trainer.metrics import compute_token_accuracy


def _ensure_trainer_dataset_namespace() -> None:
    datasets_module = getattr(trainer_module, "datasets", None)
    if datasets_module is not None and hasattr(datasets_module, "Dataset"):
        return

    class _PlaceholderDataset:
        pass

    # The repo already has a top-level `datasets/` package, which shadows the HF package name.
    # Trainer only needs a `Dataset` type for an isinstance check in our list-backed smoke runs.
    trainer_module.datasets = types.SimpleNamespace(Dataset=_PlaceholderDataset)


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
    logging_steps: int = 10,
    eval_steps: int = 50,
    save_steps: int = 50,
) -> Trainer:
    _ensure_trainer_dataset_namespace()
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps if max_steps is not None else -1,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
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
    )


def _load_token_list(vocab_path: str | Path) -> list[str]:
    vocab_path = Path(vocab_path)
    payload = json.loads(vocab_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "tokens" in payload:
        return list(payload["tokens"])
    raise ValueError(f"Unsupported vocab format: {vocab_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run small-scale spectral pretraining.")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--train-config", required=True)
    args = parser.parse_args()

    model_yaml = load_yaml_config(args.model_config)
    train_yaml = load_yaml_config(args.train_config)

    token_list = _load_token_list(train_yaml["data"]["vocab_path"])
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
    train_dataset = load_split_records(train_yaml["data"]["dataset_dir"], "train")
    eval_dataset = load_split_records(train_yaml["data"]["dataset_dir"], "val")
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
        logging_steps=train_yaml["training"].get("logging_steps", 10),
        eval_steps=train_yaml["training"].get("eval_steps", 50),
        save_steps=train_yaml["training"].get("save_steps", 50),
    )
    trainer.train()


if __name__ == "__main__":
    main()
