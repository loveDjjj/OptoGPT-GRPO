import os
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from our_work.pretrain.dataset.hf_dataset import load_parquet_records
from our_work.pretrain.monitoring import PretrainVisualizationCallback
from our_work.pretrain.model.configuration_spectral_gpt import SpectralGPTConfig
from our_work.pretrain.model.modeling_spectral_gpt import SpectralGPTForCausalLM
from our_work.pretrain.trainer.metrics import compute_token_accuracy, preprocess_logits_for_metrics
from our_work.pretrain.scripts.run_pretrain import (
    _distributed_training_requested,
    build_trainer,
    build_trainer_components,
    validate_record_spectrum_dim,
)


def test_build_trainer_components_and_load_parquet_records(tmp_path: Path):
    shard_path = tmp_path / "smoke.parquet"
    pd.DataFrame(
        [
            {
                "sample_id": "sample-000",
                "layer_count": 5,
                "structure_tokens": ["Ge_10"],
                "token_ids": [1, 4, 2],
                "materials": ["Ge"],
                "thickness_nm": [10],
                "spectrum_rt": [0.1] * 2048,
            }
        ]
    ).to_parquet(shard_path, index=False)

    records = load_parquet_records([str(shard_path)])
    assert len(records) == 1

    components = build_trainer_components(
        model_config={
            "vocab_size": 5,
            "spectrum_dim": 2048,
            "prefix_length": 2,
            "n_positions": 16,
            "n_embd": 16,
            "n_layer": 1,
            "n_head": 2,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
        token_list=["[PAD]", "[BOS]", "[EOS]", "[UNK]", "Ge_10"],
    )
    trainer = build_trainer(
        model=components["model"],
        train_dataset=records,
        eval_dataset=records,
        collator=components["collator"],
        output_dir=str(tmp_path / "trainer-out"),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        max_steps=1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.01,
        gradient_accumulation_steps=2,
        dataloader_num_workers=1,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        ddp_find_unused_parameters=False,
        save_total_limit=2,
        max_grad_norm=1.0,
    )
    assert trainer is not None
    assert trainer.args.gradient_accumulation_steps == 2
    assert trainer.args.dataloader_num_workers == 1
    assert trainer.args.dataloader_pin_memory is True
    assert trainer.args.dataloader_persistent_workers is True
    assert trainer.args.ddp_find_unused_parameters is False
    assert trainer.args.save_total_limit == 2
    assert str(trainer.args.lr_scheduler_type) == "SchedulerType.COSINE"
    assert trainer.args.warmup_ratio == pytest.approx(0.01)
    assert trainer.args.max_grad_norm == pytest.approx(1.0)
    assert trainer.preprocess_logits_for_metrics is preprocess_logits_for_metrics
    batch = next(iter(trainer.get_train_dataloader()))
    assert batch["spectra"].shape == (1, 2048)
    train_result = trainer.train()
    assert trainer.state.global_step == 1
    assert train_result.training_loss >= 0.0


def test_validate_record_spectrum_dim_rejects_mismatched_model_input_width() -> None:
    records = [
        {
            "sample_id": "sample-000",
            "layer_count": 5,
            "structure_tokens": ["Ge_10"],
            "spectrum_rt": [0.1] * 64,
        }
    ]

    with pytest.raises(ValueError, match="spectrum_dim"):
        validate_record_spectrum_dim(records, split_name="train", spectrum_dim=32)


def test_build_trainer_ignores_ddp_settings_without_real_distributed_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    shard_path = tmp_path / "smoke.parquet"
    pd.DataFrame(
        [
            {
                "sample_id": "sample-000",
                "layer_count": 5,
                "structure_tokens": ["Ge_10"],
                "token_ids": [1, 4, 2],
                "materials": ["Ge"],
                "thickness_nm": [10],
                "spectrum_rt": [0.1] * 2048,
            }
        ]
    ).to_parquet(shard_path, index=False)

    records = load_parquet_records([str(shard_path)])
    components = build_trainer_components(
        model_config={
            "vocab_size": 5,
            "spectrum_dim": 2048,
            "prefix_length": 2,
            "n_positions": 16,
            "n_embd": 16,
            "n_layer": 1,
            "n_head": 2,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
        token_list=["[PAD]", "[BOS]", "[EOS]", "[UNK]", "Ge_10"],
    )

    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")

    assert _distributed_training_requested() is False

    trainer = build_trainer(
        model=components["model"],
        train_dataset=records,
        eval_dataset=records,
        collator=components["collator"],
        output_dir=str(tmp_path / "trainer-ddp-out"),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        max_steps=1,
        ddp_find_unused_parameters=False,
        ddp_backend="nccl",
    )

    assert trainer.args.ddp_backend is None
    assert trainer.args.ddp_find_unused_parameters is None
    assert "LOCAL_RANK" not in os.environ
    assert "WORLD_SIZE" not in os.environ


def test_compute_token_accuracy_accepts_tuple_predictions() -> None:
    logits = np.asarray([
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
    ], dtype=np.float32)
    labels = np.asarray([
        [1, 2],
        [1, -100],
    ], dtype=np.int64)
    fake_past_key_values = ("ignored-cache",)

    metrics = compute_token_accuracy(((logits, fake_past_key_values), labels))

    assert metrics["token_accuracy"] == pytest.approx(1.0)


def test_compute_token_accuracy_accepts_preprocessed_token_ids() -> None:
    predicted_token_ids = np.asarray(
        [
            [1, 2],
            [1, 0],
        ],
        dtype=np.int64,
    )
    labels = np.asarray(
        [
            [1, 2],
            [1, -100],
        ],
        dtype=np.int64,
    )

    metrics = compute_token_accuracy((predicted_token_ids, labels))

    assert metrics["token_accuracy"] == pytest.approx(1.0)


def test_preprocess_logits_for_metrics_returns_token_ids_for_tuple_logits() -> None:
    logits = torch.tensor(
        [
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    fake_past_key_values = ("ignored-cache",)

    token_ids = preprocess_logits_for_metrics((logits, fake_past_key_values), labels=None)

    assert token_ids.shape == (2, 2)
    assert torch.equal(token_ids, torch.tensor([[1, 2], [1, 0]], dtype=torch.int64))


def test_model_forward_disables_cache_when_labels_are_present() -> None:
    model = SpectralGPTForCausalLM(
        SpectralGPTConfig(
            vocab_size=5,
            spectrum_dim=8,
            prefix_length=2,
            n_positions=8,
            n_embd=8,
            n_layer=1,
            n_head=2,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
    )
    outputs = model(
        spectra=torch.zeros((1, 8), dtype=torch.float32),
        input_ids=torch.tensor([[1, 4, 2]], dtype=torch.long),
        attention_mask=torch.ones((1, 5), dtype=torch.long),
        labels=torch.tensor([[-100, -100, 1, 4, 2]], dtype=torch.long),
    )

    assert outputs.past_key_values is None


def test_pretrain_visualization_callback_writes_metric_files(tmp_path: Path) -> None:
    callback = PretrainVisualizationCallback(
        output_dir=tmp_path / "run",
        enable_tensorboard=False,
        enable_jsonl=True,
        enable_csv=True,
        save_plots=False,
        plot_every_eval=False,
    )
    args = SimpleNamespace(
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        world_size=1,
    )
    state = SimpleNamespace(
        is_world_process_zero=True,
        global_step=200,
        epoch=0.1,
    )

    callback.on_train_begin(args, state, control=None)
    callback.on_log(
        args,
        state,
        control=None,
        logs={"loss": 9.5, "grad_norm": 0.8, "learning_rate": 1.0e-4, "epoch": 0.1},
    )
    callback.on_evaluate(
        args,
        state,
        control=None,
        metrics={"eval_loss": 9.2, "eval_token_accuracy": 0.37, "eval_runtime": 12.5},
    )
    callback.on_train_end(args, state, control=None)

    train_jsonl = tmp_path / "run" / "metrics" / "train_metrics.jsonl"
    eval_jsonl = tmp_path / "run" / "metrics" / "eval_metrics.jsonl"
    train_csv = tmp_path / "run" / "metrics" / "train_metrics.csv"
    eval_csv = tmp_path / "run" / "metrics" / "eval_metrics.csv"

    assert train_jsonl.exists()
    assert eval_jsonl.exists()
    assert train_csv.exists()
    assert eval_csv.exists()
    assert '"loss": 9.5' in train_jsonl.read_text(encoding="utf-8")
    assert '"eval_loss": 9.2' in eval_jsonl.read_text(encoding="utf-8")
