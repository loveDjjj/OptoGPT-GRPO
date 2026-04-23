from __future__ import annotations

import json
from pathlib import Path

import torch

from our_work.pretrain.dataset.tokenizer import SpectralStructureTokenizer
from our_work.pretrain.model.configuration_spectral_gpt import SpectralGPTConfig
from our_work.pretrain.model.modeling_spectral_gpt import SpectralGPTForCausalLM
from our_work.rl.dataset import SpectralRecordDataset
from our_work.rl.policy import RolloutSample
from our_work.rl.trainer import RLComponents, SpectralGRPOTrainer
from utils.dist import DistributedContext


def _tiny_components() -> RLComponents:
    tokenizer = SpectralStructureTokenizer(tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]", "Ge_10", "SiO2_20"])
    config = SpectralGPTConfig(
        vocab_size=len(tokenizer.tokens),
        spectrum_dim=16,
        prefix_length=2,
        n_positions=16,
        n_embd=16,
        n_layer=1,
        n_head=2,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    raw_model = SpectralGPTForCausalLM(config)
    return RLComponents(model=raw_model, raw_model=raw_model, tokenizer=tokenizer)


def _tiny_config(output_dir: str) -> dict:
    return {
        "data": {
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        },
        "training": {
            "output_dir": output_dir,
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "epochs": 1,
            "learning_rate": 1.0e-5,
            "weight_decay": 0.0,
            "grad_clip_norm": 1.0,
            "log_steps": 1,
            "eval_steps": 0,
            "save_steps": 0,
            "clip_epsilon": 0.2,
            "advantage_mode": "zscore",
            "advantage_eps": 1.0e-6,
            "bf16": False,
            "resume_from_checkpoint": None,
        },
        "rollout": {
            "group_size": 2,
            "decode": "sample",
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
            "max_new_tokens": 4,
            "batch_size": 2,
        },
        "scoring": {"batch_size": 2},
        "reward": {
            "spectrum_metric": "rt_rmse",
            "invalid_structure_penalty": 1.0,
            "tmm": {
                "database_path": "database",
                "wavelength_range_um": [2.0, 15.0],
                "num_points": 8,
                "incident_angle": 0.0,
                "polarization": 0,
                "tolerance": 1e-3,
                "complex_dtype": "complex128",
                "batch_size": 2,
                "device": "cpu",
            },
        },
        "monitoring": {
            "tensorboard": False,
            "jsonl": False,
            "csv": False,
            "save_plots": False,
            "plot_every_eval": False,
            "flush_secs": 1,
        },
    }


def _dist_ctx() -> DistributedContext:
    return DistributedContext(
        enabled=False,
        rank=0,
        world_size=1,
        local_rank=0,
        device=torch.device("cpu"),
    )


def test_save_checkpoint_writes_optimizer_scheduler_and_trainer_state(tmp_path: Path) -> None:
    trainer = SpectralGRPOTrainer(
        components=_tiny_components(),
        config=_tiny_config(str(tmp_path / "run")),
        run_dir=tmp_path / "run",
        dist_ctx=_dist_ctx(),
    )

    trainer.global_step = 7
    trainer.resume_epoch = 2
    trainer.resume_batch_index = 3
    trainer._save_checkpoint(step=7)

    checkpoint_dir = tmp_path / "run" / "checkpoints" / "checkpoint-7"
    assert (checkpoint_dir / "optimizer.pt").exists()
    assert (checkpoint_dir / "scheduler.pt").exists()
    state = json.loads((checkpoint_dir / "trainer_state.json").read_text(encoding="utf-8"))
    assert state["global_step"] == 7
    assert state["resume_epoch"] == 2
    assert state["resume_batch_index"] == 3


def test_load_checkpoint_state_restores_training_progress(tmp_path: Path) -> None:
    components = _tiny_components()
    trainer = SpectralGRPOTrainer(
        components=components,
        config=_tiny_config(str(tmp_path / "run")),
        run_dir=tmp_path / "run",
        dist_ctx=_dist_ctx(),
    )
    trainer.global_step = 5
    trainer.resume_epoch = 1
    trainer.resume_batch_index = 4
    trainer._save_checkpoint(step=5)

    resumed = SpectralGRPOTrainer(
        components=_tiny_components(),
        config=_tiny_config(str(tmp_path / "run")),
        run_dir=tmp_path / "run2",
        dist_ctx=_dist_ctx(),
        resume_checkpoint=tmp_path / "run" / "checkpoints" / "checkpoint-5",
    )

    assert resumed.global_step == 5
    assert resumed.resume_epoch == 1
    assert resumed.resume_batch_index == 4


def test_train_restores_model_to_train_mode_after_eval(tmp_path: Path, monkeypatch) -> None:
    trainer = SpectralGRPOTrainer(
        components=_tiny_components(),
        config=_tiny_config(str(tmp_path / "run")),
        run_dir=tmp_path / "run",
        dist_ctx=_dist_ctx(),
    )
    trainer.raw_model.train()

    dataset = SpectralRecordDataset(
        [
            {
                "sample_id": "sample-0",
                "spectrum_rt": [0.1] * 16,
                "structure_tokens": ["Ge_10"],
            }
        ]
    )

    monkeypatch.setattr(
        "our_work.rl.trainer.sample_structure_rollouts",
        lambda *args, **kwargs: [],
    )

    metrics = trainer._evaluate(trainer._make_dataloader(dataset, shuffle=False)[0])

    assert "mean_eval_reward" in metrics
    assert trainer.raw_model.training is True


def test_reward_kwargs_resolve_auto_device_to_runtime_device(tmp_path: Path) -> None:
    trainer = SpectralGRPOTrainer(
        components=_tiny_components(),
        config=_tiny_config(str(tmp_path / "run")),
        run_dir=tmp_path / "run",
        dist_ctx=_dist_ctx(),
    )

    trainer.reward_tmm_cfg["device"] = "auto"

    kwargs = trainer._reward_kwargs()

    assert kwargs["device"] == "cpu"


def test_scheduler_configuration_uses_cosine_and_warmup(tmp_path: Path) -> None:
    config = _tiny_config(str(tmp_path / "run"))
    config["training"]["lr_scheduler_type"] = "cosine"
    config["training"]["warmup_ratio"] = 0.1
    trainer = SpectralGRPOTrainer(
        components=_tiny_components(),
        config=config,
        run_dir=tmp_path / "run",
        dist_ctx=_dist_ctx(),
    )
    dataset = SpectralRecordDataset(
        [
            {"sample_id": "sample-0", "spectrum_rt": [0.1] * 16, "structure_tokens": ["Ge_10"]},
            {"sample_id": "sample-1", "spectrum_rt": [0.1] * 16, "structure_tokens": ["Ge_10"]},
        ]
    )
    train_loader, _ = trainer._make_dataloader(dataset, shuffle=True)
    update_steps_per_epoch = max(1, (len(train_loader) + trainer.gradient_accumulation_steps - 1) // trainer.gradient_accumulation_steps)
    trainer.total_training_steps = max(1, trainer.epochs * update_steps_per_epoch)
    trainer.warmup_steps = int(round(trainer.total_training_steps * trainer.warmup_ratio))

    assert trainer.lr_scheduler_type == "cosine"
    assert trainer.warmup_steps >= 0


def test_train_keeps_policy_in_eval_mode_for_rollout_and_scoring(tmp_path: Path, monkeypatch) -> None:
    trainer = SpectralGRPOTrainer(
        components=_tiny_components(),
        config=_tiny_config(str(tmp_path / "run")),
        run_dir=tmp_path / "run",
        dist_ctx=_dist_ctx(),
    )

    dataset = SpectralRecordDataset(
        [
            {
                "sample_id": "sample-0",
                "spectrum_rt": [0.1] * 16,
                "structure_tokens": ["Ge_10"],
            }
        ]
    )
    observed_modes: list[tuple[str, bool, bool]] = []

    def fake_rollouts(model, tokenizer, spectra, sample_ids, *, group_size, config):
        observed_modes.append(("rollout", bool(trainer.raw_model.training), bool(model.training)))
        return [
            RolloutSample(
                sample_id="sample-0",
                target_index=0,
                candidate_index=0,
                token_ids=[tokenizer.token_to_id["Ge_10"], tokenizer.eos_token_id],
                structure_tokens=["Ge_10"],
                sequence_logprob=0.0,
                terminated_by_eos=True,
            ),
            RolloutSample(
                sample_id="sample-0",
                target_index=0,
                candidate_index=1,
                token_ids=[tokenizer.token_to_id["SiO2_20"], tokenizer.eos_token_id],
                structure_tokens=["SiO2_20"],
                sequence_logprob=0.0,
                terminated_by_eos=True,
            ),
        ]

    def fake_rewards(**kwargs):
        return {
            "rewards": torch.tensor([1.0, 0.0], dtype=torch.float32),
            "spectrum_losses": torch.tensor([0.0, 1.0], dtype=torch.float32),
            "ok_mask": torch.tensor([True, True]),
        }

    def fake_logprobs(model, tokenizer, spectra, token_id_groups, *, batch_size, rollout_config=None, suppress_non_structural_tokens=True):
        observed_modes.append(("scoring", bool(trainer.raw_model.training), bool(model.training)))
        base = trainer.raw_model.lm_head.weight[0, 0]
        values = torch.stack([base + (0.1 * index) for index in range(len(token_id_groups))])
        return values, torch.ones((len(token_id_groups), 1), dtype=torch.bool, device=values.device)

    monkeypatch.setattr("our_work.rl.trainer.sample_structure_rollouts", fake_rollouts)
    monkeypatch.setattr("our_work.rl.trainer.compute_rollout_rewards", lambda **kwargs: fake_rewards(**kwargs))
    monkeypatch.setattr("our_work.rl.trainer.batch_sequence_logprobs", fake_logprobs)

    trainer.train(train_dataset=dataset, eval_dataset=None)

    assert observed_modes == [("rollout", False, False), ("scoring", False, False)]
    assert trainer.raw_model.training is False
    assert trainer.model.training is False


def test_train_does_not_perform_trailing_step_without_pending_gradients(tmp_path: Path, monkeypatch) -> None:
    base_trainer = SpectralGRPOTrainer(
        components=_tiny_components(),
        config=_tiny_config(str(tmp_path / "run")),
        run_dir=tmp_path / "run",
        dist_ctx=_dist_ctx(),
    )
    base_trainer.global_step = 3
    base_trainer.resume_epoch = 1
    base_trainer.resume_batch_index = 0
    base_trainer._save_checkpoint(step=3)

    config = _tiny_config(str(tmp_path / "run2"))
    config["training"]["gradient_accumulation_steps"] = 2
    resumed = SpectralGRPOTrainer(
        components=_tiny_components(),
        config=config,
        run_dir=tmp_path / "run2",
        dist_ctx=_dist_ctx(),
        resume_checkpoint=tmp_path / "run" / "checkpoints" / "checkpoint-3",
    )
    dataset = SpectralRecordDataset(
        [
            {
                "sample_id": "sample-0",
                "spectrum_rt": [0.1] * 16,
                "structure_tokens": ["Ge_10"],
            }
        ]
    )
    step_calls: list[int] = []
    original_step = resumed.optimizer.step

    def wrapped_step(*args, **kwargs):
        step_calls.append(1)
        return original_step(*args, **kwargs)

    monkeypatch.setattr(resumed.optimizer, "step", wrapped_step)

    resumed.train(train_dataset=dataset, eval_dataset=None)

    assert resumed.global_step == 3
    assert step_calls == []


def test_train_writes_monitoring_artifacts_and_diagnostics(tmp_path: Path, monkeypatch) -> None:
    config = _tiny_config(str(tmp_path / "run"))
    config["training"]["eval_steps"] = 1
    config["monitoring"] = {
        "tensorboard": False,
        "jsonl": True,
        "csv": True,
        "save_plots": True,
        "plot_every_eval": True,
        "flush_secs": 1,
    }
    trainer = SpectralGRPOTrainer(
        components=_tiny_components(),
        config=config,
        run_dir=tmp_path / "run",
        dist_ctx=_dist_ctx(),
    )
    dataset = SpectralRecordDataset(
        [
            {
                "sample_id": "sample-0",
                "spectrum_rt": [0.1] * 16,
                "structure_tokens": ["Ge_10"],
            }
        ]
    )

    def fake_rollouts(model, tokenizer, spectra, sample_ids, *, group_size, config):
        return [
            RolloutSample(
                sample_id="sample-0",
                target_index=0,
                candidate_index=0,
                token_ids=[tokenizer.token_to_id["Ge_10"], tokenizer.eos_token_id],
                structure_tokens=["Ge_10"],
                sequence_logprob=0.0,
                terminated_by_eos=True,
            ),
            RolloutSample(
                sample_id="sample-0",
                target_index=0,
                candidate_index=1,
                token_ids=[tokenizer.token_to_id["SiO2_20"], tokenizer.eos_token_id],
                structure_tokens=["SiO2_20"],
                sequence_logprob=0.0,
                terminated_by_eos=True,
            ),
        ]

    def fake_rewards(**kwargs):
        return {
            "rewards": torch.tensor([1.0, 0.0], dtype=torch.float32),
            "spectrum_losses": torch.tensor([0.0, 1.0], dtype=torch.float32),
            "ok_mask": torch.tensor([True, True]),
        }

    def fake_logprobs(model, tokenizer, spectra, token_id_groups, *, batch_size, rollout_config=None, suppress_non_structural_tokens=True):
        base = trainer.raw_model.lm_head.weight[0, 0]
        values = torch.stack([base + (0.1 * index) for index in range(len(token_id_groups))])
        return values, torch.ones((len(token_id_groups), 1), dtype=torch.bool, device=values.device)

    monkeypatch.setattr("our_work.rl.trainer.sample_structure_rollouts", fake_rollouts)
    monkeypatch.setattr("our_work.rl.trainer.compute_rollout_rewards", lambda **kwargs: fake_rewards(**kwargs))
    monkeypatch.setattr("our_work.rl.trainer.batch_sequence_logprobs", fake_logprobs)
    monkeypatch.setattr("our_work.rl.trainer.SpectralGRPOTrainer._evaluate", lambda self, dataloader: {"mean_eval_reward": 0.75})

    trainer.train(train_dataset=dataset, eval_dataset=dataset)

    metrics_dir = tmp_path / "run" / "metrics"
    plots_dir = tmp_path / "run" / "plots"
    train_rows = [
        json.loads(line)
        for line in (metrics_dir / "train_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert train_rows
    assert "mean_ratio" in train_rows[0]
    assert "clip_fraction" in train_rows[0]
    assert "mean_approx_kl" in train_rows[0]
    assert "learning_rate" in train_rows[0]
    assert "grad_norm" in train_rows[0]
    assert (metrics_dir / "train_metrics.csv").exists()
    assert (metrics_dir / "eval_metrics.csv").exists()
    assert (plots_dir / "train_loss.png").exists()
    assert (plots_dir / "train_mean_reward.png").exists()
    assert (plots_dir / "train_valid_ratio.png").exists()
    assert (plots_dir / "learning_rate.png").exists()
    assert (plots_dir / "grad_norm.png").exists()
    assert (plots_dir / "eval_mean_reward.png").exists()
    assert (plots_dir / "overview.png").exists()
