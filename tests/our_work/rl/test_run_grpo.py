from __future__ import annotations

from pathlib import Path

import yaml

from our_work.rl.scripts.run_grpo import main, prepare_run_dir, resolve_checkpoint_dir


def test_resolve_checkpoint_dir_picks_highest_numeric_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "checkpoint-9").mkdir(parents=True)
    (run_dir / "checkpoint-10").mkdir(parents=True)

    resolved = resolve_checkpoint_dir(run_dir)

    assert resolved == run_dir / "checkpoint-10"


def test_prepare_run_dir_uses_timestamped_child_by_default(monkeypatch, tmp_path: Path) -> None:
    class DummyCtx:
        enabled = False
        rank = 0
        world_size = 1
        local_rank = 0
        device = "cpu"
        is_main = True

    base_output_dir = tmp_path / "output"
    config = {
        "training": {
            "output_dir": str(base_output_dir),
        }
    }
    monkeypatch.setattr("our_work.rl.scripts.run_grpo._current_run_timestamp", lambda: "20260423-101112")

    run_dir = prepare_run_dir(config, dist_ctx=DummyCtx(), resume_checkpoint=None)

    assert run_dir == base_output_dir / "20260423-101112"
    assert run_dir.exists()


def test_prepare_run_dir_reuses_existing_run_on_resume(tmp_path: Path) -> None:
    class DummyCtx:
        enabled = False
        rank = 0
        world_size = 1
        local_rank = 0
        device = "cpu"
        is_main = True

    checkpoint_dir = tmp_path / "output" / "20260423-101112" / "checkpoints" / "checkpoint-50"
    checkpoint_dir.mkdir(parents=True)
    config = {
        "training": {
            "output_dir": str(tmp_path / "output"),
        }
    }

    run_dir = prepare_run_dir(config, dist_ctx=DummyCtx(), resume_checkpoint=checkpoint_dir)

    assert run_dir == checkpoint_dir.parent.parent


def test_prepare_run_dir_overwrite_mode_cleans_known_generated_outputs(tmp_path: Path) -> None:
    class DummyCtx:
        enabled = False
        rank = 0
        world_size = 1
        local_rank = 0
        device = "cpu"
        is_main = True

    run_dir = tmp_path / "output"
    (run_dir / "metrics").mkdir(parents=True)
    (run_dir / "plots").mkdir(parents=True)
    (run_dir / "tensorboard").mkdir(parents=True)
    (run_dir / "checkpoints" / "checkpoint-10").mkdir(parents=True)
    (run_dir / "metrics" / "train_metrics.jsonl").write_text("{}", encoding="utf-8")
    (run_dir / "plots" / "overview.png").write_text("png", encoding="utf-8")
    (run_dir / "tensorboard" / "events.out.tfevents.fake").write_text("tb", encoding="utf-8")
    (run_dir / "checkpoints" / "checkpoint-10" / "model.safetensors").write_text("ckpt", encoding="utf-8")
    (run_dir / "config.snapshot.yaml").write_text("training: {}", encoding="utf-8")
    (run_dir / "keep.txt").write_text("keep", encoding="utf-8")
    config = {
        "training": {
            "output_dir": str(run_dir),
            "overwrite_output_dir": True,
        }
    }

    resolved = prepare_run_dir(config, dist_ctx=DummyCtx(), resume_checkpoint=None)

    assert resolved == run_dir
    assert not (run_dir / "metrics").exists()
    assert not (run_dir / "plots").exists()
    assert not (run_dir / "tensorboard").exists()
    assert not (run_dir / "checkpoints").exists()
    assert not (run_dir / "config.snapshot.yaml").exists()
    assert (run_dir / "keep.txt").exists()


def test_run_grpo_main_forwards_distributed_backend(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "grpo.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "device": "cpu",
                "model": {"checkpoint_dir": str(tmp_path / "checkpoint")},
                "data": {"dataset_dir": str(tmp_path / "dataset")},
                "training": {"output_dir": str(tmp_path / "output")},
                "rollout": {},
                "reward": {
                    "tmm": {
                        "database_path": "database",
                        "wavelength_range_um": [2.0, 15.0],
                        "num_points": 8,
                    }
                },
                "distributed": {"backend": "gloo", "timeout_minutes": 9},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    class DummyCtx:
        enabled = False
        rank = 0
        world_size = 1
        local_rank = 0
        device = "cpu"
        is_main = True

    monkeypatch.setattr(
        "our_work.rl.scripts.run_grpo.init_distributed",
        lambda *, device, timeout_minutes, backend=None: captured.update(
            {"device": device, "timeout_minutes": timeout_minutes, "backend": backend}
        )
        or DummyCtx(),
    )
    monkeypatch.setattr(
        "our_work.rl.scripts.run_grpo.load_rl_components",
        lambda checkpoint_dir, device: object(),
    )
    monkeypatch.setattr(
        "our_work.rl.scripts.run_grpo.load_rl_split_records",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr("our_work.rl.scripts.run_grpo._current_run_timestamp", lambda: "20260423-101112")

    class DummyTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            captured["run_dir"] = kwargs["run_dir"]

        def train(self, **kwargs):
            captured["train_called"] = True

    monkeypatch.setattr("our_work.rl.scripts.run_grpo.SpectralGRPOTrainer", DummyTrainer)
    monkeypatch.setattr("our_work.rl.scripts.run_grpo.barrier", lambda: None)
    monkeypatch.setattr("our_work.rl.scripts.run_grpo.cleanup_distributed", lambda: None)

    main(["--config", str(config_path)])

    assert captured["backend"] == "gloo"
    assert captured["timeout_minutes"] == 9
    assert captured["train_called"] is True
    assert Path(captured["run_dir"]) == tmp_path / "output" / "20260423-101112"


def test_run_grpo_configs_target_a100_outputs() -> None:
    base = yaml.safe_load(Path("our_work/rl/configs/grpo/a100_4gpu.yaml").read_text(encoding="utf-8"))
    assert base["model"]["checkpoint_dir"] == "outputs/our_work/pretrain/a100_4gpu"
    assert base["data"]["dataset_dir"] == "outputs/our_work/data_gen/a100_4gpu"


def test_run_grpo_main_sets_seed_with_rank_offset(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "grpo.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "seed": 123,
                "device": "cpu",
                "model": {"checkpoint_dir": str(tmp_path / "checkpoint")},
                "data": {"dataset_dir": str(tmp_path / "dataset")},
                "training": {"output_dir": str(tmp_path / "output")},
                "rollout": {},
                "reward": {
                    "tmm": {
                        "database_path": "database",
                        "wavelength_range_um": [2.0, 15.0],
                        "num_points": 8,
                    }
                },
                "distributed": {"backend": "gloo", "timeout_minutes": 9},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    class DummyCtx:
        enabled = False
        rank = 3
        world_size = 4
        local_rank = 3
        device = "cpu"
        is_main = False

    monkeypatch.setattr(
        "our_work.rl.scripts.run_grpo.init_distributed",
        lambda *, device, timeout_minutes, backend=None: DummyCtx(),
    )
    monkeypatch.setattr(
        "our_work.rl.scripts.run_grpo.load_rl_components",
        lambda checkpoint_dir, device: object(),
    )
    monkeypatch.setattr(
        "our_work.rl.scripts.run_grpo.load_rl_split_records",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "our_work.rl.scripts.run_grpo.set_global_seed",
        lambda seed, rank_offset=0: captured.update({"seed": seed, "rank_offset": rank_offset}),
        raising=False,
    )

    class DummyTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def train(self, **kwargs):
            captured["train_called"] = True

    monkeypatch.setattr("our_work.rl.scripts.run_grpo.SpectralGRPOTrainer", DummyTrainer)
    monkeypatch.setattr("our_work.rl.scripts.run_grpo.barrier", lambda: None)
    monkeypatch.setattr("our_work.rl.scripts.run_grpo.cleanup_distributed", lambda: None)

    main(["--config", str(config_path)])

    assert captured["seed"] == 123
    assert captured["rank_offset"] == 3
