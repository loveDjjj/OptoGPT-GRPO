from __future__ import annotations

from pathlib import Path

import yaml

from our_work.rl.scripts.run_grpo import main, resolve_checkpoint_dir


def test_resolve_checkpoint_dir_picks_highest_numeric_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "checkpoint-9").mkdir(parents=True)
    (run_dir / "checkpoint-10").mkdir(parents=True)

    resolved = resolve_checkpoint_dir(run_dir)

    assert resolved == run_dir / "checkpoint-10"


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

    class DummyTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def train(self, **kwargs):
            captured["train_called"] = True

    monkeypatch.setattr("our_work.rl.scripts.run_grpo.SpectralGRPOTrainer", DummyTrainer)
    monkeypatch.setattr("our_work.rl.scripts.run_grpo.barrier", lambda: None)
    monkeypatch.setattr("our_work.rl.scripts.run_grpo.cleanup_distributed", lambda: None)

    main(["--config", str(config_path)])

    assert captured["backend"] == "gloo"
    assert captured["timeout_minutes"] == 9
    assert captured["train_called"] is True


def test_run_grpo_configs_target_a100_outputs() -> None:
    base = yaml.safe_load(Path("our_work/rl/configs/grpo/a100_4gpu.yaml").read_text(encoding="utf-8"))
    assert base["model"]["checkpoint_dir"] == "outputs/our_work/pretrain/a100_4gpu"
    assert base["data"]["dataset_dir"] == "outputs/our_work/data_gen/a100_4gpu"
