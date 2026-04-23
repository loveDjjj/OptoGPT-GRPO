from __future__ import annotations

import numpy as np
import torch

from our_work.rl.reward import compute_rollout_rewards


def test_compute_rollout_rewards_penalizes_invalid_tokens() -> None:
    outputs = compute_rollout_rewards(
        structure_token_groups=[["[UNK]"]],
        target_spectra=np.zeros((1, 16), dtype=np.float32),
        database_path="database",
        wavelength_range_um=(2.0, 15.0),
        num_points=8,
        incident_angle=0.0,
        polarization=0,
        tolerance=1e-3,
        complex_dtype="complex128",
        batch_size=1,
        invalid_structure_penalty=1.0,
        spectrum_metric="rt_rmse",
        device="cpu",
    )

    assert torch.allclose(outputs["rewards"], torch.tensor([-1.0]))
    assert torch.equal(outputs["ok_mask"], torch.tensor([False]))


def test_compute_rollout_rewards_converts_simulator_failures_to_invalid_penalty(monkeypatch) -> None:
    def raise_failure(*args, **kwargs):
        raise RuntimeError("simulator failure")

    monkeypatch.setattr("our_work.rl.reward.simulate_structure_batch", raise_failure)

    outputs = compute_rollout_rewards(
        structure_token_groups=[["Ge_10"]],
        target_spectra=np.zeros((1, 16), dtype=np.float32),
        database_path="database",
        wavelength_range_um=(2.0, 15.0),
        num_points=8,
        incident_angle=0.0,
        polarization=0,
        tolerance=1e-3,
        complex_dtype="complex128",
        batch_size=1,
        invalid_structure_penalty=1.0,
        spectrum_metric="rt_rmse",
        device="cpu",
    )

    assert torch.allclose(outputs["rewards"], torch.tensor([-1.0]))
    assert torch.allclose(outputs["spectrum_losses"], torch.tensor([1.0]))
    assert torch.equal(outputs["ok_mask"], torch.tensor([False]))
