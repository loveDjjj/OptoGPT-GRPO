from pathlib import Path

import numpy as np
import torch

from our_work._shared.physics.optical_calculator import calculate_optical_properties_indexed_batch_torch
from our_work.data_gen.pipeline.simulator import simulate_structure_batch


def test_indexed_tmm_matches_token_based_simulation(tmp_path: Path):
    database_dir = tmp_path / "database"
    database_dir.mkdir()
    for material in ["Ge", "SiO2"]:
        (database_dir / f"{material}.csv").write_text("wl,n,k\n2.0,2.0,0.1\n15.0,2.0,0.1\n", encoding="utf-8")

    token_groups = [["Ge_100", "SiO2_200"], ["SiO2_200", "Ge_100"]]
    _, reflection_ref, transmission_ref, ok_mask = simulate_structure_batch(
        token_groups,
        database_path=str(database_dir),
        wavelength_range_um=(2.0, 15.0),
        num_points=16,
        incident_angle=0.0,
        polarization=0,
        tolerance=1.0e-3,
        complex_dtype="complex128",
        device="cpu",
    )
    assert ok_mask.tolist() == [True, True]

    material_idx = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    thickness_nm = torch.tensor([[100.0, 200.0], [200.0, 100.0]], dtype=torch.float32)
    _, reflection_new, transmission_new = calculate_optical_properties_indexed_batch_torch(
        material_indices=material_idx,
        thickness_nm=thickness_nm,
        material_names=["Ge", "SiO2"],
        database_path=str(database_dir),
        wavelength_range=(2.0, 15.0),
        num_points=16,
        incident_angle=0.0,
        polarization=0,
        device="cpu",
        complex_dtype="complex128",
    )

    assert torch.allclose(reflection_new.cpu(), torch.from_numpy(np.stack(reflection_ref, axis=0)), atol=1.0e-6, rtol=1.0e-6)
    assert torch.allclose(transmission_new.cpu(), torch.from_numpy(np.stack(transmission_ref, axis=0)), atol=1.0e-6, rtol=1.0e-6)
