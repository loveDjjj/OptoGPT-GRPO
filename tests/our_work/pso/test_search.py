import numpy as np
import torch

from our_work.pso.search import (
    AcceptedStructure,
    PSOSearchConfig,
    particles_to_structure_tokens,
    run_pso_search,
)
from our_work.pso.targets import TargetProfile


def test_particles_to_structure_tokens_clips_indices_and_discretizes_thickness():
    particles = torch.tensor(
        [
            [-2.0, 0.49, 3.9, 14.0, 26.0, 999.0],
            [1.2, 1.8, 0.1, 501.0, 9.0, 30.0],
        ],
        dtype=torch.float32,
    )

    tokens = particles_to_structure_tokens(
        particles,
        material_names=["Ge", "SiO2", "TiO2"],
        thickness_values_nm=[10, 20, 30],
        layer_count=3,
    )

    assert tokens == [
        ["Ge_10", "Ge_30", "TiO2_30"],
        ["SiO2_30", "TiO2_10", "Ge_30"],
    ]


def test_run_pso_search_keeps_unique_structures_below_threshold():
    target = TargetProfile(
        target_id="demo",
        family="fixed",
        absorption=np.zeros((4,), dtype=np.float32),
    )

    def fake_evaluator(token_groups, target_profile):
        accepted = []
        scores = []
        for tokens in token_groups:
            loss = 0.01 if tokens[0] == "Ge_10" else 0.5
            scores.append(-loss)
            if loss < 0.05:
                accepted.append(
                    AcceptedStructure(
                        structure_tokens=list(tokens),
                        reflection=np.zeros((4,), dtype=np.float32),
                        transmission=np.ones((4,), dtype=np.float32),
                        target_mse=loss,
                        target_id=target_profile.target_id,
                        target_family=target_profile.family,
                        target_center_um=target_profile.center_um,
                        target_fwhm_um=target_profile.fwhm_um,
                        pso_seed=7,
                        pso_restart_index=0,
                    )
                )
        return np.asarray(scores, dtype=np.float32), accepted

    result = run_pso_search(
        target=target,
        material_names=["Ge", "SiO2"],
        thickness_values_nm=[10],
        layer_count=1,
        config=PSOSearchConfig(
            population_size=8,
            iterations=3,
            batch_size=4,
            max_accepted=2,
            acceptance_mse_threshold=0.05,
            max_stagnant_iterations=2,
            max_restarts=1,
            seed=7,
            device="cpu",
        ),
        evaluator=fake_evaluator,
    )

    assert len(result.accepted) == 1
    assert result.accepted[0].structure_tokens == ["Ge_10"]
    assert result.shortfall == 1
    assert result.total_evaluated > 0
