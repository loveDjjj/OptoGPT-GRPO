import numpy as np
import torch

from our_work.ga.search import (
    GASearchConfig,
    GAStructure,
    build_initial_population,
    build_initial_population_tensors,
    compute_masked_mse,
    run_seeded_ga_search,
    tensor_population_to_token_groups,
)
from our_work.ga.targets import GATargetProfile


def test_compute_masked_mse_ignores_unmasked_wavelengths():
    target = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    mask = np.array([True, False, True, False])
    predicted = np.array([0.5, 999.0, 0.5, 999.0], dtype=np.float32)

    assert compute_masked_mse(predicted, target, mask) == 0.25


def test_build_initial_population_mutates_seed_with_allowed_materials_and_thicknesses():
    target = GATargetProfile(
        target_id="demo",
        family="seeded",
        absorption=np.ones((4,), dtype=np.float32),
        loss_mask=np.ones((4,), dtype=bool),
        seed_tokens=["Ge_100", "SiO2_200"],
    )

    population = build_initial_population(
        target=target,
        material_names=["Ge", "SiO2", "Au"],
        thickness_values_nm=[100, 200, 300],
        population_size=8,
        material_mutation_rate=0.8,
        thickness_mutation_rate=0.8,
        thickness_mutation_steps=2,
        seed=123,
    )

    assert population[0] == ["Ge_100", "SiO2_200"]
    assert len(population) == 8
    assert all(len(tokens) == 2 for tokens in population)
    for tokens in population:
        for token in tokens:
            material, thickness = token.rsplit("_", 1)
            assert material in {"Ge", "SiO2", "Au"}
            assert int(thickness) in {100, 200, 300}


def test_build_initial_population_preprocesses_seed_layers_above_500nm():
    target = GATargetProfile(
        target_id="demo",
        family="seeded",
        absorption=np.ones((4,), dtype=np.float32),
        loss_mask=np.ones((4,), dtype=bool),
        seed_tokens=["YbF3_870", "Au_100"],
    )

    population = build_initial_population(
        target=target,
        material_names=["YbF3", "Au"],
        thickness_values_nm=list(range(10, 501, 10)),
        population_size=2,
        material_mutation_rate=0.0,
        thickness_mutation_rate=0.0,
        thickness_mutation_steps=1,
        seed=123,
    )

    assert population[0] == ["YbF3_430", "YbF3_440", "Au_100"]


def test_build_initial_population_tensors_preserve_seed_tokens_and_allowed_ranges():
    target = GATargetProfile(
        target_id="demo",
        family="seeded",
        absorption=np.ones((4,), dtype=np.float32),
        loss_mask=np.ones((4,), dtype=bool),
        seed_tokens=["Ge_100", "SiO2_200"],
    )

    material_idx, thickness_idx = build_initial_population_tensors(
        target=target,
        material_names=["Ge", "SiO2", "Au"],
        thickness_values_nm=[100, 200, 300],
        population_size=8,
        material_mutation_rate=0.8,
        thickness_mutation_rate=0.8,
        thickness_mutation_steps=2,
        seed=123,
        device=torch.device("cpu"),
    )
    population = tensor_population_to_token_groups(
        material_idx,
        thickness_idx,
        material_names=["Ge", "SiO2", "Au"],
        thickness_values_nm=[100, 200, 300],
    )

    assert population[0] == ["Ge_100", "SiO2_200"]
    assert material_idx.shape == (8, 2)
    assert thickness_idx.shape == (8, 2)
    assert torch.all((material_idx >= 0) & (material_idx <= 2))
    assert torch.all((thickness_idx >= 0) & (thickness_idx <= 2))


def test_run_seeded_ga_search_runs_full_budget_and_replaces_worse_samples():
    target = GATargetProfile(
        target_id="demo",
        family="seeded",
        absorption=np.zeros((4,), dtype=np.float32),
        loss_mask=np.ones((4,), dtype=bool),
        seed_tokens=["Ge_100"],
    )

    call_counter = {"value": 0}

    def fake_evaluator(material_idx, thickness_idx, target_profile, threshold):
        scores = []
        accepted = []
        population = tensor_population_to_token_groups(
            material_idx,
            thickness_idx,
            material_names=["Ge", "SiO2"],
            thickness_values_nm=[100],
        )
        for tokens in population:
            current_call = call_counter["value"]
            if current_call < 2:
                loss = 0.02
            elif current_call < 4:
                loss = 0.01
            else:
                loss = 0.003
            call_counter["value"] += 1
            scores.append(-loss)
            if loss < threshold:
                accepted.append(
                    GAStructure(
                        structure_tokens=list(tokens),
                        reflection=np.zeros((4,), dtype=np.float32),
                        transmission=np.ones((4,), dtype=np.float32),
                        target_mse=loss,
                        target_id=target_profile.target_id,
                        target_family=target_profile.family,
                        ga_seed=7,
                        ga_restart_index=0,
                        ga_generation=0,
                    )
                )
        return torch.tensor(scores, dtype=torch.float32), accepted

    result = run_seeded_ga_search(
        target=target,
        material_names=["Ge", "SiO2"],
        thickness_values_nm=[100],
        config=GASearchConfig(
            population_size=4,
            generations_per_restart=2,
            restart_count=2,
            batch_size=2,
            max_samples_per_target=1,
            acceptance_floor_mse=0.05,
            elite_fraction=0.5,
            tournament_size=2,
            crossover_rate=0.8,
            material_mutation_rate=0.0,
            thickness_mutation_rate=0.0,
            thickness_mutation_steps=1,
            random_injection_rate=0.0,
            seed=7,
            device="cpu",
        ),
        evaluator=fake_evaluator,
    )

    assert len(result.accepted) == 1
    assert result.accepted[0].structure_tokens == ["Ge_100"]
    assert result.accepted[0].target_mse == 0.003
    assert result.restarts_used == 2
    assert result.total_evaluated == 16
    assert result.replacement_count >= 2
    assert result.duplicate_accepted > 0
