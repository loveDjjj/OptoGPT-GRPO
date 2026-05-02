import numpy as np
import torch

from our_work.ga.search import (
    GASearchConfig,
    GAEvaluatedCandidate,
    GAStructure,
    build_numeric_structure_keys,
    build_initial_population,
    build_initial_population_tensors,
    compute_masked_mse,
    run_seeded_ga_search,
    tensor_population_to_token_groups,
    unique_population_rows,
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


def test_build_numeric_structure_keys_uses_index_rows_not_tokens():
    material_idx = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    thickness_idx = torch.tensor([[2, 3], [3, 2]], dtype=torch.long)

    keys = build_numeric_structure_keys(material_idx, thickness_idx)

    assert keys == [((0, 1), (2, 3)), ((1, 0), (3, 2))]


def test_unique_population_rows_deduplicates_batch_on_device():
    material_idx = torch.tensor([[0, 1], [0, 1], [1, 0], [1, 0]], dtype=torch.long)
    thickness_idx = torch.tensor([[2, 3], [2, 3], [3, 2], [3, 2]], dtype=torch.long)

    unique_material_idx, unique_thickness_idx, keep_indices = unique_population_rows(material_idx, thickness_idx)

    assert unique_material_idx.tolist() == [[0, 1], [1, 0]]
    assert unique_thickness_idx.tolist() == [[2, 3], [3, 2]]
    assert keep_indices.tolist() == [0, 2]


def test_run_seeded_ga_search_runs_full_budget_and_replaces_worse_samples():
    target = GATargetProfile(
        target_id="demo",
        family="seeded",
        absorption=np.zeros((4,), dtype=np.float32),
        loss_mask=np.ones((4,), dtype=bool),
        seed_tokens=["Ge_100"],
    )

    call_counter = {"value": 0}

    def fake_evaluator(material_idx, thickness_idx, target_profile, threshold, candidate_limit=None):
        scores = []
        accepted = []
        population = tensor_population_to_token_groups(
            material_idx,
            thickness_idx,
            material_names=["Ge", "SiO2"],
            thickness_values_nm=[100],
        )
        numeric_keys = build_numeric_structure_keys(material_idx, thickness_idx)
        for row_index, _tokens in enumerate(population):
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
                    GAEvaluatedCandidate(
                        numeric_key=numeric_keys[row_index],
                        material_indices=(0,),
                        thickness_indices=(0,),
                        reflection=np.zeros((4,), dtype=np.float32),
                        transmission=np.ones((4,), dtype=np.float32),
                        target_mse=loss,
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
    assert result.total_evaluated == 8
    assert result.replacement_count >= 2
    assert result.duplicate_accepted > 0


def test_run_seeded_ga_search_uses_dynamic_cutoff_after_pool_is_full():
    target = GATargetProfile(
        target_id="demo",
        family="seeded",
        absorption=np.zeros((4,), dtype=np.float32),
        loss_mask=np.ones((4,), dtype=bool),
        seed_tokens=["Ge_100"],
    )
    seen_thresholds: list[float] = []

    def fake_evaluator(material_idx, thickness_idx, target_profile, threshold, candidate_limit=None):
        seen_thresholds.append(float(threshold))
        population = tensor_population_to_token_groups(
            material_idx,
            thickness_idx,
            material_names=["Ge"],
            thickness_values_nm=[100],
        )
        accepted = []
        scores = []
        numeric_keys = build_numeric_structure_keys(material_idx, thickness_idx)
        for row_index, tokens in enumerate(population):
            loss = 0.01 if len(seen_thresholds) == 1 else 0.003
            scores.append(-loss)
            if loss < float(threshold):
                accepted.append(
                    GAEvaluatedCandidate(
                        numeric_key=numeric_keys[row_index],
                        material_indices=(0,),
                        thickness_indices=(0,),
                        reflection=np.zeros((4,), dtype=np.float32),
                        transmission=np.ones((4,), dtype=np.float32),
                        target_mse=loss,
                    )
                )
        return torch.tensor(scores, dtype=torch.float32), accepted

    result = run_seeded_ga_search(
        target=target,
        material_names=["Ge"],
        thickness_values_nm=[100],
        config=GASearchConfig(
            population_size=4,
            generations_per_restart=1,
            restart_count=1,
            batch_size=2,
            max_samples_per_target=1,
            acceptance_floor_mse=0.05,
            elite_fraction=0.5,
            tournament_size=2,
            crossover_rate=0.0,
            material_mutation_rate=0.0,
            thickness_mutation_rate=0.0,
            thickness_mutation_steps=1,
            random_injection_rate=0.0,
            seed=7,
            device="cpu",
        ),
        evaluator=fake_evaluator,
    )

    assert seen_thresholds == [0.05, 0.01]
    assert len(result.accepted) == 1
    assert result.accepted[0].target_mse == 0.003
