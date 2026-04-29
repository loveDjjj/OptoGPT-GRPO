import numpy as np

from our_work.ga.search import (
    GASearchConfig,
    GAStructure,
    build_initial_population,
    compute_masked_mse,
    run_seeded_ga_search,
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


def test_run_seeded_ga_search_keeps_unique_structures_below_threshold():
    target = GATargetProfile(
        target_id="demo",
        family="seeded",
        absorption=np.zeros((4,), dtype=np.float32),
        loss_mask=np.ones((4,), dtype=bool),
        seed_tokens=["Ge_100"],
    )

    def fake_evaluator(token_groups, target_profile, threshold):
        scores = []
        accepted = []
        for tokens in token_groups:
            loss = 0.001 if tokens[0] == "Ge_100" else 0.2
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
        return np.asarray(scores, dtype=np.float32), accepted

    result = run_seeded_ga_search(
        target=target,
        material_names=["Ge", "SiO2"],
        thickness_values_nm=[100],
        config=GASearchConfig(
            population_size=6,
            generations=3,
            batch_size=3,
            max_accepted=2,
            acceptance_mse_threshold=0.005,
            elite_fraction=0.5,
            tournament_size=2,
            crossover_rate=0.8,
            material_mutation_rate=0.0,
            thickness_mutation_rate=0.0,
            thickness_mutation_steps=1,
            random_injection_rate=0.0,
            max_stagnant_generations=2,
            max_restarts=1,
            seed=7,
            device="cpu",
        ),
        evaluator=fake_evaluator,
    )

    assert len(result.accepted) == 1
    assert result.accepted[0].structure_tokens == ["Ge_100"]
    assert result.duplicate_accepted > 0
    assert result.shortfall == 1
