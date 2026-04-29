from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from our_work.data_gen.pipeline.simulator import simulate_structure_batch
from our_work.ga.targets import GATargetProfile


@dataclass(frozen=True)
class GAStructure:
    structure_tokens: list[str]
    reflection: np.ndarray
    transmission: np.ndarray
    target_mse: float
    target_id: str
    target_family: str
    ga_seed: int
    ga_restart_index: int
    ga_generation: int


@dataclass(frozen=True)
class GASearchConfig:
    population_size: int
    generations: int
    batch_size: int
    max_accepted: int
    acceptance_mse_threshold: float
    elite_fraction: float
    tournament_size: int
    crossover_rate: float
    material_mutation_rate: float
    thickness_mutation_rate: float
    thickness_mutation_steps: int
    random_injection_rate: float
    max_stagnant_generations: int
    max_restarts: int
    seed: int
    device: str = "auto"


@dataclass(frozen=True)
class TMMEvaluationConfig:
    database_path: str
    wavelength_range_um: tuple[float, float]
    num_points: int
    incident_angle: float
    polarization: int
    tolerance: float
    complex_dtype: str
    batch_size: int
    device: str | None = None


@dataclass(frozen=True)
class GASearchResult:
    accepted: list[GAStructure]
    target_id: str
    layer_count: int
    total_evaluated: int
    duplicate_accepted: int
    restarts_used: int
    shortfall: int


Evaluator = Callable[[list[list[str]], GATargetProfile, float], tuple[np.ndarray, list[GAStructure]]]


def _split_token(token: str) -> tuple[str, int]:
    material, thickness = str(token).rsplit("_", 1)
    return material, int(thickness)


def _nearest_allowed_thickness(value: int, thickness_values_nm: list[int]) -> int:
    return min(thickness_values_nm, key=lambda allowed: abs(int(allowed) - int(value)))


def _normalize_tokens(
    tokens: list[str],
    *,
    material_names: list[str],
    thickness_values_nm: list[int],
) -> list[str]:
    if not material_names:
        raise ValueError("material_names must not be empty")
    if not thickness_values_nm:
        raise ValueError("thickness_values_nm must not be empty")
    material_set = set(material_names)
    normalized: list[str] = []
    for token in tokens:
        material, thickness = _split_token(token)
        if material not in material_set:
            raise ValueError(f"seed material not found in material_names: {material}")
        normalized.append(f"{material}_{_nearest_allowed_thickness(thickness, thickness_values_nm)}")
    return normalized


def _mutate_tokens(
    tokens: list[str],
    *,
    material_names: list[str],
    thickness_values_nm: list[int],
    material_mutation_rate: float,
    thickness_mutation_rate: float,
    thickness_mutation_steps: int,
    rng: np.random.Generator,
) -> list[str]:
    thickness_index = {int(value): index for index, value in enumerate(thickness_values_nm)}
    mutated: list[str] = []
    for token in tokens:
        material, thickness = _split_token(token)
        if rng.random() < float(material_mutation_rate):
            material = str(rng.choice(material_names))
        resolved_thickness = _nearest_allowed_thickness(thickness, thickness_values_nm)
        if rng.random() < float(thickness_mutation_rate):
            current_index = thickness_index[int(resolved_thickness)]
            delta = int(rng.integers(-int(thickness_mutation_steps), int(thickness_mutation_steps) + 1))
            current_index = max(0, min(len(thickness_values_nm) - 1, current_index + delta))
            resolved_thickness = int(thickness_values_nm[current_index])
        mutated.append(f"{material}_{int(resolved_thickness)}")
    return mutated


def build_initial_population(
    *,
    target: GATargetProfile,
    material_names: list[str],
    thickness_values_nm: list[int],
    population_size: int,
    material_mutation_rate: float,
    thickness_mutation_rate: float,
    thickness_mutation_steps: int,
    seed: int,
) -> list[list[str]]:
    if int(population_size) <= 0:
        raise ValueError("population_size must be positive")
    rng = np.random.default_rng(int(seed))
    seed_tokens = _normalize_tokens(
        list(target.seed_tokens),
        material_names=material_names,
        thickness_values_nm=thickness_values_nm,
    )
    population = [seed_tokens]
    while len(population) < int(population_size):
        population.append(
            _mutate_tokens(
                seed_tokens,
                material_names=material_names,
                thickness_values_nm=thickness_values_nm,
                material_mutation_rate=material_mutation_rate,
                thickness_mutation_rate=thickness_mutation_rate,
                thickness_mutation_steps=thickness_mutation_steps,
                rng=rng,
            )
        )
    return population


def compute_masked_mse(predicted_absorption: np.ndarray, target_absorption: np.ndarray, loss_mask: np.ndarray) -> float:
    mask = np.asarray(loss_mask, dtype=bool)
    if not bool(np.any(mask)):
        raise ValueError("loss_mask must contain at least one True value")
    diff = np.asarray(predicted_absorption, dtype=np.float32)[mask] - np.asarray(target_absorption, dtype=np.float32)[mask]
    return float(np.mean(np.square(diff)))


def evaluate_tokens_with_tmm(
    token_groups: list[list[str]],
    target: GATargetProfile,
    *,
    tmm_config: TMMEvaluationConfig,
    acceptance_mse_threshold: float,
) -> tuple[np.ndarray, list[GAStructure]]:
    scores: list[float] = []
    accepted: list[GAStructure] = []
    for start in range(0, len(token_groups), int(tmm_config.batch_size)):
        chunk = token_groups[start : start + int(tmm_config.batch_size)]
        if not chunk:
            continue
        _, reflections, transmissions, ok_mask = simulate_structure_batch(
            chunk,
            database_path=tmm_config.database_path,
            wavelength_range_um=tmm_config.wavelength_range_um,
            num_points=tmm_config.num_points,
            incident_angle=tmm_config.incident_angle,
            polarization=tmm_config.polarization,
            tolerance=tmm_config.tolerance,
            complex_dtype=tmm_config.complex_dtype,
            device=tmm_config.device,
        )
        for tokens, reflection, transmission, ok in zip(chunk, reflections, transmissions, ok_mask):
            if not bool(ok):
                scores.append(-float("inf"))
                continue
            reflection_arr = np.asarray(reflection, dtype=np.float32)
            transmission_arr = np.asarray(transmission, dtype=np.float32)
            absorption = 1.0 - reflection_arr - transmission_arr
            loss = compute_masked_mse(absorption, target.absorption, target.loss_mask)
            scores.append(-loss)
            if loss < float(acceptance_mse_threshold):
                accepted.append(
                    GAStructure(
                        structure_tokens=list(tokens),
                        reflection=reflection_arr,
                        transmission=transmission_arr,
                        target_mse=loss,
                        target_id=target.target_id,
                        target_family=target.family,
                        ga_seed=0,
                        ga_restart_index=0,
                        ga_generation=0,
                    )
                )
    return np.asarray(scores, dtype=np.float32), accepted


def make_tmm_evaluator(tmm_config: TMMEvaluationConfig) -> Evaluator:
    def _evaluate(token_groups: list[list[str]], target: GATargetProfile, threshold: float) -> tuple[np.ndarray, list[GAStructure]]:
        return evaluate_tokens_with_tmm(
            token_groups,
            target,
            tmm_config=tmm_config,
            acceptance_mse_threshold=threshold,
        )

    return _evaluate


def _select_parent(population: list[list[str]], scores: np.ndarray, *, tournament_size: int, rng: np.random.Generator) -> list[str]:
    candidate_indices = rng.integers(0, len(population), size=max(1, int(tournament_size)))
    best_index = int(candidate_indices[np.argmax(scores[candidate_indices])])
    return list(population[best_index])


def _crossover(parent_a: list[str], parent_b: list[str], *, crossover_rate: float, rng: np.random.Generator) -> list[str]:
    if len(parent_a) != len(parent_b):
        raise ValueError("GA crossover requires fixed layer count within a target")
    if rng.random() >= float(crossover_rate):
        return list(parent_a)
    mask = rng.random(len(parent_a)) < 0.5
    return [parent_b[index] if bool(mask[index]) else parent_a[index] for index in range(len(parent_a))]


def _next_population(
    population: list[list[str]],
    scores: np.ndarray,
    *,
    target: GATargetProfile,
    material_names: list[str],
    thickness_values_nm: list[int],
    config: GASearchConfig,
    rng: np.random.Generator,
) -> list[list[str]]:
    population_size = len(population)
    elite_count = max(1, min(population_size, int(round(population_size * float(config.elite_fraction)))))
    elite_indices = np.argsort(scores)[::-1][:elite_count]
    next_pop = [list(population[int(index)]) for index in elite_indices]
    while len(next_pop) < population_size:
        if rng.random() < float(config.random_injection_rate):
            seed_tokens = _normalize_tokens(
                list(target.seed_tokens),
                material_names=material_names,
                thickness_values_nm=thickness_values_nm,
            )
            child = _mutate_tokens(
                seed_tokens,
                material_names=material_names,
                thickness_values_nm=thickness_values_nm,
                material_mutation_rate=1.0,
                thickness_mutation_rate=1.0,
                thickness_mutation_steps=config.thickness_mutation_steps,
                rng=rng,
            )
        else:
            parent_a = _select_parent(population, scores, tournament_size=config.tournament_size, rng=rng)
            parent_b = _select_parent(population, scores, tournament_size=config.tournament_size, rng=rng)
            child = _crossover(parent_a, parent_b, crossover_rate=config.crossover_rate, rng=rng)
            child = _mutate_tokens(
                child,
                material_names=material_names,
                thickness_values_nm=thickness_values_nm,
                material_mutation_rate=config.material_mutation_rate,
                thickness_mutation_rate=config.thickness_mutation_rate,
                thickness_mutation_steps=config.thickness_mutation_steps,
                rng=rng,
            )
        next_pop.append(child)
    return next_pop


def run_seeded_ga_search(
    *,
    target: GATargetProfile,
    material_names: list[str],
    thickness_values_nm: list[int],
    config: GASearchConfig,
    evaluator: Evaluator,
) -> GASearchResult:
    if config.population_size <= 0 or config.generations <= 0 or config.batch_size <= 0:
        raise ValueError("population_size, generations, and batch_size must be positive")
    if config.max_accepted <= 0:
        raise ValueError("max_accepted must be positive")

    accepted: list[GAStructure] = []
    seen: set[tuple[str, ...]] = set()
    duplicate_accepted = 0
    total_evaluated = 0
    restarts_used = 0

    for restart_index in range(max(1, int(config.max_restarts))):
        restarts_used += 1
        restart_seed = int(config.seed) + restart_index
        rng = np.random.default_rng(restart_seed)
        population = build_initial_population(
            target=target,
            material_names=material_names,
            thickness_values_nm=thickness_values_nm,
            population_size=config.population_size,
            material_mutation_rate=config.material_mutation_rate,
            thickness_mutation_rate=config.thickness_mutation_rate,
            thickness_mutation_steps=config.thickness_mutation_steps,
            seed=restart_seed,
        )
        stagnant_generations = 0
        for generation in range(int(config.generations)):
            score_chunks: list[np.ndarray] = []
            generation_new = 0
            for start in range(0, len(population), int(config.batch_size)):
                chunk = population[start : start + int(config.batch_size)]
                scores_np, candidates = evaluator(chunk, target, float(config.acceptance_mse_threshold))
                score_chunks.append(scores_np)
                total_evaluated += len(chunk)
                for candidate in candidates:
                    key = tuple(candidate.structure_tokens)
                    if key in seen:
                        duplicate_accepted += 1
                        continue
                    seen.add(key)
                    accepted.append(
                        replace(
                            candidate,
                            ga_seed=restart_seed,
                            ga_restart_index=restart_index,
                            ga_generation=generation,
                        )
                    )
                    generation_new += 1
                    if len(accepted) >= int(config.max_accepted):
                        return GASearchResult(
                            accepted=accepted,
                            target_id=target.target_id,
                            layer_count=len(target.seed_tokens),
                            total_evaluated=total_evaluated,
                            duplicate_accepted=duplicate_accepted,
                            restarts_used=restarts_used,
                            shortfall=0,
                        )
            scores = np.concatenate(score_chunks) if score_chunks else np.empty((0,), dtype=np.float32)
            if scores.size == 0:
                break
            if generation_new == 0:
                stagnant_generations += 1
            else:
                stagnant_generations = 0
            if stagnant_generations >= int(config.max_stagnant_generations):
                break
            population = _next_population(
                population,
                scores,
                target=target,
                material_names=material_names,
                thickness_values_nm=thickness_values_nm,
                config=config,
                rng=rng,
            )

    shortfall = max(0, int(config.max_accepted) - len(accepted))
    return GASearchResult(
        accepted=accepted,
        target_id=target.target_id,
        layer_count=len(target.seed_tokens),
        total_evaluated=total_evaluated,
        duplicate_accepted=duplicate_accepted,
        restarts_used=restarts_used,
        shortfall=shortfall,
    )
