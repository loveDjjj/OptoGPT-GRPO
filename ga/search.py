from __future__ import annotations

import heapq
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import torch

from _shared.physics.optical_calculator import calculate_optical_properties_indexed_batch_torch
from ga.targets import GATargetProfile, preprocess_seed_tokens


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
    generations_per_restart: int
    restart_count: int
    batch_size: int
    max_samples_per_target: int
    acceptance_floor_mse: float
    elite_fraction: float
    tournament_size: int
    crossover_rate: float
    material_mutation_rate: float
    thickness_mutation_rate: float
    thickness_mutation_steps: int
    random_injection_rate: float
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
    replacement_count: int
    restarts_used: int
    shortfall: int


@dataclass(frozen=True)
class GAEvaluatedCandidate:
    numeric_key: tuple[tuple[int, ...], tuple[int, ...]]
    material_indices: tuple[int, ...]
    thickness_indices: tuple[int, ...]
    reflection: np.ndarray
    transmission: np.ndarray
    target_mse: float


Evaluator = Callable[
    [torch.Tensor, torch.Tensor, GATargetProfile, float, int | None],
    tuple[torch.Tensor, list[GAEvaluatedCandidate]],
]
ProgressCallback = Callable[[dict[str, int | float | str]], None]


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


def _resolve_device(device: str) -> torch.device:
    resolved = str(device).strip().lower()
    if resolved == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def _seed_token_indices(
    *,
    target: GATargetProfile,
    material_names: list[str],
    thickness_values_nm: list[int],
) -> tuple[list[int], list[int]]:
    seed_tokens = _normalize_tokens(
        preprocess_seed_tokens(list(target.seed_tokens)),
        material_names=material_names,
        thickness_values_nm=thickness_values_nm,
    )
    material_to_idx = {name: index for index, name in enumerate(material_names)}
    thickness_to_idx = {int(value): index for index, value in enumerate(thickness_values_nm)}
    material_idx: list[int] = []
    thickness_idx: list[int] = []
    for token in seed_tokens:
        material, thickness = _split_token(token)
        material_idx.append(int(material_to_idx[material]))
        thickness_idx.append(int(thickness_to_idx[int(thickness)]))
    return material_idx, thickness_idx


def tensor_population_to_token_groups(
    material_idx: torch.Tensor,
    thickness_idx: torch.Tensor,
    *,
    material_names: list[str],
    thickness_values_nm: list[int],
) -> list[list[str]]:
    material_rows = material_idx.detach().cpu().tolist()
    thickness_rows = thickness_idx.detach().cpu().tolist()
    return [
        [f"{material_names[int(material)]}_{int(thickness_values_nm[int(thickness)])}" for material, thickness in zip(material_row, thickness_row)]
        for material_row, thickness_row in zip(material_rows, thickness_rows)
    ]


def build_numeric_structure_keys(
    material_idx: torch.Tensor,
    thickness_idx: torch.Tensor,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    material_rows = material_idx.detach().cpu().tolist()
    thickness_rows = thickness_idx.detach().cpu().tolist()
    return [
        (
            tuple(int(value) for value in material_row),
            tuple(int(value) for value in thickness_row),
        )
        for material_row, thickness_row in zip(material_rows, thickness_rows)
    ]


def unique_population_rows(
    material_idx: torch.Tensor,
    thickness_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if material_idx.ndim != 2 or thickness_idx.ndim != 2 or material_idx.shape != thickness_idx.shape:
        raise ValueError("material_idx and thickness_idx must have the same 2D shape")
    if material_idx.shape[0] == 0:
        empty_idx = torch.empty((0,), dtype=torch.long, device=material_idx.device)
        return material_idx, thickness_idx, empty_idx

    combined = torch.cat([material_idx, thickness_idx], dim=1)
    unique_rows, inverse = torch.unique(combined, dim=0, sorted=False, return_inverse=True)
    row_indices = torch.arange(combined.shape[0], device=combined.device, dtype=torch.long)
    first_indices = torch.full((unique_rows.shape[0],), combined.shape[0], dtype=torch.long, device=combined.device)
    first_indices.scatter_reduce_(0, inverse, row_indices, reduce="amin", include_self=True)
    keep_indices = torch.sort(first_indices).values
    unique_rows = combined[keep_indices]
    layer_count = material_idx.shape[1]
    return unique_rows[:, :layer_count], unique_rows[:, layer_count:], keep_indices


def build_initial_population_tensors(
    *,
    target: GATargetProfile,
    material_names: list[str],
    thickness_values_nm: list[int],
    population_size: int,
    material_mutation_rate: float,
    thickness_mutation_rate: float,
    thickness_mutation_steps: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(population_size) <= 0:
        raise ValueError("population_size must be positive")
    seed_material_idx, seed_thickness_idx = _seed_token_indices(
        target=target,
        material_names=material_names,
        thickness_values_nm=thickness_values_nm,
    )
    layer_count = len(seed_material_idx)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    material_idx = torch.tensor(seed_material_idx, dtype=torch.long, device=device).view(1, layer_count).repeat(int(population_size), 1)
    thickness_idx = torch.tensor(seed_thickness_idx, dtype=torch.long, device=device).view(1, layer_count).repeat(int(population_size), 1)

    if int(population_size) > 1:
        mutable_rows = slice(1, None)
        random_materials = torch.randint(
            low=0,
            high=len(material_names),
            size=material_idx[mutable_rows].shape,
            device=device,
            generator=generator,
        )
        material_mask = torch.rand(material_idx[mutable_rows].shape, device=device, generator=generator) < float(material_mutation_rate)
        material_idx[mutable_rows] = torch.where(material_mask, random_materials, material_idx[mutable_rows])

        delta = torch.randint(
            low=-int(thickness_mutation_steps),
            high=int(thickness_mutation_steps) + 1,
            size=thickness_idx[mutable_rows].shape,
            device=device,
            generator=generator,
        )
        thickness_mask = torch.rand(thickness_idx[mutable_rows].shape, device=device, generator=generator) < float(thickness_mutation_rate)
        thickness_idx[mutable_rows] = torch.where(thickness_mask, thickness_idx[mutable_rows] + delta, thickness_idx[mutable_rows])
        thickness_idx[mutable_rows].clamp_(0, len(thickness_values_nm) - 1)

    return material_idx, thickness_idx


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
    material_idx, thickness_idx = build_initial_population_tensors(
        target=target,
        material_names=material_names,
        thickness_values_nm=thickness_values_nm,
        population_size=population_size,
        material_mutation_rate=material_mutation_rate,
        thickness_mutation_rate=thickness_mutation_rate,
        thickness_mutation_steps=thickness_mutation_steps,
        seed=seed,
        device=torch.device("cpu"),
    )
    return tensor_population_to_token_groups(
        material_idx,
        thickness_idx,
        material_names=material_names,
        thickness_values_nm=thickness_values_nm,
    )


def compute_masked_mse(predicted_absorption: np.ndarray, target_absorption: np.ndarray, loss_mask: np.ndarray) -> float:
    mask = np.asarray(loss_mask, dtype=bool)
    if not bool(np.any(mask)):
        raise ValueError("loss_mask must contain at least one True value")
    diff = np.asarray(predicted_absorption, dtype=np.float32)[mask] - np.asarray(target_absorption, dtype=np.float32)[mask]
    return float(np.mean(np.square(diff)))


def _validate_rt_tensors(reflection: torch.Tensor, transmission: torch.Tensor, tolerance: float) -> torch.Tensor:
    finite = torch.isfinite(reflection).all(dim=1) & torch.isfinite(transmission).all(dim=1)
    reflection_ok = (reflection.min(dim=1).values >= -float(tolerance)) & (reflection.max(dim=1).values <= 1.0 + float(tolerance))
    transmission_ok = (transmission.min(dim=1).values >= -float(tolerance)) & (transmission.max(dim=1).values <= 1.0 + float(tolerance))
    energy_ok = ((reflection + transmission).max(dim=1).values <= 1.0 + float(tolerance))
    return finite & reflection_ok & transmission_ok & energy_ok


def evaluate_population_tensors_with_tmm(
    material_idx: torch.Tensor,
    thickness_idx: torch.Tensor,
    target: GATargetProfile,
    *,
    material_names: list[str],
    thickness_values_nm: list[int],
    tmm_config: TMMEvaluationConfig,
    acceptance_floor_mse: float,
    candidate_limit: int | None = None,
    material_bank_t: torch.Tensor | None = None,
    wavelengths_tensor: torch.Tensor | None = None,
    k_tensor: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[GAEvaluatedCandidate]]:
    thickness_values_t = torch.tensor(thickness_values_nm, dtype=torch.float32, device=material_idx.device)
    thickness_nm = thickness_values_t[thickness_idx]
    _, reflection_t, transmission_t = calculate_optical_properties_indexed_batch_torch(
        material_indices=material_idx,
        thickness_nm=thickness_nm,
        material_names=material_names,
        database_path=tmm_config.database_path,
        wavelength_range=tmm_config.wavelength_range_um,
        num_points=tmm_config.num_points,
        incident_angle=tmm_config.incident_angle,
        polarization=tmm_config.polarization,
        device=tmm_config.device or str(material_idx.device),
        complex_dtype=tmm_config.complex_dtype,
        material_bank_t=material_bank_t,
        wavelengths_tensor=wavelengths_tensor,
        k_tensor=k_tensor,
    )
    reflection_f = reflection_t.to(torch.float32)
    transmission_f = transmission_t.to(torch.float32)
    ok_mask = _validate_rt_tensors(reflection_f, transmission_f, tmm_config.tolerance)

    absorption = 1.0 - reflection_f - transmission_f
    target_absorption = torch.as_tensor(target.absorption, dtype=torch.float32, device=material_idx.device)
    loss_mask = torch.as_tensor(target.loss_mask, dtype=torch.bool, device=material_idx.device)
    masked_diff = absorption[:, loss_mask] - target_absorption[loss_mask].view(1, -1)
    losses = torch.mean(masked_diff.square(), dim=1)
    scores = -losses
    scores = torch.where(ok_mask, scores, torch.full_like(scores, -float("inf")))

    accepted_indices = torch.nonzero(ok_mask & (losses < float(acceptance_floor_mse)), as_tuple=False).reshape(-1)
    if candidate_limit is not None and int(candidate_limit) > 0 and int(accepted_indices.numel()) > int(candidate_limit):
        candidate_losses = losses[accepted_indices]
        best_positions = torch.topk(candidate_losses, k=int(candidate_limit), largest=False).indices
        accepted_indices = accepted_indices[best_positions]

    accepted: list[GAEvaluatedCandidate] = []
    if int(accepted_indices.numel()) > 0:
        accepted_material_idx = material_idx[accepted_indices]
        accepted_thickness_idx = thickness_idx[accepted_indices]
        numeric_keys = build_numeric_structure_keys(accepted_material_idx, accepted_thickness_idx)
        material_rows = accepted_material_idx.detach().cpu().tolist()
        thickness_rows = accepted_thickness_idx.detach().cpu().tolist()
        reflection_np = reflection_f[accepted_indices].detach().cpu().numpy()
        transmission_np = transmission_f[accepted_indices].detach().cpu().numpy()
        losses_np = losses[accepted_indices].detach().cpu().numpy()
        for numeric_key, material_row, thickness_row, reflection, transmission, loss in zip(
            numeric_keys,
            material_rows,
            thickness_rows,
            reflection_np,
            transmission_np,
            losses_np,
        ):
            accepted.append(
                GAEvaluatedCandidate(
                    numeric_key=numeric_key,
                    material_indices=tuple(int(value) for value in material_row),
                    thickness_indices=tuple(int(value) for value in thickness_row),
                    reflection=np.asarray(reflection, dtype=np.float32),
                    transmission=np.asarray(transmission, dtype=np.float32),
                    target_mse=float(loss),
                )
            )

    return scores, accepted


def make_tmm_evaluator(
    tmm_config: TMMEvaluationConfig,
    *,
    material_names: list[str],
    thickness_values_nm: list[int],
) -> Evaluator:
    cache: dict[str, torch.Tensor] = {}

    def _evaluate(
        material_idx: torch.Tensor,
        thickness_idx: torch.Tensor,
        target: GATargetProfile,
        threshold: float,
        candidate_limit: int | None = None,
    ) -> tuple[torch.Tensor, list[GAEvaluatedCandidate]]:
        if "material_bank_t" not in cache:
            complex_dtype = torch.complex128 if str(tmm_config.complex_dtype).strip().lower() in {"complex128", "torch.complex128", "c128"} else torch.complex64
            real_dtype = torch.float64 if complex_dtype == torch.complex128 else torch.float32
            from _shared.physics.optical_calculator import _get_material_refractive_index, _resolve_database_path

            wavelengths_np = np.linspace(
                tmm_config.wavelength_range_um[0],
                tmm_config.wavelength_range_um[1],
                int(tmm_config.num_points),
                dtype=np.float64,
            )
            resolved_database_path = _resolve_database_path(tmm_config.database_path)
            material_bank = []
            for material in material_names:
                ri = _get_material_refractive_index(material, resolved_database_path, wavelengths_np)
                if ri is None:
                    raise RuntimeError(f"Missing refractive index data for material: {material}")
                material_bank.append(torch.from_numpy(ri).to(device=material_idx.device, dtype=complex_dtype))
            cache["material_bank_t"] = torch.stack(material_bank, dim=0)
            cache["wavelengths_tensor"] = torch.tensor(wavelengths_np, dtype=real_dtype, device=material_idx.device)
            cache["k_tensor"] = (2 * torch.pi / cache["wavelengths_tensor"]).to(dtype=real_dtype)
        return evaluate_population_tensors_with_tmm(
            material_idx,
            thickness_idx,
            target,
            material_names=material_names,
            thickness_values_nm=thickness_values_nm,
            tmm_config=tmm_config,
            acceptance_floor_mse=threshold,
            candidate_limit=candidate_limit,
            material_bank_t=cache["material_bank_t"],
            wavelengths_tensor=cache["wavelengths_tensor"],
            k_tensor=cache["k_tensor"],
        )

    return _evaluate


def _select_parent_indices(
    scores: torch.Tensor,
    *,
    count: int,
    tournament_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    candidates = torch.randint(
        low=0,
        high=int(scores.shape[0]),
        size=(int(count), max(1, int(tournament_size))),
        device=scores.device,
        generator=generator,
    )
    candidate_scores = scores[candidates]
    best_pos = torch.argmax(candidate_scores, dim=1)
    row_index = torch.arange(int(count), device=scores.device)
    return candidates[row_index, best_pos]


def _next_population_tensors(
    material_idx: torch.Tensor,
    thickness_idx: torch.Tensor,
    scores: torch.Tensor,
    *,
    seed_material_idx: torch.Tensor,
    seed_thickness_idx: torch.Tensor,
    material_count: int,
    thickness_count: int,
    config: GASearchConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    population_size, layer_count = material_idx.shape
    elite_count = max(1, min(population_size, int(round(population_size * float(config.elite_fraction)))))
    elite_indices = torch.argsort(scores, descending=True)[:elite_count]
    elite_material = material_idx[elite_indices].clone()
    elite_thickness = thickness_idx[elite_indices].clone()

    child_count = population_size - elite_count
    if child_count <= 0:
        return elite_material, elite_thickness

    parent_a_idx = _select_parent_indices(scores, count=child_count, tournament_size=config.tournament_size, generator=generator)
    parent_b_idx = _select_parent_indices(scores, count=child_count, tournament_size=config.tournament_size, generator=generator)
    parent_a_material = material_idx[parent_a_idx]
    parent_a_thickness = thickness_idx[parent_a_idx]
    parent_b_material = material_idx[parent_b_idx]
    parent_b_thickness = thickness_idx[parent_b_idx]

    use_crossover = torch.rand((child_count, 1), device=material_idx.device, generator=generator) < float(config.crossover_rate)
    crossover_mask = (torch.rand((child_count, layer_count), device=material_idx.device, generator=generator) < 0.5) & use_crossover
    child_material = torch.where(crossover_mask, parent_b_material, parent_a_material)
    child_thickness = torch.where(crossover_mask, parent_b_thickness, parent_a_thickness)

    random_injection = torch.rand((child_count,), device=material_idx.device, generator=generator) < float(config.random_injection_rate)
    if bool(torch.any(random_injection)):
        injected_count = int(random_injection.sum().item())
        child_material[random_injection] = seed_material_idx.view(1, -1).repeat(injected_count, 1)
        child_thickness[random_injection] = seed_thickness_idx.view(1, -1).repeat(injected_count, 1)

        injected_materials = torch.randint(
            low=0,
            high=material_count,
            size=(injected_count, layer_count),
            device=material_idx.device,
            generator=generator,
        )
        injected_delta = torch.randint(
            low=-int(config.thickness_mutation_steps),
            high=int(config.thickness_mutation_steps) + 1,
            size=(injected_count, layer_count),
            device=material_idx.device,
            generator=generator,
        )
        child_material[random_injection] = injected_materials
        child_thickness[random_injection] = (child_thickness[random_injection] + injected_delta).clamp(0, thickness_count - 1)

    material_mask = torch.rand(child_material.shape, device=material_idx.device, generator=generator) < float(config.material_mutation_rate)
    random_materials = torch.randint(
        low=0,
        high=material_count,
        size=child_material.shape,
        device=material_idx.device,
        generator=generator,
    )
    child_material = torch.where(material_mask, random_materials, child_material)

    thickness_mask = torch.rand(child_thickness.shape, device=material_idx.device, generator=generator) < float(config.thickness_mutation_rate)
    delta = torch.randint(
        low=-int(config.thickness_mutation_steps),
        high=int(config.thickness_mutation_steps) + 1,
        size=child_thickness.shape,
        device=material_idx.device,
        generator=generator,
    )
    child_thickness = torch.where(thickness_mask, child_thickness + delta, child_thickness).clamp(0, thickness_count - 1)

    return torch.cat([elite_material, child_material], dim=0), torch.cat([elite_thickness, child_thickness], dim=0)


def _candidate_limit_when_pool_full(max_samples_per_target: int) -> int:
    return max(16, min(256, int(max_samples_per_target)))


def _materialize_candidate_structure(
    candidate: GAEvaluatedCandidate,
    *,
    material_names: list[str],
    thickness_values_nm: list[int],
    target: GATargetProfile,
    restart_seed: int,
    restart_index: int,
    generation: int,
) -> GAStructure:
    structure_tokens = [
        f"{material_names[int(material)]}_{int(thickness_values_nm[int(thickness)])}"
        for material, thickness in zip(candidate.material_indices, candidate.thickness_indices)
    ]
    return GAStructure(
        structure_tokens=structure_tokens,
        reflection=np.asarray(candidate.reflection, dtype=np.float32),
        transmission=np.asarray(candidate.transmission, dtype=np.float32),
        target_mse=float(candidate.target_mse),
        target_id=target.target_id,
        target_family=target.family,
        ga_seed=int(restart_seed),
        ga_restart_index=int(restart_index),
        ga_generation=int(generation),
    )


def run_seeded_ga_search(
    *,
    target: GATargetProfile,
    material_names: list[str],
    thickness_values_nm: list[int],
    config: GASearchConfig,
    evaluator: Evaluator,
    progress_callback: ProgressCallback | None = None,
) -> GASearchResult:
    if config.population_size <= 0 or config.generations_per_restart <= 0 or config.batch_size <= 0:
        raise ValueError("population_size, generations_per_restart, and batch_size must be positive")
    if config.restart_count <= 0:
        raise ValueError("restart_count must be positive")
    if config.max_samples_per_target <= 0:
        raise ValueError("max_samples_per_target must be positive")

    device = _resolve_device(config.device)
    accepted_map: dict[tuple[tuple[int, ...], tuple[int, ...]], GAStructure] = {}
    worst_heap: list[tuple[float, tuple[tuple[int, ...], tuple[int, ...]]]] = []
    duplicate_accepted = 0
    replacement_count = 0
    total_evaluated = 0
    restarts_used = 0

    def current_worst_entry() -> tuple[tuple[tuple[int, ...], tuple[int, ...]] | None, GAStructure | None]:
        while worst_heap:
            neg_mse, key = worst_heap[0]
            item = accepted_map.get(key)
            if item is None or not np.isclose(float(item.target_mse), -float(neg_mse)):
                heapq.heappop(worst_heap)
                continue
            return key, item
        return None, None

    def emit_progress(*, restart_index: int, generation: int) -> None:
        if progress_callback is None:
            return
        kept_count = len(accepted_map)
        best_mse = min((float(item.target_mse) for item in accepted_map.values()), default=float("nan"))
        worst_kept_mse = max((float(item.target_mse) for item in accepted_map.values()), default=float("nan"))
        progress_callback(
            {
                "target_id": target.target_id,
                "restart_index": int(restart_index),
                "restart_count": int(config.restart_count),
                "generation": int(generation),
                "generations_per_restart": int(config.generations_per_restart),
                "kept_count": kept_count,
                "max_samples_per_target": int(config.max_samples_per_target),
                "best_mse": best_mse,
                "worst_kept_mse": worst_kept_mse,
                "replacement_count": int(replacement_count),
                "duplicate_accepted": int(duplicate_accepted),
                "total_evaluated": int(total_evaluated),
            }
        )

    seed_material_list, seed_thickness_list = _seed_token_indices(
        target=target,
        material_names=material_names,
        thickness_values_nm=thickness_values_nm,
    )
    seed_material_idx = torch.tensor(seed_material_list, dtype=torch.long, device=device)
    seed_thickness_idx = torch.tensor(seed_thickness_list, dtype=torch.long, device=device)

    for restart_index in range(int(config.restart_count)):
        restarts_used += 1
        restart_seed = int(config.seed) + restart_index
        generator = torch.Generator(device=device)
        generator.manual_seed(restart_seed)
        material_idx, thickness_idx = build_initial_population_tensors(
            target=target,
            material_names=material_names,
            thickness_values_nm=thickness_values_nm,
            population_size=config.population_size,
            material_mutation_rate=config.material_mutation_rate,
            thickness_mutation_rate=config.thickness_mutation_rate,
            thickness_mutation_steps=config.thickness_mutation_steps,
            seed=restart_seed,
            device=device,
        )
        for generation in range(int(config.generations_per_restart)):
            score_chunks: list[torch.Tensor] = []
            for start in range(0, int(config.population_size), int(config.batch_size)):
                end = min(start + int(config.batch_size), int(config.population_size))
                chunk_material_idx = material_idx[start:end]
                chunk_thickness_idx = thickness_idx[start:end]
                chunk_material_idx, chunk_thickness_idx, _ = unique_population_rows(chunk_material_idx, chunk_thickness_idx)
                cutoff = float(config.acceptance_floor_mse)
                candidate_limit = None
                if len(accepted_map) >= int(config.max_samples_per_target):
                    _, worst_item = current_worst_entry()
                    if worst_item is not None:
                        cutoff = min(cutoff, float(worst_item.target_mse))
                    candidate_limit = _candidate_limit_when_pool_full(int(config.max_samples_per_target))
                scores_t, candidates = evaluator(
                    chunk_material_idx,
                    chunk_thickness_idx,
                    target,
                    cutoff,
                    candidate_limit,
                )
                score_chunks.append(scores_t)
                total_evaluated += int(chunk_material_idx.shape[0])
                for candidate in candidates:
                    key = candidate.numeric_key
                    existing = accepted_map.get(key)
                    if existing is not None:
                        duplicate_accepted += 1
                        if float(candidate.target_mse) < float(existing.target_mse):
                            accepted_map[key] = _materialize_candidate_structure(
                                candidate,
                                material_names=material_names,
                                thickness_values_nm=thickness_values_nm,
                                target=target,
                                restart_seed=restart_seed,
                                restart_index=restart_index,
                                generation=generation,
                            )
                            heapq.heappush(worst_heap, (-float(candidate.target_mse), key))
                            replacement_count += 1
                        continue
                    if len(accepted_map) < int(config.max_samples_per_target):
                        accepted_map[key] = _materialize_candidate_structure(
                            candidate,
                            material_names=material_names,
                            thickness_values_nm=thickness_values_nm,
                            target=target,
                            restart_seed=restart_seed,
                            restart_index=restart_index,
                            generation=generation,
                        )
                        heapq.heappush(worst_heap, (-float(candidate.target_mse), key))
                        continue
                    worst_key, worst_item = current_worst_entry()
                    if worst_key is not None and worst_item is not None and float(candidate.target_mse) < float(worst_item.target_mse):
                        del accepted_map[worst_key]
                        accepted_map[key] = _materialize_candidate_structure(
                            candidate,
                            material_names=material_names,
                            thickness_values_nm=thickness_values_nm,
                            target=target,
                            restart_seed=restart_seed,
                            restart_index=restart_index,
                            generation=generation,
                        )
                        heapq.heappush(worst_heap, (-float(candidate.target_mse), key))
                        replacement_count += 1
            scores = torch.cat(score_chunks, dim=0) if score_chunks else torch.empty((0,), dtype=torch.float32, device=device)
            if int(scores.numel()) == 0:
                break
            material_idx, thickness_idx = _next_population_tensors(
                material_idx,
                thickness_idx,
                scores,
                seed_material_idx=seed_material_idx,
                seed_thickness_idx=seed_thickness_idx,
                material_count=len(material_names),
                thickness_count=len(thickness_values_nm),
                config=config,
                generator=generator,
            )
            emit_progress(restart_index=restart_index, generation=generation)

    accepted = sorted(accepted_map.values(), key=lambda item: (float(item.target_mse), tuple(item.structure_tokens)))
    shortfall = max(0, int(config.max_samples_per_target) - len(accepted))
    return GASearchResult(
        accepted=accepted,
        target_id=target.target_id,
        layer_count=len(target.seed_tokens),
        total_evaluated=total_evaluated,
        duplicate_accepted=duplicate_accepted,
        replacement_count=replacement_count,
        restarts_used=restarts_used,
        shortfall=shortfall,
    )
