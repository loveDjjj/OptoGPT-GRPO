from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import torch

from our_work.data_gen.pipeline.simulator import simulate_structure_batch
from our_work.pso.targets import TargetProfile


@dataclass(frozen=True)
class AcceptedStructure:
    structure_tokens: list[str]
    reflection: np.ndarray
    transmission: np.ndarray
    target_mse: float
    target_id: str
    target_family: str
    target_center_um: float | None
    target_fwhm_um: float | None
    pso_seed: int
    pso_restart_index: int


@dataclass(frozen=True)
class PSOSearchConfig:
    population_size: int
    iterations: int
    batch_size: int
    max_accepted: int
    acceptance_mse_threshold: float
    max_stagnant_iterations: int
    max_restarts: int
    seed: int
    device: str = "auto"
    inertia: float = 0.7
    cognitive: float = 1.5
    social: float = 1.5


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
class BestSearchCandidate:
    structure_tokens: list[str]
    target_mse: float
    pso_seed: int
    pso_restart_index: int


@dataclass(frozen=True)
class PSOSearchResult:
    accepted: list[AcceptedStructure]
    target_id: str
    layer_count: int
    total_evaluated: int
    duplicate_accepted: int
    restarts_used: int
    shortfall: int
    best_candidate: BestSearchCandidate | None


Evaluator = Callable[[list[list[str]], TargetProfile], tuple[np.ndarray, list[AcceptedStructure]]]


def _resolve_device(device: str) -> torch.device:
    resolved = str(device).strip().lower()
    if resolved == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _nearest_thickness_values(values: torch.Tensor, thickness_values_nm: list[int]) -> torch.Tensor:
    allowed = torch.tensor(thickness_values_nm, dtype=values.dtype, device=values.device)
    distances = torch.abs(values.unsqueeze(-1) - allowed.view(1, 1, -1))
    nearest_indices = torch.argmin(distances, dim=-1)
    return allowed[nearest_indices].to(torch.long)


def particles_to_structure_tokens(
    particles: torch.Tensor,
    *,
    material_names: list[str],
    thickness_values_nm: list[int],
    layer_count: int,
) -> list[list[str]]:
    if layer_count <= 0:
        raise ValueError("layer_count must be positive")
    if particles.ndim != 2 or particles.shape[1] != layer_count * 2:
        raise ValueError("particles must have shape (batch, layer_count * 2)")
    if not material_names:
        raise ValueError("material_names must not be empty")
    if not thickness_values_nm:
        raise ValueError("thickness_values_nm must not be empty")

    material_idx = torch.round(particles[:, :layer_count]).clamp(0, len(material_names) - 1).to(torch.long)
    thickness_nm = _nearest_thickness_values(particles[:, layer_count:], thickness_values_nm)

    material_rows = material_idx.detach().cpu().tolist()
    thickness_rows = thickness_nm.detach().cpu().tolist()
    return [
        [f"{material_names[int(material)]}_{int(thickness)}" for material, thickness in zip(material_row, thickness_row)]
        for material_row, thickness_row in zip(material_rows, thickness_rows)
    ]


def evaluate_tokens_with_tmm(
    token_groups: list[list[str]],
    target: TargetProfile,
    *,
    tmm_config: TMMEvaluationConfig,
    acceptance_mse_threshold: float,
) -> tuple[np.ndarray, list[AcceptedStructure]]:
    scores: list[float] = []
    accepted: list[AcceptedStructure] = []
    target_absorption = np.asarray(target.absorption, dtype=np.float32)

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
            loss = float(np.mean((absorption - target_absorption) ** 2))
            scores.append(-loss)
            if loss < float(acceptance_mse_threshold):
                accepted.append(
                    AcceptedStructure(
                        structure_tokens=list(tokens),
                        reflection=reflection_arr,
                        transmission=transmission_arr,
                        target_mse=loss,
                        target_id=target.target_id,
                        target_family=target.family,
                        target_center_um=target.center_um,
                        target_fwhm_um=target.fwhm_um,
                        pso_seed=0,
                        pso_restart_index=0,
                    )
                )
    return np.asarray(scores, dtype=np.float32), accepted


def make_tmm_evaluator(tmm_config: TMMEvaluationConfig, *, acceptance_mse_threshold: float) -> Evaluator:
    def _evaluate(token_groups: list[list[str]], target: TargetProfile) -> tuple[np.ndarray, list[AcceptedStructure]]:
        return evaluate_tokens_with_tmm(
            token_groups,
            target,
            tmm_config=tmm_config,
            acceptance_mse_threshold=acceptance_mse_threshold,
        )

    return _evaluate


def _initialize_particles(
    *,
    config: PSOSearchConfig,
    material_count: int,
    thickness_values_nm: list[int],
    layer_count: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    population = int(config.population_size)
    dim = layer_count * 2
    particles = torch.empty((population, dim), dtype=torch.float32, device=device)
    particles[:, :layer_count] = torch.rand((population, layer_count), device=device, generator=generator) * max(0, material_count - 1)

    min_thickness = float(min(thickness_values_nm))
    max_thickness = float(max(thickness_values_nm))
    particles[:, layer_count:] = min_thickness + torch.rand((population, layer_count), device=device, generator=generator) * (
        max_thickness - min_thickness
    )
    velocity = torch.randn((population, dim), dtype=torch.float32, device=device, generator=generator) * 0.1
    return particles, velocity


def run_pso_search(
    *,
    target: TargetProfile,
    material_names: list[str],
    thickness_values_nm: list[int],
    layer_count: int,
    config: PSOSearchConfig,
    evaluator: Evaluator,
) -> PSOSearchResult:
    if config.population_size <= 0 or config.iterations <= 0 or config.batch_size <= 0:
        raise ValueError("population_size, iterations, and batch_size must be positive")
    if config.max_accepted <= 0:
        raise ValueError("max_accepted must be positive")

    device = _resolve_device(config.device)
    accepted: list[AcceptedStructure] = []
    seen: set[tuple[str, ...]] = set()
    total_evaluated = 0
    duplicate_accepted = 0
    restarts_used = 0
    best_candidate: BestSearchCandidate | None = None

    for restart_index in range(max(1, int(config.max_restarts))):
        restarts_used += 1
        generator = torch.Generator(device=device)
        generator.manual_seed(int(config.seed) + restart_index)
        particles, velocity = _initialize_particles(
            config=config,
            material_count=len(material_names),
            thickness_values_nm=thickness_values_nm,
            layer_count=layer_count,
            device=device,
            generator=generator,
        )
        pbest = particles.clone()
        pbest_score = torch.full((config.population_size,), -float("inf"), dtype=torch.float32, device=device)
        gbest = particles[0].clone()
        gbest_score = -float("inf")
        stagnant_iterations = 0

        for _ in range(int(config.iterations)):
            iteration_new = 0
            score_chunks: list[np.ndarray] = []
            for start in range(0, int(config.population_size), int(config.batch_size)):
                end = min(start + int(config.batch_size), int(config.population_size))
                token_groups = particles_to_structure_tokens(
                    particles[start:end],
                    material_names=material_names,
                    thickness_values_nm=thickness_values_nm,
                    layer_count=layer_count,
                )
                scores_np, candidates = evaluator(token_groups, target)
                if len(scores_np) != len(token_groups):
                    raise ValueError("evaluator scores must align with token_groups")
                finite_indices = np.flatnonzero(np.isfinite(scores_np))
                if finite_indices.size:
                    local_index = int(finite_indices[np.argmax(scores_np[finite_indices])])
                    local_mse = -float(scores_np[local_index])
                    if best_candidate is None or local_mse < best_candidate.target_mse:
                        best_candidate = BestSearchCandidate(
                            structure_tokens=list(token_groups[local_index]),
                            target_mse=local_mse,
                            pso_seed=int(config.seed) + restart_index,
                            pso_restart_index=restart_index,
                        )
                score_chunks.append(scores_np)
                total_evaluated += len(token_groups)
                for candidate in candidates:
                    key = tuple(candidate.structure_tokens)
                    if key in seen:
                        duplicate_accepted += 1
                        continue
                    seen.add(key)
                    accepted.append(
                        replace(
                            candidate,
                            pso_seed=int(config.seed) + restart_index,
                            pso_restart_index=restart_index,
                        )
                    )
                    iteration_new += 1
                    if len(accepted) >= int(config.max_accepted):
                        shortfall = max(0, int(config.max_accepted) - len(accepted))
                        return PSOSearchResult(
                            accepted=accepted,
                            target_id=target.target_id,
                            layer_count=layer_count,
                            total_evaluated=total_evaluated,
                            duplicate_accepted=duplicate_accepted,
                            restarts_used=restarts_used,
                            shortfall=shortfall,
                            best_candidate=best_candidate,
                        )

            scores = np.concatenate(score_chunks) if score_chunks else np.empty((0,), dtype=np.float32)
            if scores.size == 0:
                break
            score_t = torch.tensor(scores, dtype=torch.float32, device=device)
            better = score_t > pbest_score
            pbest[better] = particles[better]
            pbest_score[better] = score_t[better]
            best_index = int(torch.argmax(score_t).item())
            best_score = float(score_t[best_index].item())
            if best_score > gbest_score:
                gbest_score = best_score
                gbest = particles[best_index].clone()

            if iteration_new == 0:
                stagnant_iterations += 1
            else:
                stagnant_iterations = 0
            if stagnant_iterations >= int(config.max_stagnant_iterations):
                break

            r1 = torch.rand(particles.shape, dtype=particles.dtype, device=device, generator=generator)
            r2 = torch.rand(particles.shape, dtype=particles.dtype, device=device, generator=generator)
            velocity = (
                float(config.inertia) * velocity
                + float(config.cognitive) * r1 * (pbest - particles)
                + float(config.social) * r2 * (gbest.view(1, -1) - particles)
            )
            particles = particles + velocity
            particles[:, :layer_count] = particles[:, :layer_count].clamp(0, len(material_names) - 1)
            particles[:, layer_count:] = particles[:, layer_count:].clamp(min(thickness_values_nm), max(thickness_values_nm))

    shortfall = max(0, int(config.max_accepted) - len(accepted))
    return PSOSearchResult(
        accepted=accepted,
        target_id=target.target_id,
        layer_count=layer_count,
        total_evaluated=total_evaluated,
        duplicate_accepted=duplicate_accepted,
        restarts_used=restarts_used,
        shortfall=shortfall,
        best_candidate=best_candidate,
    )
