from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from our_work.pso.targets import TargetProfile

if TYPE_CHECKING:
    from our_work.pso.search import BestSearchCandidate, TMMEvaluationConfig


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "target"


def _split_structure_tokens(tokens: list[str]) -> tuple[list[str], list[int]]:
    materials: list[str] = []
    thickness_nm: list[int] = []
    for token in tokens:
        material, thickness = str(token).rsplit("_", 1)
        materials.append(material)
        thickness_nm.append(int(thickness))
    return materials, thickness_nm


def _simulate_best_candidates(
    candidates: dict[str, BestSearchCandidate],
    *,
    tmm_config: TMMEvaluationConfig,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    # Keep the heavy physics import lazy so plotting helpers remain importable in
    # lightweight analysis environments without initializing Torch/CUDA.
    from our_work.data_gen.pipeline.simulator import simulate_structure_batch

    spectra: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    buckets: dict[int, list[tuple[str, BestSearchCandidate]]] = {}
    for target_id, candidate in candidates.items():
        buckets.setdefault(len(candidate.structure_tokens), []).append((target_id, candidate))

    for bucket in buckets.values():
        for start in range(0, len(bucket), max(1, int(tmm_config.batch_size))):
            chunk = bucket[start : start + max(1, int(tmm_config.batch_size))]
            token_groups = [candidate.structure_tokens for _, candidate in chunk]
            _, reflections, transmissions, ok_mask = simulate_structure_batch(
                token_groups,
                database_path=tmm_config.database_path,
                wavelength_range_um=tmm_config.wavelength_range_um,
                num_points=tmm_config.num_points,
                incident_angle=tmm_config.incident_angle,
                polarization=tmm_config.polarization,
                tolerance=tmm_config.tolerance,
                complex_dtype=tmm_config.complex_dtype,
                device=tmm_config.device,
            )
            for (target_id, _), reflection, transmission, ok in zip(
                chunk, reflections, transmissions, ok_mask
            ):
                if bool(ok):
                    spectra[target_id] = (
                        np.asarray(reflection, dtype=np.float32),
                        np.asarray(transmission, dtype=np.float32),
                    )
    return spectra


def save_best_target_plots(
    *,
    output_dir: str | Path,
    wavelengths_um: np.ndarray,
    targets: dict[str, TargetProfile],
    candidates: dict[str, BestSearchCandidate],
    tmm_config: TMMEvaluationConfig,
    dpi: int = 220,
    include_rt: bool = True,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    matplotlib_cache = output_path / ".matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wavelengths = np.asarray(wavelengths_um, dtype=np.float32)
    spectra = _simulate_best_candidates(candidates, tmm_config=tmm_config)
    manifest: dict[str, Any] = {"target_count": len(targets), "plotted_count": 0, "targets": {}}

    for target_id, target in targets.items():
        candidate = candidates.get(target_id)
        spectrum = spectra.get(target_id)
        if candidate is None or spectrum is None:
            manifest["targets"][target_id] = {"status": "no_valid_candidate"}
            continue

        reflection, transmission = spectrum
        absorption = 1.0 - reflection - transmission
        target_absorption = np.asarray(target.absorption, dtype=np.float32)
        target_mse = float(np.mean((absorption - target_absorption) ** 2))
        materials, thickness_nm = _split_structure_tokens(candidate.structure_tokens)
        safe_name = _safe_name(target_id)
        plot_path = output_path / f"{safe_name}.png"
        json_path = output_path / f"{safe_name}.json"

        fig, (spectrum_ax, structure_ax) = plt.subplots(
            2,
            1,
            figsize=(10, 8),
            gridspec_kw={"height_ratios": [3.2, 1.8]},
        )
        spectrum_ax.plot(wavelengths, target_absorption, "k--", linewidth=2.0, label="Target A")
        spectrum_ax.plot(wavelengths, absorption, color="#c23b33", linewidth=1.8, label="Best A")
        if include_rt:
            spectrum_ax.plot(wavelengths, reflection, color="#276fbf", linewidth=1.1, alpha=0.85, label="R")
            spectrum_ax.plot(wavelengths, transmission, color="#2d8a56", linewidth=1.1, alpha=0.85, label="T")
        spectrum_ax.set_title(f"{target_id} | best MSE={target_mse:.6g}")
        spectrum_ax.set_xlabel("Wavelength (um)")
        spectrum_ax.set_ylabel("R / T / A")
        spectrum_ax.set_xlim(float(wavelengths[0]), float(wavelengths[-1]))
        spectrum_ax.set_ylim(-0.05, 1.05)
        spectrum_ax.grid(alpha=0.25)
        spectrum_ax.legend(ncol=4 if include_rt else 2, fontsize=9)

        structure_ax.axis("off")
        structure_ax.set_title("Best structure (incident side to substrate)", fontsize=11, pad=8)
        table = structure_ax.table(
            cellText=[
                [str(index), material, str(thickness)]
                for index, (material, thickness) in enumerate(zip(materials, thickness_nm), start=1)
            ],
            colLabels=["Layer", "Material", "Thickness (nm)"],
            cellLoc="center",
            colLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.15)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=max(72, int(dpi)), bbox_inches="tight")
        plt.close(fig)

        payload = {
            "status": "ok",
            "target_id": target_id,
            "target_family": target.family,
            "target_mse": target_mse,
            "search_target_mse": float(candidate.target_mse),
            "layer_count": len(candidate.structure_tokens),
            "structure_tokens": list(candidate.structure_tokens),
            "materials": materials,
            "thickness_nm": thickness_nm,
            "pso_seed": int(candidate.pso_seed),
            "pso_restart_index": int(candidate.pso_restart_index),
            "plot": plot_path.name,
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["targets"][target_id] = payload
        manifest["plotted_count"] += 1

    (output_path / "best_structures.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest
