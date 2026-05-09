from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _shared.io.config import resolve_repo_path
from ga.targets import build_default_ga_targets


def _to_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        return list(value.tolist())
    return list(value)


def _extract_absorption(spectrum_rt: Any) -> np.ndarray:
    values = np.asarray(_to_list(spectrum_rt), dtype=np.float32).reshape(-1)
    if values.size == 0 or values.size % 2 != 0:
        raise ValueError("spectrum_rt must contain [R..., T...] with an even positive length")
    half = values.size // 2
    reflection = values[:half]
    transmission = values[half:]
    return (1.0 - reflection - transmission).astype(np.float32)


def _resolve_target(target_id: str, wavelengths_um: np.ndarray):
    targets = {target.target_id: target for target in build_default_ga_targets(wavelengths_um)}
    if target_id not in targets:
        raise ValueError(f"unsupported target_id {target_id!r}; available: {sorted(targets)}")
    return targets[target_id]


def plot_random_parquet_spectra(
    *,
    shard_path: str | Path,
    output_path: str | Path,
    sample_count: int = 10,
    seed: int = 42,
    wavelength_min_um: float = 2.0,
    wavelength_max_um: float = 15.0,
    target_id: str = "broad_3_13_high",
) -> list[dict[str, Any]]:
    shard = Path(shard_path)
    if int(sample_count) <= 0:
        raise ValueError("sample_count must be positive")
    frame = pd.read_parquet(shard)
    required_columns = {"sample_id", "spectrum_rt"}
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"missing required parquet columns: {missing}")
    if len(frame) == 0:
        raise ValueError(f"no records found in shard: {shard}")

    take_count = min(int(sample_count), len(frame))
    sampled = frame.sample(n=take_count, random_state=int(seed), replace=False).reset_index(drop=True)
    first_absorption = _extract_absorption(sampled.iloc[0]["spectrum_rt"])
    wavelengths = np.linspace(float(wavelength_min_um), float(wavelength_max_um), len(first_absorption), dtype=np.float32)
    target = _resolve_target(target_id, wavelengths)

    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(wavelengths, target.absorption, color="black", linestyle="--", linewidth=2.0, label="target absorption")
    ax.fill_between(wavelengths, 0.0, 1.0, where=target.loss_mask, color="#f2c94c", alpha=0.12, label="loss bands")

    selected: list[dict[str, Any]] = []
    for index, (_, row) in enumerate(sampled.iterrows()):
        absorption = _extract_absorption(row["spectrum_rt"])
        if len(absorption) != len(wavelengths):
            raise ValueError("all sampled spectrum_rt rows must have the same length")
        sample_id = str(row["sample_id"])
        ax.plot(
            wavelengths,
            absorption,
            linewidth=1.4,
            alpha=0.85,
            label="sampled spectra" if index == 0 else None,
        )
        selected.append(
            {
                "sample_id": sample_id,
                "target_id": str(row["target_id"]) if "target_id" in row and not pd.isna(row["target_id"]) else None,
                "structure_tokens": _to_list(row["structure_tokens"]) if "structure_tokens" in row else None,
            }
        )

    ax.set_title(f"{target_id} random {take_count} spectra from {shard.name}")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Absorption")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)

    selected_path = output.with_suffix(".selected.json")
    selected_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    return selected


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot random absorption spectra from a data_gen parquet shard.")
    parser.add_argument("--shard-path", default="outputs/our_work/data_gen/ga_seeded_absorbers/shards/shard-00000.parquet")
    parser.add_argument("--output-path", default="outputs/our_work/data_gen/ga_seeded_absorbers/figures/random_10_broad_3_13_absorption.png")
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wavelength-min-um", type=float, default=2.0)
    parser.add_argument("--wavelength-max-um", type=float, default=15.0)
    parser.add_argument("--target-id", default="broad_3_13_high")
    args = parser.parse_args(argv)

    selected = plot_random_parquet_spectra(
        shard_path=resolve_repo_path(args.shard_path, project_root=PROJECT_ROOT),
        output_path=resolve_repo_path(args.output_path, project_root=PROJECT_ROOT),
        sample_count=args.sample_count,
        seed=args.seed,
        wavelength_min_um=args.wavelength_min_um,
        wavelength_max_um=args.wavelength_max_um,
        target_id=args.target_id,
    )
    print(json.dumps({"selected_count": len(selected), "output_path": args.output_path}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
