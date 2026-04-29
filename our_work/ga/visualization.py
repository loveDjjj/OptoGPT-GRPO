from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from our_work.ga.search import GAStructure
from our_work.ga.targets import GATargetProfile


def save_ga_spectrum_plots(
    *,
    accepted: list[GAStructure],
    targets: list[GATargetProfile],
    wavelengths_um: np.ndarray,
    output_dir: str | Path,
    top_k: int = 20,
) -> list[str]:
    if not accepted:
        return []
    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    figures_dir = output_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    target_by_id = {target.target_id: target for target in targets}
    accepted_by_target: dict[str, list[GAStructure]] = defaultdict(list)
    for item in accepted:
        accepted_by_target[item.target_id].append(item)

    artifacts: list[str] = []
    wavelengths = np.asarray(wavelengths_um, dtype=np.float32)
    for target_id, items in accepted_by_target.items():
        target = target_by_id[target_id]
        ranked = sorted(items, key=lambda item: float(item.target_mse))[: int(top_k)]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(wavelengths, target.absorption, color="black", linewidth=2.0, linestyle="--", label="target absorption")
        for index, item in enumerate(ranked):
            absorption = 1.0 - np.asarray(item.reflection, dtype=np.float32) - np.asarray(item.transmission, dtype=np.float32)
            ax.plot(wavelengths, absorption, alpha=0.25 if index else 0.95, linewidth=1.2 if index else 2.0, label="best" if index == 0 else None)
        ax.fill_between(wavelengths, 0.0, 1.0, where=target.loss_mask, color="#f2c94c", alpha=0.12, label="loss bands")
        ax.set_title(f"{target_id} accepted GA spectra")
        ax.set_xlabel("Wavelength (um)")
        ax.set_ylabel("Absorption")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        spectrum_name = f"{target_id}_accepted_absorption_topk.png"
        fig.savefig(figures_dir / spectrum_name, dpi=180)
        plt.close(fig)
        artifacts.append(str(Path("figures") / spectrum_name))

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist([float(item.target_mse) for item in items], bins=min(30, max(1, len(items))))
        ax.set_title(f"{target_id} GA accepted MSE")
        ax.set_xlabel("Masked absorption MSE")
        ax.set_ylabel("Count")
        fig.tight_layout()
        hist_name = f"{target_id}_mse_hist.png"
        fig.savefig(figures_dir / hist_name, dpi=180)
        plt.close(fig)
        artifacts.append(str(Path("figures") / hist_name))
    return artifacts

