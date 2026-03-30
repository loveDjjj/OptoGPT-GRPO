"""对比 SFT 前后 OptoGPT 隐表示的 t-SNE 入口。"""

from __future__ import annotations

import argparse
import inspect
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

# 允许直接通过 `python runners/run_tsne_compare.py ...` 运行。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.optogpt import OptoGPTModel, resolve_device
from utils.logging import make_run_dir, write_json
from utils.seed import set_global_seed


# 沿用 analysis_optogpt.ipynb 中的材料顺序，保证颜色和图例语义一致。
NOTEBOOK_MATERIALS = [
    "Al",
    "Ag",
    "Al2O3",
    "AlN",
    "Ge",
    "HfO2",
    "ITO",
    "MgF2",
    "MgO",
    "Si",
    "Si3N4",
    "SiO2",
    "Ta2O5",
    "TiN",
    "TiO2",
    "ZnO",
    "ZnS",
    "ZnSe",
]
NOTEBOOK_LABELS = [
    "$Al$",
    "$Ag$",
    "$Al_2O_3$",
    "$AlN$",
    "$Ge$",
    "$HfO_2$",
    "$ITO$",
    "$MgF_2$",
    "$MgO$",
    "$Si$",
    "$Si_3N_4$",
    "$SiO_2$",
    "$Ta_2O_5$",
    "$TiN$",
    "$TiO_2$",
    "$ZnO$",
    "$ZnS$",
    "$ZnSe$",
]
NOTEBOOK_COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
    "magenta",
    "lightcoral",
    "red",
    "peru",
    "gold",
    "lawngreen",
    "aqua",
    "blue",
]


def _load_spectrum_array(path: str | Path) -> np.ndarray:
    """加载 `.npy/.pkl` 光谱数组。"""

    array_path = Path(path)
    suffix = array_path.suffix.lower()
    if suffix == ".npy":
        return np.load(array_path, mmap_mode="r")
    if suffix == ".pkl":
        with array_path.open("rb") as handle:
            return pickle.load(handle)
    raise ValueError(f"不支持的光谱文件格式: {array_path}")


def _select_spectra(path: str | Path, max_spectra: int | None) -> np.ndarray:
    """按 notebook 的习惯直接取前 N 条光谱做可视化。"""

    spectrum_array = _load_spectrum_array(path)
    if spectrum_array.ndim != 2:
        raise ValueError(f"光谱数组必须是二维，当前形状为 {tuple(spectrum_array.shape)}")
    count = spectrum_array.shape[0] if max_spectra is None else min(int(max_spectra), int(spectrum_array.shape[0]))
    return np.asarray(spectrum_array[:count], dtype=np.float32)


def _extract_structure_embeddings(model: OptoGPTModel) -> np.ndarray:
    """提取结构 token embedding，等价于 notebook 里的 `hidedn_struc`。"""

    embed_layer = model.raw_model.tgt_embed[0]
    if not hasattr(embed_layer, "lut"):
        raise AttributeError("当前 checkpoint 的 tgt_embed[0] 不包含 `lut`，无法提取结构 embedding。")
    return embed_layer.lut.weight.detach().cpu().numpy().astype(np.float32, copy=False)


@torch.inference_mode()
def _extract_spectrum_embeddings(
    model: OptoGPTModel,
    spectra: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """提取光谱经过 `fc` 后的隐藏表示，等价于 notebook 里的 `hidedn_spec`。"""

    hidden_chunks: list[np.ndarray] = []
    total = int(spectra.shape[0])
    for start in range(0, total, batch_size):
        batch_np = spectra[start : start + batch_size]
        batch_tensor = model.targets_to_tensor_batch(batch_np)
        hidden = model.raw_model.fc(batch_tensor)
        if hidden.dim() == 3 and hidden.size(1) == 1:
            hidden = hidden[:, 0, :]
        elif hidden.dim() != 2:
            hidden = hidden.reshape(hidden.size(0), -1)
        hidden_chunks.append(hidden.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(hidden_chunks, axis=0) if hidden_chunks else np.empty((0, model.ckpt_args.d_model), dtype=np.float32)


def _build_material_token_indices(struc_word_dict: dict[str, int]) -> dict[str, list[int]]:
    """构造 notebook 里 `mat_index` 等价映射。"""

    material_indices: dict[str, list[int]] = {}
    for material in NOTEBOOK_MATERIALS:
        token_ids = []
        for thickness in range(10, 501, 10):
            token = f"{material}_{thickness}"
            token_id = struc_word_dict.get(token)
            if token_id is not None:
                token_ids.append(int(token_id))
        material_indices[material] = token_ids
    return material_indices


def _build_tsne(perplexity: float, iterations: int, seed: int) -> TSNE:
    """兼容不同 sklearn 版本的 TSNE 参数名。"""

    signature = inspect.signature(TSNE)
    kwargs = {
        "n_components": 2,
        "perplexity": float(perplexity),
        "random_state": int(seed),
        "init": "pca",
        "learning_rate": "auto",
    }
    if "max_iter" in signature.parameters:
        kwargs["max_iter"] = int(iterations)
    else:
        kwargs["n_iter"] = int(iterations)
    return TSNE(**kwargs)


def _split_tsne_coordinates(
    joint_coords: np.ndarray,
    before_structure_count: int,
    before_spectrum_count: int,
    after_structure_count: int,
    after_spectrum_count: int,
) -> dict[str, np.ndarray]:
    """把共享 t-SNE 结果拆回 before/after 两组。"""

    cursor = 0
    before_structure = joint_coords[cursor : cursor + before_structure_count]
    cursor += before_structure_count
    before_spectrum = joint_coords[cursor : cursor + before_spectrum_count]
    cursor += before_spectrum_count
    after_structure = joint_coords[cursor : cursor + after_structure_count]
    cursor += after_structure_count
    after_spectrum = joint_coords[cursor : cursor + after_spectrum_count]
    return {
        "before_structure": before_structure,
        "before_spectrum": before_spectrum,
        "after_structure": after_structure,
        "after_spectrum": after_spectrum,
    }


def _plot_panel(
    axis,
    *,
    structure_coords: np.ndarray,
    spectrum_coords: np.ndarray,
    material_indices: dict[str, list[int]],
    title: str,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> None:
    """绘制单个 checkpoint 的 t-SNE 面板。"""

    for material, label, color in zip(NOTEBOOK_MATERIALS, NOTEBOOK_LABELS, NOTEBOOK_COLORS):
        token_ids = material_indices.get(material, [])
        if not token_ids:
            continue
        points = structure_coords[np.asarray(token_ids, dtype=np.int64)]
        sizes = np.linspace(10.0, 34.0, len(token_ids), dtype=np.float32)
        axis.scatter(points[:, 0], points[:, 1], s=sizes, label=label, color=color, alpha=0.95)

    if spectrum_coords.size > 0:
        axis.scatter(
            spectrum_coords[:, 0],
            spectrum_coords[:, 1],
            color="tab:green",
            marker="+",
            s=18,
            linewidths=0.8,
            label="Spectrum",
            alpha=0.85,
        )

    axis.set_title(title, fontsize=16, pad=12)
    axis.set_xlabel("Dimension 1")
    axis.set_ylabel("Dimension 2")
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.tick_params(axis="both", direction="in")
    axis.grid(alpha=0.18)


def _save_tsne_compare_plot(
    path: str | Path,
    *,
    coordinates: dict[str, np.ndarray],
    material_indices: dict[str, list[int]],
    before_label: str,
    after_label: str,
    dpi: int,
) -> None:
    """保存共享坐标系下的 before/after t-SNE 对比图。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_coords = np.concatenate(
        [
            coordinates["before_structure"],
            coordinates["before_spectrum"],
            coordinates["after_structure"],
            coordinates["after_spectrum"],
        ],
        axis=0,
    )
    x_margin = 0.05 * max(1.0, float(np.ptp(all_coords[:, 0])))
    y_margin = 0.05 * max(1.0, float(np.ptp(all_coords[:, 1])))
    x_limits = (float(all_coords[:, 0].min() - x_margin), float(all_coords[:, 0].max() + x_margin))
    y_limits = (float(all_coords[:, 1].min() - y_margin), float(all_coords[:, 1].max() + y_margin))

    figure, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=dpi, sharex=True, sharey=True)
    _plot_panel(
        axes[0],
        structure_coords=coordinates["before_structure"],
        spectrum_coords=coordinates["before_spectrum"],
        material_indices=material_indices,
        title=before_label,
        x_limits=x_limits,
        y_limits=y_limits,
    )
    _plot_panel(
        axes[1],
        structure_coords=coordinates["after_structure"],
        spectrum_coords=coordinates["after_spectrum"],
        material_indices=material_indices,
        title=after_label,
        x_limits=x_limits,
        y_limits=y_limits,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncols=7, fontsize=10)
    figure.suptitle("OptoGPT Hidden Representation t-SNE Comparison", fontsize=18)
    figure.tight_layout(rect=(0, 0.05, 1, 0.96))
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="对比 SFT 前后 OptoGPT 隐表示的 t-SNE。")
    parser.add_argument("--before-checkpoint", required=True, help="SFT 前 checkpoint 路径。")
    parser.add_argument("--after-checkpoint", required=True, help="SFT 后 checkpoint 路径。")
    parser.add_argument("--spectrum-path", required=True, help="用于提取光谱隐藏表示的数据文件路径（.npy/.pkl）。")
    parser.add_argument("--output-dir", default="outputs/analysis", help="输出根目录。")
    parser.add_argument("--name", default="tsne_compare", help="实验名称。")
    parser.add_argument("--device", default="auto", help="运行设备，如 auto/cpu/cuda:0。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument("--max-spectra", type=int, default=1000, help="最多取多少条光谱做对比。")
    parser.add_argument("--batch-size", type=int, default=256, help="提取光谱隐藏表示时的 batch 大小。")
    parser.add_argument("--perplexity", type=float, default=5.0, help="t-SNE perplexity。")
    parser.add_argument("--n-iter", type=int, default=1500, help="t-SNE 迭代轮数。")
    parser.add_argument("--dpi", type=int, default=180, help="输出图片 DPI。")
    args = parser.parse_args()

    set_global_seed(int(args.seed))
    device = resolve_device(args.device)
    run_dir = make_run_dir(args.output_dir, args.name)

    print(f"[tsne] loading spectra from {args.spectrum_path}")
    spectra = _select_spectra(args.spectrum_path, args.max_spectra)
    print(f"[tsne] selected spectra: {spectra.shape[0]}")

    print(f"[tsne] loading before checkpoint: {args.before_checkpoint}")
    before_model = OptoGPTModel(args.before_checkpoint, device=device)
    print(f"[tsne] loading after checkpoint: {args.after_checkpoint}")
    after_model = OptoGPTModel(args.after_checkpoint, device=device)

    if before_model.struc_word_dict != after_model.struc_word_dict:
        raise ValueError("before/after checkpoint 的结构词表不一致，当前脚本无法直接对齐比较。")

    before_structure = _extract_structure_embeddings(before_model)
    after_structure = _extract_structure_embeddings(after_model)
    before_spectrum = _extract_spectrum_embeddings(before_model, spectra, batch_size=int(args.batch_size))
    after_spectrum = _extract_spectrum_embeddings(after_model, spectra, batch_size=int(args.batch_size))

    all_design = np.concatenate(
        [before_structure, before_spectrum, after_structure, after_spectrum],
        axis=0,
    )
    all_design_norm = MinMaxScaler().fit_transform(all_design)

    time_start = time.time()
    tsne = _build_tsne(perplexity=float(args.perplexity), iterations=int(args.n_iter), seed=int(args.seed))
    joint_coords = tsne.fit_transform(all_design_norm)
    elapsed = time.time() - time_start
    print(f"[tsne] finished in {elapsed:.2f}s")

    coordinates = _split_tsne_coordinates(
        joint_coords,
        before_structure_count=before_structure.shape[0],
        before_spectrum_count=before_spectrum.shape[0],
        after_structure_count=after_structure.shape[0],
        after_spectrum_count=after_spectrum.shape[0],
    )
    material_indices = _build_material_token_indices(before_model.struc_word_dict)

    plot_path = run_dir / "tsne_compare.png"
    before_label = f"Before: {Path(args.before_checkpoint).stem}"
    after_label = f"After: {Path(args.after_checkpoint).stem}"
    _save_tsne_compare_plot(
        plot_path,
        coordinates=coordinates,
        material_indices=material_indices,
        before_label=before_label,
        after_label=after_label,
        dpi=int(args.dpi),
    )

    np.savez(
        run_dir / "tsne_points.npz",
        before_structure=coordinates["before_structure"],
        before_spectrum=coordinates["before_spectrum"],
        after_structure=coordinates["after_structure"],
        after_spectrum=coordinates["after_spectrum"],
    )
    write_json(
        run_dir / "metadata.json",
        {
            "before_checkpoint": str(args.before_checkpoint),
            "after_checkpoint": str(args.after_checkpoint),
            "spectrum_path": str(args.spectrum_path),
            "selected_spectra": int(spectra.shape[0]),
            "structure_token_count": int(before_structure.shape[0]),
            "perplexity": float(args.perplexity),
            "n_iter": int(args.n_iter),
            "seed": int(args.seed),
            "elapsed_seconds": float(elapsed),
        },
    )
    print(f"[tsne] plot saved to {plot_path}")


if __name__ == "__main__":
    main()
