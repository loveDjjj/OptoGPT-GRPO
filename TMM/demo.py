import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from TMM import (
    calculate_absorption,
    calculate_optical_properties_batch,
    calculate_optical_properties_batch_torch,
)


def list_database_materials():
    db_dir = THIS_DIR / "database"
    materials = sorted(p.stem for p in db_dir.glob("*.xlsx"))
    if not materials:
        raise RuntimeError(f"未在 {db_dir} 找到任何 .xlsx 材料库文件")
    return materials


def maybe_plot(path, wavelengths, curves, title):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] 跳过绘图（matplotlib 不可用）: {exc}")
        return False

    plt.figure(figsize=(8, 5))
    for y, label in curves:
        plt.plot(wavelengths, y, label=label)
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Value")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return True


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def single_structure_demo(output_dir, wl_min, wl_max, num_points):
    print("[single] 开始单结构演示（通过批量接口传入 1 个结构）")
    structure = {
        "materials": ["SiO2", "Ge", "SiO2"],
        "thicknesses": [0.45, 0.18, 0.30],  # um
        "database_path": "database",
    }

    wavelengths, R, T = calculate_optical_properties_batch(
        [structure],
        wavelength_range=(wl_min, wl_max),
        num_points=num_points,
        incident_angle=np.deg2rad(15.0),
        polarization=0,  # TE/s
    )
    if wavelengths is None:
        raise RuntimeError("单结构计算失败，请检查输入参数和数据库。")

    R = R[0]
    T = T[0]
    A = calculate_absorption(wavelengths, R, T)

    csv_data = np.column_stack([wavelengths, R, T, A])
    csv_path = output_dir / "single_structure_spectrum.csv"
    np.savetxt(csv_path, csv_data, delimiter=",", header="wavelength_um,R,T,A", comments="")

    png_path = output_dir / "single_structure_spectrum.png"
    maybe_plot(
        png_path,
        wavelengths,
        [(R, "R"), (T, "T"), (A, "A")],
        "Single Structure Spectrum (TE, 15 deg)",
    )

    summary = {
        "structure": structure,
        "wavelength_range_um": [float(wl_min), float(wl_max)],
        "num_points": int(num_points),
        "R_mean": float(np.mean(R)),
        "T_mean": float(np.mean(T)),
        "A_mean": float(np.mean(A)),
        "saved_csv": str(csv_path),
        "saved_plot": str(png_path),
    }
    save_json(output_dir / "single_structure_summary.json", summary)
    print(f"[single] 完成: {csv_path}")
    return summary


def generate_random_structures(materials, num_structures, num_layers, rng):
    structures = []
    for _ in range(num_structures):
        mat = rng.choice(materials, size=num_layers, replace=True).tolist()
        th = rng.uniform(0.05, 1.2, size=num_layers).tolist()  # um
        structures.append(
            {
                "materials": mat,
                "thicknesses": th,
                "database_path": "database",
            }
        )
    return structures


def batch_structure_demo(output_dir, wl_min, wl_max, num_points, batch_size, num_layers, seed):
    print("[batch] 开始批量演示")
    rng = np.random.default_rng(seed)
    materials = list_database_materials()
    structures = generate_random_structures(materials, batch_size, num_layers, rng)

    wavelengths, R, T = calculate_optical_properties_batch(
        structures,
        wavelength_range=(wl_min, wl_max),
        num_points=num_points,
        incident_angle=np.deg2rad(0.0),
        polarization=0,  # TE/s
    )
    if wavelengths is None:
        raise RuntimeError("批量计算失败，请检查输入参数和数据库。")

    A = calculate_absorption(wavelengths, R, T)
    npz_path = output_dir / "batch_result.npz"
    np.savez_compressed(
        npz_path,
        wavelengths=wavelengths,
        R=R,
        T=T,
        A=A,
    )

    mean_R = np.mean(R, axis=0)
    mean_T = np.mean(T, axis=0)
    mean_A = np.mean(A, axis=0)
    png_path = output_dir / "batch_mean_spectrum.png"
    maybe_plot(
        png_path,
        wavelengths,
        [(mean_R, "mean R"), (mean_T, "mean T"), (mean_A, "mean A")],
        f"Batch Mean Spectrum (N={batch_size}, layers={num_layers})",
    )

    summary = {
        "num_structures": int(batch_size),
        "num_layers": int(num_layers),
        "seed": int(seed),
        "wavelength_range_um": [float(wl_min), float(wl_max)],
        "num_points": int(num_points),
        "R_global_mean": float(np.mean(R)),
        "T_global_mean": float(np.mean(T)),
        "A_global_mean": float(np.mean(A)),
        "sample_structure_0": structures[0],
        "saved_npz": str(npz_path),
        "saved_plot": str(png_path),
    }
    save_json(output_dir / "batch_summary.json", summary)
    print(f"[batch] 完成: {npz_path}")
    return summary


def autograd_demo(output_dir, wl_min, wl_max, num_points):
    print("[autograd] 开始梯度演示（使用 calculate_optical_properties_batch_torch）")
    materials = ["SiO2", "Al2O3", "HfO2"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thickness_param = torch.tensor([0.35, 0.22, 0.18], dtype=torch.float64, device=device, requires_grad=True)

    structure = {
        "materials": materials,
        "thicknesses": thickness_param,  # 保持 requires_grad=True
        "database_path": "database",
    }
    theta = torch.tensor(np.deg2rad(20.0), dtype=torch.float64, device=device)
    wavelengths_t, R, T = calculate_optical_properties_batch_torch(
        [structure],
        wavelength_range=(wl_min, wl_max),
        num_points=num_points,
        incident_angle=theta,
        polarization=0,
        device=device,
        complex_dtype=torch.complex128,
        keep_grad=True,
    )
    if wavelengths_t is None:
        raise RuntimeError("autograd 计算失败，请检查输入或数据库。")

    target_mask = (wavelengths_t >= 4.8) & (wavelengths_t <= 5.2)
    if not torch.any(target_mask):
        raise RuntimeError("目标波段掩码为空，请检查波长设置。")

    idx = torch.where(target_mask)[0]
    loss = -torch.mean(R[0, idx])  # 目标：提高目标波段反射率
    loss.backward()

    grad = thickness_param.grad.detach().cpu().numpy()
    summary = {
        "materials": materials,
        "initial_thickness_um": thickness_param.detach().cpu().numpy().tolist(),
        "loss": float(loss.detach().cpu().item()),
        "thickness_grad_dloss_dum": grad.tolist(),
        "comment": "梯度单位为 d(loss)/d(thickness_um)。这里 thickness 以 um 输入，内部会换算为 m。",
    }
    save_json(output_dir / "autograd_summary.json", summary)
    print(f"[autograd] 完成: 梯度 = {grad}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="TMM 用法演示（单结构、批量、自动求导）")
    parser.add_argument("--mode", choices=["single", "batch", "autograd", "all"], default="all")
    parser.add_argument("--wl-min", type=float, default=3.0, help="最小波长 (um)")
    parser.add_argument("--wl-max", type=float, default=8.0, help="最大波长 (um)")
    parser.add_argument("--num-points", type=int, default=500, help="波长采样点数")
    parser.add_argument("--batch-size", type=int, default=32, help="批量演示中的结构数量")
    parser.add_argument("--num-layers", type=int, default=10, help="批量演示中的层数")
    parser.add_argument("--seed", type=int, default=20260205, help="随机种子")
    parser.add_argument("--output-dir", type=str, default="demo_outputs", help="输出目录")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_single = args.mode in ("single", "all")
    run_batch = args.mode in ("batch", "all")
    run_autograd = args.mode in ("autograd", "all")

    outputs = {}
    if run_single:
        outputs["single"] = single_structure_demo(output_dir, args.wl_min, args.wl_max, args.num_points)
    if run_batch:
        outputs["batch"] = batch_structure_demo(
            output_dir,
            args.wl_min,
            args.wl_max,
            args.num_points,
            args.batch_size,
            args.num_layers,
            args.seed,
        )
    if run_autograd:
        outputs["autograd"] = autograd_demo(output_dir, args.wl_min, args.wl_max, args.num_points)

    save_json(output_dir / "demo_run_summary.json", outputs)
    print(f"[done] 所有结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
