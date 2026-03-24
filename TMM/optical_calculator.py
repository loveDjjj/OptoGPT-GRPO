"""Batch TMM utilities used by the GRPO reward pipeline.

This module is performance-sensitive: it handles material caching, wavelength
grid reuse, material-table loading, and the bridge from high-level structure
configs into the low-level TMM solver.
"""

import os
import numpy as np
import torch
try:
    from .TMM import TMM_solver
except ImportError:
    from TMM import TMM_solver

# 全局缓存，避免重复读取/插值/生成波长网格
_material_cache = {}
_wavelength_cache = {}
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_MODULE_DIR, ".."))
_DEFAULT_DATABASE_PATH = "nk"


def _resolve_database_path(database_path=None):
    """Resolve the nk database path relative to the project when needed."""

    path = _DEFAULT_DATABASE_PATH if database_path in (None, "") else str(database_path)
    if os.path.isabs(path):
        return os.path.normpath(path)

    candidates = [
        os.path.normpath(os.path.abspath(path)),
        os.path.normpath(os.path.join(_PROJECT_ROOT, path)),
        os.path.normpath(os.path.join(_MODULE_DIR, path)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def _find_column(columns, candidates):
    """Find the first matching column name under case-insensitive aliases."""

    normalized = {str(col).strip().lower(): col for col in columns}
    for candidate in candidates:
        match = normalized.get(candidate.lower())
        if match is not None:
            return match
    return None


def load_material_data(filename, database_path=None):
    """
    从 Excel 文件加载材料数据
    格式: 波长(微米) 实部折射率 虚部折射率
    """
    try:
        import pandas as pd

        resolved_database_path = _resolve_database_path(database_path)
        file_path = os.path.join(resolved_database_path, filename)

        if not os.path.exists(file_path):
            print(f"Could not find {filename} in database path: {resolved_database_path}")
            return None, None, None

        suffix = os.path.splitext(file_path)[1].lower()
        if suffix == ".csv":
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        wl_col = _find_column(df.columns, ["wl", "wavelength", "wavelength_um", "lambda", "um"])
        nm_col = _find_column(df.columns, ["nm", "wavelength_nm"])
        n_col = _find_column(df.columns, ["n", "n_real", "real_n"])
        k_col = _find_column(df.columns, ["k", "n_imag", "imag_k", "imag_n"])

        if wl_col is not None and n_col is not None and k_col is not None:
            wavelengths = df[wl_col].to_numpy(dtype=np.float64)
            n_real = df[n_col].to_numpy(dtype=np.float64)
            n_imag = df[k_col].to_numpy(dtype=np.float64)
        elif nm_col is not None and n_col is not None and k_col is not None:
            wavelengths = df[nm_col].to_numpy(dtype=np.float64) / 1000.0
            n_real = df[n_col].to_numpy(dtype=np.float64)
            n_imag = df[k_col].to_numpy(dtype=np.float64)
        else:
            wavelengths = df.iloc[:, 0].to_numpy(dtype=np.float64)
            n_real = df.iloc[:, 1].to_numpy(dtype=np.float64)
            n_imag = df.iloc[:, 2].to_numpy(dtype=np.float64)
            first_col_name = str(df.columns[0]).strip().lower()
            if "nm" in first_col_name or np.nanmax(wavelengths) > 50.0:
                wavelengths = wavelengths / 1000.0

        return wavelengths, n_real, n_imag
    except Exception as e:
        print(f"Cannot load material data {filename}: {e}")
        return None, None, None


def interpolate_refractive_index(wavelengths, n_real, n_imag, target_wavelengths):
    """
    插值折射率数据
    """
    from scipy.interpolate import interp1d

    n_real_interp = interp1d(wavelengths, n_real, kind="linear", bounds_error=False, fill_value="extrapolate")
    n_imag_interp = interp1d(wavelengths, n_imag, kind="linear", bounds_error=False, fill_value="extrapolate")

    n_real_interpolated = n_real_interp(target_wavelengths)
    n_imag_interpolated = n_imag_interp(target_wavelengths)
    refractive_indices = n_real_interpolated + 1j * n_imag_interpolated
    return refractive_indices


def _get_wavelength_grid(wavelength_range, num_points):
    key = (float(wavelength_range[0]), float(wavelength_range[1]), int(num_points))
    if key not in _wavelength_cache:
        wl = np.linspace(wavelength_range[0], wavelength_range[1], num_points)
        k = 2 * np.pi / wl
        _wavelength_cache[key] = (wl, k)
    return _wavelength_cache[key]


def _get_material_refractive_index(material, database_path, wavelengths):
    """Load and cache the complex refractive index for one material."""

    resolved_database_path = _resolve_database_path(database_path)
    key = (resolved_database_path, material, float(wavelengths[0]), float(wavelengths[-1]), len(wavelengths))
    if key in _material_cache:
        return _material_cache[key]

    if str(material).strip().lower() == "air":
        ri = np.ones_like(wavelengths, dtype=np.complex128)
        _material_cache[key] = ri
        return ri

    csv_name = f"{material}.csv"
    xlsx_name = f"{material}.xlsx"
    filename = csv_name if os.path.exists(os.path.join(resolved_database_path, csv_name)) else xlsx_name
    wl, n_real, n_imag = load_material_data(filename, resolved_database_path)
    if wl is None:
        return None
    ri = interpolate_refractive_index(wl, n_real, n_imag, wavelengths)
    _material_cache[key] = ri
    return ri


def calculate_optical_properties_batch(
    structure_configs,
    wavelength_range=(2, 15),
    num_points=1000,
    incident_angle=0.0,
    polarization=0,
    plot_results=False,
    debug=False,
):
    """
    批量计算多层结构的反射率和透射率（GPU并行处理）
    注意：同一批次的结构必须具有相同的层数

    参数:
        structure_configs: 结构配置列表，每个元素包含:
            {
                'materials': ['Ge', 'SiO2', ...],   # 从上到下的材料
                'thicknesses': [0.6, 0.1, ...],    # 厚度，单位微米
                'database_path': 'database'         # 材料数据库路径（相对当前模块）
            }
        wavelength_range: 波长范围 (min, max)，单位微米
        num_points: 计算的波长点数
        incident_angle: 入射角，弧度 (0 表示垂直入射)
        polarization: 0=TE, 1=TM
    """
    wavelengths_t, reflections_t, transmissions_t = calculate_optical_properties_batch_torch(
        structure_configs=structure_configs,
        wavelength_range=wavelength_range,
        num_points=num_points,
        incident_angle=incident_angle,
        polarization=polarization,
        device=None,
        complex_dtype=torch.complex128,
        keep_grad=False,
        debug=debug,
    )
    if wavelengths_t is None:
        return None, None, None

    wavelengths_np = wavelengths_t.detach().cpu().numpy()
    reflections_np = reflections_t.detach().cpu().numpy()
    transmissions_np = transmissions_t.detach().cpu().numpy()
    return wavelengths_np, reflections_np, transmissions_np


def calculate_optical_properties_batch_torch(
    structure_configs,
    wavelength_range=(2, 15),
    num_points=1000,
    incident_angle=0.0,
    polarization=0,
    device=None,
    complex_dtype=torch.complex128,
    keep_grad=False,
    debug=False,
):
    """
    训练友好的批量计算接口（返回 torch.Tensor，可选保留梯度）

    参数:
        structure_configs: 结构配置列表。每个元素:
            {
                'materials': ['Ge', 'SiO2', ...],
                'thicknesses': [0.6, 0.1, ...] 或 torch.Tensor([..], requires_grad=True),
                'database_path': 'database'
            }
        wavelength_range: 波长范围 (min, max)，单位微米
        num_points: 波长点数（>=2）
        incident_angle: 入射角（弧度），可为 float 或 torch.Tensor
        polarization: 0=TE, 1=TM
        device: 设备，None 时自动选择 cuda/cpu
        complex_dtype: torch.complex64 或 torch.complex128
        keep_grad: True 时启用梯度图（用于反向传播）

    返回:
        wavelengths_t: (num_points,) torch.Tensor
        reflections_t: (batch_size, num_points) torch.Tensor
        transmissions_t: (batch_size, num_points) torch.Tensor
    """
    if complex_dtype not in (torch.complex64, torch.complex128):
        print("错误: complex_dtype 只能是 torch.complex64 或 torch.complex128")
        return None, None, None
    real_dtype = torch.float64 if complex_dtype == torch.complex128 else torch.float32

    if not isinstance(num_points, (int, np.integer)) or int(num_points) < 2:
        print("错误: num_points 必须是大于等于 2 的整数")
        return None, None, None
    num_points = int(num_points)

    if not isinstance(wavelength_range, (tuple, list, np.ndarray)) or len(wavelength_range) != 2:
        print("错误: wavelength_range 必须是长度为 2 的序列")
        return None, None, None
    try:
        wl_min = float(wavelength_range[0])
        wl_max = float(wavelength_range[1])
    except (TypeError, ValueError):
        print("错误: wavelength_range 必须是数值")
        return None, None, None
    if not np.isfinite(wl_min) or not np.isfinite(wl_max):
        print("错误: wavelength_range 必须为有限数值")
        return None, None, None
    if wl_min <= 0 or wl_min >= wl_max:
        print("错误: wavelength_range 必须满足 0 < min < max")
        return None, None, None

    if polarization not in (0, 1):
        print("错误: polarization 只能是 0(TE) 或 1(TM)")
        return None, None, None

    if torch.is_tensor(incident_angle):
        incident_angle_value = incident_angle
        if not torch.all(torch.isfinite(incident_angle_value)):
            print("错误: incident_angle tensor 必须是有限数值")
            return None, None, None
    else:
        try:
            incident_angle_value = float(incident_angle)
        except (TypeError, ValueError):
            print("错误: incident_angle 必须是数值")
            return None, None, None
        if not np.isfinite(incident_angle_value):
            print("错误: incident_angle 必须是有限数值")
            return None, None, None

    if len(structure_configs) == 0:
        print("错误: 没有提供结构配置")
        return None, None, None

    if "materials" not in structure_configs[0] or "thicknesses" not in structure_configs[0]:
        print("错误: 每个结构配置必须包含 materials 和 thicknesses")
        return None, None, None

    try:
        first_num_layers = len(structure_configs[0]["materials"])
        first_num_thicknesses = len(structure_configs[0]["thicknesses"])
    except TypeError:
        print("错误: materials 与 thicknesses 必须是可迭代对象")
        return None, None, None

    if first_num_thicknesses != first_num_layers:
        print("错误: 第一个结构的 thicknesses 数量与 materials 层数不匹配")
        return None, None, None

    database_path = _resolve_database_path(structure_configs[0].get("database_path", "database"))
    for i, config in enumerate(structure_configs):
        if "materials" not in config or "thicknesses" not in config:
            print(f"错误: 结构 {i} 缺少 materials 或 thicknesses")
            return None, None, None
        try:
            num_materials = len(config["materials"])
            num_thicknesses = len(config["thicknesses"])
        except TypeError:
            print(f"错误: 结构 {i} 的 materials 或 thicknesses 不是可迭代对象")
            return None, None, None

        if num_materials != first_num_layers:
            print(f"错误: 结构 {i} 的层数({num_materials}) 与第一个结构({first_num_layers}) 不匹配")
            return None, None, None
        if num_thicknesses != num_materials:
            print(
                f"错误: 结构 {i} 的 thicknesses 数量({num_thicknesses}) "
                f"与 materials 层数({num_materials}) 不匹配"
            )
            return None, None, None

        config_database_path = _resolve_database_path(config.get("database_path", "database"))
        if config_database_path != database_path:
            print(
                f"错误: 结构 {i} 的 database_path ({config_database_path}) 与第一个结构 ({database_path}) 不一致"
            )
            return None, None, None

    wavelengths, k = _get_wavelength_grid(wavelength_range, num_points)
    all_materials = set()
    for config in structure_configs:
        all_materials.update(config["materials"])

    refractive_indices_dict = {}
    for material in all_materials:
        ri = _get_material_refractive_index(material, database_path, wavelengths)
        if ri is None:
            print(f"错误: 没有 {material} 的数据")
            return None, None, None
        refractive_indices_dict[material] = ri

    if device is None:
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)

    batch_size = len(structure_configs)
    wl_len = len(wavelengths)
    zero_th = torch.zeros((1,), dtype=real_dtype, device=torch_device)
    air_n = torch.ones((wl_len,), dtype=complex_dtype, device=torch_device)

    thickness_rows = []
    refractive_rows = []
    for i, config in enumerate(structure_configs):
        thicknesses_input = config["thicknesses"]
        if torch.is_tensor(thicknesses_input):
            thickness_tensor = thicknesses_input.reshape(-1).to(device=torch_device, dtype=real_dtype)
        else:
            thickness_np = np.asarray(thicknesses_input, dtype=np.float64 if real_dtype == torch.float64 else np.float32).reshape(-1)
            thickness_tensor = torch.tensor(thickness_np, dtype=real_dtype, device=torch_device)

        if thickness_tensor.numel() != first_num_layers:
            print(f"错误: 结构 {i} 的 thicknesses 元素数量与层数不匹配")
            return None, None, None
        if not torch.all(torch.isfinite(thickness_tensor)):
            print(f"错误: 结构 {i} 的 thicknesses 包含非有限数值")
            return None, None, None

        th_full = torch.cat([zero_th, thickness_tensor, zero_th], dim=0)
        thickness_rows.append(th_full.view(1, -1).expand(wl_len, -1).to(complex_dtype))

        n_list = [air_n]
        for material in config["materials"]:
            n_tensor = torch.from_numpy(refractive_indices_dict[material]).to(device=torch_device, dtype=complex_dtype)
            n_list.append(n_tensor)
        n_list.append(air_n)
        refractive_rows.append(torch.stack(n_list, dim=0))

    thicknesses_batch = torch.stack(thickness_rows, dim=0)
    refractive_indices_batch = torch.stack(refractive_rows, dim=0)
    k_tensor = torch.tensor(k, dtype=real_dtype, device=torch_device)
    wavelengths_tensor = torch.tensor(wavelengths, dtype=real_dtype, device=torch_device)

    if torch.is_tensor(incident_angle_value):
        incident_angle_value = incident_angle_value.to(device=torch_device, dtype=real_dtype)

    grad_context = torch.enable_grad() if keep_grad else torch.no_grad()
    with grad_context:
        reflections, transmissions = TMM_solver(
            thicknesses_batch,
            refractive_indices_batch,
            k_tensor,
            incident_angle_value,
            polarization,
            debug=debug,
        )

    return wavelengths_tensor, reflections, transmissions


def calculate_absorption(wavelengths, reflection, transmission):
    """
    从反射率和透射率计算吸收率
    A = 1 - R - T
    """
    absorption = 1 - reflection - transmission
    return absorption
