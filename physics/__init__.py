"""薄膜光学物理模块。

这里直接承接原来的 TMM 代码，不再额外维护第二套实现。
"""

from .TMM import TMM_solver
from .optical_calculator import (
    calculate_absorption,
    calculate_optical_properties_batch,
    calculate_optical_properties_batch_torch,
    interpolate_refractive_index,
    resolve_complex_dtype,
    load_material_data,
)
from .spectrum import (
    absorption_curve,
    absorption_curve_torch,
    flatten_rt,
    is_physical_spectrum,
    physical_spectrum_mask_torch,
    spectrum_error,
    spectrum_error_torch,
    split_rt_spectrum,
    split_rt_spectrum_torch,
)
from .structure import (
    bucket_indices_by_layer_count,
    pad_tmm_config,
    pad_tmm_configs_to_max_layers,
    split_structure_token,
    structure_key,
    tokens_to_tmm_config,
)

__all__ = [
    "TMM_solver",
    "absorption_curve",
    "bucket_indices_by_layer_count",
    "calculate_absorption",
    "calculate_optical_properties_batch",
    "calculate_optical_properties_batch_torch",
    "resolve_complex_dtype",
    "flatten_rt",
    "interpolate_refractive_index",
    "is_physical_spectrum",
    "physical_spectrum_mask_torch",
    "load_material_data",
    "pad_tmm_config",
    "pad_tmm_configs_to_max_layers",
    "split_rt_spectrum",
    "split_rt_spectrum_torch",
    "split_structure_token",
    "spectrum_error_torch",
    "spectrum_error",
    "structure_key",
    "tokens_to_tmm_config",
    "absorption_curve_torch",
]
