"""
TMM (Transfer Matrix Method) module
Optical calculation and dataset generation for multilayer thin films
"""

from .TMM import TMM_solver
from .optical_calculator import (
    load_material_data,
    interpolate_refractive_index,
    calculate_optical_properties_batch,
    calculate_optical_properties_batch_torch,
    calculate_absorption
)

__all__ = [
    'TMM_solver',
    'load_material_data',
    'interpolate_refractive_index',
    'calculate_optical_properties_batch',
    'calculate_optical_properties_batch_torch',
    'calculate_absorption'
]
