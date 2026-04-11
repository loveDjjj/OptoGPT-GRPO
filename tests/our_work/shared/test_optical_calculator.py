from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from our_work._shared.physics.optical_calculator import load_material_data


def test_load_material_data_reads_headerless_xlsx_as_um_n_k(tmp_path: Path) -> None:
    material_path = tmp_path / "SiO2.xlsx"
    pd.DataFrame(
        [
            [0.280347, 1.49404, 1e-9],
            [0.289360, 1.49099, 1e-9],
            [500.0, 1.83900, 6.596e-2],
        ]
    ).to_excel(material_path, header=False, index=False)

    wavelengths, n_real, n_imag = load_material_data(material_path.name, tmp_path)

    assert np.isclose(wavelengths[0], 0.280347)
    assert np.isclose(wavelengths[-1], 500.0)
    assert np.isclose(n_real[0], 1.49404)
    assert np.isclose(n_imag[-1], 6.596e-2)


def test_load_material_data_reads_headerless_csv_as_um_n_k(tmp_path: Path) -> None:
    material_path = tmp_path / "Ge.csv"
    material_path.write_text(
        "2.0,4.10845,0.0\n2.12,4.09481,0.0\n15.0,4.0019,0.0\n",
        encoding="utf-8",
    )

    wavelengths, n_real, n_imag = load_material_data(material_path.name, tmp_path)

    assert np.isclose(wavelengths[0], 2.0)
    assert np.isclose(wavelengths[-1], 15.0)
    assert np.isclose(n_real[0], 4.10845)
    assert np.isclose(n_imag[-1], 0.0)
