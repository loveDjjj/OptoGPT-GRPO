from pathlib import Path

import numpy as np
import pandas as pd

from our_work.ga.scripts.plot_random_parquet_spectra import plot_random_parquet_spectra


def test_plot_random_parquet_spectra_samples_records_and_saves_png(tmp_path: Path):
    records = []
    for index in range(12):
        reflection = np.full((4,), 0.1 + index * 0.001, dtype=np.float32)
        transmission = np.full((4,), 0.2, dtype=np.float32)
        records.append(
            {
                "sample_id": f"sample-{index}",
                "structure_tokens": ["Ge_100"],
                "spectrum_rt": reflection.tolist() + transmission.tolist(),
                "target_id": "broad_3_13_high",
            }
        )
    shard_path = tmp_path / "shard-00000.parquet"
    pd.DataFrame.from_records(records).to_parquet(shard_path, index=False)
    output_path = tmp_path / "random_absorption.png"

    selected = plot_random_parquet_spectra(
        shard_path=shard_path,
        output_path=output_path,
        sample_count=3,
        seed=42,
        wavelength_min_um=2.0,
        wavelength_max_um=15.0,
        target_id="broad_3_13_high",
    )

    assert len(selected) == 3
    assert output_path.exists()
    assert output_path.stat().st_size > 0
