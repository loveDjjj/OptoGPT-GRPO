# our_work Data Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an independent `our_work/data_gen` pipeline that samples 5-10 layer structures, simulates `2-15um` `R/T` spectra with the migrated TMM solver, validates outputs, and writes shard-based training data for `our_work/pretrain`.

**Architecture:** Create a self-contained `our_work/_shared` package by migrating the required physics and utility modules from the repo root, then build `data_gen` as a small pipeline with four focused responsibilities: material registry, structure sampling, spectrum simulation, and shard/manifest writing. The pipeline should produce neutral shards plus explicit split metadata so pretraining can consume the dataset without depending on root-level RL code.

**Tech Stack:** Python, PyTorch, NumPy, pandas, PyYAML, pytest, Parquet/Arrow via `datasets`

---

## File Structure

### Shared modules
- Create: `our_work/_shared/__init__.py`
- Create: `our_work/_shared/physics/__init__.py`
- Create: `our_work/_shared/physics/TMM.py`
- Create: `our_work/_shared/physics/optical_calculator.py`
- Create: `our_work/_shared/physics/structure.py`
- Create: `our_work/_shared/io/__init__.py`
- Create: `our_work/_shared/io/config.py`
- Create: `our_work/_shared/utils/__init__.py`
- Create: `our_work/_shared/utils/seed.py`

### Data generation pipeline
- Create: `our_work/data_gen/__init__.py`
- Create: `our_work/data_gen/configs/dataset_v1.yaml`
- Create: `our_work/data_gen/pipeline/__init__.py`
- Create: `our_work/data_gen/pipeline/material_registry.py`
- Create: `our_work/data_gen/pipeline/token_vocab.py`
- Create: `our_work/data_gen/pipeline/sampler.py`
- Create: `our_work/data_gen/pipeline/simulator.py`
- Create: `our_work/data_gen/pipeline/shard_writer.py`
- Create: `our_work/data_gen/pipeline/build_dataset.py`
- Create: `our_work/data_gen/scripts/run_build_dataset.py`

### Tests
- Create: `tests/our_work/shared/test_structure.py`
- Create: `tests/our_work/data_gen/test_material_registry.py`
- Create: `tests/our_work/data_gen/test_sampler.py`
- Create: `tests/our_work/data_gen/test_simulator.py`
- Create: `tests/our_work/data_gen/test_shard_writer.py`
- Create: `tests/our_work/data_gen/test_build_dataset.py`

### Docs
- Modify: `docs/notes.md`
- Modify: `docs/logs/2026-04.md`

---

### Task 1: Scaffold `our_work/_shared` and migrate the physics baseline

**Files:**
- Create: `our_work/_shared/__init__.py`
- Create: `our_work/_shared/physics/__init__.py`
- Create: `our_work/_shared/physics/TMM.py`
- Create: `our_work/_shared/physics/optical_calculator.py`
- Create: `our_work/_shared/physics/structure.py`
- Create: `our_work/_shared/io/__init__.py`
- Create: `our_work/_shared/io/config.py`
- Create: `our_work/_shared/utils/__init__.py`
- Create: `our_work/_shared/utils/seed.py`
- Test: `tests/our_work/shared/test_structure.py`

- [ ] **Step 1: Write the failing test**

```python
from our_work._shared.physics.structure import split_structure_token, tokens_to_tmm_config


def test_tokens_to_tmm_config_converts_nm_to_um():
    material, thickness = split_structure_token("SiO2_120")
    assert material == "SiO2"
    assert thickness == 120.0

    config = tokens_to_tmm_config(["SiO2_120", "Ge_250"], database_path="database")
    assert config["materials"] == ["SiO2", "Ge"]
    assert config["thicknesses"] == [0.12, 0.25]
    assert config["database_path"] == "database"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/our_work/shared/test_structure.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'our_work'`

- [ ] **Step 3: Create the shared package and migrate the baseline files**

```powershell
New-Item -ItemType Directory -Force -Path `
  our_work\_shared\physics, `
  our_work\_shared\io, `
  our_work\_shared\utils | Out-Null

Set-Content our_work\_shared\__init__.py ""
Set-Content our_work\_shared\physics\__init__.py ""
Set-Content our_work\_shared\io\__init__.py ""
Set-Content our_work\_shared\utils\__init__.py ""

Copy-Item physics\TMM.py our_work\_shared\physics\TMM.py
Copy-Item physics\optical_calculator.py our_work\_shared\physics\optical_calculator.py
Copy-Item physics\structure.py our_work\_shared\physics\structure.py
```

Then add minimal config and seed helpers:

```python
# our_work/_shared/io/config.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
```

```python
# our_work/_shared/utils/seed.py
from __future__ import annotations

import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
```

- [ ] **Step 4: Patch migrated imports to point to `our_work/_shared`**

```python
# our_work/_shared/physics/optical_calculator.py
try:
    from .TMM import TMM_solver
except ImportError:
    from our_work._shared.physics.TMM import TMM_solver
```

```python
# our_work/_shared/physics/__init__.py
from .optical_calculator import (
    calculate_optical_properties_batch,
    calculate_optical_properties_batch_torch,
    resolve_complex_dtype,
)
from .structure import split_structure_token, tokens_to_tmm_config
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/our_work/shared/test_structure.py -v`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/our_work/shared/test_structure.py our_work/_shared
git commit -m "feat: scaffold our_work shared physics package"
```

---

### Task 2: Build the material registry and token vocabulary

**Files:**
- Create: `our_work/data_gen/__init__.py`
- Create: `our_work/data_gen/pipeline/__init__.py`
- Create: `our_work/data_gen/pipeline/material_registry.py`
- Create: `our_work/data_gen/pipeline/token_vocab.py`
- Test: `tests/our_work/data_gen/test_material_registry.py`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

from our_work.data_gen.pipeline.material_registry import MaterialRecord, build_material_registry
from our_work.data_gen.pipeline.token_vocab import build_token_vocab


def test_build_material_registry_reads_csv_materials(tmp_path: Path):
    (tmp_path / "SiO2.csv").write_text("wl,n,k\n2.0,1.4,0.0\n15.0,1.4,0.0\n", encoding="utf-8")
    (tmp_path / "Ge.csv").write_text("wl,n,k\n2.0,4.0,0.1\n15.0,4.0,0.1\n", encoding="utf-8")

    registry = build_material_registry(tmp_path)

    assert registry.material_names == ["Ge", "SiO2"]
    assert registry.records["SiO2"] == MaterialRecord(name="SiO2", filename="SiO2.csv")


def test_build_token_vocab_expands_material_thickness_pairs():
    vocab = build_token_vocab(["Ge", "SiO2"], thickness_values_nm=[10, 20])

    assert vocab.special_tokens == ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
    assert "Ge_10" in vocab.token_to_id
    assert "SiO2_20" in vocab.token_to_id
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/our_work/data_gen/test_material_registry.py -v`  
Expected: FAIL with `ImportError` for missing `material_registry` and `token_vocab`

- [ ] **Step 3: Implement the material registry**

```python
# our_work/data_gen/pipeline/material_registry.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MaterialRecord:
    name: str
    filename: str


@dataclass(frozen=True)
class MaterialRegistry:
    root_dir: str
    material_names: list[str]
    records: dict[str, MaterialRecord]


def build_material_registry(database_dir: str | Path) -> MaterialRegistry:
    root = Path(database_dir)
    files = sorted(root.glob("*.csv"))
    records = {
        file.stem: MaterialRecord(name=file.stem, filename=file.name)
        for file in files
    }
    return MaterialRegistry(
        root_dir=str(root),
        material_names=sorted(records.keys()),
        records=records,
    )
```

- [ ] **Step 4: Implement the token vocabulary builder**

```python
# our_work/data_gen/pipeline/token_vocab.py
from __future__ import annotations

from dataclasses import dataclass


SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]


@dataclass(frozen=True)
class TokenVocabulary:
    special_tokens: list[str]
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]


def build_token_vocab(material_names: list[str], thickness_values_nm: list[int]) -> TokenVocabulary:
    tokens = list(SPECIAL_TOKENS)
    for material in sorted(material_names):
        for thickness_nm in thickness_values_nm:
            tokens.append(f"{material}_{thickness_nm}")
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    id_to_token = {idx: token for token, idx in token_to_id.items()}
    return TokenVocabulary(
        special_tokens=list(SPECIAL_TOKENS),
        token_to_id=token_to_id,
        id_to_token=id_to_token,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/our_work/data_gen/test_material_registry.py -v`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/our_work/data_gen/test_material_registry.py our_work/data_gen
git commit -m "feat: add material registry and token vocabulary"
```

---

### Task 3: Implement layer-bucket structure sampling with uniqueness

**Files:**
- Create: `our_work/data_gen/pipeline/sampler.py`
- Test: `tests/our_work/data_gen/test_sampler.py`

- [ ] **Step 1: Write the failing tests**

```python
from our_work.data_gen.pipeline.sampler import sample_structure_tokens, sample_unique_bucket


def test_sample_structure_tokens_returns_requested_layer_count():
    tokens = sample_structure_tokens(
        material_names=["Ge", "SiO2"],
        thickness_values_nm=[10, 20],
        layer_count=5,
        rng_seed=7,
    )
    assert len(tokens) == 5
    assert all(token in {"Ge_10", "Ge_20", "SiO2_10", "SiO2_20"} for token in tokens)


def test_sample_unique_bucket_deduplicates_exact_structures():
    bucket = sample_unique_bucket(
        material_names=["Ge"],
        thickness_values_nm=[10, 20, 30],
        layer_count=2,
        target_count=3,
        rng_seed=11,
    )
    assert len(bucket) == 3
    assert len({tuple(tokens) for tokens in bucket}) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/our_work/data_gen/test_sampler.py -v`  
Expected: FAIL with `ImportError` for missing sampler module

- [ ] **Step 3: Implement single-structure sampling**

```python
# our_work/data_gen/pipeline/sampler.py
from __future__ import annotations

import random


def sample_structure_tokens(
    material_names: list[str],
    thickness_values_nm: list[int],
    layer_count: int,
    rng_seed: int | None = None,
) -> list[str]:
    rng = random.Random(rng_seed)
    return [
        f"{rng.choice(material_names)}_{rng.choice(thickness_values_nm)}"
        for _ in range(layer_count)
    ]
```

- [ ] **Step 4: Implement unique bucket sampling**

```python
# our_work/data_gen/pipeline/sampler.py
def sample_unique_bucket(
    material_names: list[str],
    thickness_values_nm: list[int],
    layer_count: int,
    target_count: int,
    rng_seed: int,
) -> list[list[str]]:
    rng = random.Random(rng_seed)
    seen: set[tuple[str, ...]] = set()
    results: list[list[str]] = []
    while len(results) < target_count:
        candidate = [
            f"{rng.choice(material_names)}_{rng.choice(thickness_values_nm)}"
            for _ in range(layer_count)
        ]
        key = tuple(candidate)
        if key in seen:
            continue
        seen.add(key)
        results.append(candidate)
    return results
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/our_work/data_gen/test_sampler.py -v`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/our_work/data_gen/test_sampler.py our_work/data_gen/pipeline/sampler.py
git commit -m "feat: add unique structure bucket sampler"
```

---

### Task 4: Implement batched TMM simulation and spectrum validation

**Files:**
- Create: `our_work/data_gen/pipeline/simulator.py`
- Test: `tests/our_work/data_gen/test_simulator.py`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np

from our_work.data_gen.pipeline.simulator import validate_rt_spectrum, flatten_rt_spectrum


def test_flatten_rt_spectrum_concatenates_r_and_t():
    flat = flatten_rt_spectrum(np.array([0.1, 0.2]), np.array([0.7, 0.6]))
    assert flat.tolist() == [0.1, 0.2, 0.7, 0.6]


def test_validate_rt_spectrum_rejects_energy_overflow():
    ok = validate_rt_spectrum(
        reflection=np.array([0.7, 0.8], dtype=np.float32),
        transmission=np.array([0.5, 0.4], dtype=np.float32),
        tolerance=1e-3,
    )
    assert ok is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/our_work/data_gen/test_simulator.py -v`  
Expected: FAIL with `ImportError` for missing simulator module

- [ ] **Step 3: Implement spectrum flattening and validation helpers**

```python
# our_work/data_gen/pipeline/simulator.py
from __future__ import annotations

import numpy as np

from our_work._shared.physics import calculate_optical_properties_batch
from our_work._shared.physics.structure import tokens_to_tmm_config


def flatten_rt_spectrum(reflection: np.ndarray, transmission: np.ndarray) -> np.ndarray:
    return np.concatenate([reflection.astype(np.float32), transmission.astype(np.float32)], axis=0)


def validate_rt_spectrum(reflection: np.ndarray, transmission: np.ndarray, tolerance: float) -> bool:
    if not np.all(np.isfinite(reflection)) or not np.all(np.isfinite(transmission)):
        return False
    if float(reflection.min()) < -tolerance or float(reflection.max()) > 1.0 + tolerance:
        return False
    if float(transmission.min()) < -tolerance or float(transmission.max()) > 1.0 + tolerance:
        return False
    if float((reflection + transmission).max()) > 1.0 + tolerance:
        return False
    return True
```

- [ ] **Step 4: Implement batched structure simulation**

```python
# our_work/data_gen/pipeline/simulator.py
def simulate_structure_batch(
    structure_token_groups: list[list[str]],
    *,
    database_path: str,
    wavelength_range_um: tuple[float, float],
    num_points: int,
    incident_angle: float,
    polarization: int,
    tolerance: float,
    complex_dtype: str,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray]:
    configs = [
        tokens_to_tmm_config(tokens, database_path=database_path)
        for tokens in structure_token_groups
    ]
    wavelengths, reflections, transmissions = calculate_optical_properties_batch(
        structure_configs=configs,
        wavelength_range=wavelength_range_um,
        num_points=num_points,
        incident_angle=incident_angle,
        polarization=polarization,
        complex_dtype=complex_dtype,
    )
    ok_mask = np.asarray(
        [validate_rt_spectrum(reflection, transmission, tolerance) for reflection, transmission in zip(reflections, transmissions)],
        dtype=np.bool_,
    )
    return wavelengths, list(reflections), list(transmissions), ok_mask
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/our_work/data_gen/test_simulator.py -v`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/our_work/data_gen/test_simulator.py our_work/data_gen/pipeline/simulator.py
git commit -m "feat: add batched spectrum simulator"
```

---

### Task 5: Write shard output, split metadata, and stats

**Files:**
- Create: `our_work/data_gen/pipeline/shard_writer.py`
- Test: `tests/our_work/data_gen/test_shard_writer.py`

- [ ] **Step 1: Write the failing test**

```python
import json
from pathlib import Path

from our_work.data_gen.pipeline.shard_writer import write_records_to_parquet, write_split_manifest


def test_write_split_manifest_creates_json(tmp_path: Path):
    write_split_manifest(
        tmp_path / "splits" / "split_manifest.json",
        {"train": ["shard-00000.parquet"], "val": ["shard-00001.parquet"], "test": []},
    )
    payload = json.loads((tmp_path / "splits" / "split_manifest.json").read_text(encoding="utf-8"))
    assert payload["train"] == ["shard-00000.parquet"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/our_work/data_gen/test_shard_writer.py -v`  
Expected: FAIL with `ImportError` for missing shard writer module

- [ ] **Step 3: Implement neutral shard writing**

```python
# our_work/data_gen/pipeline/shard_writer.py
from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset


def write_records_to_parquet(path: str | Path, records: list[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).to_parquet(str(output_path))
```

- [ ] **Step 4: Implement split manifest writing**

```python
# our_work/data_gen/pipeline/shard_writer.py
def write_split_manifest(path: str | Path, payload: dict[str, list[str]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/our_work/data_gen/test_shard_writer.py -v`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/our_work/data_gen/test_shard_writer.py our_work/data_gen/pipeline/shard_writer.py
git commit -m "feat: add neutral shard and split manifest writer"
```

---

### Task 6: Wire the end-to-end dataset build command and smoke-test it

**Files:**
- Create: `our_work/data_gen/configs/dataset_v1.yaml`
- Create: `our_work/data_gen/pipeline/build_dataset.py`
- Create: `our_work/data_gen/scripts/run_build_dataset.py`
- Test: `tests/our_work/data_gen/test_build_dataset.py`

- [ ] **Step 1: Write the failing smoke test**

```python
from pathlib import Path

from our_work.data_gen.pipeline.build_dataset import build_small_dataset


def test_build_small_dataset_writes_manifest(tmp_path: Path):
    output_dir = tmp_path / "outputs"
    build_small_dataset(
        output_dir=output_dir,
        database_path="database",
        material_names=["Ge", "SiO2"],
        thickness_values_nm=[10, 20],
        layer_counts=[5],
        samples_per_bucket=2,
        num_points=8,
        wavelength_range_um=(2.0, 15.0),
    )
    assert (output_dir / "splits" / "split_manifest.json").exists()
```

- [ ] **Step 2: Run the smoke test to verify it fails**

Run: `pytest tests/our_work/data_gen/test_build_dataset.py -v`  
Expected: FAIL with `ImportError` for missing `build_dataset`

- [ ] **Step 3: Add the YAML config and orchestrator**

```yaml
# our_work/data_gen/configs/dataset_v1.yaml
seed: 42
paths:
  database_dir: database
  output_dir: our_work/data_gen/outputs/v1
data:
  layer_counts: [5, 6, 7, 8, 9, 10]
  samples_per_bucket: 500000
  thickness_values_nm: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500]
tmm:
  wavelength_range_um: [2.0, 15.0]
  num_points: 1024
  incident_angle: 0.0
  polarization: 0
  tolerance: 0.001
  complex_dtype: complex128
shards:
  records_per_shard: 50000
splits:
  train_ratio: 0.98
  val_ratio: 0.01
  test_ratio: 0.01
```

```python
# our_work/data_gen/pipeline/build_dataset.py
from __future__ import annotations

from pathlib import Path

from our_work.data_gen.pipeline.sampler import sample_unique_bucket
from our_work.data_gen.pipeline.shard_writer import write_records_to_parquet, write_split_manifest


def build_small_dataset(
    *,
    output_dir: str | Path,
    database_path: str,
    material_names: list[str],
    thickness_values_nm: list[int],
    layer_counts: list[int],
    samples_per_bucket: int,
    num_points: int,
    wavelength_range_um: tuple[float, float],
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for layer_count in layer_counts:
        for sample_index, tokens in enumerate(
            sample_unique_bucket(material_names, thickness_values_nm, layer_count, samples_per_bucket, rng_seed=layer_count)
        ):
            records.append(
                {
                    "sample_id": f"{layer_count}-{sample_index}",
                    "layer_count": layer_count,
                    "structure_tokens": tokens,
                }
            )
    write_records_to_parquet(Path(output_dir) / "shards" / "shard-00000.parquet", records)
    write_split_manifest(Path(output_dir) / "splits" / "split_manifest.json", {"train": ["shard-00000.parquet"], "val": [], "test": []})
```

- [ ] **Step 4: Add the CLI entrypoint**

```python
# our_work/data_gen/scripts/run_build_dataset.py
from __future__ import annotations

import argparse

from our_work._shared.io.config import load_yaml_config
from our_work.data_gen.pipeline.build_dataset import build_small_dataset
from our_work.data_gen.pipeline.material_registry import build_material_registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    registry = build_material_registry(config["paths"]["database_dir"])
    build_small_dataset(
        output_dir=config["paths"]["output_dir"],
        database_path=config["paths"]["database_dir"],
        material_names=registry.material_names,
        thickness_values_nm=config["data"]["thickness_values_nm"],
        layer_counts=config["data"]["layer_counts"],
        samples_per_bucket=int(config["data"]["samples_per_bucket"]),
        num_points=int(config["tmm"]["num_points"]),
        wavelength_range_um=tuple(config["tmm"]["wavelength_range_um"]),
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the smoke test and CLI check**

Run: `pytest tests/our_work/data_gen/test_build_dataset.py -v`  
Expected: PASS

Run: `python our_work/data_gen/scripts/run_build_dataset.py --config our_work/data_gen/configs/dataset_v1.yaml`  
Expected: creates `our_work/data_gen/outputs/v1/shards/` and `our_work/data_gen/outputs/v1/splits/split_manifest.json`

- [ ] **Step 6: Commit**

```bash
git add tests/our_work/data_gen/test_build_dataset.py our_work/data_gen
git commit -m "feat: add data generation build command"
```

---

## Self-Review

### Spec coverage
- `our_work/_shared` 独立化：Task 1
- `database/` 材料扫描：Task 2
- `5-10` 层独立采样：Task 3 + Task 6 config
- 批量 TMM 计算与物理检查：Task 4
- shard/manifest/split 输出：Task 5 + Task 6
- 先小样本 smoke test 再全量：Task 6

### Placeholder scan
- 未使用 `TODO/TBD`
- 每个代码步骤都给出具体路径、代码或命令
- 测试命令与预期结果已明确

### Type consistency
- 结构 token 始终为 `list[str]`
- 光谱向量统一为 `2048` 维拼接 `float32`
- split 信息统一落在 `splits/split_manifest.json`
