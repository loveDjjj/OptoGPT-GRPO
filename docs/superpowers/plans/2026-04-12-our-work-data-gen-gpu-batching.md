# our_work data_gen GPU Batching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add GPU chunked structure sampling, chunked TMM execution, YAML-configurable `sampling.batch_size` / `tmm.batch_size`, and bucket/chunk progress feedback to `our_work/data_gen` without changing the output dataset schema.

**Architecture:** Keep `pipeline` boundaries intact: `sampler.py` owns candidate generation, `simulator.py` owns one TMM batch, and `build_dataset.py` orchestrates bucket-level accumulation, deduplication, chunking, and record writing. YAML parsing stays in `run_build_dataset.py`, which expands thickness ranges and forwards the new sampling/TMM config into the pipeline. Tests stay focused on entrypoint parsing, sampler behavior, and build orchestration chunking rather than full large-scale generation.

**Tech Stack:** Python, PyTorch, NumPy, pandas, PyYAML, tqdm, pytest

---

## File Structure

- Modify: `our_work/data_gen/configs/dataset_v1.yaml`
  - Add `sampling.device`, `sampling.batch_size`, `sampling.max_duplicate_retry`, and `tmm.batch_size`.
- Modify: `our_work/data_gen/scripts/run_build_dataset.py`
  - Parse new YAML fields.
  - Expand thickness ranges into a list of ints.
  - Pass sampling/TMM settings into `build_small_dataset(...)`.
- Modify: `our_work/data_gen/pipeline/sampler.py`
  - Keep current CPU helpers.
  - Add a GPU-capable batched sampling helper that emits candidate structures in chunks.
- Modify: `our_work/data_gen/pipeline/build_dataset.py`
  - Replace “entire bucket at once” logic with iterative chunked sampling + chunked TMM.
  - Keep bucket-global deduplication.
  - Surface chunk/bucket progress in tqdm postfix.
- Modify: `our_work/data_gen/pipeline/simulator.py`
  - Keep one-batch responsibility but accept chunk-sized input cleanly.
- Modify: `tests/our_work/data_gen/test_sampler.py`
  - Cover batched sampling output shape and token validity.
- Modify: `tests/our_work/data_gen/test_build_dataset.py`
  - Cover range parsing, chunked orchestration, dedup/retry, and TMM chunking.
- Modify: `README.md`
  - Document the new `sampling` and `tmm.batch_size` fields.
- Modify: `docs/notes.md`
  - Overwrite with the final implementation summary after code changes.
- Modify: `docs/logs/2026-04.md`
  - Append the final implementation record after code changes.

---

### Task 1: Add Config Parsing and Failing Tests

**Files:**
- Modify: `our_work/data_gen/scripts/run_build_dataset.py`
- Modify: `our_work/data_gen/configs/dataset_v1.yaml`
- Modify: `tests/our_work/data_gen/test_build_dataset.py`

- [ ] **Step 1: Write the failing tests for sampling/TMM config parsing**

```python
def test_resolve_thickness_values_nm_expands_inclusive_range_config():
    values = resolve_thickness_values_nm(
        {
            "thickness_range_nm": {"min": 10, "max": 500, "step": 10},
        }
    )
    assert values[0] == 10
    assert values[-1] == 500
    assert len(values) == 50


def test_resolve_data_gen_runtime_config_reads_sampling_and_tmm_batch_sizes():
    config = {
        "data": {
            "thickness_range_nm": {"min": 10, "max": 30, "step": 10},
            "layer_counts": [5],
            "samples_per_bucket": 4,
        },
        "sampling": {
            "device": "cuda:0",
            "batch_size": 8,
            "max_duplicate_retry": 9,
        },
        "tmm": {
            "num_points": 8,
            "wavelength_range_um": [2.0, 15.0],
            "incident_angle": 0.0,
            "polarization": 0,
            "tolerance": 1e-3,
            "complex_dtype": "complex128",
            "batch_size": 2,
        },
    }

    runtime = resolve_data_gen_runtime_config(config)

    assert runtime["thickness_values_nm"] == [10, 20, 30]
    assert runtime["sampling_device"] == "cuda:0"
    assert runtime["sampling_batch_size"] == 8
    assert runtime["max_duplicate_retry"] == 9
    assert runtime["tmm_batch_size"] == 2
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `python -m pytest tests/our_work/data_gen/test_build_dataset.py -q --basetemp C:/Users/15450/.codex/memories/pytest-data-gen-config`

Expected: FAIL because `resolve_data_gen_runtime_config` does not exist and new assertions are unmet.

- [ ] **Step 3: Add minimal config parsing helpers**

```python
def resolve_data_gen_runtime_config(config: dict) -> dict:
    data_cfg = config["data"]
    sampling_cfg = config.get("sampling", {})
    tmm_cfg = config["tmm"]
    return {
        "thickness_values_nm": resolve_thickness_values_nm(data_cfg),
        "sampling_device": str(sampling_cfg.get("device", "auto")),
        "sampling_batch_size": int(sampling_cfg.get("batch_size", 65536)),
        "max_duplicate_retry": int(sampling_cfg.get("max_duplicate_retry", 1000)),
        "tmm_batch_size": int(tmm_cfg.get("batch_size", 2048)),
    }
```

- [ ] **Step 4: Replace the YAML defaults with server-ready batching fields**

```yaml
sampling:
  device: auto
  batch_size: 65536
  max_duplicate_retry: 1000

tmm:
  wavelength_range_um: [2.0, 15.0]
  num_points: 1024
  incident_angle: 0.0
  polarization: 0
  tolerance: 0.001
  complex_dtype: complex128
  batch_size: 2048
```

- [ ] **Step 5: Run the focused tests to verify they pass**

Run: `python -m pytest tests/our_work/data_gen/test_build_dataset.py -q --basetemp C:/Users/15450/.codex/memories/pytest-data-gen-config`

Expected: PASS for the new parsing/config tests.

- [ ] **Step 6: Commit**

```bash
git add our_work/data_gen/scripts/run_build_dataset.py our_work/data_gen/configs/dataset_v1.yaml tests/our_work/data_gen/test_build_dataset.py
git commit -m "feat: add data_gen batching config parsing"
```

---

### Task 2: Add Batched GPU Sampling

**Files:**
- Modify: `our_work/data_gen/pipeline/sampler.py`
- Modify: `tests/our_work/data_gen/test_sampler.py`

- [ ] **Step 1: Write failing tests for batched sampler behavior**

```python
def test_sample_structure_token_batch_returns_requested_shape():
    batch = sample_structure_token_batch(
        material_names=["Ge", "SiO2"],
        thickness_values_nm=[10, 20],
        layer_count=3,
        batch_size=4,
        device="cpu",
        rng_seed=7,
    )
    assert len(batch) == 4
    assert all(len(tokens) == 3 for tokens in batch)


def test_sample_structure_token_batch_uses_valid_material_thickness_pairs():
    batch = sample_structure_token_batch(
        material_names=["Ge"],
        thickness_values_nm=[10, 20, 30],
        layer_count=2,
        batch_size=5,
        device="cpu",
        rng_seed=11,
    )
    allowed = {"Ge_10", "Ge_20", "Ge_30"}
    assert all(token in allowed for tokens in batch for token in tokens)
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `python -m pytest tests/our_work/data_gen/test_sampler.py -q --basetemp C:/Users/15450/.codex/memories/pytest-data-gen-sampler`

Expected: FAIL because `sample_structure_token_batch` does not exist.

- [ ] **Step 3: Implement the minimal batched sampler**

```python
def sample_structure_token_batch(
    material_names: list[str],
    thickness_values_nm: list[int],
    layer_count: int,
    batch_size: int,
    device: str = "auto",
    rng_seed: int | None = None,
) -> list[list[str]]:
    torch_device = resolve_sampling_device(device)
    generator = torch.Generator(device=torch_device)
    if rng_seed is not None:
        generator.manual_seed(int(rng_seed))
    material_idx = torch.randint(0, len(material_names), (batch_size, layer_count), generator=generator, device=torch_device)
    thickness_idx = torch.randint(0, len(thickness_values_nm), (batch_size, layer_count), generator=generator, device=torch_device)
    material_idx = material_idx.cpu().tolist()
    thickness_idx = thickness_idx.cpu().tolist()
    return [
        [f"{material_names[m]}_{thickness_values_nm[t]}" for m, t in zip(material_row, thickness_row)]
        for material_row, thickness_row in zip(material_idx, thickness_idx)
    ]
```

- [ ] **Step 4: Add device resolution and validation**

```python
def resolve_sampling_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print("sampling.device requested CUDA but CUDA is unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(device)
```

- [ ] **Step 5: Run the focused tests to verify they pass**

Run: `python -m pytest tests/our_work/data_gen/test_sampler.py -q --basetemp C:/Users/15450/.codex/memories/pytest-data-gen-sampler`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add our_work/data_gen/pipeline/sampler.py tests/our_work/data_gen/test_sampler.py
git commit -m "feat: add batched gpu-capable data_gen sampler"
```

---

### Task 3: Chunk TMM Execution and Bucket-Global Deduplication

**Files:**
- Modify: `our_work/data_gen/pipeline/build_dataset.py`
- Modify: `our_work/data_gen/pipeline/simulator.py`
- Modify: `tests/our_work/data_gen/test_build_dataset.py`

- [ ] **Step 1: Write failing orchestration tests**

```python
def test_build_small_dataset_chunks_tmm_calls(monkeypatch, tmp_path: Path):
    calls = []

    def fake_sample_batch(**kwargs):
        return [
            ["Ge_10", "SiO2_20"],
            ["Ge_10", "SiO2_20"],
            ["SiO2_20", "Ge_10"],
            ["Ge_20", "SiO2_10"],
        ]

    def fake_simulate(groups, **kwargs):
        calls.append(len(groups))
        refl = [np.zeros((8,), dtype=np.float32) for _ in groups]
        tran = [np.ones((8,), dtype=np.float32) for _ in groups]
        mask = np.ones((len(groups),), dtype=np.bool_)
        return np.arange(8, dtype=np.float32), refl, tran, mask

    monkeypatch.setattr("our_work.data_gen.pipeline.build_dataset.sample_structure_token_batch", fake_sample_batch)
    monkeypatch.setattr("our_work.data_gen.pipeline.build_dataset.simulate_structure_batch", fake_simulate)

    build_small_dataset(
        output_dir=tmp_path / "outputs",
        database_path="database",
        material_names=["Ge", "SiO2"],
        thickness_values_nm=[10, 20],
        layer_counts=[2],
        samples_per_bucket=3,
        sampling_batch_size=4,
        tmm_batch_size=2,
        max_duplicate_retry=4,
        sampling_device="cpu",
        num_points=8,
        wavelength_range_um=(2.0, 15.0),
        show_progress=False,
    )

    assert calls == [2, 1]
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `python -m pytest tests/our_work/data_gen/test_build_dataset.py -q --basetemp C:/Users/15450/.codex/memories/pytest-data-gen-build`

Expected: FAIL because `build_small_dataset` does not accept batching parameters and does not chunk TMM calls.

- [ ] **Step 3: Extend `build_small_dataset(...)` signature with batching controls**

```python
def build_small_dataset(
    *,
    output_dir: str | Path,
    database_path: str,
    material_names: list[str],
    thickness_values_nm: list[int],
    layer_counts: list[int],
    samples_per_bucket: int,
    sampling_batch_size: int,
    tmm_batch_size: int,
    max_duplicate_retry: int,
    sampling_device: str,
    ...
) -> dict[str, list[str]]:
```

- [ ] **Step 4: Replace the one-shot bucket loop with chunked accumulation**

```python
seen: set[tuple[str, ...]] = set()
accepted_records: list[dict[str, Any]] = []
retry_count = 0
while len(accepted_records) < samples_per_bucket:
    sampled_groups = sample_structure_token_batch(...)
    unique_groups = []
    duplicate_count = 0
    for tokens in sampled_groups:
        key = tuple(tokens)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        unique_groups.append(tokens)
    for start in range(0, len(unique_groups), tmm_batch_size):
        chunk = unique_groups[start : start + tmm_batch_size]
        _, reflections, transmissions, ok_mask = simulate_structure_batch(chunk, ...)
        ...
        if len(accepted_records) >= samples_per_bucket:
            break
    retry_count += 1
    if retry_count > max_duplicate_retry and len(accepted_records) < samples_per_bucket:
        raise RuntimeError(...)
```

- [ ] **Step 5: Update the tqdm postfix to expose chunk progress**

```python
layer_iterable.set_postfix(
    {
        "layer_count": int(layer_count),
        "bucket_kept": int(len(accepted_records)),
        "bucket_target": int(samples_per_bucket),
        "sample_batch": int(sampling_batch_size),
        "tmm_batch": int(tmm_batch_size),
        "duplicates_skipped": int(duplicate_count),
    },
    refresh=False,
)
```

- [ ] **Step 6: Run the focused tests to verify they pass**

Run: `python -m pytest tests/our_work/data_gen/test_build_dataset.py -q --basetemp C:/Users/15450/.codex/memories/pytest-data-gen-build`

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add our_work/data_gen/pipeline/build_dataset.py our_work/data_gen/pipeline/simulator.py tests/our_work/data_gen/test_build_dataset.py
git commit -m "feat: chunk data_gen sampling and tmm execution"
```

---

### Task 4: Wire CLI/YAML End-to-End and Update Docs

**Files:**
- Modify: `our_work/data_gen/scripts/run_build_dataset.py`
- Modify: `README.md`
- Modify: `docs/notes.md`
- Modify: `docs/logs/2026-04.md`

- [ ] **Step 1: Add the new runtime config fields to the CLI handoff**

```python
runtime = resolve_data_gen_runtime_config(config)
build_small_dataset(
    output_dir=config["paths"]["output_dir"],
    database_path=config["paths"]["database_dir"],
    material_names=registry.material_names,
    thickness_values_nm=runtime["thickness_values_nm"],
    layer_counts=[int(value) for value in config["data"]["layer_counts"]],
    samples_per_bucket=int(config["data"]["samples_per_bucket"]),
    sampling_batch_size=runtime["sampling_batch_size"],
    tmm_batch_size=runtime["tmm_batch_size"],
    max_duplicate_retry=runtime["max_duplicate_retry"],
    sampling_device=runtime["sampling_device"],
    ...
)
```

- [ ] **Step 2: Update README deployment guidance**

```markdown
- `sampling.batch_size`
  - 单次 GPU 采样的候选结构数
- `tmm.batch_size`
  - 单次送入 TMM 的结构数
- `sampling.max_duplicate_retry`
  - bucket 内全局去重补采的最大重试轮数
```

- [ ] **Step 3: Run the end-to-end smoke command from a non-root working directory**

Run:

```bash
python ..\data_gen\scripts\run_build_dataset.py --config .tmp_server_check\dataset_smoke.yaml
```

Workdir: `our_work\pretrain`

Expected:
- terminal shows `data_gen buckets`
- output contains shards, split manifest, vocab

- [ ] **Step 4: Update docs summary and monthly log**

```markdown
- `docs/notes.md`
  - overwrite with the implementation summary
- `docs/logs/2026-04.md`
  - append implementation files, verification, and results
```

- [ ] **Step 5: Run the targeted verification set**

Run:

```bash
python -m compileall our_work/data_gen tests/our_work/data_gen
python -m pytest tests/our_work/data_gen/test_sampler.py -q --basetemp C:/Users/15450/.codex/memories/pytest-data-gen-sampler
python -m pytest tests/our_work/data_gen/test_build_dataset.py -q --basetemp C:/Users/15450/.codex/memories/pytest-data-gen-build
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add our_work/data_gen/scripts/run_build_dataset.py our_work/data_gen/configs/dataset_v1.yaml README.md docs/notes.md docs/logs/2026-04.md
git commit -m "feat: add gpu chunked data generation pipeline"
```

---

## Self-Review

### Spec coverage
- `sampling.device / sampling.batch_size / max_duplicate_retry`：Task 1 + Task 4
- `tmm.batch_size`：Task 1 + Task 3 + Task 4
- GPU 批量结构采样：Task 2
- bucket 内全局严格唯一补采：Task 3
- chunk 级进度反馈：Task 3
- 非仓库根目录 smoke：Task 4

### Placeholder scan
- No `TODO/TBD/待确认`
- Every task has concrete files, tests, commands, and expected outputs

### Type consistency
- `build_small_dataset(...)` batching args are introduced once in Task 3 and reused consistently in Task 4
- `resolve_data_gen_runtime_config(...)` returns the same key names consumed by the CLI handoff
