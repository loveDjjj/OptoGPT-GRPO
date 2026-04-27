# PSO Supplement Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a PSO-based supplement dataset generator that searches target absorption profiles and writes main-data-compatible parquet records.

**Architecture:** Add new modular code under `our_work/pso` while keeping `PSO_lisan.py` as a prototype. Reuse `our_work/data_gen` vocabulary, shard writing, material registry, and shared TMM simulator compatibility instead of inventing a separate output format.

**Tech Stack:** Python, PyTorch, NumPy, pandas/pyarrow parquet, PyYAML, existing `our_work._shared.physics` TMM utilities.

---

### Task 1: Target Profile Library

**Files:**
- Create: `our_work/pso/targets.py`
- Test: `tests/our_work/pso/test_targets.py`

- [ ] Write tests for fixed band profiles and Lorentzian center generation.
- [ ] Implement target dataclass plus `build_target_profiles()`.
- [ ] Verify `4 + 129 = 133` default targets.

### Task 2: Particle Conversion And Batch Evaluation

**Files:**
- Create: `our_work/pso/search.py`
- Test: `tests/our_work/pso/test_search.py`

- [ ] Write tests for thickness discretization, material index clipping, and token conversion.
- [ ] Implement PSO config dataclasses and particle-to-token conversion.
- [ ] Implement vectorized PSO search with chunked TMM evaluation.
- [ ] Ensure accepted structures are deduplicated by ordered token tuple.

### Task 3: Dataset Writer

**Files:**
- Create: `our_work/pso/dataset_writer.py`
- Test: `tests/our_work/pso/test_dataset_writer.py`

- [ ] Write tests for accepted record serialization and shard output.
- [ ] Implement records compatible with existing data_gen schema.
- [ ] Add target metadata columns without breaking the required fields.

### Task 4: CLI And Config

**Files:**
- Create: `our_work/pso/configs/pso_supplement.yaml`
- Create: `our_work/pso/scripts/run_pso_dataset.py`
- Test: `tests/our_work/pso/test_run_pso_dataset.py`

- [ ] Write tests for config parsing and path resolution.
- [ ] Implement root-resolved CLI entrypoint.
- [ ] Add detailed YAML comments for dataset, target, PSO, TMM, stopping, output, and distributed slicing.

### Task 5: Smoke Verification And Docs

**Files:**
- Modify: `docs/notes.md`
- Modify: `docs/logs/2026-04.md`

- [ ] Run targeted tests.
- [ ] Run compile check for new PSO modules and tests.
- [ ] Run a tiny smoke build with mocked or small TMM settings when feasible.
- [ ] Update docs with modified files, verification, branch, and commit message.
