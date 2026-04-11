# run_eval Batch Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `our_work/pretrain/scripts/run_eval.py` so one batch evaluation run writes normalized JSON artifacts and PNG visualizations for later offline analysis.

**Architecture:** Keep `run_eval.py` as the only public entrypoint, but split its new responsibilities into focused helper modules. One helper owns result serialization and summary aggregation, one helper owns plot generation and sample selection, and the script wires model inference, physics replay, artifact writing, and CLI flags together.

**Tech Stack:** Python, PyTorch, NumPy, matplotlib, pathlib, json, pytest

---

## File Structure

### Artifact and summary helpers
- Create: `our_work/pretrain/eval_outputs.py`
  - Writes `results.jsonl`
  - Builds `summary.json`
  - Creates timestamped run directories
  - Tracks artifact paths and skipped-artifact reasons

### Plot helpers
- Create: `our_work/pretrain/eval_plots.py`
  - Selects `worst + random` valid samples
  - Writes histogram/bar plots
  - Writes per-sample R/T comparison figures

### Script integration
- Modify: `our_work/pretrain/scripts/run_eval.py`
  - Adds output-dir and plotting CLI
  - Calls helper modules
  - Persists evaluation artifacts

### Tests
- Create: `tests/our_work/pretrain/test_eval_outputs.py`
- Create: `tests/our_work/pretrain/test_eval_plots.py`
- Modify: `tests/our_work/pretrain/test_eval.py`

### Docs
- Modify: `docs/notes.md`
- Modify: `docs/logs/2026-04.md`

---

### Task 1: Add result serialization and summary aggregation

**Files:**
- Create: `our_work/pretrain/eval_outputs.py`
- Create: `tests/our_work/pretrain/test_eval_outputs.py`

- [ ] **Step 1: Write the failing tests**

```python
import json
from pathlib import Path

from our_work.pretrain.eval_outputs import (
    build_summary_payload,
    create_eval_run_dir,
    write_results_jsonl,
)


def test_write_results_jsonl_writes_one_json_object_per_line(tmp_path: Path):
    rows = [
        {"sample_id": "a", "target_layer_count": 5, "generated_valid": True},
        {"sample_id": "b", "target_layer_count": 6, "generated_valid": False},
    ]
    output_path = tmp_path / "results.jsonl"
    write_results_jsonl(rows, output_path)
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["sample_id"] == "a"


def test_build_summary_payload_includes_global_and_per_layer_metrics():
    rows = [
        {
            "sample_id": "a",
            "target_layer_count": 5,
            "generated_valid": True,
            "token_exact_match": True,
            "spectrum_rmse": 0.1,
            "spectrum_mae": 0.05,
        },
        {
            "sample_id": "b",
            "target_layer_count": 5,
            "generated_valid": False,
            "token_exact_match": False,
            "spectrum_rmse": None,
            "spectrum_mae": None,
        },
        {
            "sample_id": "c",
            "target_layer_count": 6,
            "generated_valid": True,
            "token_exact_match": False,
            "spectrum_rmse": 0.3,
            "spectrum_mae": 0.2,
        },
    ]
    payload = build_summary_payload(
        rows=rows,
        metadata={"split": "val"},
        artifacts={"summary": "summary.json"},
        skipped_artifacts={"rmse_hist": "not enough valid rows"},
    )
    assert payload["global_metrics"]["sample_count"] == 3
    assert payload["global_metrics"]["valid_generation_count"] == 2
    assert payload["global_metrics"]["exact_match_rate"] == 1 / 3
    assert payload["per_target_layer_count"]["5"]["sample_count"] == 2
    assert payload["per_target_layer_count"]["6"]["mean_spectrum_rmse"] == 0.3


def test_create_eval_run_dir_creates_timestamped_subdirectories(tmp_path: Path):
    run_dir = create_eval_run_dir(tmp_path, run_name="base_run", timestamp="20260411-120000")
    assert run_dir.name == "20260411-120000"
    assert (run_dir / "plots").exists()
    assert (run_dir / "samples").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/our_work/pretrain/test_eval_outputs.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'our_work.pretrain.eval_outputs'`

- [ ] **Step 3: Write the minimal implementation**

```python
# our_work/pretrain/eval_outputs.py
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np


def create_eval_run_dir(output_root: str | Path, *, run_name: str, timestamp: str | None = None) -> Path:
    stamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output_root) / run_name / "eval_runs" / stamp
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    return run_dir


def write_results_jsonl(rows: list[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _metric_mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return float(np.mean(clean)) if clean else None


def build_summary_payload(
    *,
    rows: list[dict],
    metadata: dict,
    artifacts: dict[str, str],
    skipped_artifacts: dict[str, str],
) -> dict:
    sample_count = len(rows)
    valid_rows = [row for row in rows if row["generated_valid"]]
    exact_match_count = sum(1 for row in rows if row["token_exact_match"])
    per_layer: dict[str, dict] = {}
    for layer_count in sorted({int(row["target_layer_count"]) for row in rows}):
        layer_rows = [row for row in rows if int(row["target_layer_count"]) == layer_count]
        layer_valid = [row for row in layer_rows if row["generated_valid"]]
        layer_exact = sum(1 for row in layer_rows if row["token_exact_match"])
        per_layer[str(layer_count)] = {
            "sample_count": len(layer_rows),
            "valid_generation_count": len(layer_valid),
            "valid_generation_rate": float(len(layer_valid) / len(layer_rows)) if layer_rows else 0.0,
            "exact_match_count": layer_exact,
            "exact_match_rate": float(layer_exact / len(layer_rows)) if layer_rows else 0.0,
            "mean_spectrum_rmse": _metric_mean([row["spectrum_rmse"] for row in layer_valid]),
            "mean_spectrum_mae": _metric_mean([row["spectrum_mae"] for row in layer_valid]),
        }
    return {
        "metadata": metadata,
        "global_metrics": {
            "sample_count": sample_count,
            "valid_generation_count": len(valid_rows),
            "valid_generation_rate": float(len(valid_rows) / sample_count) if sample_count else 0.0,
            "exact_match_count": exact_match_count,
            "exact_match_rate": float(exact_match_count / sample_count) if sample_count else 0.0,
            "mean_spectrum_rmse": _metric_mean([row["spectrum_rmse"] for row in valid_rows]),
            "mean_spectrum_mae": _metric_mean([row["spectrum_mae"] for row in valid_rows]),
        },
        "per_target_layer_count": per_layer,
        "artifacts": artifacts,
        "skipped_artifacts": skipped_artifacts,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/our_work/pretrain/test_eval_outputs.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/our_work/pretrain/test_eval_outputs.py our_work/pretrain/eval_outputs.py
git commit -m "feat: add eval artifact serialization helpers"
```

---

### Task 2: Add plot generation and sample selection helpers

**Files:**
- Create: `our_work/pretrain/eval_plots.py`
- Create: `tests/our_work/pretrain/test_eval_plots.py`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

import numpy as np

from our_work.pretrain.eval_plots import (
    plot_metric_histogram,
    plot_sample_spectrum,
    select_sample_plot_rows,
)


def test_select_sample_plot_rows_returns_worst_then_random_without_overlap():
    rows = [
        {"sample_id": "a", "generated_valid": True, "spectrum_rmse": 0.1},
        {"sample_id": "b", "generated_valid": True, "spectrum_rmse": 0.4},
        {"sample_id": "c", "generated_valid": True, "spectrum_rmse": 0.3},
        {"sample_id": "d", "generated_valid": False, "spectrum_rmse": None},
    ]
    selected = select_sample_plot_rows(rows, worst_count=1, random_count=1, seed=7)
    assert selected["worst"][0]["sample_id"] == "b"
    assert len({row["sample_id"] for bucket in selected.values() for row in bucket}) == 2


def test_plot_metric_histogram_writes_png(tmp_path: Path):
    output_path = tmp_path / "rmse_hist.png"
    plot_metric_histogram(
        values=[0.1, 0.2, 0.4],
        title="RMSE",
        xlabel="rmse",
        output_path=output_path,
    )
    assert output_path.exists()


def test_plot_sample_spectrum_writes_png(tmp_path: Path):
    output_path = tmp_path / "sample.png"
    row = {
        "sample_id": "sample-1",
        "target_layer_count": 5,
        "prediction_layer_count": 6,
        "token_exact_match": False,
        "spectrum_rmse": 0.2,
        "target_spectrum_rt": list(np.linspace(0.1, 0.9, 8)),
        "predicted_spectrum_rt": list(np.linspace(0.2, 0.8, 8)),
    }
    plot_sample_spectrum(row=row, output_path=output_path, num_points=4)
    assert output_path.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/our_work/pretrain/test_eval_plots.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'our_work.pretrain.eval_plots'`

- [ ] **Step 3: Write the minimal implementation**

```python
# our_work/pretrain/eval_plots.py
from __future__ import annotations

from pathlib import Path
import random

import matplotlib.pyplot as plt


def select_sample_plot_rows(rows: list[dict], *, worst_count: int, random_count: int, seed: int = 42) -> dict[str, list[dict]]:
    valid_rows = [row for row in rows if row["generated_valid"] and row["spectrum_rmse"] is not None]
    worst_rows = sorted(valid_rows, key=lambda row: row["spectrum_rmse"], reverse=True)[:worst_count]
    remaining = [row for row in valid_rows if row["sample_id"] not in {item["sample_id"] for item in worst_rows}]
    rng = random.Random(seed)
    random_rows = rng.sample(remaining, k=min(random_count, len(remaining)))
    return {"worst": worst_rows, "random": random_rows}


def plot_metric_histogram(*, values: list[float], title: str, xlabel: str, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=min(20, max(1, len(values))))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_sample_spectrum(*, row: dict, output_path: str | Path, num_points: int) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target = row["target_spectrum_rt"]
    predicted = row["predicted_spectrum_rt"]
    target_r = target[:num_points]
    target_t = target[num_points:]
    pred_r = predicted[:num_points]
    pred_t = predicted[num_points:]
    x = list(range(num_points))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, target_r, label="target_R")
    ax.plot(x, pred_r, label="pred_R")
    ax.plot(x, target_t, label="target_T")
    ax.plot(x, pred_t, label="pred_T")
    ax.set_title(
        f"{row['sample_id']} | target={row['target_layer_count']} | pred={row['prediction_layer_count']} | "
        f"exact={row['token_exact_match']} | rmse={row['spectrum_rmse']}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/our_work/pretrain/test_eval_plots.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/our_work/pretrain/test_eval_plots.py our_work/pretrain/eval_plots.py
git commit -m "feat: add eval plotting helpers"
```

---

### Task 3: Integrate run_eval.py batch artifacts and end-to-end smoke output

**Files:**
- Modify: `our_work/pretrain/scripts/run_eval.py`
- Modify: `tests/our_work/pretrain/test_eval.py`

- [ ] **Step 1: Write the failing tests**

```python
import json
from pathlib import Path

import pandas as pd

from our_work.pretrain.scripts.run_eval import main


def test_run_eval_main_writes_summary_results_and_plots(tmp_path: Path, monkeypatch):
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "shards").mkdir(parents=True)
    (dataset_dir / "splits").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "sample_id": "sample-000",
                "layer_count": 1,
                "structure_tokens": ["Ge_10"],
                "spectrum_rt": [0.1] * 32,
            }
        ]
    ).to_parquet(dataset_dir / "shards" / "shard-00000.parquet", index=False)
    (dataset_dir / "splits" / "split_manifest.json").write_text(
        json.dumps({"val": ["shard-00000.parquet"]}),
        encoding="utf-8",
    )
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "eval-output"

    monkeypatch.setattr(
        "our_work.pretrain.scripts.run_eval.load_eval_components",
        lambda *args, **kwargs: (object(), object(), "cpu"),
    )
    monkeypatch.setattr(
        "our_work.pretrain.scripts.run_eval.evaluate_records",
        lambda **kwargs: [
            {
                "sample_id": "sample-000",
                "target_layer_count": 1,
                "prediction_layer_count": 1,
                "target_tokens": ["Ge_10"],
                "predicted_tokens": ["Ge_10"],
                "token_exact_match": True,
                "generated_valid": True,
                "spectrum_rmse": 0.0,
                "spectrum_mae": 0.0,
                "target_spectrum_rt": [0.1] * 32,
                "predicted_spectrum_rt": [0.1] * 32,
            }
        ],
    )

    main(
        [
            "--checkpoint-dir", str(checkpoint_dir),
            "--dataset-dir", str(dataset_dir),
            "--database-dir", str(tmp_path),
            "--split", "val",
            "--max-samples", "1",
            "--num-points", "16",
            "--output-dir", str(output_dir),
            "--worst-sample-plots", "1",
            "--random-sample-plots", "0",
        ]
    )

    run_dirs = list((output_dir / "eval_runs").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "results.jsonl").exists()
    assert (run_dir / "plots" / "rmse_hist.png").exists()
    assert list((run_dir / "samples").glob("worst-*.png"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/our_work/pretrain/test_eval.py -v`  
Expected: FAIL because `main()` does not accept argv input and does not write batch artifacts

- [ ] **Step 3: Write the minimal implementation**

```python
# our_work/pretrain/scripts/run_eval.py
from our_work.pretrain.eval_outputs import (
    build_summary_payload,
    create_eval_run_dir,
    write_results_jsonl,
)
from our_work.pretrain.eval_plots import (
    plot_metric_histogram,
    plot_sample_spectrum,
    select_sample_plot_rows,
)


def _write_eval_artifacts(*, run_dir: Path, rows: list[dict], metadata: dict, num_points: int, worst_sample_plots: int, random_sample_plots: int, disable_plots: bool) -> dict:
    artifacts: dict[str, str] = {}
    skipped_artifacts: dict[str, str] = {}
    write_results_jsonl(rows, run_dir / "results.jsonl")
    artifacts["results_jsonl"] = "results.jsonl"
    valid_rmse = [row["spectrum_rmse"] for row in rows if row["spectrum_rmse"] is not None]
    valid_mae = [row["spectrum_mae"] for row in rows if row["spectrum_mae"] is not None]
    if disable_plots:
        skipped_artifacts["plots"] = "disabled by cli"
    else:
        if valid_rmse:
            plot_metric_histogram(values=valid_rmse, title="Spectrum RMSE", xlabel="rmse", output_path=run_dir / "plots" / "rmse_hist.png")
            artifacts["rmse_hist"] = "plots/rmse_hist.png"
        else:
            skipped_artifacts["rmse_hist"] = "no valid rmse values"
        if valid_mae:
            plot_metric_histogram(values=valid_mae, title="Spectrum MAE", xlabel="mae", output_path=run_dir / "plots" / "mae_hist.png")
            artifacts["mae_hist"] = "plots/mae_hist.png"
        else:
            skipped_artifacts["mae_hist"] = "no valid mae values"
        selections = select_sample_plot_rows(rows, worst_count=worst_sample_plots, random_count=random_sample_plots)
        for bucket_name, bucket_rows in selections.items():
            for index, row in enumerate(bucket_rows, start=1):
                rel_path = f"samples/{bucket_name}-{index}-{row['sample_id']}.png"
                plot_sample_spectrum(row=row, output_path=run_dir / rel_path, num_points=num_points)
                row["sample_figure_path"] = rel_path
                row["selection_bucket"] = bucket_name
    summary = build_summary_payload(rows=rows, metadata=metadata, artifacts=artifacts, skipped_artifacts=skipped_artifacts)
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(...)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--worst-sample-plots", type=int, default=5)
    parser.add_argument("--random-sample-plots", type=int, default=5)
    parser.add_argument("--disable-plots", action="store_true")
    args = parser.parse_args(argv)
    run_root = Path(args.output_dir) if args.output_dir else resolve_repo_path("our_work/pretrain/outputs")
    checkpoint_name = resolve_checkpoint_dir(args.checkpoint_dir).parent.name
    run_dir = create_eval_run_dir(run_root, run_name=checkpoint_name)
    ...
    summary = _write_eval_artifacts(
        run_dir=run_dir,
        rows=results,
        metadata={...},
        num_points=args.num_points,
        worst_sample_plots=args.worst_sample_plots,
        random_sample_plots=args.random_sample_plots,
        disable_plots=args.disable_plots,
    )
    payload = {"summary": summary["global_metrics"], "results": results, "run_dir": str(run_dir)}
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    return payload
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/our_work/pretrain/test_eval.py -v`  
Expected: PASS

Run: `pytest tests/our_work/pretrain -v`  
Expected: PASS

Run: `D:\anaconda\envs\oneday\python.exe our_work\pretrain\scripts\run_eval.py --checkpoint-dir our_work\pretrain\outputs\base_run --dataset-dir our_work\data_gen\outputs\v1_smoke --database-dir database --split val --max-samples 2 --num-points 1024 --worst-sample-plots 1 --random-sample-plots 1`
Expected: exit 0 and a fresh `eval_runs/<timestamp>/` directory containing `summary.json`, `results.jsonl`, `plots/`, and `samples/`

- [ ] **Step 5: Commit**

```bash
git add tests/our_work/pretrain/test_eval.py our_work/pretrain/scripts/run_eval.py our_work/pretrain/eval_outputs.py our_work/pretrain/eval_plots.py
git commit -m "feat: add batch eval artifacts and plots"
```

---

### Task 4: Update docs for the new evaluation workflow

**Files:**
- Modify: `docs/notes.md`
- Modify: `docs/logs/2026-04.md`

- [ ] **Step 1: Write the doc updates**

```markdown
# 本次修改摘要

## 需求
- 为 `our_work/pretrain/scripts/run_eval.py` 增加批量结果落盘与可视化能力。

## 实际修改
- `our_work/pretrain/eval_outputs.py`
  - 新增结果 JSONL、summary 聚合与评测运行目录管理。
- `our_work/pretrain/eval_plots.py`
  - 新增 RMSE/MAE 分布图、层数分布图和样本光谱对比图。
- `our_work/pretrain/scripts/run_eval.py`
  - 新增 `--output-dir`、`--worst-sample-plots`、`--random-sample-plots`、`--disable-plots`
  - 评测运行现在会输出 `summary.json`、`results.jsonl`、`plots/`、`samples/`
```

- [ ] **Step 2: Commit**

```bash
git add docs/notes.md docs/logs/2026-04.md
git commit -m "docs: record batch eval artifact workflow"
```

---

## Self-Review

### Spec coverage
- 输出目录结构：Task 1 + Task 3
- `summary.json` / `results.jsonl` schema：Task 1
- 分布图与样本图：Task 2 + Task 3
- 最差 + 随机样本策略：Task 2
- 分层统计：Task 1
- worktree 相对路径与脚本集成：Task 3
- 文档同步：Task 4

### Placeholder scan
- 未使用 `TODO`、`TBD`、`待确认`
- 每个代码任务都包含测试、失败验证、最小实现、通过验证与提交步骤
- 所有命令均给出具体路径与预期结果

### Type consistency
- 结果行统一使用 `target_layer_count`、`prediction_layer_count`、`generated_valid`、`spectrum_rmse`、`spectrum_mae`
- `summary.json` 聚合统一围绕 `rows: list[dict]`
- `main(argv: list[str] | None = None)` 与现有脚本入口兼容
