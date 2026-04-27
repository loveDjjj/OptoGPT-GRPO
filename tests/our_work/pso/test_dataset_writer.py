import json
from pathlib import Path

import numpy as np
import pandas as pd

from our_work.data_gen.pipeline.token_vocab import build_token_vocab
from our_work.pso.dataset_writer import serialize_accepted_structure, write_pso_supplement_dataset
from our_work.pso.search import AcceptedStructure


def _accepted(tokens: list[str], target_id: str = "demo") -> AcceptedStructure:
    return AcceptedStructure(
        structure_tokens=tokens,
        reflection=np.array([0.1, 0.2], dtype=np.float32),
        transmission=np.array([0.7, 0.6], dtype=np.float32),
        target_mse=0.01,
        target_id=target_id,
        target_family="fixed",
        target_center_um=None,
        target_fwhm_um=None,
        pso_seed=123,
        pso_restart_index=0,
    )


def test_serialize_accepted_structure_matches_data_gen_schema():
    vocab = build_token_vocab(["Ge", "SiO2"], [10, 20])

    record = serialize_accepted_structure(
        _accepted(["Ge_10", "SiO2_20"]),
        sample_id="pso-0",
        token_to_id=vocab.token_to_id,
        acceptance_mse_threshold=0.05,
    )

    assert record["sample_id"] == "pso-0"
    assert record["layer_count"] == 2
    assert record["structure_tokens"] == ["Ge_10", "SiO2_20"]
    assert record["materials"] == ["Ge", "SiO2"]
    assert record["thickness_nm"] == [10, 20]
    assert record["spectrum_rt"] == [0.1, 0.2, 0.7, 0.6]
    assert record["target_mse"] == 0.01


def test_write_pso_supplement_dataset_writes_shards_manifest_vocab_and_summary(tmp_path: Path):
    vocab = build_token_vocab(["Ge", "SiO2"], [10, 20])

    manifest = write_pso_supplement_dataset(
        output_dir=tmp_path,
        accepted=[_accepted(["Ge_10"]), _accepted(["SiO2_20"], target_id="other")],
        token_to_id=vocab.token_to_id,
        vocab_tokens=vocab.special_tokens + [token for token in vocab.token_to_id if token not in vocab.special_tokens],
        records_per_shard=1,
        acceptance_mse_threshold=0.05,
        train_ratio=0.5,
        val_ratio=0.5,
        seed=42,
    )

    assert len(manifest["train"]) == 1
    assert len(manifest["val"]) == 1
    assert (tmp_path / "vocab" / "vocab.json").exists()
    assert (tmp_path / "stats" / "summary.json").exists()
    frame = pd.read_parquet(tmp_path / "shards" / manifest["train"][0])
    assert set(["sample_id", "structure_tokens", "spectrum_rt", "target_id"]).issubset(frame.columns)
    summary = json.loads((tmp_path / "stats" / "summary.json").read_text(encoding="utf-8"))
    assert summary["accepted_count"] == 2
