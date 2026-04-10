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
