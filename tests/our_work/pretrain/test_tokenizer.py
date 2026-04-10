from pathlib import Path

from our_work.pretrain.dataset.tokenizer import SpectralStructureTokenizer


def test_tokenizer_round_trip():
    tokenizer = SpectralStructureTokenizer(
        tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]", "Ge_10", "SiO2_20"]
    )
    ids = tokenizer.encode(["Ge_10", "SiO2_20"])
    assert ids == [1, 4, 5, 2]
    assert tokenizer.decode(ids) == ["Ge_10", "SiO2_20"]


def test_tokenizer_maps_unknown_token_to_unk():
    tokenizer = SpectralStructureTokenizer(
        tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]", "Ge_10"]
    )
    assert tokenizer.encode(["Missing_30"]) == [1, 3, 2]


def test_tokenizer_save_and_load_round_trip(tmp_path: Path):
    tokenizer = SpectralStructureTokenizer(
        tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]", "Ge_10", "SiO2_20"]
    )
    tokenizer.save_pretrained(tmp_path)

    restored = SpectralStructureTokenizer.from_pretrained(tmp_path)
    assert restored.tokens == tokenizer.tokens
    assert restored.encode(["Ge_10"]) == [1, 4, 2]
