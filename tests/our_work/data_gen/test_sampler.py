from our_work.data_gen.pipeline.sampler import (
    resolve_sampling_device,
    sample_structure_token_batch,
    sample_structure_tokens,
    sample_unique_bucket,
)


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


def test_resolve_sampling_device_falls_back_to_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    device = resolve_sampling_device("cuda:0")

    assert str(device) == "cpu"
