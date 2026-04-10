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
