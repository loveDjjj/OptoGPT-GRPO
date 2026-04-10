from pathlib import Path

import pandas as pd

from our_work.pretrain.dataset.hf_dataset import load_parquet_records
from our_work.pretrain.scripts.run_pretrain import (
    build_trainer,
    build_trainer_components,
)


def test_build_trainer_components_and_load_parquet_records(tmp_path: Path):
    shard_path = tmp_path / "smoke.parquet"
    pd.DataFrame(
        [
            {
                "sample_id": "sample-000",
                "layer_count": 5,
                "structure_tokens": ["Ge_10"],
                "token_ids": [1, 4, 2],
                "materials": ["Ge"],
                "thickness_nm": [10],
                "spectrum_rt": [0.1] * 2048,
            }
        ]
    ).to_parquet(shard_path, index=False)

    records = load_parquet_records([str(shard_path)])
    assert len(records) == 1

    components = build_trainer_components(
        model_config={
            "vocab_size": 5,
            "spectrum_dim": 2048,
            "prefix_length": 2,
            "n_positions": 16,
            "n_embd": 16,
            "n_layer": 1,
            "n_head": 2,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
        token_list=["[PAD]", "[BOS]", "[EOS]", "[UNK]", "Ge_10"],
    )
    trainer = build_trainer(
        model=components["model"],
        train_dataset=records,
        eval_dataset=records,
        collator=components["collator"],
        output_dir=str(tmp_path / "trainer-out"),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        max_steps=1,
    )
    assert trainer is not None
    batch = next(iter(trainer.get_train_dataloader()))
    assert batch["spectra"].shape == (1, 2048)
    train_result = trainer.train()
    assert trainer.state.global_step == 1
    assert train_result.training_loss >= 0.0
