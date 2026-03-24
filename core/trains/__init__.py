"""Compatibility package for legacy checkpoints serialized under `core.trains`."""

from .train import (
    LabelSmoothing,
    NoamOpt,
    SimpleLossCompute,
    count_params,
    get_std_opt,
    run_epoch,
    run_epoch_I,
    save_checkpoint,
    train,
    train_I,
)

__all__ = [
    "LabelSmoothing",
    "NoamOpt",
    "SimpleLossCompute",
    "count_params",
    "get_std_opt",
    "run_epoch",
    "run_epoch_I",
    "save_checkpoint",
    "train",
    "train_I",
]
