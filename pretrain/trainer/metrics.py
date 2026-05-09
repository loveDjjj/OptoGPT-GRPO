from __future__ import annotations

import numpy as np


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    if hasattr(logits, "argmax"):
        return logits.argmax(dim=-1)
    return np.asarray(logits).argmax(axis=-1)


def compute_token_accuracy(eval_prediction) -> dict[str, float]:
    logits, labels = eval_prediction
    if isinstance(logits, (tuple, list)):
        # HF Trainer may pass a tuple like (logits, past_key_values, ...).
        # Token accuracy only depends on the logits tensor.
        logits = logits[0]
    predictions = logits if np.ndim(logits) == np.ndim(labels) else np.argmax(logits, axis=-1)
    valid_mask = labels != -100
    if not np.any(valid_mask):
        return {"token_accuracy": 0.0}
    accuracy = (predictions[valid_mask] == labels[valid_mask]).mean()
    return {"token_accuracy": float(accuracy)}
