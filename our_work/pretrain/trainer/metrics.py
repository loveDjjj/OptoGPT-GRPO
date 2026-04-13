from __future__ import annotations

import numpy as np


def compute_token_accuracy(eval_prediction) -> dict[str, float]:
    logits, labels = eval_prediction
    if isinstance(logits, (tuple, list)):
        # HF Trainer may pass a tuple like (logits, past_key_values, ...).
        # Token accuracy only depends on the logits tensor.
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    valid_mask = labels != -100
    if not np.any(valid_mask):
        return {"token_accuracy": 0.0}
    accuracy = (predictions[valid_mask] == labels[valid_mask]).mean()
    return {"token_accuracy": float(accuracy)}
