from __future__ import annotations

import numpy as np


def compute_token_accuracy(eval_prediction) -> dict[str, float]:
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    valid_mask = labels != -100
    if not np.any(valid_mask):
        return {"token_accuracy": 0.0}
    accuracy = (predictions[valid_mask] == labels[valid_mask]).mean()
    return {"token_accuracy": float(accuracy)}
