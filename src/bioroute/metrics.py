from __future__ import annotations

import numpy as np


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """ECE over max-class confidence."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if not np.any(mask):
            continue
        acc = (pred[mask] == y_true[mask]).mean()
        ece += mask.mean() * abs(acc - conf[mask].mean())
    return float(ece)


def brier_multiclass(probs: np.ndarray, y_true: np.ndarray, n_classes: int) -> float:
    onehot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def abstain_coverage(
    probs: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.55,
) -> dict:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    keep = conf >= threshold
    coverage = float(keep.mean())
    if not np.any(keep):
        return {"coverage": coverage, "precision_kept": 0.0, "abstain_rate": 1.0}
    precision = float((pred[keep] == y_true[keep]).mean())
    return {
        "coverage": coverage,
        "precision_kept": precision,
        "abstain_rate": float(1.0 - coverage),
    }
