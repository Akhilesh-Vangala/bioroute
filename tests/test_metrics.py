from __future__ import annotations

import numpy as np

from bioroute.metrics import abstain_coverage, brier_multiclass, expected_calibration_error


def test_ece_confident_and_correct_lower_than_overconfident():
    good = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9]])
    y_good = np.array([0, 1, 2])
    bad = np.array([[0.95, 0.025, 0.025], [0.95, 0.025, 0.025], [0.95, 0.025, 0.025]])
    y_bad = np.array([1, 1, 1])
    assert expected_calibration_error(good, y_good) < expected_calibration_error(bad, y_bad)


def test_ece_overconfident():
    probs = np.array([[0.95, 0.025, 0.025], [0.95, 0.025, 0.025]])
    y = np.array([1, 1])  # always wrong top class
    assert expected_calibration_error(probs, y) > 0.5


def test_brier_bounds():
    probs = np.eye(3)
    y = np.array([0, 1, 2])
    assert brier_multiclass(probs, y, 3) == 0.0


def test_abstain_coverage():
    probs = np.array([[0.9, 0.05, 0.05], [0.4, 0.3, 0.3]])
    y = np.array([0, 0])
    out = abstain_coverage(probs, y, threshold=0.55)
    assert out["coverage"] == 0.5
    assert out["precision_kept"] == 1.0
