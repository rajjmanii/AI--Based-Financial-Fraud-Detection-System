import numpy as np
import pytest

from src.models.thresholding import flag_scores, select_threshold, threshold_sweep


def test_flag_is_inclusive_at_threshold():
    flags = flag_scores([0.2, 0.5, 0.8], 0.5)
    assert flags.tolist() == [False, True, True]


def test_lower_threshold_flags_at_least_as_many():
    rng = np.random.default_rng(1)
    scores = rng.random(500)
    assert flag_scores(scores, 0.3).sum() >= flag_scores(scores, 0.7).sum()


def test_sweep_recall_never_increases_with_threshold():
    rng = np.random.default_rng(2)
    y = (rng.random(1000) < 0.1).astype(int)
    scores = np.clip(y * 0.4 + rng.random(1000) * 0.6, 0, 1)
    sweep = threshold_sweep(y, scores, np.linspace(0.05, 0.95, 19))
    assert (np.diff(sweep["recall"].to_numpy()) <= 1e-12).all()
    assert (np.diff(sweep["flagged"].to_numpy()) <= 0).all()


def test_selected_threshold_separates_a_clean_problem():
    y = np.array([0] * 90 + [1] * 10)
    scores = np.concatenate([np.linspace(0.0, 0.4, 90), np.linspace(0.6, 1.0, 10)])
    t = select_threshold(y, scores)
    assert 0.4 < t <= 0.6 + 1e-9
    assert flag_scores(scores, t)[y == 1].all()


def test_threshold_selection_needs_fraud_examples():
    with pytest.raises(ValueError):
        select_threshold([0, 0, 0], [0.1, 0.2, 0.3])


def test_threshold_selection_is_independent_of_other_data():
    y = np.array([0] * 90 + [1] * 10)
    scores = np.concatenate([np.linspace(0.0, 0.4, 90), np.linspace(0.6, 1.0, 10)])
    assert select_threshold(y, scores) == select_threshold(y.copy(), scores.copy())
