import pandas as pd
import pytest

from src.monitoring.alert_engine import (assign_risk_category, build_review_queue,
                                         category_bounds, category_counts)


def test_category_boundaries_for_threshold_40():
    assert category_bounds(40) == {"medium_from": 20.0, "high_from": 40.0, "critical_from": 70.0}
    risks = [0, 19.99, 20, 39.99, 40, 69.99, 70, 100]
    expected = ["Low", "Low", "Medium", "Medium", "High", "High", "Critical", "Critical"]
    assert assign_risk_category(risks, 40).tolist() == expected


def test_flagged_means_high_or_critical():
    risks = pd.Series(range(0, 101))
    flagged = risks >= 35
    cats = assign_risk_category(risks, 35)
    assert set(cats[flagged]) <= {"High", "Critical"}
    assert set(cats[~flagged]) <= {"Low", "Medium"}


def test_categories_shift_with_threshold():
    assert assign_risk_category([50], 30)[0] == "High"
    assert assign_risk_category([50], 60)[0] == "Medium"


def test_invalid_threshold_rejected():
    for bad in (0, -5, 101):
        with pytest.raises(ValueError):
            category_bounds(bad)


def test_review_queue_sorted_and_filtered():
    df = pd.DataFrame({"row_id": range(6), "risk_score": [5, 45, 90, 30, 60, 10]})
    queue = build_review_queue(df, 40)
    assert queue["row_id"].tolist() == [2, 4, 1]
    assert queue["risk_score"].is_monotonic_decreasing
    watch = build_review_queue(df, 40, include_watchlist=True)
    assert 3 in watch["row_id"].tolist() and len(watch) == 4


def test_category_counts_include_all_categories():
    counts = category_counts(["Low", "Low", "High"])
    assert counts == {"Low": 2, "Medium": 0, "High": 1, "Critical": 0}
