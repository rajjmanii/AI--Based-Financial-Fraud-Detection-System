import pytest

from src.database.history_repository import HistoryRepository


def _record(**kw):
    base = dict(source="single", model_name="Random Forest", threshold=0.3, risk_score=72.5,
                risk_category="High", flagged=True, amount=12.5, time_seconds=100.0)
    return {**base, **kw}


def test_database_is_created_on_first_use(tmp_path):
    path = tmp_path / "sub" / "history.db"
    HistoryRepository(path)
    assert path.is_file()


def test_add_and_search(tmp_path):
    repo = HistoryRepository(tmp_path / "h.db")
    repo.add_prediction(_record())
    repo.add_many([_record(source="bulk", row_ref="file.csv:7", risk_category="Critical")])
    assert repo.count() == 2
    assert len(repo.search(source="bulk")) == 1
    assert len(repo.search(category="High")) == 1
    assert len(repo.search(text="file.csv")) == 1
    assert repo.search(text="nothing-matches").empty


def test_timestamp_and_default_status(tmp_path):
    repo = HistoryRepository(tmp_path / "h.db")
    repo.add_prediction(_record())
    row = repo.search().iloc[0]
    assert row["review_status"] == "Pending" and "T" in row["created_at"]


def test_status_update_and_validation(tmp_path):
    repo = HistoryRepository(tmp_path / "h.db")
    rid = repo.add_prediction(_record())
    assert repo.update_status(rid, "Reviewed") is True
    assert repo.search(status="Reviewed").shape[0] == 1
    assert repo.update_status(9999, "Reviewed") is False
    with pytest.raises(ValueError):
        repo.update_status(rid, "Approved")


def test_review_queue_upsert_does_not_duplicate(tmp_path):
    repo = HistoryRepository(tmp_path / "h.db")
    rec = _record(row_ref="holdout:5")
    repo.set_review_status(rec, "Reviewed")
    repo.set_review_status(rec, "Dismissed")
    assert repo.count() == 1
    assert repo.queue_statuses(["holdout:5", "holdout:6"]) == {"holdout:5": "Dismissed"}


def test_clear_history(tmp_path):
    repo = HistoryRepository(tmp_path / "h.db")
    repo.add_many([_record(), _record()])
    assert repo.clear_all() == 2 and repo.count() == 0


def test_invalid_status_in_record_rejected(tmp_path):
    repo = HistoryRepository(tmp_path / "h.db")
    with pytest.raises(ValueError):
        repo.add_prediction(_record(review_status="Nope"))
