import json

import numpy as np

from src.models.evaluate import all_normal_baseline, compute_metrics, pr_curve_points
from src.models.model_registry import ArtifactsMissingError, load_bundle
from src.models.train import candidate_models

import pytest


def test_training_writes_expected_artifacts(trained_dir):
    for name in ("selected_model.joblib", "metadata.json", "validation_scores.csv",
                 "holdout_scores.csv", "examples.csv"):
        assert (trained_dir / name).is_file()


def test_metadata_contents(metadata):
    assert set(metadata["models"]) == set(candidate_models())
    sel = metadata["selection"]
    assert sel["selected_model"] in metadata["models"]
    best_ap = max(m["validation_metrics"]["average_precision"] for m in metadata["models"].values())
    chosen_ap = metadata["models"][sel["selected_model"]]["validation_metrics"]["average_precision"]
    assert round(chosen_ap, 4) == round(best_ap, 4)
    assert metadata["is_demo"] is True
    json.dumps(metadata)  # serialisable


def test_splits_do_not_overlap_in_saved_score_tables(trained_dir):
    import pandas as pd
    val = pd.read_csv(trained_dir / "validation_scores.csv")
    test = pd.read_csv(trained_dir / "holdout_scores.csv")
    assert not set(val["row_id"]) & set(test["row_id"])


def test_test_metrics_use_validation_threshold(metadata):
    assert metadata["test_metrics"]["threshold"] == metadata["selection"]["threshold"]


def test_test_metrics_match_recomputation_from_saved_scores(trained_dir, metadata):
    import pandas as pd
    test = pd.read_csv(trained_dir / "holdout_scores.csv")
    again = compute_metrics(test["Class"], test["score"], metadata["selection"]["threshold"])
    assert again["tp"] == metadata["test_metrics"]["tp"]
    assert again["fp"] == metadata["test_metrics"]["fp"]


def test_metrics_on_known_values():
    y = [1, 1, 0, 0, 0, 0]
    s = [0.9, 0.4, 0.8, 0.2, 0.1, 0.05]
    m = compute_metrics(y, s, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 3)
    assert m["precision"] == 0.5 and m["recall"] == 0.5
    assert 0 < m["average_precision"] <= 1 and 0 <= m["roc_auc"] <= 1


def test_metrics_handle_single_class_without_error():
    m = compute_metrics([0, 0, 0], [0.1, 0.2, 0.3], 0.5)
    assert m["roc_auc"] is None and m["average_precision"] is None and m["recall"] == 0.0


def test_accuracy_is_misleading_baseline():
    y = np.array([0] * 990 + [1] * 10)
    base = all_normal_baseline(y)
    assert base["accuracy"] == 0.99 and base["recall"] == 0.0


def test_pr_curve_is_downsampled():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    curve = pr_curve_points(y, rng.random(2000), max_points=50)
    assert len(curve["precision"]) <= 50 and len(curve["precision"]) == len(curve["recall"])


def test_model_beats_trivial_baseline(metadata):
    assert metadata["test_metrics"]["recall"] > 0.5
    assert metadata["test_metrics"]["average_precision"] > 0.5


def test_missing_artifacts_raise_helpful_error(tmp_path):
    with pytest.raises(ArtifactsMissingError) as info:
        load_bundle(tmp_path)
    assert "train_models.py" in str(info.value)


def test_corrupt_model_file_raises_helpful_error(trained_dir, tmp_path):
    import shutil
    for f in trained_dir.iterdir():
        shutil.copy(f, tmp_path / f.name)
    (tmp_path / "selected_model.joblib").write_bytes(b"not a model")
    with pytest.raises(ArtifactsMissingError):
        load_bundle(tmp_path)
