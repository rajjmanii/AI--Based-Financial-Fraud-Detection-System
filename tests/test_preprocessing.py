import numpy as np
import pandas as pd
import pytest

from src.data.data_validation import DataValidationError
from src.data.preprocessing import build_pipeline, prepare_training_frame, split_dataset
from src.features.feature_schema import FEATURE_COLUMNS
from sklearn.linear_model import LogisticRegression


def test_duplicates_removed_before_splitting(synthetic_df):
    doubled = pd.concat([synthetic_df, synthetic_df.head(100)], ignore_index=True)
    clean, info = prepare_training_frame(doubled)
    assert info["n_duplicates_removed"] == 100
    assert not clean.duplicated().any()


def test_too_few_fraud_rows_rejected(synthetic_df):
    few = synthetic_df[synthetic_df["Class"] == 0].head(500).copy()
    few.iloc[:5, few.columns.get_loc("Class")] = 1
    with pytest.raises(DataValidationError):
        prepare_training_frame(few)


def test_split_is_disjoint_stratified_and_complete(synthetic_df):
    clean, _ = prepare_training_frame(synthetic_df)
    splits = split_dataset(clean, random_state=1)
    idx = {k: set(v[0].index) for k, v in splits.items()}
    assert not (idx["train"] & idx["val"]) and not (idx["train"] & idx["test"]) and not (idx["val"] & idx["test"])
    assert sum(len(v) for v in idx.values()) == len(clean)
    rates = [v[1].mean() for v in splits.values()]
    assert max(rates) - min(rates) < 0.01


def test_split_is_reproducible(synthetic_df):
    clean, _ = prepare_training_frame(synthetic_df)
    a = split_dataset(clean, random_state=3)["test"][0].index
    b = split_dataset(clean, random_state=3)["test"][0].index
    assert list(a) == list(b)


def test_scaler_is_fitted_on_training_data_only(synthetic_df):
    clean, _ = prepare_training_frame(synthetic_df)
    (X_train, y_train), (X_val, _), _ = (split_dataset(clean)[k] for k in ("train", "val", "test"))
    pipe = build_pipeline(LogisticRegression(max_iter=300)).fit(X_train, y_train)
    scaler = pipe.named_steps["prep"].named_transformers_["scale"]
    assert np.allclose(scaler.mean_, X_train[["Time", "Amount"]].mean().to_numpy())
    assert not np.allclose(scaler.mean_, clean[["Time", "Amount"]].mean().to_numpy())


def test_pipeline_output_uses_same_feature_schema(synthetic_df):
    clean, _ = prepare_training_frame(synthetic_df)
    X_train, y_train = split_dataset(clean)["train"]
    pipe = build_pipeline(LogisticRegression(max_iter=300)).fit(X_train, y_train)
    shuffled = X_train.head(5)[FEATURE_COLUMNS[::-1]]  # column order must not matter
    assert np.allclose(pipe.predict_proba(shuffled), pipe.predict_proba(X_train.head(5)))
