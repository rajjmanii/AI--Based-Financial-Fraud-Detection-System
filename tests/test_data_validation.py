import io

import numpy as np
import pandas as pd
import pytest

from src.data.data_validation import (DataValidationError, read_transactions_csv,
                                      validate_transactions)
from src.data.dataset_downloader import DatasetError, require_dataset
from src.features.feature_schema import FEATURE_COLUMNS
from src.utils.validation_helpers import out_of_range_features, validate_single_transaction


def test_valid_dataframe_passes(synthetic_df):
    report = validate_transactions(synthetic_df, require_label=True)
    assert report.ok
    assert report.stats["n_fraud"] == int(synthetic_df["Class"].sum())


def test_missing_columns_reported(synthetic_df):
    report = validate_transactions(synthetic_df.drop(columns=["V3", "Amount"]))
    assert not report.ok
    assert "V3" in report.errors[0] and "Amount" in report.errors[0]


def test_missing_values_and_text_detected(synthetic_df):
    df = synthetic_df.copy()
    df["V1"] = df["V1"].astype(object)
    df.loc[0, "V1"] = None
    df.loc[1, "V2"] = np.nan
    df["V5"] = df["V5"].astype(object)
    df.loc[2, "V5"] = "abc"
    errors = " ".join(validate_transactions(df).errors)
    assert "V1" in errors and "V2" in errors and "non-numeric" in errors


def test_infinite_and_negative_values_detected(synthetic_df):
    df = synthetic_df.copy()
    df.loc[0, "V4"] = np.inf
    df.loc[1, "Amount"] = -5.0
    errors = " ".join(validate_transactions(df).errors)
    assert "infinite" in errors and "negative" in errors


def test_invalid_labels_detected(synthetic_df):
    df = synthetic_df.copy()
    df.loc[0, "Class"] = 2
    assert not validate_transactions(df).ok


def test_label_required_only_when_requested(synthetic_df):
    unlabeled = synthetic_df.drop(columns=["Class"])
    assert validate_transactions(unlabeled).ok
    assert not validate_transactions(unlabeled, require_label=True).ok


def test_duplicates_are_warnings_not_errors(synthetic_df):
    df = pd.concat([synthetic_df.head(50), synthetic_df.head(3)], ignore_index=True)
    report = validate_transactions(df)
    assert report.ok and report.warnings


def test_empty_dataframe_rejected(synthetic_df):
    assert not validate_transactions(synthetic_df.iloc[0:0]).ok


def test_csv_reader_errors(tmp_path):
    with pytest.raises(DataValidationError):
        read_transactions_csv(tmp_path / "missing.csv")
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(DataValidationError):
        read_transactions_csv(empty)
    header_only = io.StringIO(",".join(FEATURE_COLUMNS) + "\n")
    with pytest.raises(DataValidationError):
        read_transactions_csv(header_only)


def test_csv_reader_reads_valid_upload(synthetic_df):
    buffer = io.StringIO(synthetic_df.head(20).to_csv(index=False))
    assert len(read_transactions_csv(buffer)) == 20


def test_missing_dataset_gives_helpful_error(tmp_path):
    with pytest.raises(DatasetError) as info:
        require_dataset(tmp_path)
    assert "creditcard.csv" in str(info.value)


def test_single_transaction_validation():
    good = {c: 0.0 for c in FEATURE_COLUMNS}
    assert validate_single_transaction(good) == []
    bad = {**good, "Amount": -1.0, "V1": float("nan")}
    assert len(validate_single_transaction(bad)) == 2
    assert validate_single_transaction({"Time": 1.0})  # missing fields


def test_out_of_range_detection():
    ranges = {"V1": [-1.0, 1.0], "V2": [-1.0, 1.0]}
    assert out_of_range_features({"V1": 10.0, "V2": 0.0}, ranges) == ["V1"]
