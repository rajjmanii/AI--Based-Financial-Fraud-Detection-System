import numpy as np
import pytest

from src.data.data_validation import DataValidationError
from src.explainability.explanations import global_importance, local_explanation
from src.features.feature_schema import FEATURE_COLUMNS
from src.models.predict import predict_single, predict_transactions, score_transactions


def test_output_format(bundle, synthetic_df):
    out = predict_transactions(bundle, synthetic_df.head(50).drop(columns=["Class"]))
    assert {"risk_score", "risk_category", "flagged_for_review"} <= set(out.columns)
    assert out["risk_score"].between(0, 100).all()
    assert set(out["risk_category"]) <= {"Low", "Medium", "High", "Critical"}
    assert len(out) == 50


def test_flag_matches_threshold(bundle, synthetic_df):
    data = synthetic_df.head(300).drop(columns=["Class"])
    scores = score_transactions(bundle, data)
    out = predict_transactions(bundle, data)
    assert (out["flagged_for_review"].to_numpy() == (scores >= bundle.threshold)).all()
    assert set(out.loc[out["flagged_for_review"], "risk_category"]) <= {"High", "Critical"}


def test_prediction_does_not_use_label_column(bundle, synthetic_df):
    labelled = synthetic_df.head(100)
    a = score_transactions(bundle, labelled)
    b = score_transactions(bundle, labelled.drop(columns=["Class"]))
    assert np.allclose(a, b)


def test_existing_prediction_columns_are_replaced(bundle, synthetic_df):
    data = synthetic_df.head(10).drop(columns=["Class"]).assign(risk_score=-1.0)
    assert (predict_transactions(bundle, data)["risk_score"] >= 0).all()


def test_invalid_input_rejected(bundle, synthetic_df):
    with pytest.raises(DataValidationError):
        score_transactions(bundle, synthetic_df.head(5).drop(columns=["V7"]))


def test_single_prediction(bundle, synthetic_df):
    row = synthetic_df.iloc[0][FEATURE_COLUMNS].to_dict()
    result = predict_single(bundle, row)
    assert 0 <= result["risk_score"] <= 100 and result["model"] == bundle.model_name
    assert isinstance(result["flagged"], bool)


def test_higher_threshold_flags_fewer(bundle, synthetic_df):
    data = synthetic_df.head(500).drop(columns=["Class"])
    low = predict_transactions(bundle, data, 0.1)["flagged_for_review"].sum()
    high = predict_transactions(bundle, data, 0.9)["flagged_for_review"].sum()
    assert low >= high


def test_local_explanation_shape_and_baseline(bundle, synthetic_df):
    row = synthetic_df.iloc[0][FEATURE_COLUMNS].to_dict()
    expl = local_explanation(bundle, row, top_n=5)
    assert len(expl) == 5 and {"feature", "contribution"} <= set(expl.columns)
    typical = {c: bundle.metadata["training_medians"][c] for c in FEATURE_COLUMNS}
    assert np.allclose(local_explanation(bundle, typical, top_n=30)["contribution"], 0)


def test_global_importance(metadata):
    imp = global_importance(metadata, top_n=5)
    assert len(imp) == 5 and imp["importance"].is_monotonic_decreasing
