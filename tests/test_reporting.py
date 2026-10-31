import io

import pandas as pd

from src.features.feature_schema import TARGET_COLUMN
from src.models.predict import predict_transactions
from src.reporting.export_service import dataframe_to_csv_bytes, summarize_screening
from src.reporting.pdf_report import build_pdf_report


def test_csv_export_round_trip(bundle, synthetic_df):
    results = predict_transactions(bundle, synthetic_df.head(40))
    restored = pd.read_csv(io.BytesIO(dataframe_to_csv_bytes(results)))
    assert len(restored) == 40 and "risk_score" in restored.columns and "V1" in restored.columns


def test_summary_without_labels_has_no_evaluation(bundle, synthetic_df):
    results = predict_transactions(bundle, synthetic_df.head(100).drop(columns=[TARGET_COLUMN]))
    summary = summarize_screening(results, bundle.threshold, bundle.model_name)
    assert summary["has_labels"] is False and summary["evaluation"] is None
    assert summary["n_flagged"] == int(results["flagged_for_review"].sum())


def test_summary_with_labels_reports_evaluation(bundle, synthetic_df):
    results = predict_transactions(bundle, synthetic_df.head(500))
    summary = summarize_screening(results, bundle.threshold, bundle.model_name)
    assert summary["evaluation"]["n_rows"] == 500


def test_pdf_report_generation(metadata):
    pdf = build_pdf_report(metadata)
    assert pdf.startswith(b"%PDF") and len(pdf) > 3000


def test_pdf_report_with_screening(bundle, metadata, synthetic_df):
    results = predict_transactions(bundle, synthetic_df.head(300))
    summary = summarize_screening(results, bundle.threshold, bundle.model_name)
    with_screening = build_pdf_report(metadata, summary)
    assert with_screening.startswith(b"%PDF")
    assert len(with_screening) > len(build_pdf_report(metadata))
