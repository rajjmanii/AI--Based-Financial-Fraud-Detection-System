"""FraudShield entry point: page registration and the home screen."""
import streamlit as st

from config.settings import APP_NAME, APP_TITLE
from src.data.dataset_downloader import find_dataset
from src.database.db_manager import DatabaseError, init_db
from src.models.model_registry import load_bundle, resolve_artifacts_dir
from src.ui.theme import apply_theme, metric_card, note_box, page_header
from src.utils.logging_config import setup_logging

st.set_page_config(page_title=APP_TITLE, page_icon=None, layout="wide")
setup_logging()
apply_theme()


def home() -> None:
    page_header("Home")
    try:
        init_db()
        db_status = "Ready"
    except DatabaseError as exc:
        db_status = f"Problem: {exc}"

    directory = resolve_artifacts_dir()
    dataset = find_dataset()
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Dataset", "Found" if dataset else "Not found",
                    "data/raw/creditcard.csv" if dataset else "See README: Dataset setup",
                    "positive" if dataset else "alert")
    with c2:
        if directory:
            bundle = load_bundle(directory)
            label = "Demo model" if bundle.is_demo else "Trained"
            metric_card("Model", label, f"{bundle.model_name}", "positive" if not bundle.is_demo else "navy")
        else:
            metric_card("Model", "Not trained", "Run scripts\\train_models.py", "alert")
    with c3:
        metric_card("History database", db_status, "Local SQLite file", "positive" if db_status == "Ready" else "alert")

    st.markdown("### What you can do here")
    st.markdown(
        "- **Executive Dashboard**: held-out test results, flagged counts, score distributions.\n"
        "- **Transaction Analyzer**: score one transaction and see which features moved the score.\n"
        "- **Bulk Screening**: upload a CSV, score every row, filter, and download results.\n"
        "- **Model Laboratory**: compare the three models with imbalance-aware metrics.\n"
        "- **Risk Monitoring**: choose the threshold and work through the review queue.\n"
        "- **Prediction History**: search saved predictions and review statuses.\n"
        "- **Reports and Exports**: download CSV files and a PDF report."
    )
    note_box("<b>Important:</b> FraudShield produces model risk scores to help a person prioritise "
             "investigation. A high score is not proof of fraud, and the application never blocks "
             "payments or contacts anyone. Data: Kaggle Credit Card Fraud Detection (anonymised "
             "European card transactions, two days in 2013).")


pages = [
    st.Page(home, title="Home", default=True, url_path="home"),
    st.Page("pages/1_Executive_Dashboard.py", title="Executive Dashboard"),
    st.Page("pages/2_Transaction_Analyzer.py", title="Transaction Analyzer"),
    st.Page("pages/3_Bulk_Screening.py", title="Bulk Screening"),
    st.Page("pages/4_Model_Laboratory.py", title="Model Laboratory"),
    st.Page("pages/5_Risk_Monitoring.py", title="Risk Monitoring"),
    st.Page("pages/6_Prediction_History.py", title="Prediction History"),
    st.Page("pages/7_Reports_Exports.py", title="Reports and Exports"),
]
st.sidebar.markdown(f"### {APP_NAME}")
st.navigation(pages).run()
