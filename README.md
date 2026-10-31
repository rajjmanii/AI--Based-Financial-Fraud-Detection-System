# FraudShield | Financial Risk Intelligence
*Machine Learning Powered Transaction Risk Analysis*

A Streamlit application that scores card transactions with a machine learning model, shows how
reliable that model is on highly imbalanced data, and helps a person work through a review queue.

**Two objectives**
1. **Intelligent fraud detection**: score each transaction (risk score 0-100) to prioritise investigation.
2. **Reliable evaluation and explainability**: imbalance-aware metrics, false-positive/false-negative
   analysis, and honest feature explanations.

> The output is a model prediction, not proof of fraud. The app never blocks payments or contacts anyone.

---

## 1. Requirements
- Windows 10/11, VS Code, **Python 3.10 or newer** (check with `py -0` or `py -3.10 --version`)
- No GPU, Docker or paid service. Internet is needed only to install packages and download the dataset.

## 2. First-time setup (PowerShell, from the project folder)
```powershell
cd C:\SHAHAN\FraudShield
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\setup_project.py
.\.venv\Scripts\python.exe scripts\verify_installation.py
```
If you have a different Python version, replace `py -3.10` with e.g. `py -3.12`.

## 3. Get the dataset (real data)
The dataset is **not** bundled. Choose one option.

**Option A: manual (simplest)**
1. Sign in at https://www.kaggle.com and open https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Click *Download*, unzip, and place **creditcard.csv** at `data\raw\creditcard.csv`.

**Option B: Kaggle API**
1. Kaggle -> *Settings* -> *API* -> *Create New Token* downloads `kaggle.json`.
2. Put it in `C:\Users\<you>\.kaggle\kaggle.json`, **or** copy `.env.example` to `.env` and fill in
   `KAGGLE_USERNAME` and `KAGGLE_KEY` (never share or commit this file).
3. Run:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-kaggle.txt
.\.venv\Scripts\python.exe scripts\download_dataset.py
```
Credentials are never stored in the code. If the API fails, use Option A.

## 4. Train the models (one time)
```powershell
.\.venv\Scripts\python.exe scripts\train_models.py
```
This validates the data, removes duplicates, splits it (60/20/20 stratified), trains three models,
selects one using validation data, tests it once on the held-out set and saves everything to
`artifacts\`. This can take several minutes on a laptop (I could not time it on the full dataset). The app never retrains by itself.

**No dataset yet? Demo mode** trains on *synthetic random data* so you can explore the interface.
The app shows a warning banner and its numbers mean nothing about real fraud:
```powershell
.\.venv\Scripts\python.exe scripts\train_models.py --demo
```
A real model in `artifacts\` always takes priority over the demo model in `artifacts_demo\`.

## 5. Run the app
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Next time you only need this last command (plus step 4 again if you want to retrain).
For Bulk Screening, `data\sample\sample_transactions_SYNTHETIC.csv` shows the file format (random
numbers; it only makes sense with the demo model).

## 6. Run the tests
```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
Tests use small synthetic data, so they need neither the Kaggle file nor internet access.

---

## What each page does
| Page | Purpose |
|---|---|
| Executive Dashboard | Held-out test partition: actual fraud vs normal, flagged count at the chosen threshold, final metrics, amount and risk distributions, key findings. |
| Transaction Analyzer | Enter Time, Amount, V1-V28 (or load a real held-out example). Shows risk score, category, outcome, and which features move the score. Optional save to history. |
| Bulk Screening | Upload a CSV, validate it, score every row, filter/sort, download results. If a `Class` column exists, evaluation metrics are shown separately; if not, none are claimed. |
| Model Laboratory | Compare Logistic Regression, Random Forest, HistGradientBoosting (PR-AUC, ROC-AUC, precision, recall, F1, FP/FN, confusion matrices, PR curves), final test result and feature importance. |
| Risk Monitoring | Threshold slider with validation-based precision/recall trade-off, plus the review queue (Pending / Reviewed / Dismissed) stored in SQLite. |
| Prediction History | Search and filter saved records, update statuses, export, clear (with confirmation). |
| Reports and Exports | PDF report (ReportLab) and CSV exports, built from real results. |

## Methodology (for your viva)
1. **Data**: Kaggle credit-card data, 284,807 rows, 492 frauds (0.172%). Features: `Time`, `V1`-`V28`
   (PCA components, anonymised), `Amount`; label `Class`. Exact duplicate rows are removed *before*
   splitting so the same transaction cannot appear in both training and test data.
2. **Split**: stratified train 60% / validation 20% / test 20%, fixed seed 42.
3. **Preprocessing**: `Time` and `Amount` are standardised inside a scikit-learn Pipeline, so the scaler
   learns its mean and spread from training data only (no leakage). V1-V28 are used as-is.
4. **Imbalance**: class weights (`balanced`), the simplest method that needs no synthetic rows. SMOTE or
   resampling was not added because class weights already work; adding it would be extra complexity
   without demonstrated benefit.
5. **Models**: Logistic Regression (interpretable baseline), Random Forest (non-linear trees),
   HistGradientBoosting (boosted trees). All are in scikit-learn, so no extra dependency.
6. **Model selection**: highest **validation PR-AUC** (average precision). It ignores the threshold and
   focuses on the rare class. Ties go to validation F2.
7. **Threshold**: chosen on validation data by maximising **F2** (recall counts twice as much as
   precision, because a missed fraud is usually costlier than an extra review). The test set is scored
   once at the end. The Risk Monitoring trade-off chart also uses validation data, so moving the slider
   never tunes on the test set.
8. **Scores**: class-weighted models do not output calibrated probabilities, so the app calls the values
   *risk scores* (probability x 100). They rank transactions but should not be read as "x% chance of fraud".
9. **Risk categories** (threshold *t* on the 0-100 scale): Low < t/2; Medium t/2 to t; High t to t+(100-t)/2;
   Critical above that. "Flagged" means High or Critical, i.e. score >= t.
10. **Explainability**: global importance = permutation importance on validation data (same method for all
    models); local explanation = change in score when one feature is replaced by its training median.
    No SHAP/LIME dependency was added because this is simpler and enough to explain behaviour.
    Importance is not causation, and V1-V28 have no published meaning.

**Why not accuracy?** Predicting "normal" for everything gives about 99.83% accuracy and catches no fraud.
The Model Laboratory computes this baseline from your data.

## Limitations (be upfront about these)
- Data is from two days in 2013, anonymised, from one region. It does not represent today's fraud patterns.
- With only roughly 100 fraud cases in the test set, precision and recall are noisy estimates.
- A random split is used (as is common for this dataset); a time-ordered split would be stricter.
- Scores are uncalibrated. The review queue uses the held-out partition of the same dataset to
  demonstrate the workflow; no live transactions are processed.
- Uploaded files containing training rows will give optimistically good metrics.

## Folder guide
`app.py` entry and router | `pages/` Streamlit pages | `src/data` loading, validation, preprocessing |
`src/models` train/evaluate/predict/threshold/registry | `src/explainability` | `src/database` SQLite |
`src/monitoring` risk categories and queue | `src/reporting` CSV/PDF | `src/ui` theme and cached loaders |
`scripts/` command-line tools | `tests/` pytest suite | `artifacts/` trained model and metadata (created by training).

## Troubleshooting
| Problem | Fix |
|---|---|
| `py` is not recognised | Install Python from python.org and tick *Add python.exe to PATH*, or use `python -m venv .venv`. |
| `No module named streamlit` (or another package) | You are not using the venv. Always call `.\.venv\Scripts\python.exe ...` as shown. |
| `Dataset not found` | Put `creditcard.csv` in `data\raw\` (exact name), then train again. |
| `Kaggle credentials were not found` | Use manual download (Option A) or set up `kaggle.json` / `.env`. |
| App says *No trained model found* | Run `scripts\train_models.py` (or add `--demo`). |
| `MemoryError` or very slow training | Close other programs and free some RAM; the full dataset has 284,807 rows. |
| Warning about scikit-learn version mismatch | Re-run `train_models.py` with the currently installed version. |
| `TypeError` mentioning `use_container_width` or an unknown argument | Install the tested range: `.\.venv\Scripts\python.exe -m pip install -r requirements.txt --upgrade`. |
| Port 8501 already in use | Add `--server.port 8502` to the streamlit command. |
| History database error | Close other programs using `data\fraudshield.db`, or delete it (history is lost). |
| `Execution policy` errors | Not needed: all commands call `python.exe` directly, so you never activate the venv. |

Stop the app with `Ctrl+C` in the terminal.
