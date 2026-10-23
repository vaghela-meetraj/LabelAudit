# LabelAudit

**LabelAudit** is a local, privacy-preserving Python tool for inspecting text classification datasets for potential data-quality issues: exact duplicates, near-duplicates, conflicting labels, and suspicious label assignments.

> **Note:** This is an academic mini-project. The sample dataset is synthetic and illustrative only — not a real-world benchmark.

---

## Features

| Feature | Details |
|---|---|
| Exact duplicates | Groups records with identical normalised text; detects label conflicts within groups |
| Near-duplicates | TF-IDF cosine similarity (word + char n-grams); block-wise to avoid N×N matrix |
| Conflicting labels | Similar texts with different labels flagged as review signals |
| Suspicious labels | Out-of-fold classifier disagreements flagged for review |
| Metrics | Accuracy, macro-F1, per-class P/R/F1, confusion matrix |
| Corruption experiment | Synthetic label errors planted and detected (EXPERIMENTAL) |
| Export | UTF-8 BOM CSVs, JSON config summary, formula-injection sanitisation |

---

## Directory structure

```
LabelAudit/
├── app.py                  Streamlit UI (rendering only)
├── AGENTS.md               Build rules and module layout
├── requirements.txt        Pinned dependency ranges
├── README.md               This file
├── .gitignore
├── .streamlit/
│   └── config.toml         Theme, upload limit, no telemetry
├── src/
│   ├── __init__.py
│   ├── data_loading.py     CSV loading, encoding, validation
│   ├── preprocessing.py    Text normalisation pipeline
│   ├── duplicate_detection.py  Methods A and B
│   ├── label_analysis.py   Method C: OOF cross-validation
│   ├── model_evaluation.py Metrics and corruption experiment
│   ├── audit.py            Orchestrator (AuditConfig + AuditResult)
│   └── reporting.py        CSV/JSON export builders
├── tests/
│   ├── conftest.py         Shared deterministic fixtures
│   ├── test_data_loading.py
│   ├── test_preprocessing.py
│   ├── test_duplicates.py
│   ├── test_label_analysis.py
│   ├── test_model_evaluation.py
│   ├── test_reporting.py
│   ├── test_sample_dataset.py
│   └── test_app.py
├── docs/
│   └── METHODOLOGY.md      Algorithm explanations and examiner Q&A
└── sample_data/
    ├── example_dataset.csv   96-row synthetic support-ticket dataset
    └── PLANTED_ISSUES.md     Documented planted issues and expected counts
```

---

## Setup

**Requirements:** Python 3.10 or later.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## Uploading and analysing a dataset

1. Open the app (`streamlit run app.py`).
2. Use the **sidebar** file uploader to upload a CSV (max 50 MB).
3. In the main-page **Audit setup** area, select the **text column** and **label column**.
4. Adjust thresholds and classifier settings in that area.
5. Press **▶ Run Audit**.
6. Explore the six tabs: Overview, Exact Duplicates, Similar Texts, Suspicious Labels, Evaluation, Export.

### CSV schema example

```
ticket_id,message,category,source
1,Network keeps dropping,technical,web
2,Charged twice this month,billing,email
```

- Any column names are accepted — select them in the main-page Audit setup area.
- Labels are treated as strings; numeric labels are converted automatically.
- `NA`, `None`, `null` are treated as real class names (not missing values).

---

## Algorithms

### TF-IDF (Term Frequency–Inverse Document Frequency)

TF-IDF converts text into numerical vectors.  Each word's score reflects how often
it appears in a document, down-weighted by how common it is across all documents.
Rare, informative words score high; common words (e.g. "the") score low.

We use word n-grams (1–2 words) with `sublinear_tf=True` (log-scaling) to reduce
the dominance of very frequent terms.

### Cosine similarity

Given two TF-IDF vectors **a** and **b**, cosine similarity is:

```
sim(a, b) = (a · b) / (|a| × |b|)
```

It measures the angle between vectors, not their magnitude.  Two documents with
the same words in the same proportions score 1.0 regardless of length.

We compute it in row blocks (500 rows at a time) against the full matrix to avoid
building an N×N dense matrix in memory.

### Character n-grams (optional)

Character n-grams (3–5 character windows, `char_wb` analyser) partially compensate
for TF-IDF's poor handling of non-Latin scripts and catch typos/morphological
variants that word n-grams miss.

When both word and character TF-IDF are used:
- Each matrix is L2-normalised independently.
- Both are scaled by 1/√2 and horizontally stacked.
- The cosine of the combined vector equals the unweighted average of the two cosines.

### Stratified k-fold cross-validation

The dataset is split into *k* non-overlapping subsets ("folds").  For each fold,
the classifier is trained on the other *k*−1 folds and predicts the held-out fold.
"Stratified" means each fold preserves the class proportion of the full dataset.

### Grouped cross-validation

Exact duplicates and near-duplicate clusters are assigned to the same fold.
If a record's twin were in the training fold, the model would effectively
have "seen" the test record — contaminating the evaluation.

`StratifiedGroupKFold` enforces this.  When too few distinct groups exist per class
(making the constraint infeasible), we fall back to `StratifiedKFold` and show a notice.

### Out-of-fold (OOF) predictions

Every record is predicted **exactly once**, by a model that was never trained on it.
This is the only unbiased way to use cross-validation for instance-level evidence.
Training on all data and predicting on training data is circular: the model
memorises training labels and flags would be meaningless.

### Evaluation metrics

| Metric | Formula | Notes |
|---|---|---|
| Accuracy | correct / total | Overall classification correctness |
| Precision (per class) | TP / (TP + FP) | Of all records predicted as class C, how many are correct |
| Recall (per class) | TP / (TP + FN) | Of all true class-C records, how many were predicted correctly |
| F1 (per class) | 2PR/(P+R) | Harmonic mean; 0 when class absent from predictions |
| Macro F1 | mean of per-class F1 | Equal weight per class regardless of size |

`zero_division=0` is used throughout so missing-class predictions do not crash.

---

## Limitations

- **Flag ≠ error**: A flag means a classifier disagreed. The supplied label may still be correct.
- **OOF accuracy < 0.6**: Many flags likely reflect model weakness, not label errors.
- **Non-Latin scripts**: The word token pattern `[a-zA-Z0-9]+` handles them poorly. Character n-grams partially compensate.
- **Scalability**: Above the pairwise limit the similarity step is skipped; prepare a smaller CSV or label subset outside the app and upload it.
- **Corruption experiment**: Synthetic random corruption ≠ real annotation errors.

---

## Privacy

All analysis runs locally on your machine.  No data is sent to any server.
`browser.gatherUsageStats = false` is set in `.streamlit/config.toml`.
No logging, no telemetry, no external API calls.

---

## Test commands

```bash
# Run all tests
pytest -q

# Run a specific module
pytest tests/test_duplicates.py -v

# Run sample dataset tests
pytest tests/test_sample_dataset.py -v
```

---

## Demonstration workflow (sample CSV)

```bash
streamlit run app.py
# → Upload sample_data/example_dataset.csv
# → Text column: message  |  Label column: category
# → Press ▶ Run Audit
# → Exact Duplicates tab: should show ≥ 2 groups, ≥ 1 with label conflict
# → Similar Texts tab: should show near-duplicate and conflict pairs
# → Suspicious Labels tab: flagged records listed with predicted labels
# → Export tab: download the full audit report CSV
```
