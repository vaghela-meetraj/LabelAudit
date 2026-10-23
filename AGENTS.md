# AGENTS.md — LabelAudit

## Run command
```
streamlit run app.py
```

## Test command
```
pytest -q
```
Run from the `LabelAudit/` directory.

## Module layout

| Path | Purpose |
|---|---|
| `app.py` | Streamlit UI — renders, handles session state, calls `audit.py`. No analysis logic here. |
| `src/__init__.py` | Package marker. |
| `src/data_loading.py` | CSV reading, encoding detection, validation, column selection, `csv_row` assignment. |
| `src/preprocessing.py` | Text normalization pipeline (Unicode NFKC → lower → trim → punct → collapse ws). |
| `src/duplicate_detection.py` | Exact-duplicate grouping (Method A) and TF-IDF cosine near-duplicate / conflict pairing (Method B). |
| `src/label_analysis.py` | Out-of-fold cross-validation classifier; suspicious-label flags (Method C). |
| `src/model_evaluation.py` | Baseline metrics from OOF predictions; controlled corruption experiment. |
| `src/audit.py` | Orchestrator: `AuditConfig` + `AuditResult` dataclasses; calls A → B → C → eval. |
| `src/reporting.py` | CSV/JSON export builders; formula-injection sanitisation. |
| `tests/` | One test file per source module plus `conftest.py`, `test_sample_dataset.py`, `test_app.py`. |
| `sample_data/` | `example_dataset.csv` and `PLANTED_ISSUES.md`. |
| `docs/METHODOLOGY.md` | Algorithm explanations, design rationale, examiner Q&A. |

## Rules (must follow)

1. **Never fake results.** If something cannot be run, say so.
2. **Build in step order**: data loading → exact dups → near-dups → label analysis → metrics → reporting → Streamlit → sample data → final verification.
3. **Run `pytest -q` after each step** and fix failures before proceeding.
4. **Tests are written alongside the module**, not at the end.
5. **Stretch items** (marked in brief) are implemented only after every must-have passes tests.
6. `app.py` must contain no analysis logic — only rendering and state management.
7. `audit.py` is the single entry point for analysis. Both `app.py` and tests call it.
8. All records have a `csv_row` identifier (`index + 2`, header = row 1). Use it everywhere.
9. Never silently discard records. Report exclusions with reasons.
10. The random seed defaults to 42 and is configurable in the main-page Audit setup form.
11. Exports use UTF-8 BOM; formula injection is sanitised when the toggle is on.
12. Never fit the vectorizer or classifier on held-out rows.
