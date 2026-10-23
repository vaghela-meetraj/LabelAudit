"""
label_analysis.py — Method C: out-of-fold suspicious-label detection.

Algorithm
---------
A baseline text classifier is trained with k-fold cross-validation.
Each record is predicted by a model that never saw that record during training
(out-of-fold = OOF predictions).  When the OOF prediction disagrees with the
supplied label, the record is flagged for review.

Why OOF?  Training on all data and predicting on training data is circular:
the model memorises its training labels, so disagreements would be rare even
when labels are wrong.  OOF avoids this bias.

Grouping rationale: exact duplicates and near-duplicates in the same connected
component must stay together in one fold.  If a record leaks into the training
fold via its twin, the prediction is contaminated and the flag is meaningless.
We use StratifiedGroupKFold to enforce this.  When groups are too small to
support StratifiedGroupKFold (i.e. fewer distinct groups than folds in some
class), we fall back to StratifiedKFold and show a notice.

Eligible records: non-empty normalised text + label whose class has at least
`min_class_size` members (default 3).  Classes with fewer members are excluded
with the reason reported.

Flags: a record is flagged when predicted_label ≠ supplied_label.
Disagreement is a review signal only — the supplied label may still be correct.
If OOF accuracy < 0.6 we warn that many flags likely reflect model weakness.

Status values (mutually exclusive, never merged):
  - "flagged"       : OOF prediction ≠ supplied label
  - "not_flagged"   : OOF prediction = supplied label
  - "excluded"      : record was not eligible (reason attached)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LabelAnalysisRecord:
    csv_row: int
    orig_text: str
    supplied_label: str
    status: str           # "flagged", "not_flagged", "excluded"
    predicted_label: Optional[str] = None
    model_score: Optional[float] = None   # uncalibrated OOF score for ranking
    exclusion_reason: Optional[str] = None


@dataclass
class LabelAnalysisResult:
    records: list[LabelAnalysisRecord]
    n_flagged: int
    n_not_flagged: int
    n_excluded: int
    n_eligible: int
    folds_used: int
    requested_folds: int
    splitter_used: str          # "StratifiedGroupKFold" or "StratifiedKFold"
    splitter_fallback_reason: Optional[str]
    oof_accuracy: Optional[float]
    low_accuracy_warning: bool
    cannot_run: bool
    cannot_run_reason: Optional[str]
    # Predictions indexed by csv_row for export
    predictions: dict[int, str] = field(default_factory=dict)
    scores: dict[int, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_groups_array(
    csv_rows: list[int],
    row_to_group: dict[int, int],
) -> np.ndarray:
    """
    Build the groups array for GroupKFold.  Records in an exact-dup or
    near-dup cluster share a group id; singletons get a unique id.
    """
    next_singleton = max(row_to_group.values(), default=0) + 1
    groups: list[int] = []
    for r in csv_rows:
        gid = row_to_group.get(r)
        if gid is None:
            groups.append(next_singleton)
            next_singleton += 1
        else:
            groups.append(gid)
    return np.array(groups)


def _can_use_grouped_split(
    labels: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
) -> tuple[bool, str]:
    """
    Check whether StratifiedGroupKFold is feasible.
    It needs at least n_folds distinct groups per class.
    Returns (feasible, reason_if_not).
    """
    for cls in np.unique(labels):
        mask = labels == cls
        n_distinct_groups = len(np.unique(groups[mask]))
        if n_distinct_groups < n_folds:
            return False, (
                f"Class '{cls}' has only {n_distinct_groups} distinct duplicate-group(s) "
                f"but {n_folds} folds were requested. "
                "Falling back to StratifiedKFold (duplicates may straddle folds)."
            )
    return True, ""


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze_labels(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    norm_col: str = "_norm_text",
    classifier_name: str = "LogisticRegression",
    class_weight: str = "balanced",
    n_folds: int = 5,
    random_seed: int = 42,
    min_class_size: int = 3,
    row_to_group: Optional[dict[int, int]] = None,
) -> LabelAnalysisResult:
    """
    Run out-of-fold cross-validation and return suspicious-label flags.

    Parameters
    ----------
    df : DataFrame with csv_row, text_col, label_col, norm_col.
    row_to_group : mapping csv_row → group_id for duplicates/near-dups.
                   Singletons are absent (they get their own unique group).
    classifier_name : "LogisticRegression" or "LinearSVC".
    class_weight : "balanced" or None.
    """
    row_to_group = row_to_group or {}
    records: list[LabelAnalysisRecord] = []
    excluded_csv_rows: set[int] = set()

    # --- Count class sizes ---
    eligible_mask = (
        (df[norm_col] != "") &
        (df[label_col].apply(lambda v: isinstance(v, str) and v.strip() != ""))
    )
    eligible_df = df[eligible_mask].copy()

    # Compute per-class sizes on eligible only
    class_counts = eligible_df[label_col].value_counts()
    small_classes = set(class_counts[class_counts < min_class_size].index)

    def _classify_row(row: pd.Series) -> LabelAnalysisRecord:
        csv_row = int(row["csv_row"])
        orig_text = str(row[text_col])
        label = str(row[label_col])
        norm = str(row[norm_col])

        # Excluded — missing text/label
        if not eligible_mask.loc[row.name]:
            reason = ""
            if not isinstance(row[text_col], str) or row[text_col].strip() == "" or norm == "":
                reason = "Missing or empty text after normalisation."
            elif not isinstance(row[label_col], str) or row[label_col].strip() == "":
                reason = "Missing label."
            return LabelAnalysisRecord(
                csv_row=csv_row, orig_text=orig_text, supplied_label=label,
                status="excluded", exclusion_reason=reason,
            )

        # Excluded — class too small
        if label in small_classes:
            return LabelAnalysisRecord(
                csv_row=csv_row, orig_text=orig_text, supplied_label=label,
                status="excluded",
                exclusion_reason=(
                    f"Class '{label}' has fewer than {min_class_size} records "
                    f"(found {class_counts.get(label, 0)})."
                ),
            )

        # Will be filled in after cross-validation
        return LabelAnalysisRecord(
            csv_row=csv_row, orig_text=orig_text, supplied_label=label,
            status="not_flagged",
        )

    for _, row in df.iterrows():
        records.append(_classify_row(row))

    # Build eligible subset (not excluded)
    eligible_records = [r for r in records if r.status != "excluded"]
    n_excluded = len(records) - len(eligible_records)
    n_eligible = len(eligible_records)

    # --- Check feasibility ---
    def _cannot(reason: str) -> LabelAnalysisResult:
        return LabelAnalysisResult(
            records=records,
            n_flagged=0, n_not_flagged=0, n_excluded=n_excluded,
            n_eligible=n_eligible, folds_used=0, requested_folds=n_folds,
            splitter_used="", splitter_fallback_reason=None,
            oof_accuracy=None, low_accuracy_warning=False,
            cannot_run=True, cannot_run_reason=reason,
        )

    if n_eligible == 0:
        return _cannot("No eligible records after applying exclusion criteria.")

    elig_labels = [r.supplied_label for r in eligible_records]
    n_classes = len(set(elig_labels))
    if n_classes < 2:
        return _cannot(
            f"Only {n_classes} eligible class(es) remain after excluding small classes. "
            "Cross-validation requires at least 2 classes."
        )

    # Smallest class size among eligible records
    from collections import Counter
    class_counter = Counter(elig_labels)
    min_eligible_class = min(class_counter.values())
    folds_used = max(2, min(n_folds, min_eligible_class))

    if folds_used < 2:
        return _cannot(
            f"The smallest eligible class has only {min_eligible_class} record(s); "
            "need at least 2 to form 2 folds."
        )

    # --- Build arrays for cross-validation ---
    elig_csv_rows = [r.csv_row for r in eligible_records]
    elig_texts = []
    for r in eligible_records:
        row_match = df[df["csv_row"] == r.csv_row]
        if not row_match.empty:
            elig_texts.append(str(row_match.iloc[0][norm_col]))
        else:
            elig_texts.append("")

    X = elig_texts
    y = np.array(elig_labels)
    groups = _build_groups_array(elig_csv_rows, row_to_group)

    # --- Choose splitter ---
    feasible, fallback_reason = _can_use_grouped_split(y, groups, folds_used)
    if feasible:
        splitter = StratifiedGroupKFold(n_splits=folds_used, shuffle=True, random_state=random_seed)
        splitter_name = "StratifiedGroupKFold"
        split_kwargs = {"groups": groups}
    else:
        splitter = StratifiedKFold(n_splits=folds_used, shuffle=True, random_state=random_seed)
        splitter_name = "StratifiedKFold"
        split_kwargs = {}

    # --- Build classifier ---
    if classifier_name == "LinearSVC":
        clf_core = LinearSVC(class_weight=class_weight if class_weight else None,
                             random_state=random_seed, max_iter=2000)
    else:
        clf_core = LogisticRegression(
            class_weight=class_weight if class_weight else None,
            max_iter=1000, random_state=random_seed, solver="lbfgs",
        )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
        )),
        ("clf", clf_core),
    ])

    # --- Out-of-fold predictions ---
    oof_preds = np.full(len(X), "", dtype=object)
    oof_scores = np.full(len(X), np.nan)

    try:
        for train_idx, test_idx in splitter.split(X, y, **split_kwargs):
            X_train = [X[i] for i in train_idx]
            y_train = y[train_idx]
            X_test = [X[i] for i in test_idx]

            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            oof_preds[test_idx] = preds

            # Scores for ranking (uncalibrated)
            if classifier_name == "LogisticRegression":
                proba = pipeline.predict_proba(X_test)
                classes_ = pipeline.classes_
                for local_j, global_j in enumerate(test_idx):
                    supplied = y[global_j]
                    if supplied in classes_:
                        cls_idx = list(classes_).index(supplied)
                        oof_scores[global_j] = proba[local_j, cls_idx]
            else:
                # LinearSVC: decision margin of supplied class minus best other
                decision = pipeline.decision_function(X_test)
                classes_ = pipeline.classes_
                for local_j, global_j in enumerate(test_idx):
                    supplied = y[global_j]
                    if supplied in classes_:
                        cls_idx = list(classes_).index(supplied)
                        if decision.ndim == 1:
                            oof_scores[global_j] = decision[local_j]
                        else:
                            best_other = max(
                                s for k, s in enumerate(decision[local_j]) if k != cls_idx
                            )
                            oof_scores[global_j] = float(decision[local_j, cls_idx]) - best_other

    except Exception as exc:
        return _cannot(f"Cross-validation failed: {exc}")

    # --- Update records ---
    predictions: dict[int, str] = {}
    scores: dict[int, float] = {}
    oof_correct = 0
    n_flagged = 0
    n_not_flagged = 0

    for i, rec in enumerate(eligible_records):
        pred = str(oof_preds[i])
        score = float(oof_scores[i]) if not np.isnan(oof_scores[i]) else None

        if pred == rec.supplied_label:
            rec.status = "not_flagged"
            oof_correct += 1
            n_not_flagged += 1
        else:
            rec.status = "flagged"
            n_flagged += 1

        rec.predicted_label = pred
        rec.model_score = score
        predictions[rec.csv_row] = pred
        if score is not None:
            scores[rec.csv_row] = score

    oof_accuracy = oof_correct / n_eligible if n_eligible > 0 else None
    low_acc_warning = oof_accuracy is not None and oof_accuracy < 0.6

    return LabelAnalysisResult(
        records=records,
        n_flagged=n_flagged,
        n_not_flagged=n_not_flagged,
        n_excluded=n_excluded,
        n_eligible=n_eligible,
        folds_used=folds_used,
        requested_folds=n_folds,
        splitter_used=splitter_name,
        splitter_fallback_reason=fallback_reason if not feasible else None,
        oof_accuracy=oof_accuracy,
        low_accuracy_warning=low_acc_warning,
        cannot_run=False,
        cannot_run_reason=None,
        predictions=predictions,
        scores=scores,
    )
