"""
model_evaluation.py — Baseline classifier metrics and the controlled corruption experiment.

Baseline metrics
----------------
Computed from the out-of-fold predictions produced by label_analysis.py.
These measure how well a simple TF-IDF classifier generalises on the dataset,
NOT how accurate the audit flags are.

Metrics: accuracy, macro-F1, per-class precision/recall/F1, confusion matrix.
- zero_division=0 throughout (missing-class predictions do not crash or shift matrix).
- Label ordering is explicit (sorted list of all eligible classes).

Corruption experiment (SYNTHETIC / EXPERIMENTAL)
-------------------------------------------------
Purpose: to estimate, on the sample data, how well Method C can detect randomly
planted label errors.  This does NOT measure real-world label-error detection
accuracy because real errors are systematic and ambiguous, not random.

Steps:
1. Take eligible records (no conflicting exact/near-dup pairs).
2. Randomly replace `corruption_rate` fraction of labels with a different class.
3. Run Method C on corrupted labels.
4. Compare flags with planted corruptions.
5. Repeat for several seeds; report mean and std.
6. Report a random-flag baseline (expected precision = corruption_rate).
7. Report false-flag rate among uncorrupted records.

Zero-division: if nothing is flagged, precision = undefined; if nothing corrupted,
recall = undefined.  In exports use empty + note column, not 0/1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .label_analysis import analyze_labels, LabelAnalysisResult


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BaselineMetrics:
    accuracy: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]   # {label: {precision, recall, f1, support}}
    confusion_matrix: np.ndarray
    labels: list[str]                         # Explicit ordering for the matrix


@dataclass
class CorruptionSeedResult:
    seed: int
    n_corrupted: int
    n_flagged: int
    tp: int                      # Flagged AND corrupted
    fp: int                      # Flagged AND NOT corrupted
    fn: int                      # NOT flagged AND corrupted
    precision: Optional[float]   # None = undefined (nothing flagged)
    recall: Optional[float]      # None = undefined (nothing corrupted)
    f1: Optional[float]
    false_flag_rate: Optional[float]   # fp / (n_eligible - n_corrupted)


@dataclass
class CorruptionExperiment:
    corruption_rate: float
    n_eligible: int
    seeds_results: list[CorruptionSeedResult]
    chosen_seed_result: CorruptionSeedResult
    # Averaged over all seeds (exclude None)
    mean_precision: Optional[float]
    std_precision: Optional[float]
    mean_recall: Optional[float]
    std_recall: Optional[float]
    mean_f1: Optional[float]
    std_f1: Optional[float]
    random_baseline_precision: float   # = corruption_rate
    cannot_run: bool
    cannot_run_reason: Optional[str]


# ---------------------------------------------------------------------------
# Baseline metrics
# ---------------------------------------------------------------------------

def compute_baseline_metrics(
    label_result: LabelAnalysisResult,
) -> Optional[BaselineMetrics]:
    """
    Compute metrics from out-of-fold predictions.
    Returns None if there are no eligible predictions.
    """
    eligible = [
        r for r in label_result.records
        if r.status in {"flagged", "not_flagged"} and r.predicted_label is not None
    ]
    if not eligible:
        return None

    y_true = [r.supplied_label for r in eligible]
    y_pred = [r.predicted_label for r in eligible]
    labels = sorted(set(y_true))

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)

    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class: dict[str, dict[str, float]] = {}
    for i, lbl in enumerate(labels):
        per_class[lbl] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": float(sup[i]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return BaselineMetrics(
        accuracy=acc,
        macro_f1=macro_f1,
        per_class=per_class,
        confusion_matrix=cm,
        labels=labels,
    )


# ---------------------------------------------------------------------------
# Corruption experiment
# ---------------------------------------------------------------------------

def _plant_corruption(
    df: pd.DataFrame,
    label_col: str,
    eligible_mask: pd.Series,
    rate: float,
    seed: int,
    exclude_csv_rows: Optional[set[int]] = None,
) -> tuple[pd.DataFrame, set[int]]:
    """
    Randomly replace `rate` fraction of eligible labels with a different class.
    Returns (modified_df, set_of_corrupted_csv_rows).
    exclude_csv_rows: records not to corrupt (e.g. from conflicting-dup groups).
    """
    rng = np.random.RandomState(seed)
    df_corrupt = df.copy()
    exclude_csv_rows = exclude_csv_rows or set()

    elig = df[eligible_mask & ~df["csv_row"].isin(exclude_csv_rows)].copy()
    all_labels = sorted(elig[label_col].unique())
    if len(all_labels) < 2:
        return df_corrupt, set()

    n_to_corrupt = max(1, int(len(elig) * rate))
    chosen_idx = rng.choice(len(elig), size=n_to_corrupt, replace=False)
    chosen_rows = elig.iloc[chosen_idx]

    corrupted_csv_rows: set[int] = set()
    for _, row in chosen_rows.iterrows():
        original_label = row[label_col]
        other_labels = [l for l in all_labels if l != original_label]
        new_label = rng.choice(other_labels)
        df_corrupt.loc[df_corrupt["csv_row"] == row["csv_row"], label_col] = new_label
        corrupted_csv_rows.add(int(row["csv_row"]))

    return df_corrupt, corrupted_csv_rows


def _run_single_corruption(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    norm_col: str,
    eligible_mask: pd.Series,
    rate: float,
    seed: int,
    exclude_csv_rows: Optional[set[int]],
    classifier_name: str,
    class_weight: str,
    n_folds: int,
    random_seed: int,
    min_class_size: int,
) -> CorruptionSeedResult:
    df_corrupt, corrupted_rows = _plant_corruption(
        df, label_col, eligible_mask, rate, seed, exclude_csv_rows
    )
    n_corrupted = len(corrupted_rows)

    result = analyze_labels(
        df_corrupt,
        text_col=text_col,
        label_col=label_col,
        norm_col=norm_col,
        classifier_name=classifier_name,
        class_weight=class_weight,
        n_folds=n_folds,
        random_seed=random_seed,
        min_class_size=min_class_size,
    )

    flagged_rows = {r.csv_row for r in result.records if r.status == "flagged"}
    n_flagged = len(flagged_rows)
    n_eligible_total = len([r for r in result.records if r.status != "excluded"])

    if n_corrupted == 0:
        return CorruptionSeedResult(
            seed=seed, n_corrupted=0, n_flagged=n_flagged,
            tp=0, fp=n_flagged, fn=0,
            precision=None, recall=None, f1=None,
            false_flag_rate=(n_flagged / (n_eligible_total - 0)) if n_eligible_total > 0 else None,
        )

    tp = len(flagged_rows & corrupted_rows)
    fp = n_flagged - tp
    fn = n_corrupted - tp
    n_clean = n_eligible_total - n_corrupted

    prec = tp / n_flagged if n_flagged > 0 else None
    rec = tp / n_corrupted if n_corrupted > 0 else None
    f1_val: Optional[float]
    if prec is not None and rec is not None and (prec + rec) > 0:
        f1_val = 2 * prec * rec / (prec + rec)
    else:
        f1_val = None

    ffr = fp / n_clean if n_clean > 0 else None

    return CorruptionSeedResult(
        seed=seed, n_corrupted=n_corrupted, n_flagged=n_flagged,
        tp=tp, fp=fp, fn=fn,
        precision=prec, recall=rec, f1=f1_val,
        false_flag_rate=ffr,
    )


def run_corruption_experiment(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    norm_col: str,
    eligible_mask: pd.Series,
    corruption_rate: float = 0.10,
    chosen_seed: int = 42,
    n_seeds: int = 5,
    exclude_csv_rows: Optional[set[int]] = None,
    classifier_name: str = "LogisticRegression",
    class_weight: str = "balanced",
    n_folds: int = 5,
    min_class_size: int = 3,
) -> CorruptionExperiment:
    """
    Run the controlled corruption experiment over multiple seeds.

    Parameters
    ----------
    eligible_mask : boolean mask on df rows that are eligible for corruption.
    exclude_csv_rows : rows to exclude from corruption (e.g. conflicting dup groups).
    """
    n_eligible = int(eligible_mask.sum())

    if n_eligible < 10:
        return CorruptionExperiment(
            corruption_rate=corruption_rate,
            n_eligible=n_eligible,
            seeds_results=[], chosen_seed_result=None,  # type: ignore[arg-type]
            mean_precision=None, std_precision=None,
            mean_recall=None, std_recall=None,
            mean_f1=None, std_f1=None,
            random_baseline_precision=corruption_rate,
            cannot_run=True,
            cannot_run_reason=(
                f"Only {n_eligible} eligible records — need at least 10 to run "
                "the corruption experiment."
            ),
        )

    # Use seeds 0, 1, 2, … for the multi-seed average, plus the chosen seed
    seeds_to_run = list(range(n_seeds))
    if chosen_seed not in seeds_to_run:
        seeds_to_run.append(chosen_seed)

    seeds_results: list[CorruptionSeedResult] = []
    for seed in seeds_to_run:
        sr = _run_single_corruption(
            df=df, text_col=text_col, label_col=label_col, norm_col=norm_col,
            eligible_mask=eligible_mask, rate=corruption_rate, seed=seed,
            exclude_csv_rows=exclude_csv_rows,
            classifier_name=classifier_name, class_weight=class_weight,
            n_folds=n_folds, random_seed=seed, min_class_size=min_class_size,
        )
        seeds_results.append(sr)

    chosen_result = next((s for s in seeds_results if s.seed == chosen_seed), seeds_results[-1])

    # Average over the first n_seeds runs (deterministic)
    avg_results = [s for s in seeds_results if s.seed in range(n_seeds)]

    def _mean_std(vals: list[Optional[float]]) -> tuple[Optional[float], Optional[float]]:
        valid = [v for v in vals if v is not None]
        if not valid:
            return None, None
        arr = np.array(valid, dtype=float)
        return float(arr.mean()), float(arr.std())

    mean_p, std_p = _mean_std([s.precision for s in avg_results])
    mean_r, std_r = _mean_std([s.recall for s in avg_results])
    mean_f, std_f = _mean_std([s.f1 for s in avg_results])

    return CorruptionExperiment(
        corruption_rate=corruption_rate,
        n_eligible=n_eligible,
        seeds_results=seeds_results,
        chosen_seed_result=chosen_result,
        mean_precision=mean_p, std_precision=std_p,
        mean_recall=mean_r, std_recall=std_r,
        mean_f1=mean_f, std_f1=std_f,
        random_baseline_precision=corruption_rate,
        cannot_run=False,
        cannot_run_reason=None,
    )
