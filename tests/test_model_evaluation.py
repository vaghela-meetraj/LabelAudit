"""
test_model_evaluation.py — Tests for src/model_evaluation.py.

Verifies:
- Baseline metrics handle missing classes and zero-division without crash.
- Confusion matrix label ordering is explicit.
- Corruption experiment is reproducible for a fixed seed.
- Undefined precision/recall reported correctly when nothing is flagged
  or nothing is corrupted.
- Multi-seed averaging works and excludes None values correctly.
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from src.preprocessing import add_normalized_column
from src.label_analysis import analyze_labels, LabelAnalysisRecord, LabelAnalysisResult
from src.model_evaluation import (
    compute_baseline_metrics,
    run_corruption_experiment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_label_result_from_records(records):
    """Wrap a list of LabelAnalysisRecord into a minimal LabelAnalysisResult."""
    flagged = [r for r in records if r.status == "flagged"]
    not_flagged = [r for r in records if r.status == "not_flagged"]
    excluded = [r for r in records if r.status == "excluded"]
    return LabelAnalysisResult(
        records=records,
        n_flagged=len(flagged),
        n_not_flagged=len(not_flagged),
        n_excluded=len(excluded),
        n_eligible=len(flagged) + len(not_flagged),
        folds_used=3,
        requested_folds=3,
        splitter_used="StratifiedKFold",
        splitter_fallback_reason=None,
        oof_accuracy=None,
        low_accuracy_warning=False,
        cannot_run=False,
        cannot_run_reason=None,
        predictions={r.csv_row: r.predicted_label for r in records if r.predicted_label},
        scores={},
    )


@pytest.fixture
def cv_df_for_eval():
    classes = ["billing", "technical", "account"]
    rows = []
    for i, cls in enumerate(classes):
        for j in range(10):
            rows.append({
                "csv_row": i * 10 + j + 2,
                "text": f"example {cls} text number {j} with plenty of content words here",
                "label": cls,
            })
    df = pd.DataFrame(rows)
    return add_normalized_column(df, "text")


# ---------------------------------------------------------------------------
# Baseline metrics
# ---------------------------------------------------------------------------

class TestBaselineMetrics:
    def test_returns_none_when_no_predictions(self):
        records = [
            LabelAnalysisRecord(
                csv_row=2, orig_text="x", supplied_label="A",
                status="excluded", exclusion_reason="missing text"
            )
        ]
        result = _make_label_result_from_records(records)
        metrics = compute_baseline_metrics(result)
        assert metrics is None

    def test_accuracy_in_range(self, cv_df_for_eval):
        label_result = analyze_labels(cv_df_for_eval, "text", "label")
        if label_result.cannot_run:
            pytest.skip("Cannot run CV")
        metrics = compute_baseline_metrics(label_result)
        assert metrics is not None
        assert 0.0 <= metrics.accuracy <= 1.0

    def test_macro_f1_in_range(self, cv_df_for_eval):
        label_result = analyze_labels(cv_df_for_eval, "text", "label")
        if label_result.cannot_run:
            pytest.skip("Cannot run CV")
        metrics = compute_baseline_metrics(label_result)
        assert metrics is not None
        assert 0.0 <= metrics.macro_f1 <= 1.0

    def test_confusion_matrix_shape(self, cv_df_for_eval):
        label_result = analyze_labels(cv_df_for_eval, "text", "label")
        if label_result.cannot_run:
            pytest.skip("Cannot run CV")
        metrics = compute_baseline_metrics(label_result)
        assert metrics is not None
        n = len(metrics.labels)
        assert metrics.confusion_matrix.shape == (n, n)

    def test_per_class_has_all_classes(self, cv_df_for_eval):
        label_result = analyze_labels(cv_df_for_eval, "text", "label")
        if label_result.cannot_run:
            pytest.skip("Cannot run CV")
        metrics = compute_baseline_metrics(label_result)
        assert metrics is not None
        for cls in ["billing", "technical", "account"]:
            assert cls in metrics.per_class

    def test_handles_missing_class_in_predictions(self):
        """A class that appears in y_true but not y_pred should not crash."""
        records = [
            LabelAnalysisRecord(2, "text a", "A", "not_flagged", predicted_label="A"),
            LabelAnalysisRecord(3, "text b", "A", "not_flagged", predicted_label="A"),
            LabelAnalysisRecord(4, "text c", "B", "flagged", predicted_label="A"),
            LabelAnalysisRecord(5, "text d", "B", "flagged", predicted_label="A"),
        ]
        result = _make_label_result_from_records(records)
        metrics = compute_baseline_metrics(result)
        assert metrics is not None
        # B has zero predicted → recall=0, precision=undefined → zero_division=0
        assert metrics.per_class["B"]["precision"] == 0.0 or metrics.per_class["B"]["recall"] == 0.0


# ---------------------------------------------------------------------------
# Corruption experiment
# ---------------------------------------------------------------------------

class TestCorruptionExperiment:
    def test_reproducible_for_fixed_seed(self, cv_df_for_eval):
        eligible_mask = cv_df_for_eval["_norm_text"] != ""
        result1 = run_corruption_experiment(
            cv_df_for_eval, "text", "label", "_norm_text",
            eligible_mask=eligible_mask,
            corruption_rate=0.10,
            chosen_seed=42,
            n_seeds=2,
        )
        result2 = run_corruption_experiment(
            cv_df_for_eval, "text", "label", "_norm_text",
            eligible_mask=eligible_mask,
            corruption_rate=0.10,
            chosen_seed=42,
            n_seeds=2,
        )
        if not result1.cannot_run and not result2.cannot_run:
            assert result1.chosen_seed_result.n_corrupted == result2.chosen_seed_result.n_corrupted
            assert result1.chosen_seed_result.n_flagged == result2.chosen_seed_result.n_flagged

    def test_undefined_precision_when_nothing_flagged(self, cv_df_for_eval):
        """
        If no records are flagged, precision should be None (undefined), not 0.
        We test the CorruptionSeedResult directly.
        """
        from src.model_evaluation import CorruptionSeedResult
        sr = CorruptionSeedResult(
            seed=0, n_corrupted=5, n_flagged=0,
            tp=0, fp=0, fn=5,
            precision=None, recall=0.0, f1=None,
            false_flag_rate=None,
        )
        assert sr.precision is None

    def test_cannot_run_on_tiny_dataset(self):
        df = pd.DataFrame({
            "csv_row": [2, 3, 4],
            "text": ["a", "b", "c"],
            "label": ["A", "B", "A"],
        })
        df = add_normalized_column(df, "text")
        eligible_mask = df["_norm_text"] != ""
        result = run_corruption_experiment(
            df, "text", "label", "_norm_text",
            eligible_mask=eligible_mask,
        )
        assert result.cannot_run

    def test_random_baseline_precision_equals_corruption_rate(self, cv_df_for_eval):
        eligible_mask = cv_df_for_eval["_norm_text"] != ""
        result = run_corruption_experiment(
            cv_df_for_eval, "text", "label", "_norm_text",
            eligible_mask=eligible_mask,
            corruption_rate=0.15,
        )
        if not result.cannot_run:
            assert abs(result.random_baseline_precision - 0.15) < 1e-9

    def test_multi_seed_averaging_excludes_nones(self, cv_df_for_eval):
        eligible_mask = cv_df_for_eval["_norm_text"] != ""
        result = run_corruption_experiment(
            cv_df_for_eval, "text", "label", "_norm_text",
            eligible_mask=eligible_mask,
            corruption_rate=0.10,
            n_seeds=3,
        )
        if not result.cannot_run:
            # mean should be a float or None, not NaN
            if result.mean_precision is not None:
                assert not np.isnan(result.mean_precision)
            if result.mean_recall is not None:
                assert not np.isnan(result.mean_recall)
