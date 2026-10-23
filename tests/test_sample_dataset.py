"""
test_sample_dataset.py — Asserts exact counts from PLANTED_ISSUES.md.

This test uses the actual sample CSV file and verifies that the audit pipeline
produces the planted issues documented in PLANTED_ISSUES.md.

The sample dataset uses:
  text_col  = "message"
  label_col = "category"
"""
from __future__ import annotations

import pathlib
import pytest
import pandas as pd

from src.data_loading import load_csv
from src.audit import AuditConfig, run_audit
from src.preprocessing import add_normalized_column


SAMPLE_CSV = pathlib.Path(__file__).parent.parent / "sample_data" / "example_dataset.csv"


@pytest.fixture(scope="module")
def sample_result():
    raw = SAMPLE_CSV.read_bytes()
    load_result = load_csv(raw, "example_dataset.csv")
    assert not load_result.errors, f"Load errors: {load_result.errors}"
    cfg = AuditConfig(
        text_col="message",
        label_col="category",
        near_dup_threshold=0.80,
        conflict_threshold=0.70,
        use_char_ngrams=True,
        n_folds=3,
        min_class_size=3,
        pairwise_limit=5000,
    )
    fingerprint = "sample"
    return run_audit(load_result.df, cfg, fingerprint)


class TestSampleDatasetCounts:
    def test_total_rows(self, sample_result):
        """Should have 96 data rows (header is row 1)."""
        assert sample_result.exclusion.n_total_rows == 96

    def test_missing_text_count(self, sample_result):
        """Exactly csv_rows 79 and 97 have empty message cells."""
        assert sample_result.exclusion.n_missing_text == 2

    def test_missing_label_count(self, sample_result):
        """No rows have missing labels."""
        assert sample_result.exclusion.n_missing_label == 0

    def test_exact_duplicate_groups_found(self, sample_result):
        """The three documented exact-duplicate groups are found."""
        assert sample_result.exact_dup.n_groups == 3
        memberships = sorted(
            sorted(group.members.csv_row.tolist())
            for group in sample_result.exact_dup.groups
        )
        assert memberships == [[2, 5, 83], [3, 15], [4, 8, 18]]

    def test_conflicting_label_group_found(self, sample_result):
        """At least 1 group has label conflicts."""
        conflicting_groups = [g for g in sample_result.exact_dup.groups if g.label_conflict]
        assert len(conflicting_groups) == 1
        assert conflicting_groups[0].members.csv_row.tolist() == [2, 5, 83]

    def test_affected_records_gt_groups(self, sample_result):
        """The documented groups contain exactly eight affected records."""
        assert sample_result.exact_dup.n_affected_records == 8

    def test_redundant_equals_affected_minus_groups(self, sample_result):
        """Exact accounting: redundant = affected − groups."""
        assert (
            sample_result.exact_dup.n_redundant_records
            == sample_result.exact_dup.n_affected_records - sample_result.exact_dup.n_groups
            == 5
        )

    def test_near_dup_pairs_found(self, sample_result):
        """At least some near-duplicate or conflict pairs detected."""
        if sample_result.near_dup_skipped:
            pytest.skip("Near-dup skipped due to limit")
        assert sample_result.near_dup is not None
        # The documented technical pair is the only finding at these thresholds.
        assert sample_result.near_dup.n_pairs_total == 1
        assert sample_result.near_dup.n_near_dup == 1
        assert sample_result.near_dup.n_conflicts == 0
        pair = sample_result.near_dup.pairs[0]
        assert (pair.csv_row_a, pair.csv_row_b) == (2, 25)
        for pair in sample_result.near_dup.pairs:
            assert sample_result.row_to_group[pair.csv_row_a] == sample_result.row_to_group[pair.csv_row_b]

    def test_label_analysis_runs(self, sample_result):
        """Method C should run (enough classes and records)."""
        la = sample_result.label_analysis
        assert not la.cannot_run, f"Label analysis cannot run: {la.cannot_run_reason}"

    def test_classes_are_billing_technical_account(self, sample_result):
        """Three expected classes present."""
        raw = SAMPLE_CSV.read_bytes()
        load_result = load_csv(raw, "example_dataset.csv")
        df = load_result.df
        classes = set(df["category"].unique()) - {""}
        assert "billing" in classes
        assert "technical" in classes
        assert "account" in classes

    def test_three_member_exact_dup_group_exists(self, sample_result):
        """There is at least one group with ≥ 3 members (the planted 3+1 group)."""
        large_groups = [g for g in sample_result.exact_dup.groups if len(g.members) >= 3]
        assert len(large_groups) >= 1

    def test_column_names_not_assumed(self, sample_result):
        """The audit ran with non-default column names (message, category)."""
        assert sample_result.config.text_col == "message"
        assert sample_result.config.label_col == "category"
