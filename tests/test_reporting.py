"""
test_reporting.py — Tests for src/reporting.py.

Verifies:
- Row report preserves csv_row and original columns.
- Boolean finding columns are present.
- Formula-injection sanitisation works.
- Encoding is UTF-8 BOM.
- Flagged report contains only flagged records.
- Config summary contains required keys.
"""
from __future__ import annotations

import io
import json
import pytest
import pandas as pd
import numpy as np

from src.audit import AuditConfig, run_audit
from src.data_loading import compute_file_fingerprint
from src.reporting import (
    build_row_report,
    build_flagged_report,
    build_pair_report,
    build_config_summary,
    df_to_csv_bytes,
    dict_to_json_bytes,
    _sanitize_cell,
    _sanitize_df,
)


# ---------------------------------------------------------------------------
# Helper: run a minimal audit
# ---------------------------------------------------------------------------

def _minimal_audit() -> "AuditResult":
    from src.audit import AuditConfig, AuditResult, run_audit
    classes = ["billing", "technical", "account"]
    rows = []
    for i, cls in enumerate(classes):
        for j in range(8):
            rows.append({
                "ticket": i * 8 + j + 1,
                "message": f"extended sample {cls} text item {j} with more words here",
                "category": cls,
            })
    df = pd.DataFrame(rows)
    df.insert(0, "csv_row", range(2, len(df) + 2))
    cfg = AuditConfig(text_col="message", label_col="category")
    fingerprint = "abc123"
    return run_audit(df, cfg, fingerprint)


# ---------------------------------------------------------------------------
# Formula injection sanitisation
# ---------------------------------------------------------------------------

class TestFormulaSanitization:
    def test_equals_sign_prefixed(self):
        assert _sanitize_cell("=SUM(A1:A10)") == "'=SUM(A1:A10)"

    def test_plus_sign_prefixed(self):
        assert _sanitize_cell("+100") == "'+100"

    def test_minus_sign_prefixed(self):
        assert _sanitize_cell("-200") == "'-200"

    def test_at_sign_prefixed(self):
        assert _sanitize_cell("@user") == "'@user"

    def test_normal_text_unchanged(self):
        assert _sanitize_cell("hello world") == "hello world"

    def test_none_unchanged(self):
        assert _sanitize_cell(None) is None

    def test_integer_unchanged(self):
        assert _sanitize_cell(42) == 42

    def test_sanitize_df_applies_to_all_string_cols(self):
        df = pd.DataFrame({
            "formula": ["=BAD", "normal", "+evil"],
            "number": [1, 2, 3],
        })
        sanitized = _sanitize_df(df)
        assert sanitized["formula"].iloc[0] == "'=BAD"
        assert sanitized["formula"].iloc[1] == "normal"
        assert sanitized["formula"].iloc[2] == "'+evil"
        assert sanitized["number"].iloc[0] == 1  # int unchanged


# ---------------------------------------------------------------------------
# Row report
# ---------------------------------------------------------------------------

class TestBuildRowReport:
    def test_csv_row_present(self):
        result = _minimal_audit()
        report = build_row_report(result, sanitize=False)
        assert "csv_row" in report.columns

    def test_original_columns_present(self):
        result = _minimal_audit()
        report = build_row_report(result, sanitize=False)
        assert "message" in report.columns
        assert "category" in report.columns
        assert "ticket" in report.columns

    def test_no_internal_columns(self):
        result = _minimal_audit()
        report = build_row_report(result, sanitize=False)
        assert "_norm_text" not in report.columns

    def test_boolean_finding_columns_present(self):
        result = _minimal_audit()
        report = build_row_report(result, sanitize=False)
        expected = [
            "is_missing_text", "is_missing_label", "is_exact_duplicate",
            "has_duplicate_label_conflict", "is_near_duplicate",
            "has_similar_label_conflict", "is_suspicious_label",
        ]
        for col in expected:
            assert col in report.columns, f"Missing column: {col}"

    def test_all_rows_included(self):
        result = _minimal_audit()
        report = build_row_report(result, sanitize=False)
        assert len(report) == result.exclusion.n_total_rows

    def test_review_reasons_column_present(self):
        result = _minimal_audit()
        report = build_row_report(result, sanitize=False)
        assert "review_reasons" in report.columns


# ---------------------------------------------------------------------------
# Flagged report
# ---------------------------------------------------------------------------

class TestBuildFlaggedReport:
    def test_flagged_only(self):
        result = _minimal_audit()
        flagged = build_flagged_report(result, sanitize=False)
        assert all(flagged["is_suspicious_label"])

    def test_subset_of_full_report(self):
        result = _minimal_audit()
        full = build_row_report(result, sanitize=False)
        flagged = build_flagged_report(result, sanitize=False)
        assert len(flagged) <= len(full)


# ---------------------------------------------------------------------------
# CSV bytes encoding
# ---------------------------------------------------------------------------

class TestCsvBytesEncoding:
    def test_utf8_bom_present(self):
        result = _minimal_audit()
        report = build_row_report(result, sanitize=False)
        raw = df_to_csv_bytes(report)
        # UTF-8 BOM is the first three bytes: EF BB BF
        assert raw[:3] == b"\xef\xbb\xbf"

    def test_readable_as_csv(self):
        result = _minimal_audit()
        report = build_row_report(result, sanitize=False)
        raw = df_to_csv_bytes(report)
        df_back = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
        assert "csv_row" in df_back.columns


# ---------------------------------------------------------------------------
# Config summary
# ---------------------------------------------------------------------------

class TestConfigSummary:
    def test_required_keys_present(self):
        result = _minimal_audit()
        summary = build_config_summary(result, {"pandas": "2.x", "scikit-learn": "1.x"})
        required_keys = [
            "dataset_fingerprint", "text_col", "label_col",
            "near_dup_threshold", "conflict_threshold",
            "n_folds_requested", "n_folds_used", "splitter_used",
            "random_seed", "pairwise_limit",
            "n_total_rows", "package_versions",
        ]
        for key in required_keys:
            assert key in summary, f"Missing key: {key}"

    def test_serializable_as_json(self):
        result = _minimal_audit()
        summary = build_config_summary(result, {"pandas": "2.x"})
        raw = dict_to_json_bytes(summary)
        parsed = json.loads(raw)
        assert "dataset_fingerprint" in parsed
