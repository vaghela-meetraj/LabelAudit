"""
reporting.py — CSV/JSON export builders for the LabelAudit audit results.

All CSV exports:
- UTF-8 with BOM (utf-8-sig) so spreadsheet software opens files correctly.
- Formula-injection sanitisation: if the toggle is on, cells that begin with
  =, +, -, or @ are prefixed with an apostrophe (standard spreadsheet defence).
  This is documented in the config summary.

Row-level report columns (in order):
  All original columns (original names and order), csv_row,
  is_missing_text, is_missing_label, is_exact_duplicate,
  has_duplicate_label_conflict, is_near_duplicate, has_similar_label_conflict,
  is_suspicious_label, duplicate_group_id, label_analysis_status,
  predicted_label_oof, model_score, review_reasons.

Internal columns (_norm_text etc.) are excluded from exports.
"""

from __future__ import annotations

import io
import json
from typing import Optional

import numpy as np
import pandas as pd

from .audit import AuditResult, AuditConfig
from .duplicate_detection import PairFinding, pairs_to_dataframe


# ---------------------------------------------------------------------------
# Formula injection guard
# ---------------------------------------------------------------------------

_FORMULA_PREFIXES = ("=", "+", "-", "@")


def _sanitize_cell(val: object) -> object:
    """Prefix formula-like strings with an apostrophe."""
    if isinstance(val, str) and val.startswith(_FORMULA_PREFIXES):
        return "'" + val
    return val


def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply formula injection sanitisation to all string columns."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(_sanitize_cell)
    return df


# ---------------------------------------------------------------------------
# Internal columns to exclude from exports
# ---------------------------------------------------------------------------

_INTERNAL_COLS = {"_norm_text"}


# ---------------------------------------------------------------------------
# Row-level audit report
# ---------------------------------------------------------------------------

def build_row_report(
    result: AuditResult,
    sanitize: bool = True,
) -> pd.DataFrame:
    """
    Build the complete row-level audit report with Boolean finding columns.

    Finding columns
    ---------------
    is_missing_text          : True when normalised text is empty.
    is_missing_label         : True when label cell is blank.
    is_exact_duplicate       : True when record belongs to any exact-dup group.
    has_duplicate_label_conflict : True when exact-dup group has label conflict.
    is_near_duplicate        : True when record appears in a near-dup pair.
    has_similar_label_conflict   : True when record appears in a conflict pair.
    is_suspicious_label      : True when OOF prediction ≠ supplied label.
    """
    df = result.df_with_norm.copy()
    cfg = result.config

    # Drop internal columns
    orig_cols = [c for c in df.columns if c not in _INTERNAL_COLS and c != "csv_row"]

    # Reconstruct with original column order: orig cols first, then csv_row, then findings
    # (csv_row was inserted first during loading; re-order so orig cols appear first)
    all_export_cols = orig_cols + ["csv_row"]

    # --- Boolean finding columns ---
    n = len(df)
    is_missing_text = df["_norm_text"] == ""
    is_missing_label = df[cfg.label_col].apply(
        lambda v: not isinstance(v, str) or v.strip() == ""
    )

    exact_dup_rows = set(result.exact_dup.row_to_group.keys())
    conflict_group_ids = {
        g.group_id for g in result.exact_dup.groups if g.label_conflict
    }
    row_to_group = result.exact_dup.row_to_group

    is_exact_dup = df["csv_row"].apply(lambda r: r in exact_dup_rows)
    has_dup_conflict = df["csv_row"].apply(
        lambda r: row_to_group.get(r) in conflict_group_ids
    )

    # Near-dup / conflict pair sets
    near_dup_rows: set[int] = set()
    conflict_pair_rows: set[int] = set()
    if result.near_dup is not None:
        for p in result.near_dup.pairs:
            if p.finding_type in {"near_duplicate", "near_dup_conflict"}:
                near_dup_rows.add(p.csv_row_a)
                near_dup_rows.add(p.csv_row_b)
            if p.finding_type in {"conflicting_label", "near_dup_conflict"}:
                conflict_pair_rows.add(p.csv_row_a)
                conflict_pair_rows.add(p.csv_row_b)

    is_near_dup = df["csv_row"].apply(lambda r: r in near_dup_rows)
    has_similar_conflict = df["csv_row"].apply(lambda r: r in conflict_pair_rows)

    # Label analysis
    status_map = {r.csv_row: r.status for r in result.label_analysis.records}
    pred_map = result.label_analysis.predictions
    score_map = result.label_analysis.scores

    is_suspicious = df["csv_row"].apply(
        lambda r: status_map.get(r, "excluded") == "flagged"
    )
    label_status = df["csv_row"].apply(
        lambda r: status_map.get(r, "excluded")
    )
    pred_label = df["csv_row"].apply(
        lambda r: pred_map.get(r, "")
    )
    model_score = df["csv_row"].apply(
        lambda r: score_map.get(r, np.nan)
    )

    # Duplicate group id
    dup_group_id = df["csv_row"].apply(
        lambda r: row_to_group.get(r, np.nan)
    )

    # Review reasons: concise human-readable summary
    def _reasons(row_idx: int) -> str:
        r = df.loc[row_idx, "csv_row"]
        reasons = []
        if is_missing_text.iloc[row_idx]:
            reasons.append("missing text")
        if is_missing_label.iloc[row_idx]:
            reasons.append("missing label")
        if is_exact_dup.iloc[row_idx]:
            gid = row_to_group.get(r)
            conf = gid in conflict_group_ids
            reasons.append(f"exact duplicate (group {gid})" + (" [label conflict]" if conf else ""))
        if is_near_dup.iloc[row_idx] and not is_exact_dup.iloc[row_idx]:
            reasons.append("near-duplicate text")
        if has_similar_conflict.iloc[row_idx]:
            reasons.append("similar text, different label")
        if is_suspicious.iloc[row_idx]:
            reasons.append(f"OOF predicted '{pred_label.iloc[row_idx]}'")
        return "; ".join(reasons) if reasons else ""

    df_out = df[all_export_cols].copy()
    df_out["is_missing_text"] = is_missing_text.values
    df_out["is_missing_label"] = is_missing_label.values
    df_out["is_exact_duplicate"] = is_exact_dup.values
    df_out["has_duplicate_label_conflict"] = has_dup_conflict.values
    df_out["is_near_duplicate"] = is_near_dup.values
    df_out["has_similar_label_conflict"] = has_similar_conflict.values
    df_out["is_suspicious_label"] = is_suspicious.values
    df_out["duplicate_group_id"] = dup_group_id.values
    df_out["label_analysis_status"] = label_status.values
    df_out["predicted_label_oof"] = pred_label.values
    df_out["model_score"] = model_score.values
    df_out["review_reasons"] = [_reasons(i) for i in range(n)]

    if sanitize:
        df_out = _sanitize_df(df_out)

    return df_out


def build_flagged_report(result: AuditResult, sanitize: bool = True) -> pd.DataFrame:
    """Subset of the row report containing only flagged records."""
    full = build_row_report(result, sanitize=sanitize)
    return full[full["is_suspicious_label"]].reset_index(drop=True)


def build_pair_report(result: AuditResult, sanitize: bool = True) -> pd.DataFrame:
    """Near-duplicate and conflict pair table."""
    if result.near_dup is None:
        return pd.DataFrame(columns=[
            "csv_row_a", "csv_row_b", "text_a", "text_b",
            "label_a", "label_b", "similarity", "finding_type", "reason",
        ])
    df = pairs_to_dataframe(result.near_dup.pairs)
    if sanitize:
        df = _sanitize_df(df)
    return df


def build_config_summary(result: AuditResult, package_versions: dict[str, str]) -> dict:
    """Configuration summary dict for JSON export."""
    la = result.label_analysis
    return {
        "dataset_fingerprint": result.dataset_fingerprint,
        "text_col": result.config.text_col,
        "label_col": result.config.label_col,
        "near_dup_threshold": result.config.near_dup_threshold,
        "conflict_threshold": result.config.conflict_threshold,
        "use_char_ngrams": result.config.use_char_ngrams,
        "classifier": result.config.classifier_name,
        "class_weight": result.config.class_weight,
        "n_folds_requested": result.config.n_folds,
        "n_folds_used": la.folds_used,
        "splitter_used": la.splitter_used,
        "random_seed": result.config.random_seed,
        "min_class_size": result.config.min_class_size,
        "pairwise_limit": result.config.pairwise_limit,
        "n_total_rows": result.exclusion.n_total_rows,
        "n_missing_text": result.exclusion.n_missing_text,
        "n_missing_label": result.exclusion.n_missing_label,
        "n_usable": result.exclusion.n_usable,
        "n_exact_dup_groups": result.exact_dup.n_groups,
        "n_exact_dup_affected": result.exact_dup.n_affected_records,
        "n_near_dup_pairs": result.near_dup.n_pairs_total if result.near_dup else None,
        "n_label_flagged": la.n_flagged,
        "n_label_eligible": la.n_eligible,
        "oof_accuracy": la.oof_accuracy,
        "package_versions": package_versions,
    }


# ---------------------------------------------------------------------------
# Bytes helpers for Streamlit download buttons
# ---------------------------------------------------------------------------

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Encode DataFrame to UTF-8 BOM CSV bytes."""
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def dict_to_json_bytes(d: dict) -> bytes:
    """Encode dict to UTF-8 JSON bytes."""
    return json.dumps(d, indent=2, default=str).encode("utf-8")
