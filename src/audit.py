"""
audit.py — Orchestrator for the full LabelAudit pipeline.

This module is the single entry point for analysis.  Both app.py (Streamlit)
and the test suite call run_audit() — no analysis logic lives elsewhere.

AuditConfig holds all user-configurable parameters.
AuditResult holds all outputs, keyed by the dataset fingerprint + config hash
so cached results can be validated before reuse.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Iterable, Optional

import pandas as pd
import numpy as np

from .preprocessing import add_normalized_column, build_exclusion_mask
from .duplicate_detection import (
    find_exact_duplicates,
    find_near_duplicates,
    ExactDupResult,
    NearDupResult,
)
from .label_analysis import analyze_labels, LabelAnalysisResult
from .model_evaluation import compute_baseline_metrics, BaselineMetrics


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AuditConfig:
    text_col: str
    label_col: str
    near_dup_threshold: float = 0.90
    conflict_threshold: float = 0.80
    use_char_ngrams: bool = True
    similarity_method: str = "tfidf"   # "semantic" or "tfidf"
    classifier_name: str = "LogisticRegression"   # or "LinearSVC"
    class_weight: str = "balanced"                 # or ""
    n_folds: int = 5
    random_seed: int = 42
    min_class_size: int = 3
    pairwise_limit: int = 5000

    def config_hash(self) -> str:
        """Stable hash of every config parameter, including selected columns."""
        d = {k: v for k, v in self.__dict__.items()}
        s = json.dumps(d, sort_keys=True)
        return hashlib.sha256(s.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class ExclusionSummary:
    n_missing_text: int = 0
    n_missing_label: int = 0
    n_total_rows: int = 0
    n_usable: int = 0
    # For Method B
    n_over_pairwise_limit: int = 0
    pairwise_limit_hit: bool = False
    # For Method C
    n_small_class_excluded: int = 0


@dataclass
class AuditResult:
    config: AuditConfig
    dataset_fingerprint: str         # SHA-256 of raw bytes
    df_with_norm: pd.DataFrame       # Original columns + csv_row + _norm_text

    # Exclusion accounting
    exclusion: ExclusionSummary

    # Method A
    exact_dup: ExactDupResult

    # Method B (None if pairwise_limit exceeded and user did not subsample)
    near_dup: Optional[NearDupResult]
    near_dup_skipped: bool
    near_dup_skip_reason: str

    # Method C
    label_analysis: LabelAnalysisResult

    # Evaluation
    baseline_metrics: Optional[BaselineMetrics]

    # Mapping csv_row → group used for cross-validation grouping
    row_to_group: dict[int, int]


def _build_cv_group_mapping(
    exact_row_to_group: dict[int, int],
    similar_pair_edges: Iterable[tuple[int, int]],
) -> dict[int, int]:
    """Map connected exact/near-duplicate rows to shared cross-validation groups."""
    parent: dict[int, int] = {}

    def find(row: int) -> int:
        parent.setdefault(row, row)
        if parent[row] != row:
            parent[row] = find(parent[row])
        return parent[row]

    def union(row_a: int, row_b: int) -> None:
        root_a = find(row_a)
        root_b = find(row_b)
        if root_a != root_b:
            parent[root_b] = root_a

    first_row_by_exact_group: dict[int, int] = {}
    for row, group_id in exact_row_to_group.items():
        row = int(row)
        group_id = int(group_id)
        first_row = first_row_by_exact_group.setdefault(group_id, row)
        union(first_row, row)

    for row_a, row_b in similar_pair_edges:
        union(int(row_a), int(row_b))

    components: dict[int, list[int]] = {}
    for row in parent:
        components.setdefault(find(row), []).append(row)

    ordered_components = sorted(
        (sorted(rows) for rows in components.values() if len(rows) > 1),
        key=lambda rows: rows[0],
    )
    return {
        row: group_id
        for group_id, rows in enumerate(ordered_components, start=1)
        for row in rows
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_audit(
    df: pd.DataFrame,
    config: AuditConfig,
    dataset_fingerprint: str,
    progress_callback=None,   # Optional callable(step: int, message: str)
) -> AuditResult:
    """
    Run the full audit pipeline and return an AuditResult.

    Parameters
    ----------
    df : DataFrame produced by load_csv() (includes csv_row).
    config : AuditConfig with all user selections.
    dataset_fingerprint : SHA-256 from compute_file_fingerprint().
    progress_callback : optional fn(step, msg) for streaming progress to UI.
    """

    def _progress(step: int, msg: str) -> None:
        if progress_callback:
            progress_callback(step, msg)

    _progress(0, "Preprocessing text…")

    # --- 1. Normalisation ---
    df_norm = add_normalized_column(df, config.text_col, norm_col="_norm_text")

    # --- 2. Exclusion accounting ---
    excl_mask = build_exclusion_mask(
        df_norm, config.text_col, config.label_col, norm_col="_norm_text"
    )
    n_missing_text = int(
        df_norm["_norm_text"].apply(lambda v: v == "").sum()
    )
    n_missing_label = int(
        df_norm[config.label_col].apply(
            lambda v: not isinstance(v, str) or v.strip() == ""
        ).sum()
    )
    n_usable = int((~excl_mask).sum())

    excl = ExclusionSummary(
        n_missing_text=n_missing_text,
        n_missing_label=n_missing_label,
        n_total_rows=len(df_norm),
        n_usable=n_usable,
    )

    _progress(1, "Finding exact duplicates…")

    # --- 3. Method A — Exact duplicates ---
    exact_dup_result = find_exact_duplicates(
        df_norm, config.text_col, config.label_col, norm_col="_norm_text"
    )

    exact_row_to_group = dict(exact_dup_result.row_to_group)

    _progress(2, "Analysing near-duplicates and conflicts…")

    # --- 4. Method B — Near-duplicates ---
    near_dup_skipped = False
    near_dup_skip_reason = ""
    near_dup_result: Optional[NearDupResult] = None

    if n_usable > config.pairwise_limit:
        near_dup_skipped = True
        near_dup_skip_reason = (
            f"The dataset has {n_usable:,} usable records, which exceeds the "
            f"pairwise analysis limit of {config.pairwise_limit:,}. "
            "Near-duplicate analysis was skipped. To proceed, either lower the limit "
            "(may be slow), analyze a random subsample, or restrict to a label subset."
        )
        excl.pairwise_limit_hit = True
        excl.n_over_pairwise_limit = n_usable - config.pairwise_limit
    else:
        near_dup_result = find_near_duplicates(
            df_norm,
            text_col=config.text_col,
            label_col=config.label_col,
            norm_col="_norm_text",
            near_dup_threshold=config.near_dup_threshold,
            conflict_threshold=config.conflict_threshold,
            use_char=config.use_char_ngrams,
            pairwise_limit=config.pairwise_limit,
            exact_dup_result=exact_dup_result,
            similarity_method=config.similarity_method,
        )
        if near_dup_result.hit_pair_cap:
            near_dup_skip_reason = (
                "The pair storage cap was reached. Results shown are partial; "
                "lower the thresholds or reduce the dataset for a complete analysis."
            )

    similar_pair_edges = (
        (pair.csv_row_a, pair.csv_row_b) for pair in near_dup_result.pairs
    ) if near_dup_result is not None else ()
    row_to_group = _build_cv_group_mapping(exact_row_to_group, similar_pair_edges)

    _progress(3, "Running out-of-fold label analysis…")

    # --- 5. Method C — Label analysis ---
    label_result = analyze_labels(
        df_norm,
        text_col=config.text_col,
        label_col=config.label_col,
        norm_col="_norm_text",
        classifier_name=config.classifier_name,
        class_weight=config.class_weight,
        n_folds=config.n_folds,
        random_seed=config.random_seed,
        min_class_size=config.min_class_size,
        row_to_group=row_to_group,
    )

    excl.n_small_class_excluded = sum(
        1 for r in label_result.records
        if r.status == "excluded"
        and r.exclusion_reason is not None
        and "fewer than" in (r.exclusion_reason or "")
    )

    _progress(4, "Computing evaluation metrics…")

    # --- 6. Baseline metrics ---
    baseline_metrics = compute_baseline_metrics(label_result)

    _progress(5, "Done.")

    return AuditResult(
        config=config,
        dataset_fingerprint=dataset_fingerprint,
        df_with_norm=df_norm,
        exclusion=excl,
        exact_dup=exact_dup_result,
        near_dup=near_dup_result,
        near_dup_skipped=near_dup_skipped,
        near_dup_skip_reason=near_dup_skip_reason,
        label_analysis=label_result,
        baseline_metrics=baseline_metrics,
        row_to_group=row_to_group,
    )
