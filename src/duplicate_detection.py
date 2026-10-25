"""
duplicate_detection.py — Method A (exact duplicates) and Method B (near-duplicates
and conflicting labels via TF-IDF cosine similarity).

Method A: exact duplicates
--------------------------
Two records are exact duplicates when their normalised texts are identical.
Grouping: every set of records sharing the same normalised text forms one group.
  - Groups where every member has the same label  → "matching labels"
  - Groups where members have at least two distinct labels → "conflicting labels"
Counting (explained in the UI):
  - duplicate_groups : number of groups with ≥ 2 members
  - affected_records : total records belonging to any such group
  - redundant_records: affected_records − duplicate_groups
                       (one representative per group is "kept"; the rest are redundant)

Method B: near-duplicates and conflicting labels
-------------------------------------------------
Uses TF-IDF vectors (word n-grams + optional character n-grams) and cosine similarity.

Representation:
  - Word TF-IDF: (1, 2) n-grams, sublinear_tf=True.
  - Char TF-IDF: char_wb analyser, (3, 5) n-grams, optional.
  - When both are used: matrices are each L2-normalised, then stacked with
    equal weight (scaled by 1/sqrt(2)), so cosine of combined equals average
    of the two cosines.  This is stated in the app's help text.

Pair classification (per pair):
  - similarity ≥ near_dup_threshold AND same label    → near_duplicate
  - similarity ≥ conflict_threshold AND different label → conflicting_label
  - similarity ≥ near_dup_threshold AND different label → near_duplicate + conflicting_label
    (the pair is listed in both categories; finding_type = "near_dup_conflict")
  Pair table columns: csv_row_a, csv_row_b, text_a, text_b, label_a, label_b,
                      similarity, finding_type, reason.

Scalability:
  - One representative per exact-duplicate group is used for similarity.
  - Similarity computed in row blocks to avoid building an N×N dense matrix.
  - Cap on stored pairs (max_pairs). If hit, the function stops and returns a flag.
  - Record limit (pairwise_limit). If exceeded, caller must offer subsampling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


# ---------------------------------------------------------------------------
# Method A — Exact duplicates
# ---------------------------------------------------------------------------

@dataclass
class ExactDupGroup:
    group_id: int
    norm_text: str                   # The shared normalised text
    members: pd.DataFrame            # Rows from the working df (with csv_row)
    label_conflict: bool             # True if >1 distinct label in group


@dataclass
class ExactDupResult:
    groups: list[ExactDupGroup]      # Only groups with ≥ 2 members
    n_groups: int
    n_affected_records: int
    n_redundant_records: int         # = n_affected_records - n_groups
    # Mapping: csv_row → group_id (for all members of a group)
    row_to_group: dict[int, int]


def find_exact_duplicates(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    norm_col: str = "_norm_text",
) -> ExactDupResult:
    """
    Group records by identical normalised text.  Only records with non-empty
    normalised text and non-empty label participate.  Records excluded for
    other reasons (missing text/label) should already be filtered by the caller
    or are simply not in any group.

    Parameters
    ----------
    df : DataFrame with csv_row, text_col, label_col, norm_col already added.
    """
    # Only eligible rows: non-empty normalised text and non-empty label
    eligible = df[
        (df[norm_col] != "") &
        (df[label_col].apply(lambda v: isinstance(v, str) and v.strip() != ""))
    ].copy()

    groups: list[ExactDupGroup] = []
    row_to_group: dict[int, int] = {}
    group_id = 0

    grouped = eligible.groupby(norm_col, sort=False)
    for norm_text, grp in grouped:
        if len(grp) < 2:
            continue  # singletons are not duplicates
        group_id += 1
        distinct_labels = grp[label_col].nunique()
        conflict = distinct_labels > 1
        groups.append(
            ExactDupGroup(
                group_id=group_id,
                norm_text=str(norm_text),
                members=grp.reset_index(drop=True),
                label_conflict=conflict,
            )
        )
        for csv_row in grp["csv_row"].tolist():
            row_to_group[csv_row] = group_id

    n_affected = sum(len(g.members) for g in groups)
    n_groups = len(groups)

    return ExactDupResult(
        groups=groups,
        n_groups=n_groups,
        n_affected_records=n_affected,
        n_redundant_records=max(0, n_affected - n_groups),
        row_to_group=row_to_group,
    )


# ---------------------------------------------------------------------------
# Method B — Near-duplicates and conflicting-label pairs
# ---------------------------------------------------------------------------

@dataclass
class PairFinding:
    csv_row_a: int
    csv_row_b: int
    text_a: str
    text_b: str
    label_a: str
    label_b: str
    similarity: float
    finding_type: str    # "near_duplicate", "conflicting_label", "near_dup_conflict"
    reason: str


@dataclass
class NearDupResult:
    pairs: list[PairFinding]
    n_pairs_total: int
    n_near_dup: int
    n_conflicts: int
    n_analyzed_records: int          # Records fed to the vectorizer
    hit_pair_cap: bool               # True if max_pairs was reached
    vectorizer_word: Optional[object] = None  # Kept for inspection
    vectorizer_char: Optional[object] = None
    similarity_method: str = "tfidf"


_BLOCK_SIZE = 500          # Rows per block in the similarity computation
DEFAULT_MAX_PAIRS = 200_000
_SENTENCE_MODEL = None


def get_sentence_model():
    """Lazily load and cache the SentenceTransformer model for semantic analysis."""
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _SENTENCE_MODEL = False
    return _SENTENCE_MODEL if _SENTENCE_MODEL is not False else None


def build_semantic_matrix(texts: list[str]) -> tuple[np.ndarray, Optional[str]]:
    """
    Encode texts with all-MiniLM-L6-v2 and return L2-normalized dense embeddings.
    Captures deep sentence meaning, intent, synonyms, and paraphrases.
    """
    model = get_sentence_model()
    if model is None:
        raise RuntimeError("SentenceTransformer is not available.")
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings, "all-MiniLM-L6-v2"


def build_tfidf_matrix(
    texts: list[str],
    use_char: bool = True,
) -> tuple[csr_matrix, TfidfVectorizer, Optional[TfidfVectorizer]]:
    """
    Fit TF-IDF vectorizers on `texts` and return the combined L2-normalised matrix.

    Word TF-IDF: (1, 2) n-grams, sublinear_tf=True.
    Char TF-IDF: char_wb, (3, 5) n-grams, sublinear_tf=True (optional).

    Combination: stack horizontally, each half scaled by 1/sqrt(2), so the
    combined cosine similarity equals the unweighted average of word and char cosines.
    """
    vect_word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
    )
    mat_word = vect_word.fit_transform(texts)
    mat_word = normalize(mat_word, norm="l2")

    vect_char: Optional[TfidfVectorizer] = None

    if use_char:
        vect_char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            min_df=1,
        )
        mat_char = vect_char.fit_transform(texts)
        mat_char = normalize(mat_char, norm="l2")
        scale = 1.0 / math.sqrt(2)
        matrix = hstack([mat_word * scale, mat_char * scale], format="csr")
    else:
        matrix = mat_word

    return matrix, vect_word, vect_char


def find_near_duplicates(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    norm_col: str = "_norm_text",
    near_dup_threshold: float = 0.90,
    conflict_threshold: float = 0.80,
    use_char: bool = True,
    pairwise_limit: int = 5000,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    exact_dup_result: Optional[ExactDupResult] = None,
    similarity_method: str = "tfidf",
) -> NearDupResult:
    """
    Find near-duplicate and conflicting-label pairs using cosine similarity.
    Supports either semantic embeddings (captures meaning & paraphrases)
    or lexical TF-IDF (word/character overlap).

    Algorithm
    ---------
    1. Take one representative per exact-duplicate group (to avoid spurious pairs).
    2. Encode eligible representative texts into semantic or TF-IDF matrix.
    3. Compute similarity in row blocks; keep only entries above candidate threshold.
    4. Classify each surviving pair against its own threshold.
    """
    candidate_threshold = min(near_dup_threshold, conflict_threshold)

    # --- Select eligible records ---
    eligible = df[
        (df[norm_col] != "") &
        (df[label_col].apply(lambda v: isinstance(v, str) and v.strip() != ""))
    ].copy()

    # --- One representative per exact-dup group ---
    # Use the first member of each group as the representative.
    if exact_dup_result is not None and exact_dup_result.n_groups > 0:
        row_to_group = exact_dup_result.row_to_group
        group_seen: set[int] = set()
        representatives: list[int] = []   # indices into eligible
        for idx, row in eligible.iterrows():
            csv_row = int(row["csv_row"])
            gid = row_to_group.get(csv_row)
            if gid is None:
                representatives.append(idx)
            elif gid not in group_seen:
                group_seen.add(gid)
                representatives.append(idx)
        eligible = eligible.loc[representatives].reset_index(drop=True)
    else:
        eligible = eligible.reset_index(drop=True)
        row_to_group = {}

    n_analyzed = len(eligible)

    texts = eligible[norm_col].tolist()
    csv_rows = eligible["csv_row"].tolist()
    orig_texts = eligible[text_col].tolist()
    labels = eligible[label_col].tolist()

    if n_analyzed < 2:
        return NearDupResult(
            pairs=[], n_pairs_total=0, n_near_dup=0, n_conflicts=0,
            n_analyzed_records=n_analyzed, hit_pair_cap=False,
            vectorizer_word=None, vectorizer_char=None,
            similarity_method=similarity_method,
        )

    # --- Build representation matrix ---
    actual_method = similarity_method
    if similarity_method == "semantic":
        try:
            matrix, model_name = build_semantic_matrix(orig_texts)
            vect_word = model_name
            vect_char = None
        except Exception:
            matrix, vect_word, vect_char = build_tfidf_matrix(texts, use_char=use_char)
            actual_method = "tfidf"
    else:
        matrix, vect_word, vect_char = build_tfidf_matrix(texts, use_char=use_char)

    # --- Block-wise similarity ---
    pairs: list[PairFinding] = []
    hit_cap = False

    for block_start in range(0, n_analyzed, _BLOCK_SIZE):
        block_end = min(block_start + _BLOCK_SIZE, n_analyzed)
        block_mat = matrix[block_start:block_end]

        # Cosine similarity of block rows against all rows
        if isinstance(matrix, np.ndarray):
            sim_block = block_mat @ matrix.T
        else:
            sim_block = (block_mat @ matrix.T).toarray()  # shape: (block, n_analyzed)

        for local_i, global_i in enumerate(range(block_start, block_end)):
            row_sims = sim_block[local_i]
            # Fast vectorized scan for candidates in the upper triangle (j > global_i)
            candidate_j_rel = np.where(row_sims[global_i + 1:] >= candidate_threshold)[0]
            if len(candidate_j_rel) == 0:
                continue

            for rel_j in candidate_j_rel:
                global_j = global_i + 1 + int(rel_j)
                sim = float(row_sims[global_j])

                same_label = labels[global_i] == labels[global_j]
                is_near_dup = sim >= near_dup_threshold
                is_conflict = sim >= conflict_threshold and not same_label

                if not is_near_dup and not is_conflict:
                    continue

                if actual_method == "semantic":
                    if is_near_dup and is_conflict:
                        ftype = "near_dup_conflict"
                        reason = (
                            f"These texts have the same meaning (semantic similarity {sim:.3f} ≥ "
                            f"near-duplicate threshold {near_dup_threshold}) but have "
                            f"different labels. Review both the deduplication and the labels."
                        )
                    elif is_near_dup:
                        ftype = "near_duplicate"
                        reason = (
                            f"These texts express the same meaning (semantic similarity {sim:.3f} ≥ "
                            f"near-duplicate threshold {near_dup_threshold}) with matching "
                            f"label '{labels[global_i]}'."
                        )
                    else:
                        ftype = "conflicting_label"
                        reason = (
                            f"These texts express similar meaning (semantic similarity {sim:.3f} ≥ "
                            f"conflict threshold {conflict_threshold}) but have different "
                            f"labels ('{labels[global_i]}' vs '{labels[global_j]}'). "
                            f"This is a review signal, not proof of an error."
                        )
                else:
                    if is_near_dup and is_conflict:
                        ftype = "near_dup_conflict"
                        reason = (
                            f"These texts are very similar (similarity {sim:.3f} ≥ "
                            f"near-duplicate threshold {near_dup_threshold}) but have "
                            f"different labels. Review both the deduplication and the labels."
                        )
                    elif is_near_dup:
                        ftype = "near_duplicate"
                        reason = (
                            f"These texts are very similar (similarity {sim:.3f} ≥ "
                            f"near-duplicate threshold {near_dup_threshold}) and share the "
                            f"same label '{labels[global_i]}'."
                        )
                    else:
                        ftype = "conflicting_label"
                        reason = (
                            f"These texts are similar (similarity {sim:.3f} ≥ "
                            f"conflict threshold {conflict_threshold}) but have different "
                            f"labels ('{labels[global_i]}' vs '{labels[global_j]}'). "
                            f"This is a review signal, not proof of an error."
                        )

                # When a representative stands for an exact-dup group, expand to
                # group member lists.  For simplicity we list the representative
                # csv_row; the UI links to the group view.
                pairs.append(PairFinding(
                    csv_row_a=csv_rows[global_i],
                    csv_row_b=csv_rows[global_j],
                    text_a=orig_texts[global_i],
                    text_b=orig_texts[global_j],
                    label_a=labels[global_i],
                    label_b=labels[global_j],
                    similarity=sim,
                    finding_type=ftype,
                    reason=reason,
                ))

                if len(pairs) >= max_pairs:
                    hit_cap = True
                    break

            if hit_cap:
                break
        if hit_cap:
            break

    n_near_dup = sum(1 for p in pairs if p.finding_type in {"near_duplicate", "near_dup_conflict"})
    n_conflicts = sum(1 for p in pairs if p.finding_type in {"conflicting_label", "near_dup_conflict"})

    return NearDupResult(
        pairs=pairs,
        n_pairs_total=len(pairs),
        n_near_dup=n_near_dup,
        n_conflicts=n_conflicts,
        n_analyzed_records=n_analyzed,
        hit_pair_cap=hit_cap,
        vectorizer_word=vect_word,
        vectorizer_char=vect_char,
        similarity_method=actual_method,
    )


def pairs_to_dataframe(pairs: list[PairFinding]) -> pd.DataFrame:
    """Convert the list of PairFinding objects to a flat DataFrame for display/export."""
    if not pairs:
        return pd.DataFrame(columns=[
            "csv_row_a", "csv_row_b", "text_a", "text_b",
            "label_a", "label_b", "similarity", "finding_type", "reason",
        ])
    rows = [
        {
            "csv_row_a": p.csv_row_a,
            "csv_row_b": p.csv_row_b,
            "text_a": p.text_a,
            "text_b": p.text_b,
            "label_a": p.label_a,
            "label_b": p.label_b,
            "similarity": p.similarity,
            "finding_type": p.finding_type,
            "reason": p.reason,
        }
        for p in pairs
    ]
    return pd.DataFrame(rows)
