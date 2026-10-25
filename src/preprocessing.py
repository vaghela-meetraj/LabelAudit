"""
preprocessing.py — Text normalisation for comparison (not for display).

Normalisation pipeline (applied in this order — document changes here):

1. Unicode NFKC normalisation: collapses compatibility characters,
   e.g. ﬁ → fi, ² → 2, full-width letters → ASCII.
2. Lowercase: makes comparison case-insensitive.
3. Trim leading/trailing whitespace.
4. Punctuation handling: replace most punctuation with a single space,
   EXCEPT apostrophes that appear between word characters (contractions
   like "don't") and digits (so "3.14" stays "3.14").
   Rationale: removes incidental punctuation differences while keeping
   meaningful token distinctions.
5. Collapse repeated whitespace to a single space and strip.

Conservative choices:
- Numbers are kept (e.g. "error code 404" differs from "error code 500").
- Negation words (not, no, never) are kept.
- Stop words are NOT removed — removal can make distinct sentences look identical.
- No stemming or lemmatisation for the same reason.
- Non-Latin scripts: the default word token pattern [a-zA-Z0-9] handles them
  poorly. Character n-grams (3–5 chars) in the TF-IDF partially compensate;
  see README for details.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

import pandas as pd


# Pre-compiled patterns for speed
#
# _PUNCT_REPLACE: replaces punctuation with a space UNLESS the character is:
#   - an apostrophe flanked by word characters (don't → don't)
#   - a period/comma flanked by digits (3.14 → 3.14)
# We use a single regex with alternation: match the "keep" cases first as
# zero-width captures (via lookahead/lookbehind), then match and replace
# the "remove" cases.
# Strategy: use a callback function with re.sub to decide per match.
_PUNCT_PATTERN = re.compile(r"((?<=\w)'(?=\w)|(?<=\d)[.,](?=\d))|[^\w\s]")
_MULTI_SPACE = re.compile(r"\s+")


def _replace_punct(text: str) -> str:
    """
    Replace punctuation with a space in a single pass, with two exceptions:
    1. Apostrophes between word chars (don't) → kept.
    2. Periods/commas between digits (3.14) → kept.
    """
    return _PUNCT_PATTERN.sub(lambda m: m.group(1) if m.group(1) else " ", text)


def normalize_text(text: str) -> str:
    """
    Apply the five-step normalisation pipeline to a single string.

    Empty string in → empty string out (the caller decides how to handle it).
    Non-string input is coerced to str first.
    """
    if not isinstance(text, str):
        text = str(text)

    # Step 1 — Unicode NFKC
    text = unicodedata.normalize("NFKC", text)

    # Step 2 — lowercase
    text = text.lower()

    # Step 3 — trim
    text = text.strip()

    # Step 4 — punctuation: replace most punct with space, keep don't and 3.14
    text = _replace_punct(text)

    # Step 5 — collapse whitespace
    text = _MULTI_SPACE.sub(" ", text).strip()

    return text


def is_empty_after_normalization(text: str) -> bool:
    """True if the text is blank or becomes empty after normalisation."""
    if not isinstance(text, str):
        return True
    return normalize_text(text) == ""


def add_normalized_column(
    df: pd.DataFrame,
    text_col: str,
    norm_col: str = "_norm_text",
) -> pd.DataFrame:
    """
    Return df with a new column `norm_col` containing normalised text.
    Rows where original text is missing or becomes empty after normalisation
    receive an empty string ""; the caller uses that to exclude them.
    This never modifies the original text column.
    """
    df = df.copy()

    def _safe_norm(val: object) -> str:
        if not isinstance(val, str) or val.strip() == "":
            return ""
        return normalize_text(val)

    df[norm_col] = df[text_col].apply(_safe_norm)
    return df


def build_exclusion_mask(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    norm_col: str = "_norm_text",
) -> pd.Series:
    """
    Return a boolean Series (True = exclude) marking records that are missing
    either usable text or a usable label.

    A text is unusable if:
    - The raw cell is blank/whitespace, OR
    - The normalised text is the empty string.

    A label is unusable if the raw cell is blank/whitespace.
    """
    missing_text = (
        df[text_col].apply(lambda v: not isinstance(v, str) or v.strip() == "")
        | (df[norm_col] == "")
    )
    missing_label = df[label_col].apply(
        lambda v: not isinstance(v, str) or v.strip() == ""
    )
    return missing_text | missing_label
