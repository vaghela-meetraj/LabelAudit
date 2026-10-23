"""
data_loading.py — CSV reading, encoding detection, validation, and column selection.

Design decisions:
- Read with dtype=str and keep_default_na=False to preserve real NA/None/null labels
  and leading zeros. Missing values are detected explicitly after reading.
- Encoding: try utf-8-sig first (handles BOM), fall back to latin-1 with a warning.
- Delimiter sniffing: try comma, semicolon, tab in that order.
- Every record receives a stable csv_row = index + 2 (header = row 1).
- Duplicate column names are renamed deterministically (col, col.1, col.2, …).
- No records are silently discarded; exclusion reasons are accumulated and returned.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

@dataclass
class LoadResult:
    """Everything produced by load_csv()."""
    df: pd.DataFrame                        # All original columns + csv_row
    filename: str
    encoding_used: str
    encoding_warning: bool                  # True if latin-1 fallback was used
    delimiter_used: str
    duplicate_columns_renamed: list[str]    # Original names that were renamed
    warnings: list[str]                     # Non-fatal notices
    errors: list[str]                       # Fatal problems (df will be empty)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_sha256(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()


def _sniff_delimiter(content: str) -> str:
    """Try comma, semicolon, tab. Return the one that produces the most columns."""
    candidates = [",", ";", "\t"]
    best, best_count = ",", 0
    first_line = content.split("\n")[0] if content else ""
    for delim in candidates:
        count = first_line.count(delim)
        if count > best_count:
            best, best_count = delim, count
    return best


def _deduplicate_columns(columns: list[str]) -> tuple[list[str], list[str]]:
    """Return (new_column_list, list_of_original_names_that_were_renamed)."""
    seen: dict[str, int] = {}
    result: list[str] = []
    renamed: list[str] = []
    for col in columns:
        if col not in seen:
            seen[col] = 0
            result.append(col)
        else:
            seen[col] += 1
            new_name = f"{col}.{seen[col]}"
            renamed.append(col)
            result.append(new_name)
    return result, renamed


def _friendly_parse_error(exc: Exception) -> str:
    """Explain inconsistent CSV field counts in terms users can fix."""
    match = re.search(r"Expected\s+(\d+)\s+fields\s+in line\s+(\d+),\s*saw\s+(\d+)", str(exc), re.IGNORECASE)
    if match:
        expected, line_number, actual = match.groups()
        return (
            f"Line {line_number} has {actual} fields, but the header has {expected}. "
            "A comma or other delimiter inside a cell may be unquoted. Wrap the entire cell in double quotes, "
            "or correct an extra separator or missing header column."
        )
    return f"Could not parse CSV: {exc}"


def _is_blank(val: object) -> bool:
    """True if val is the empty string or whitespace-only."""
    if not isinstance(val, str):
        return False
    return val.strip() == ""


# ---------------------------------------------------------------------------
# Main loading function
# ---------------------------------------------------------------------------

def load_csv(raw_bytes: bytes, filename: str) -> LoadResult:
    """
    Load a CSV from raw bytes, sniff encoding and delimiter, assign csv_row.

    Parameters
    ----------
    raw_bytes : bytes
        Raw file content from the uploader.
    filename : str
        Original filename (for display only).

    Returns
    -------
    LoadResult
        Populated regardless of success; check .errors for fatal problems.
    """
    warnings: list[str] = []
    errors: list[str] = []
    encoding_warning = False

    if not raw_bytes:
        errors.append("Uploaded file is empty.")
        return LoadResult(
            df=pd.DataFrame(), filename=filename,
            encoding_used="", encoding_warning=False,
            delimiter_used="", duplicate_columns_renamed=[],
            warnings=warnings, errors=errors,
        )

    # --- Encoding detection ---
    encoding_used = "utf-8-sig"
    try:
        content = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            content = raw_bytes.decode("latin-1")
            encoding_used = "latin-1"
            encoding_warning = True
            warnings.append(
                "File could not be decoded as UTF-8. Fell back to latin-1 — "
                "some characters may display incorrectly."
            )
        except Exception as exc:
            errors.append(f"Cannot decode file: {exc}")
            return LoadResult(
                df=pd.DataFrame(), filename=filename,
                encoding_used="unknown", encoding_warning=False,
                delimiter_used="", duplicate_columns_renamed=[],
                warnings=warnings, errors=errors,
            )

    delimiter = _sniff_delimiter(content)

    # Read the raw header before Pandas can silently mangle duplicate names.
    try:
        header_reader = csv.reader(io.StringIO(content, newline=""), delimiter=delimiter, strict=True)
        original_columns = next(header_reader)
    except StopIteration:
        errors.append("The file contains no header row.")
        return LoadResult(
            df=pd.DataFrame(), filename=filename,
            encoding_used=encoding_used, encoding_warning=encoding_warning,
            delimiter_used=delimiter, duplicate_columns_renamed=[],
            warnings=warnings, errors=errors,
        )
    except csv.Error as exc:
        errors.append(f"Malformed CSV header: {exc}")
        return LoadResult(
            df=pd.DataFrame(), filename=filename,
            encoding_used=encoding_used, encoding_warning=encoding_warning,
            delimiter_used=delimiter, duplicate_columns_renamed=[],
            warnings=warnings, errors=errors,
        )
    new_columns, renamed = _deduplicate_columns(original_columns)

    # --- Parse CSV ---
    try:
        df = pd.read_csv(
            io.StringIO(content),
            sep=delimiter,
            header=0,
            names=new_columns,
            dtype=str,
            keep_default_na=False,
        )
    except pd.errors.EmptyDataError:
        errors.append("The file contains no data (empty or header-only).")
        return LoadResult(
            df=pd.DataFrame(), filename=filename,
            encoding_used=encoding_used, encoding_warning=encoding_warning,
            delimiter_used=delimiter, duplicate_columns_renamed=[],
            warnings=warnings, errors=errors,
        )
    except Exception as exc:
        errors.append(_friendly_parse_error(exc))
        return LoadResult(
            df=pd.DataFrame(), filename=filename,
            encoding_used=encoding_used, encoding_warning=encoding_warning,
            delimiter_used=delimiter, duplicate_columns_renamed=[],
            warnings=warnings, errors=errors,
        )

    if df.empty or len(df.columns) == 0:
        errors.append("File has no usable columns.")
        return LoadResult(
            df=pd.DataFrame(), filename=filename,
            encoding_used=encoding_used, encoding_warning=encoding_warning,
            delimiter_used=delimiter, duplicate_columns_renamed=[],
            warnings=warnings, errors=errors,
        )

    if len(df) == 0:
        errors.append("File has a header row but no data rows.")
        return LoadResult(
            df=pd.DataFrame(), filename=filename,
            encoding_used=encoding_used, encoding_warning=encoding_warning,
            delimiter_used=delimiter, duplicate_columns_renamed=[],
            warnings=warnings, errors=errors,
        )

    # --- Report deterministic duplicate-header renames ---
    if renamed:
        warnings.append(
            f"Duplicate column names detected and renamed: {renamed}. "
            "The renamed columns now have a numeric suffix (.1, .2, …)."
        )

    # --- Assign stable row identifier ---
    # csv_row = 1-based data row number; header is row 1, first data row is row 2.
    df.insert(0, "csv_row", range(2, len(df) + 2))
    df["csv_row"] = df["csv_row"].astype(int)

    return LoadResult(
        df=df,
        filename=filename,
        encoding_used=encoding_used,
        encoding_warning=encoding_warning,
        delimiter_used=delimiter,
        duplicate_columns_renamed=renamed,
        warnings=warnings,
        errors=errors,
    )


def compute_file_fingerprint(raw_bytes: bytes) -> str:
    """SHA-256 hex digest of the raw file bytes, used to key cached results."""
    return _compute_sha256(raw_bytes)


# ---------------------------------------------------------------------------
# Column validation
# ---------------------------------------------------------------------------

@dataclass
class ColumnValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Counts of usable / missing records in the selected columns
    n_missing_text: int = 0
    n_missing_label: int = 0
    n_usable: int = 0
    # Label analysis helpers
    unique_labels: list[str] = field(default_factory=list)
    label_case_conflicts: list[tuple[str, str]] = field(default_factory=list)


def validate_columns(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
) -> ColumnValidationResult:
    """
    Validate the user's column selections and return label-quality warnings.

    Rules
    -----
    - text_col == label_col → invalid.
    - Both columns must exist in df.
    - Missing text = empty or whitespace-only cell.
    - Missing label = empty or whitespace-only cell.
    - Warn on case/whitespace duplicates in label names.
    - Warn on very many distinct labels (> 50).
    - Warn when the label column looks like free text (mean token length > 5).
    """
    result = ColumnValidationResult(valid=True)

    if text_col == label_col:
        result.valid = False
        result.errors.append("Text column and label column must be different.")
        return result

    missing_cols = [c for c in [text_col, label_col] if c not in df.columns]
    if missing_cols:
        result.valid = False
        result.errors.append(f"Column(s) not found in dataset: {missing_cols}.")
        return result

    # Count missing values
    texts = df[text_col]
    labels = df[label_col]

    result.n_missing_text = int(texts.apply(_is_blank).sum())
    result.n_missing_label = int(labels.apply(_is_blank).sum())
    result.n_usable = int(
        ((~texts.apply(_is_blank)) & (~labels.apply(_is_blank))).sum()
    )

    if result.n_usable == 0:
        result.valid = False
        result.errors.append(
            "No records have both a non-empty text and a non-empty label. "
            "Check your column selections."
        )
        return result

    # Label quality warnings
    unique_labels = [
        lbl for lbl in labels.unique()
        if not _is_blank(lbl)
    ]
    result.unique_labels = sorted(unique_labels)

    if len(unique_labels) > 50:
        result.warnings.append(
            f"Found {len(unique_labels)} distinct label values — unusually many. "
            "Verify that the label column is a category, not free text."
        )

    # Detect labels that differ only by case or surrounding whitespace
    normalised_map: dict[str, list[str]] = {}
    for lbl in unique_labels:
        key = lbl.strip().lower()
        normalised_map.setdefault(key, []).append(lbl)
    for key, variants in normalised_map.items():
        if len(variants) > 1:
            result.label_case_conflicts.append(tuple(variants))  # type: ignore[arg-type]
            result.warnings.append(
                f"Labels differ only by case/whitespace and will be treated as "
                f"distinct classes: {variants}. Consider standardising them."
            )

    # Heuristic: label column looks like free text when average word-count > 5
    avg_words = labels.apply(lambda x: len(str(x).split())).mean()
    if avg_words > 5:
        result.warnings.append(
            f"The label column has an average of {avg_words:.1f} words per cell, "
            "which suggests it may be free text rather than a class name."
        )

    return result
