"""
test_data_loading.py — Tests for src/data_loading.py.
"""
from __future__ import annotations

import pytest
from tests.conftest import make_csv_bytes
from src.data_loading import load_csv, validate_columns


# ---------------------------------------------------------------------------
# load_csv: basic success cases
# ---------------------------------------------------------------------------

class TestLoadCsvBasic:
    def test_loads_simple_csv(self):
        csv = "text,label\nhello world,positive\ngoodbye,negative\n"
        result = load_csv(make_csv_bytes(csv), "test.csv")
        assert not result.errors
        assert len(result.df) == 2
        assert "csv_row" in result.df.columns

    def test_csv_row_starts_at_2(self):
        """csv_row = index + 2 (header is row 1, first data row is row 2)."""
        csv = "text,label\na,x\nb,y\nc,z\n"
        result = load_csv(make_csv_bytes(csv), "test.csv")
        assert list(result.df["csv_row"]) == [2, 3, 4]

    def test_preserves_original_columns(self):
        csv = "message,category,source\nhello,billing,web\nworld,technical,email\n"
        result = load_csv(make_csv_bytes(csv), "test.csv")
        assert "message" in result.df.columns
        assert "category" in result.df.columns
        assert "source" in result.df.columns

    def test_delimiter_semicolon(self):
        csv = "text;label\nhello world;positive\ngoodbye;negative\n"
        result = load_csv(make_csv_bytes(csv), "test.csv")
        assert not result.errors
        assert result.delimiter_used == ";"
        assert len(result.df) == 2

    def test_delimiter_tab(self):
        csv = "text\tlabel\nhello world\tpositive\ngoodbye\tnegative\n"
        result = load_csv(make_csv_bytes(csv), "test.csv")
        assert not result.errors
        assert result.delimiter_used == "\t"

    def test_na_label_preserved(self):
        """'NA', 'None', 'null' are real label values, not missing."""
        csv = "text,label\nhello,NA\nworld,None\nfoo,null\n"
        result = load_csv(make_csv_bytes(csv, encoding="utf-8-sig"), "test.csv")
        assert not result.errors
        labels = list(result.df["label"])
        assert "NA" in labels
        assert "None" in labels
        assert "null" in labels

    def test_leading_zeros_preserved(self):
        """dtype=str ensures leading zeros survive."""
        csv = "id,text,label\n007,hello,positive\n042,world,negative\n"
        result = load_csv(make_csv_bytes(csv), "test.csv")
        assert not result.errors
        ids = list(result.df["id"])
        assert "007" in ids

    def test_utf8_bom_detected(self):
        csv = "text,label\nhello,positive\n"
        result = load_csv(make_csv_bytes(csv, encoding="utf-8-sig"), "test.csv")
        assert result.encoding_used == "utf-8-sig"
        assert not result.encoding_warning

    def test_latin1_fallback(self):
        """File with latin-1 encoding triggers fallback warning."""
        csv = "text,label\ncafé,positive\n"
        raw = csv.encode("latin-1")
        result = load_csv(raw, "test.csv")
        assert not result.errors
        assert result.encoding_warning
        assert result.encoding_used == "latin-1"


# ---------------------------------------------------------------------------
# load_csv: error cases
# ---------------------------------------------------------------------------

class TestLoadCsvErrors:
    def test_empty_file(self):
        result = load_csv(b"", "empty.csv")
        assert result.errors
        assert result.df.empty

    def test_header_only(self):
        csv = "text,label\n"
        result = load_csv(make_csv_bytes(csv), "header_only.csv")
        assert result.errors


    def test_extra_unquoted_delimiter_reports_line_and_quoting_fix(self):
        csv = (
            "id,text,label,source\n"
            "1,first message,yes,email\n"
            "2,second message,no,email\n"
            "3,third message,yes,email\n"
            "4,fourth message,no,email\n"
            "5,fifth message,yes,email\n"
            "6,This text has, an unquoted comma,no,email\n"
        )
        result = load_csv(make_csv_bytes(csv), "bad_row.csv")
        assert result.errors
        assert "line 7 has 5 fields, but the header has 4" in result.errors[0].lower()
        assert "wrap the entire cell in double quotes" in result.errors[0].lower()

    def test_duplicate_column_names_renamed(self):
        csv = "text,label,text\nhello,positive,extra\nworld,negative,more\n"
        result = load_csv(make_csv_bytes(csv), "dup_cols.csv")
        assert not result.errors
        # Original 'text' → 'text', second 'text' → 'text.1'
        assert "text" in result.df.columns
        assert "text.1" in result.df.columns
        assert result.duplicate_columns_renamed  # list is non-empty


# ---------------------------------------------------------------------------
# validate_columns
# ---------------------------------------------------------------------------

class TestValidateColumns:
    def _make_df(self):
        import pandas as pd
        return pd.DataFrame({
            "csv_row": [2, 3, 4, 5],
            "text": ["hello there", "goodbye world", "foo bar", "baz qux"],
            "label": ["positive", "negative", "positive", "negative"],
        })

    def test_same_column_invalid(self):
        df = self._make_df()
        result = validate_columns(df, "text", "text")
        assert not result.valid
        assert result.errors

    def test_missing_column(self):
        df = self._make_df()
        result = validate_columns(df, "nonexistent", "label")
        assert not result.valid

    def test_valid_columns(self):
        df = self._make_df()
        result = validate_columns(df, "text", "label")
        assert result.valid
        assert result.n_usable == 4
        assert result.n_missing_text == 0
        assert result.n_missing_label == 0

    def test_case_conflict_warning(self):
        import pandas as pd
        df = pd.DataFrame({
            "csv_row": [2, 3, 4, 5],
            "text": ["a", "b", "c", "d"],
            "label": ["Positive", "positive", "Negative", "negative"],
        })
        result = validate_columns(df, "text", "label")
        assert result.valid
        assert result.label_case_conflicts  # at least one group

    def test_many_classes_warning(self):
        import pandas as pd
        labels = [f"class_{i}" for i in range(60)]
        df = pd.DataFrame({
            "csv_row": list(range(2, 62)),
            "text": [f"text {i}" for i in range(60)],
            "label": labels,
        })
        result = validate_columns(df, "text", "label")
        assert any("unusually many" in w for w in result.warnings)

    def test_all_missing_text_invalid(self):
        import pandas as pd
        df = pd.DataFrame({
            "csv_row": [2, 3],
            "text": ["", "   "],
            "label": ["positive", "negative"],
        })
        result = validate_columns(df, "text", "label")
        assert not result.valid
