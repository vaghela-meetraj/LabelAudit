"""
test_preprocessing.py — Tests for src/preprocessing.py.
"""
from __future__ import annotations

import pytest
from src.preprocessing import normalize_text, is_empty_after_normalization, add_normalized_column


class TestNormalizeText:
    def test_lowercase(self):
        assert normalize_text("Hello World") == "hello world"

    def test_unicode_nfkc(self):
        # ﬁ (ligature) → fi
        assert normalize_text("ﬁle") == "file"

    def test_trim_whitespace(self):
        assert normalize_text("  hello  ") == "hello"

    def test_collapse_repeated_whitespace(self):
        assert normalize_text("hello   world") == "hello world"

    def test_punctuation_replaced_with_space(self):
        result = normalize_text("hello! world?")
        assert "!" not in result
        assert "?" not in result

    def test_apostrophe_in_word_preserved(self):
        """Contractions must survive: don't → don't (not don t)."""
        result = normalize_text("don't stop")
        assert "don't" in result

    def test_decimal_point_preserved(self):
        """3.14 should survive."""
        result = normalize_text("version 3.14 released")
        assert "3.14" in result

    def test_digits_preserved(self):
        result = normalize_text("error code 404")
        assert "404" in result

    def test_negation_words_preserved(self):
        result = normalize_text("not working never helps no solution")
        assert "not" in result
        assert "never" in result
        assert "no" in result

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_whitespace_only(self):
        assert normalize_text("   ") == ""

    def test_non_string_coerced(self):
        # Should not raise
        result = normalize_text(42)  # type: ignore[arg-type]
        assert result == "42"

    def test_idempotent(self):
        """Normalising twice gives the same result as once."""
        text = "Hello, World! Don't stop at 3.14"
        once = normalize_text(text)
        twice = normalize_text(once)
        assert once == twice


class TestIsEmptyAfterNormalization:
    def test_empty_string(self):
        assert is_empty_after_normalization("") is True

    def test_whitespace(self):
        assert is_empty_after_normalization("   ") is True

    def test_punctuation_only(self):
        # Punctuation is replaced by spaces → empty after strip
        assert is_empty_after_normalization("!!!???") is True

    def test_normal_text(self):
        assert is_empty_after_normalization("hello world") is False

    def test_non_string(self):
        assert is_empty_after_normalization(None) is True  # type: ignore[arg-type]


class TestAddNormalizedColumn:
    def test_adds_column(self, simple_df):
        result = add_normalized_column(simple_df, "text")
        assert "_norm_text" in result.columns

    def test_original_text_unchanged(self, simple_df):
        original_texts = list(simple_df["text"])
        result = add_normalized_column(simple_df, "text")
        assert list(result["text"]) == original_texts

    def test_empty_text_becomes_empty_string(self, missing_values_df):
        result = add_normalized_column(missing_values_df, "text")
        norms = result["_norm_text"].tolist()
        assert norms[0] == ""  # empty text → ""
        assert norms[1] == ""  # whitespace-only → ""

    def test_does_not_modify_original_df(self, simple_df):
        import pandas as pd
        original_cols = list(simple_df.columns)
        _ = add_normalized_column(simple_df, "text")
        assert list(simple_df.columns) == original_cols
