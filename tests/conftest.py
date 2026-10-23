"""
conftest.py — Shared fixtures for the LabelAudit test suite.

All fixtures use small, deterministic datasets with known expected outputs.
No mocking, no invented results.
"""
from __future__ import annotations

import io
import pytest
import pandas as pd


# ---------------------------------------------------------------------------
# Raw CSV bytes helpers
# ---------------------------------------------------------------------------

def make_csv_bytes(content: str, encoding: str = "utf-8-sig") -> bytes:
    """Encode a CSV string to bytes as the file uploader would deliver."""
    return content.encode(encoding)


# ---------------------------------------------------------------------------
# Small deterministic DataFrames
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_df():
    """
    6-row DataFrame with 3 classes, no duplicates, no missing values.
    csv_row column is pre-assigned (as load_csv would do).
    """
    data = {
        "csv_row": [2, 3, 4, 5, 6, 7],
        "text": [
            "The network is down and I cannot connect",
            "I was charged twice this month please refund",
            "Cannot log in password reset fails",
            "The server is completely unreachable",
            "Double charge on my bill this month",
            "My login keeps failing every time",
        ],
        "label": ["technical", "billing", "account", "technical", "billing", "account"],
        "source": ["web", "email", "web", "phone", "email", "web"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def exact_dup_df():
    """
    DataFrame with one group of 5 identical texts (all same label) and
    one group of 2 identical texts (conflicting labels).
    Plus 3 singletons.
    """
    data = {
        "csv_row": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "text": [
            "The router keeps rebooting",   # group A (5 members, same label)
            "The router keeps rebooting",
            "The router keeps rebooting",
            "The router keeps rebooting",
            "The router keeps rebooting",
            "Account locked out completely", # group B (2 members, conflicting)
            "Account locked out completely",
            "Please send the invoice now",   # singleton
            "Billing address update needed", # singleton
            "Password is not working here",  # singleton
        ],
        "label": [
            "technical", "technical", "technical", "technical", "technical",  # group A
            "account", "billing",                                             # group B conflict
            "billing", "billing", "account",                                  # singletons
        ],
    }
    return pd.DataFrame(data)


@pytest.fixture
def near_dup_df():
    """
    DataFrame for near-duplicate testing.
    Pair (row 2, row 3): very similar text, same label → near_duplicate
    Pair (row 4, row 5): similar text, different labels → conflicting_label
    Row 6: unrelated text → no pairs
    """
    data = {
        "csv_row": [2, 3, 4, 5, 6],
        "text": [
            "The internet connection drops frequently at home",
            "Internet connection drops often at my house",     # similar to row 2
            "Cannot access my account login fails",
            "Account login is not working for me",              # similar to row 4, different label
            "The quarterly report is ready for review",         # unrelated
        ],
        "label": ["technical", "technical", "account", "billing", "billing"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def tiny_df():
    """2-row single-class dataset — cross-validation must not crash."""
    data = {
        "csv_row": [2, 3],
        "text": ["Something technical here", "More technical content"],
        "label": ["technical", "technical"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def missing_values_df():
    """
    DataFrame with:
    - One row with empty text
    - One row with whitespace text
    - One row with empty label
    - One row with the label value "NA" (must survive as a real label)
    """
    data = {
        "csv_row": [2, 3, 4, 5, 6, 7],
        "text": [
            "",                         # missing text
            "   ",                      # whitespace-only text
            "Normal text here",
            "Another normal text",
            "Third normal text",
            "Fourth normal text",
        ],
        "label": [
            "technical",
            "billing",
            "",                         # missing label
            "NA",                       # real label, must survive
            "account",
            "account",
        ],
    }
    return pd.DataFrame(data)
