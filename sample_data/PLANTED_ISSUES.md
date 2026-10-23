# PLANTED_ISSUES.md — Deliberately Seeded Problems in `example_dataset.csv`

This document lists every deliberately planted data-quality issue in the sample dataset.
`test_sample_dataset.py` asserts the exact counts and row memberships listed here.

**Column mapping (non-default names):**
- Text column: `message`
- Label column: `category`

---

## A. Exact Duplicate Groups

### Group 1 — Conflicting labels (3 members)
| csv_row | message | category |
|---|---|---|
| 2 | My internet connection keeps dropping… | technical |
| 5 | My internet connection keeps dropping… | technical |
| 83 | My internet connection keeps dropping… | account |

These rows share the same normalized text. Row 83 has the conflicting `account` label.

### Group 2 — Matching labels (2 members, billing)
| csv_row | category |
|---|---|
| 3 | billing |
| 15 | billing |

### Group 3 — Matching labels (3 members, account)
| csv_row | category |
|---|---|
| 4 | account |
| 8 | account |
| 18 | account |

Row 18 differs in punctuation, which normalization removes; it is therefore an exact duplicate for this audit.

Expected exact-duplicate totals: **3 groups, 8 affected records, 5 redundant records, 1 conflicting group.**

---

## B. Near-Duplicate Pairs

### Near-duplicate — same label (technical)
- csv_row 2: *My internet connection keeps dropping every few minutes. I have restarted the router but the problem persists.*
- csv_row 25: *My internet connection keeps dropping every few minutes and I have restarted the router.*

At the documented test thresholds (near-duplicate 0.80, conflict 0.70), this pair is detected as a same-label near-duplicate (cosine similarity about 0.81).

### Similar-looking pair below the configured threshold (not reported)
- csv_row 3 and csv_row 84 are both billing messages about a duplicate subscription charge. Their cosine similarity is about 0.49, below both test thresholds, so the audit does not report this pair.

The account messages in rows 4, 8, and 18 are already grouped as exact duplicates after normalization; they do not appear again as near-duplicate pairs.

---

## C. Missing Values

| csv_row | issue |
|---|---|
| 79 | Empty `message` cell — missing text |
| 97 | Empty `message` cell — missing text |

**Note:** csv_row 79 maps to the row with empty message; csv_row 97 maps to the last row with empty message.  
These are counted in `n_missing_text`.

---

## D. Expected Aggregate Counts

| Metric | Expected value |
|---|---|
| Total rows | 96 |
| Rows with missing text | 2 (csv_rows 79, 97) |
| Rows with missing label | 0 |
| Exact-duplicate groups | 3 |
| Exact-dup groups with label conflict | 1 |
| Exact-dup affected records | 8 |
| Exact-dup redundant records | 5 |
| Near-duplicate pairs at test thresholds | 1 |
| Conflicting near-duplicate pairs at test thresholds | 0 |

> **These exact expectations are for the fixed synthetic sample file.** If its text or normalization rules change, update this document and the corresponding assertions together.

---

## E. Statement of Purpose

This dataset is **illustrative data for testing and demonstration only**.  
It is not a real-world benchmark. All messages are synthetic support-ticket texts.  
Do not use this data to draw conclusions about real annotation quality.
