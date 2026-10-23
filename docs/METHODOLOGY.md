# METHODOLOGY.md — LabelAudit Algorithm Documentation

This document provides a plain-language explanation of each algorithm used in
LabelAudit, the rationale for each design decision, the limitations of each method,
and a set of questions a project examiner might ask.

---

## 1. Text normalisation (`preprocessing.py`)

### What it does

Applies five transformations in sequence:

1. **Unicode NFKC** — collapses compatibility characters (e.g. `ﬁ → fi`, `² → 2`, full-width letters → ASCII).
2. **Lowercase** — makes comparison case-insensitive.
3. **Trim** — removes surrounding whitespace.
4. **Punctuation → space** — replaces most punctuation with a space. *Except:* apostrophes between word characters (`don't` stays intact) and decimal points between digits (`3.14` stays intact).
5. **Collapse whitespace** — reduces multiple spaces to one.

### Why these choices?

- **No stop-word removal**: Removing "not", "no", "never" can make semantically opposite sentences look identical. Better to keep them.
- **No stemming/lemmatisation**: "running" and "run" convey the same concept, but removing morphological endings can also merge unrelated words. The TF-IDF feature weights handle frequency differences adequately.
- **No number removal**: "error code 404" differs meaningfully from "error code 500".
- **Conservative**: The goal is to group texts that are *visually* the same, not texts that *mean* the same. Semantic similarity is left to the cosine score.

### Limitations

- The default TF-IDF word token pattern `[a-zA-Z0-9]+` skips non-Latin scripts (Chinese, Arabic, Devanagari, etc.). Character n-grams (3–5 chars) partially compensate.
- Very short texts (1–2 words) produce sparse TF-IDF vectors with low cosine similarity even when semantically equivalent.

---

## 2. Method A — Exact duplicates (`duplicate_detection.py`)

### What it does

Groups all records by their normalised text. Any group with two or more members is a duplicate group.

**Counting:**
- `n_groups` = number of groups with ≥ 2 members.
- `n_affected_records` = total records in those groups.
- `n_redundant_records` = affected − groups (one per group is the "kept" representative).

A group where all members share one label → **matching labels**.
A group where members have two or more distinct labels → **conflicting labels** (identical text, different annotation).

### Why group-level counting?

If five records share the same text, that is one duplicate situation, not five.
Counting five groups would inflate the problem report and confuse reviewers.

### Limitations

- Only detects *normalised* equality; slight paraphrasing is not caught here (see Method B).
- A label conflict within an exact group is reported here; Method B only reports conflicts between *different* groups.

---

## 3. Method B — Near-duplicates and conflicting labels (`duplicate_detection.py`)

### What it does

1. Fits TF-IDF vectorisers on eligible texts.
2. Computes pairwise cosine similarity in row blocks (500 rows at a time) to avoid an N×N dense matrix.
3. Classifies each pair:
   - `similarity ≥ near_dup_threshold` AND same label → **near_duplicate**
   - `similarity ≥ conflict_threshold` AND different label → **conflicting_label**
   - Both conditions → **near_dup_conflict**

**Representatives:** Only one record per exact-duplicate group participates in the similarity step, preventing the group from generating O(n²) redundant pairs.

### Why block-wise computation?

An N×N dense similarity matrix requires O(N²) memory. For N = 5,000 that is 25 million float64 values ≈ 200 MB; for N = 20,000 it is 3.2 GB. Block-wise computation keeps only one block in memory at a time.

### Why combined word + character TF-IDF?

Character n-grams catch:
- Near-identical texts that differ by a single word or character.
- Morphological variants and misspellings.
- Non-Latin script texts (partially).

The two matrices are each L2-normalised and scaled by 1/√2 before stacking.
This ensures the combined cosine equals the unweighted average of the two cosines.

### Why independent thresholds?

A pair might be a near-duplicate with the same label (a data redundancy issue) but
not a label conflict. Using a single threshold would force a trade-off that loses one
type of finding. Independent thresholds let reviewers tune each separately.

### Limitations

- TF-IDF is a lexical measure; semantically equivalent paraphrases with different
  words score low.
- Short texts have low cosine similarity even when nearly identical (sparse vectors).
- Above the pairwise limit the analysis is skipped entirely — no silent empty result.

---

## 4. Method C — Suspicious-label detection (`label_analysis.py`)

### What it does

Trains a TF-IDF + classifier pipeline with k-fold cross-validation.
Each record's label is predicted by a model that never saw that record during training
(out-of-fold prediction).  When the OOF prediction disagrees with the supplied label,
the record is flagged for review.

### Why out-of-fold predictions?

If you train on all data and predict on training data, the model memorises training labels.
A correct label would rarely be "disagreed with" even if it looked wrong, and the flag signal
would be meaningless.  OOF avoids this circularity.

### Why group duplicates in cross-validation?

If record A and its exact twin B are in different folds, the model trained on A "knows"
B's label through memorisation — not through generalisation.  The OOF prediction for B
would be contaminated.  `StratifiedGroupKFold` keeps twins together.

Detected near-duplicate and conflicting-label pairs are also joined into shared
components before cross-validation when Method B completes. If Method B is skipped
by the pairwise limit, only exact-duplicate groups can be kept together; if the pair
storage cap is reached, grouping uses the detected pairs returned before the cap.

### Fallback to StratifiedKFold

When the grouped split is infeasible (e.g. a class has all its records in one group,
so no split can keep them together and form k folds), we fall back to `StratifiedKFold`
and show a notice.  The trade-off is accepted; the alternative would be to refuse to run.

### Honest framing

- A flag is a review signal. The supplied label may still be correct.
- Low OOF accuracy (< 0.6) warns that flags likely reflect model weakness.
- We never call the model score a "confidence" or "probability of error".

### Limitations

- A simple TF-IDF + LR/SVM classifier may have low accuracy on short or ambiguous texts.
- Small datasets produce noisy OOF estimates.
- The classifier is not calibrated; model scores are ordering tools only.

---

## 5. Corruption experiment (`model_evaluation.py`)

### What it does

1. Selects eligible records (no conflicting dup/near-dup).
2. Randomly replaces a configurable fraction of labels with a different class.
3. Runs Method C on corrupted labels.
4. Compares flagged records with planted corruptions.
5. Repeats with several seeds; reports mean, std, and a random-flag baseline.

### Why a random baseline?

If the classifier flags records at random (same count as actual flags), the expected
precision equals the corruption rate.  Comparing against this baseline shows whether
the classifier does better than chance.

### Limitations

- **Synthetic ≠ real**: Real annotation errors are often systematic (e.g. annotators
  consistently confuse two classes), ambiguous (border-line examples), or clustered.
  Random corruption is none of these.
- **Ground truth assumption**: The experiment assumes original labels are correct.
  If the uploaded CSV already contains label errors, the experiment is misleading.
- **Only one dataset**: Results from one synthetic run do not generalise.

---

## 6. Examiner Q&A

**Q1: Why don't you remove stop words and stem?**
Stop words carry semantic meaning (negation, conditionals). Stemming merges distinct
concepts. Both decisions could make distinct examples look identical, creating false
duplicate findings. We prefer conservative normalisation.

**Q2: How do you prevent data leakage in cross-validation?**
`StratifiedGroupKFold` assigns all records sharing a duplicate/near-duplicate group ID
to the same fold.  The vectoriser and classifier are both fitted *inside* each fold's
`Pipeline`, so they only see training data.

**Q3: Why is a flag not proof of a wrong label?**
The classifier learns from patterns in the dataset.  If the majority label for a text
type is `A`, a record labelled `B` will be predicted as `A` even if `B` is correct.
The classifier is a signal, not an oracle.

**Q4: Why cosine similarity instead of Euclidean distance?**
Cosine similarity is document-length-invariant: a short document and its verbatim
copy padded with an extra sentence score 1.0 on cosine but different on Euclidean.
This is exactly what we want for duplicate detection.

**Q5: What happens when the pairwise limit is exceeded?**
Method B is skipped entirely.  The UI shows a clear message and suggests alternatives, such as preparing a random
subsample or restricting the data to selected labels before uploading again. Those
subsets are prepared outside the app. It never returns an empty result silently,
which would be misleading.

**Q6: Why does the corruption experiment run separately from the main audit?**
Planted errors would contaminate the main audit results.  Mixing them would mean
the reported flags include synthetic errors the user did not ask about.  The separation
also makes it clear that experiment results are SYNTHETIC and not part of the real audit.

**Q7: What is the character n-gram scaling, and why?**
Each TF-IDF matrix is L2-normalised (unit vectors), then scaled by 1/√2.
Horizontal stacking produces a combined vector whose L2 norm is also 1 (since
|1/√2 · a|² + |1/√2 · b|² = ½ + ½ = 1 for unit vectors a, b).  The cosine of the
combined vector equals the dot product of the combined vectors, which is
½(a·c) + ½(b·d) = average of the two cosines.  This gives equal weight to both
representations without any tuning.

**Q8: Why StratifiedGroupKFold instead of GroupKFold?**
Pure `GroupKFold` does not guarantee equal class proportions per fold.
For imbalanced datasets, some folds might contain no examples of a rare class,
making both training and evaluation unreliable.  The stratified variant maintains
class proportions while still honouring group constraints.
