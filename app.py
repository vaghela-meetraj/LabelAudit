"""
app.py — LabelAudit Streamlit interface.

Rendering and state management only.  All analysis logic is in src/audit.py.

Architecture:
- Session state holds the loaded dataset, configuration, and results.
- Settings are inside a st.form so changing widgets does not trigger a rerun
  until the user presses "Run Audit".
- Results are keyed by (dataset_fingerprint, config_hash) to detect staleness.
- Expensive operations run only on form submission.
"""

from __future__ import annotations

import hashlib
import io
import json
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.audit import AuditConfig, AuditResult, run_audit
from src.data_loading import (
    load_csv,
    validate_columns,
    compute_file_fingerprint,
)
from src.model_evaluation import (
    run_corruption_experiment,
    BaselineMetrics,
)
from src.reporting import (
    build_row_report,
    build_flagged_report,
    build_pair_report,
    build_config_summary,
    df_to_csv_bytes,
    dict_to_json_bytes,
    pairs_to_dataframe,
)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="LabelAudit",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Package versions (for config export)
# ---------------------------------------------------------------------------

def _get_package_versions() -> dict[str, str]:
    try:
        import importlib.metadata as meta
        pkgs = ["streamlit", "pandas", "scikit-learn", "plotly", "numpy"]
        return {p: meta.version(p) for p in pkgs}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "load_result": None,
        "file_fingerprint": None,
        "audit_result": None,
        "audit_key": None,          # fingerprint + config_hash
        "corruption_result": None,
        "settings_changed": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _results_stale() -> bool:
    """True if settings or data changed since the last audit run."""
    return st.session_state.get("settings_changed", False)


# ---------------------------------------------------------------------------
# Header and status
# ---------------------------------------------------------------------------

def _render_header() -> None:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🔍 LabelAudit")
        st.caption("Inspect text datasets for duplicates and potential label issues.")
    with col2:
        st.write("")
        if st.session_state.audit_result is not None:
            if _results_stale():
                st.warning("⚠️ Results out of date — settings changed.")
            else:
                st.success("✅ Audit complete")
        elif st.session_state.load_result is not None:
            st.info("📂 Dataset loaded")
        else:
            st.info("⬆️ No dataset")


# ---------------------------------------------------------------------------
# Sidebar upload + main-workspace settings form
# ---------------------------------------------------------------------------

def _render_sidebar() -> Optional[AuditConfig]:
    """
    Render the dataset uploader and main-workspace settings form.
    Returns an AuditConfig if the form was submitted, else None.
    """
    st.sidebar.header("Dataset")

    # --- File upload (outside form so it triggers immediately) ---
    uploaded = st.sidebar.file_uploader(
        "Upload CSV (max 50 MB)",
        type=["csv"],
        help="Accepts comma-, semicolon-, or tab-delimited CSV files.",
    )

    if uploaded is not None:
        raw = uploaded.read()
        fingerprint = compute_file_fingerprint(raw)

        # New file → reload
        if fingerprint != st.session_state.file_fingerprint:
            load_result = load_csv(raw, uploaded.name)
            st.session_state.load_result = load_result
            st.session_state.file_fingerprint = fingerprint
            st.session_state.audit_result = None
            st.session_state.audit_key = None
            st.session_state.corruption_result = None
            st.session_state.settings_changed = False

            if load_result.errors:
                for err in load_result.errors:
                    st.sidebar.error(err)
            else:
                if load_result.encoding_warning:
                    st.sidebar.warning(load_result.warnings[0])
                st.sidebar.success(
                    f"Loaded **{load_result.filename}** "
                    f"({load_result.n_usable_rows if hasattr(load_result, 'n_usable_rows') else len(load_result.df)} rows) "
                    f"• delimiter: `{load_result.delimiter_used!r}` "
                    f"• encoding: `{load_result.encoding_used}`"
                )

    load_result = st.session_state.load_result

    if load_result is None or load_result.df.empty:
        st.sidebar.info("Upload a CSV file to begin.")
        return None

    df = load_result.df
    all_cols = [c for c in df.columns if c != "csv_row"]

    # --- Settings form ---
    submitted_config: Optional[AuditConfig] = None

    st.subheader("Audit setup")

    with st.form("settings_form"):
        st.subheader("Column selection")
        text_col = st.selectbox(
            "Text column",
            all_cols,
            index=0,
            key="form_text_col",
        )
        label_col = st.selectbox(
            "Label column",
            all_cols,
            index=min(1, len(all_cols) - 1),
            key="form_label_col",
        )

        run_clicked = st.form_submit_button(
            "▶ Run Audit",
            use_container_width=True,
            type="primary",
        )

        st.subheader("Similarity settings")
        st.caption(
            "Word TF-IDF (1–2-grams) + optional character TF-IDF (3–5-grams). "
            "When both are used, cosine similarity = average of word and character cosines."
        )
        near_dup_thresh = st.slider(
            "Near-duplicate threshold",
            min_value=0.50, max_value=1.00, value=0.90, step=0.01,
            help="Pairs at or above this threshold with the same label are near-duplicates.",
        )
        conflict_thresh = st.slider(
            "Conflicting-label threshold",
            min_value=0.50, max_value=1.00, value=0.80, step=0.01,
            help="Pairs at or above this threshold with different labels are conflicts.",
        )
        use_char = st.checkbox("Use character n-grams", value=True)
        pairwise_limit = st.number_input(
            "Pairwise analysis limit",
            min_value=100, max_value=20000, value=5000, step=100,
            help="Skip near-dup analysis if usable records exceed this count.",
        )

        st.subheader("Classifier settings")
        classifier = st.selectbox(
            "Classifier",
            ["LogisticRegression", "LinearSVC"],
        )
        class_weight = st.checkbox("Balanced class weights", value=True)
        n_folds = st.slider("Cross-validation folds", min_value=2, max_value=10, value=5)
        random_seed = st.number_input("Random seed", min_value=0, value=42, step=1)
        min_class_size = st.number_input(
            "Min class size for CV",
            min_value=2, max_value=20, value=3, step=1,
            help="Classes with fewer records are excluded from label analysis.",
        )


    if run_clicked:
        val = validate_columns(df, text_col, label_col)
        if not val.valid:
            for err in val.errors:
                st.error(err)
            return None
        submitted_config = AuditConfig(
            text_col=text_col,
            label_col=label_col,
            near_dup_threshold=near_dup_thresh,
            conflict_threshold=conflict_thresh,
            use_char_ngrams=use_char,
            classifier_name=classifier,
            class_weight="balanced" if class_weight else "",
            n_folds=n_folds,
            random_seed=int(random_seed),
            min_class_size=int(min_class_size),
            pairwise_limit=int(pairwise_limit),
        )
        st.session_state.settings_changed = False

    return submitted_config


# ---------------------------------------------------------------------------
# Tab 1 — Dataset overview
# ---------------------------------------------------------------------------

def _tab_overview(result: AuditResult) -> None:
    cfg = result.config
    excl = result.exclusion
    df = result.df_with_norm

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total rows", excl.n_total_rows)
    col2.metric("Usable rows", excl.n_usable)
    col3.metric("Missing text", excl.n_missing_text)
    col4.metric("Missing label", excl.n_missing_label)

    st.subheader("Dataset preview")
    preview_cols = [c for c in df.columns if c not in {"_norm_text"}]
    st.dataframe(
        df[preview_cols].head(20),
        column_config={
            cfg.text_col: st.column_config.TextColumn(width="large"),
        },
        use_container_width=True,
        height=300,
    )

    st.subheader("Class distribution")
    label_counts = (
        df[~df["_norm_text"].eq("")]
        [cfg.label_col]
        .value_counts()
        .reset_index()
    )
    label_counts.columns = ["label", "count"]
    fig = px.bar(
        label_counts,
        x="label", y="count",
        color="label",
        color_discrete_sequence=px.colors.qualitative.Safe,
        labels={"label": cfg.label_col, "count": "Records"},
        title=f"Records per class (column: {cfg.label_col})",
    )
    fig.update_layout(showlegend=False, plot_bgcolor="white", height=300)
    st.plotly_chart(fig, use_container_width=True)

    # Exclusion summary
    with st.expander("Exclusion summary"):
        rows = []
        if excl.n_missing_text > 0:
            rows.append({"Reason": "Empty/blank text after normalisation",
                         "Count": excl.n_missing_text})
        if excl.n_missing_label > 0:
            rows.append({"Reason": "Empty/blank label", "Count": excl.n_missing_label})
        if excl.n_small_class_excluded > 0:
            rows.append({"Reason": f"Class has fewer than {cfg.min_class_size} records (excluded from CV only)",
                         "Count": excl.n_small_class_excluded})
        if excl.pairwise_limit_hit:
            rows.append({"Reason": f"Over pairwise limit ({cfg.pairwise_limit:,}) — near-dup skipped",
                         "Count": excl.n_over_pairwise_limit})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No records were excluded.")


# ---------------------------------------------------------------------------
# Tab 2 — Exact duplicates
# ---------------------------------------------------------------------------

def _tab_exact_dups(result: AuditResult) -> None:
    ex = result.exact_dup
    cfg = result.config

    col1, col2, col3 = st.columns(3)
    col1.metric("Duplicate groups", ex.n_groups,
                help="Each group has ≥ 2 records with identical normalised text.")
    col2.metric("Affected records", ex.n_affected_records,
                help="Total records belonging to any duplicate group.")
    col3.metric("Redundant records", ex.n_redundant_records,
                help="Affected records minus one representative per group.")

    if ex.n_groups == 0:
        st.info("✅ No exact duplicates found.")
        return

    n_conflict_groups = sum(1 for g in ex.groups if g.label_conflict)
    if n_conflict_groups:
        st.warning(
            f"⚠️ **{n_conflict_groups} group(s)** contain records with conflicting labels — "
            "identical text but different supplied labels. These warrant manual review."
        )

    for i, group in enumerate(ex.groups):
        label = "⚠️ Label conflict" if group.label_conflict else "✓ Matching labels"
        with st.expander(
            f"Group {group.group_id} — {len(group.members)} records — {label}",
            expanded=(i < 3),
        ):
            show_cols = [c for c in group.members.columns if c not in {"_norm_text"}]
            st.dataframe(
                group.members[show_cols],
                column_config={
                    cfg.text_col: st.column_config.TextColumn(width="large"),
                },
                use_container_width=True,
            )


# ---------------------------------------------------------------------------
# Tab 3 — Near-duplicates
# ---------------------------------------------------------------------------

def _tab_near_dups(result: AuditResult) -> None:
    if result.near_dup_skipped:
        st.warning(f"⚠️ {result.near_dup_skip_reason}")
        return

    nd = result.near_dup
    cfg = result.config

    if nd is None:
        st.info("Near-duplicate analysis did not run.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Total pairs", nd.n_pairs_total)
    col2.metric("Near-duplicate pairs", nd.n_near_dup)
    col3.metric("Conflict pairs", nd.n_conflicts)

    st.caption(
        f"Near-duplicate threshold: **{cfg.near_dup_threshold}** | "
        f"Conflict threshold: **{cfg.conflict_threshold}** | "
        f"Records analysed: **{nd.n_analyzed_records}**"
    )

    if nd.hit_pair_cap:
        st.warning("⚠️ Pair storage cap reached — results shown are partial.")

    if nd.n_pairs_total == 0:
        st.info("✅ No near-duplicate or conflicting pairs found at the current thresholds.")
        return

    pair_df = pairs_to_dataframe(nd.pairs)

    # Filters
    st.subheader("Filters")
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        type_filter = st.multiselect(
            "Finding type",
            options=["near_duplicate", "conflicting_label", "near_dup_conflict"],
            default=["near_duplicate", "conflicting_label", "near_dup_conflict"],
        )
    with fcol2:
        min_sim = st.slider("Min similarity", 0.0, 1.0, float(min(cfg.near_dup_threshold, cfg.conflict_threshold)), 0.01)

    filtered = pair_df[
        pair_df["finding_type"].isin(type_filter) &
        (pair_df["similarity"] >= min_sim)
    ]

    st.subheader(f"Pairs ({len(filtered)} shown)")
    st.dataframe(
        filtered.assign(similarity=filtered["similarity"].round(4)),
        column_config={
            "text_a": st.column_config.TextColumn(width="large"),
            "text_b": st.column_config.TextColumn(width="large"),
        },
        use_container_width=True,
        height=400,
    )

    # Similarity histogram
    st.subheader("Similarity distribution")
    fig = px.histogram(
        pair_df, x="similarity", color="finding_type",
        nbins=40,
        color_discrete_map={
            "near_duplicate": "#2563EB",
            "conflicting_label": "#DC2626",
            "near_dup_conflict": "#D97706",
        },
        labels={"similarity": "Cosine similarity", "count": "Pairs"},
        title="Distribution of pairwise cosine similarities",
    )
    fig.update_layout(plot_bgcolor="white", height=280)
    st.plotly_chart(fig, use_container_width=True)

    # Detail view
    if len(filtered) > 0:
        st.subheader("Pair detail view")
        row_options = list(filtered.index)
        selected_idx = st.selectbox("Select row index to inspect", row_options)
        row = filtered.loc[selected_idx]
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.markdown(f"**csv_row {row['csv_row_a']}** — label: `{row['label_a']}`")
            st.text_area("Text A", row["text_a"], height=150, disabled=True, key=f"ta_{selected_idx}")
        with dcol2:
            st.markdown(f"**csv_row {row['csv_row_b']}** — label: `{row['label_b']}`")
            st.text_area("Text B", row["text_b"], height=150, disabled=True, key=f"tb_{selected_idx}")
        st.markdown(f"**Similarity:** `{row['similarity']:.4f}` | **Type:** `{row['finding_type']}`")
        st.markdown(f"**Reason:** {row['reason']}")


# ---------------------------------------------------------------------------
# Tab 4 — Suspicious labels
# ---------------------------------------------------------------------------

def _tab_suspicious_labels(result: AuditResult) -> None:
    la = result.label_analysis

    if la.cannot_run:
        st.error(f"Label analysis could not run: {la.cannot_run_reason}")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Flagged", la.n_flagged)
    col2.metric("Not flagged", la.n_not_flagged)
    col3.metric("Excluded", la.n_excluded)
    col4.metric(
        "OOF accuracy",
        f"{la.oof_accuracy:.2%}" if la.oof_accuracy is not None else "—",
    )

    splitter_note = f"Splitter: **{la.splitter_used}** ({la.folds_used} folds)"
    if la.splitter_fallback_reason:
        st.warning(f"⚠️ {la.splitter_fallback_reason}")
    st.caption(splitter_note)

    if la.low_accuracy_warning:
        st.warning(
            "⚠️ OOF accuracy is below 0.6. Many flags likely reflect model weakness, "
            "not label errors. Review with care."
        )

    st.info(
        "**How to read flags:** A model that was not trained on each record predicted "
        "a different label. This is a review signal only — the supplied label may still "
        "be correct. Disagreement does not prove an error."
    )

    if la.n_flagged == 0:
        st.success("✅ No records flagged by the out-of-fold classifier.")

    # Build display dataframe
    flagged_records = [r for r in la.records if r.status == "flagged"]
    if not flagged_records:
        return

    flagged_data = [
        {
            "csv_row": r.csv_row,
            "text": r.orig_text,
            "supplied_label": r.supplied_label,
            "predicted_label": r.predicted_label,
            "model_score": f"{r.model_score:.4f}" if r.model_score is not None else "—",
        }
        for r in flagged_records
    ]
    flagged_df = pd.DataFrame(flagged_data)

    # Filters
    supplied_options = sorted(flagged_df["supplied_label"].unique())
    pred_options = sorted(flagged_df["predicted_label"].unique())
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        sl_filter = st.multiselect("Supplied label", supplied_options, default=supplied_options)
    with fcol2:
        pl_filter = st.multiselect("Predicted label", pred_options, default=pred_options)

    filtered = flagged_df[
        flagged_df["supplied_label"].isin(sl_filter) &
        flagged_df["predicted_label"].isin(pl_filter)
    ]

    st.subheader(f"Flagged records ({len(filtered)} shown)")
    st.dataframe(
        filtered,
        column_config={"text": st.column_config.TextColumn(width="large")},
        use_container_width=True,
        height=400,
    )

    # Flags per supplied class chart
    by_class = flagged_df.groupby("supplied_label").size().reset_index(name="flags")
    fig = px.bar(
        by_class, x="supplied_label", y="flags",
        color_discrete_sequence=["#DC2626"],
        labels={"supplied_label": "Supplied label", "flags": "Flagged records"},
        title="Flagged records per supplied class",
    )
    fig.update_layout(showlegend=False, plot_bgcolor="white", height=280)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 5 — Classifier evaluation
# ---------------------------------------------------------------------------

def _tab_evaluation(result: AuditResult) -> None:
    metrics = result.baseline_metrics
    la = result.label_analysis

    st.subheader("Baseline classifier metrics")
    st.caption(
        "These metrics measure how well a simple TF-IDF classifier generalises on "
        "this dataset — NOT how accurate the audit flags are."
    )

    if la.cannot_run:
        st.error(f"Classifier could not run: {la.cannot_run_reason}")
        return

    if metrics is None:
        st.warning("No metrics available — no eligible predictions were produced.")
        return

    col1, col2 = st.columns(2)
    col1.metric("OOF accuracy", f"{metrics.accuracy:.2%}")
    col2.metric("Macro F1", f"{metrics.macro_f1:.3f}")

    # Per-class F1 chart
    class_data = [
        {"class": cls, **vals}
        for cls, vals in metrics.per_class.items()
    ]
    class_df = pd.DataFrame(class_data)
    fig = px.bar(
        class_df, x="class", y="f1",
        color_discrete_sequence=["#2563EB"],
        labels={"class": "Class", "f1": "F1 score"},
        title="Per-class F1 (out-of-fold)",
    )
    fig.update_layout(showlegend=False, plot_bgcolor="white", height=280)
    st.plotly_chart(fig, use_container_width=True)

    # Per-class table
    st.dataframe(class_df.set_index("class").round(3), use_container_width=True)

    # Confusion matrix
    st.subheader("Confusion matrix")
    cm_df = pd.DataFrame(
        metrics.confusion_matrix,
        index=metrics.labels,
        columns=metrics.labels,
    )
    fig_cm = go.Figure(data=go.Heatmap(
        z=metrics.confusion_matrix,
        x=metrics.labels,
        y=metrics.labels,
        colorscale="Blues",
        text=metrics.confusion_matrix,
        texttemplate="%{text}",
        showscale=True,
    ))
    fig_cm.update_layout(
        xaxis_title="Predicted",
        yaxis_title="True",
        height=350,
        yaxis_autorange="reversed",
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    # --- Corruption experiment tab ---
    st.markdown("---")
    st.subheader("🧪 Corruption Experiment — SYNTHETIC / EXPERIMENTAL")
    st.warning(
        "⚠️ This experiment plants random label errors and measures how well "
        "the classifier detects them. **Synthetic random corruption does not represent "
        "real annotation errors**, which are often systematic and ambiguous. "
        "Results here do not establish real-world label-error detection accuracy."
    )

    with st.expander("Configure and run experiment", expanded=False):
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            corruption_rate = st.slider(
                "Corruption rate", 0.01, 0.40, 0.10, 0.01,
                key="exp_rate",
            )
            n_seeds = st.slider("Seeds for averaging", 2, 10, 5, key="exp_seeds")
        with exp_col2:
            exp_seed = st.number_input("Chosen seed", 0, 999, 42, key="exp_seed")

        run_exp = st.button("Run experiment", key="run_experiment")

    if run_exp:
        cfg = result.config
        df = result.df_with_norm
        eligible_mask = (
            (df["_norm_text"] != "") &
            (df[cfg.label_col].apply(lambda v: isinstance(v, str) and v.strip() != ""))
        )
        # Exclude records from conflicting exact-dup groups
        conflict_rows: set[int] = set()
        for g in result.exact_dup.groups:
            if g.label_conflict:
                conflict_rows.update(g.members["csv_row"].tolist())

        with st.spinner("Running corruption experiment…"):
            exp_result = run_corruption_experiment(
                df,
                text_col=cfg.text_col,
                label_col=cfg.label_col,
                norm_col="_norm_text",
                eligible_mask=eligible_mask,
                corruption_rate=corruption_rate,
                chosen_seed=int(exp_seed),
                n_seeds=n_seeds,
                exclude_csv_rows=conflict_rows,
                classifier_name=cfg.classifier_name,
                class_weight=cfg.class_weight,
                n_folds=cfg.n_folds,
                min_class_size=cfg.min_class_size,
            )
        st.session_state.corruption_result = exp_result

    exp_result = st.session_state.corruption_result
    if exp_result is not None:
        if exp_result.cannot_run:
            st.error(f"Experiment cannot run: {exp_result.cannot_run_reason}")
        else:
            sr = exp_result.chosen_seed_result

            def _fmt(v):
                return f"{v:.3f}" if v is not None else "undefined"

            ecol1, ecol2, ecol3 = st.columns(3)
            ecol1.metric("Precision (chosen seed)", _fmt(sr.precision),
                         help="TP / (TP + FP) among flagged records")
            ecol2.metric("Recall (chosen seed)", _fmt(sr.recall),
                         help="TP / (TP + FN) among corrupted records")
            ecol3.metric("F1 (chosen seed)", _fmt(sr.f1))

            st.markdown(
                f"Corrupted: **{sr.n_corrupted}** | "
                f"Flagged: **{sr.n_flagged}** | "
                f"TP: **{sr.tp}** | FP: **{sr.fp}** | FN: **{sr.fn}**"
            )
            st.markdown(
                f"**Random baseline precision** (= corruption rate): "
                f"`{exp_result.random_baseline_precision:.3f}`"
            )
            if exp_result.mean_precision is not None:
                st.markdown(
                    f"Multi-seed mean precision: `{exp_result.mean_precision:.3f}` "
                    f"±`{exp_result.std_precision:.3f}` | "
                    f"mean recall: `{exp_result.mean_recall:.3f}` "
                    f"±`{exp_result.std_recall:.3f}`"
                )


# ---------------------------------------------------------------------------
# Tab 6 — Export
# ---------------------------------------------------------------------------

def _tab_export(result: AuditResult, sanitize: bool = True) -> None:
    cfg = result.config

    st.subheader("Downloads")
    sanitize_toggle = st.checkbox(
        "Guard against spreadsheet formula injection",
        value=True,
        help=(
            "Cells beginning with =, +, - or @ are prefixed with an apostrophe. "
            "Standard protection for opening CSVs in Excel or Google Sheets."
        ),
    )

    col1, col2, col3 = st.columns(3)

    # Full row report
    with col1:
        full_report = build_row_report(result, sanitize=sanitize_toggle)
        st.download_button(
            "📥 Complete audit report",
            data=df_to_csv_bytes(full_report),
            file_name="labelaudit_full_report.csv",
            mime="text/csv",
        )

    # Flagged records
    with col2:
        flagged_report = build_flagged_report(result, sanitize=sanitize_toggle)
        st.download_button(
            "📥 Flagged records only",
            data=df_to_csv_bytes(flagged_report),
            file_name="labelaudit_flagged.csv",
            mime="text/csv",
        )

    # Pair report
    with col3:
        pair_report = build_pair_report(result, sanitize=sanitize_toggle)
        st.download_button(
            "📥 Near-dup / conflict pairs",
            data=df_to_csv_bytes(pair_report),
            file_name="labelaudit_pairs.csv",
            mime="text/csv",
        )

    # Config summary
    pkg_versions = _get_package_versions()
    config_summary = build_config_summary(result, pkg_versions)
    st.download_button(
        "📥 Configuration summary (JSON)",
        data=dict_to_json_bytes(config_summary),
        file_name="labelaudit_config.json",
        mime="application/json",
    )

    if st.session_state.corruption_result is not None:
        exp = st.session_state.corruption_result
        if not exp.cannot_run:
            st.download_button(
                "📥 Corruption experiment results (JSON) — SYNTHETIC",
                data=dict_to_json_bytes({
                    "WARNING": "Synthetic random corruption. Not a real-world benchmark.",
                    "corruption_rate": exp.corruption_rate,
                    "n_eligible": exp.n_eligible,
                    "random_baseline_precision": exp.random_baseline_precision,
                    "chosen_seed": {
                        "seed": exp.chosen_seed_result.seed,
                        "n_corrupted": exp.chosen_seed_result.n_corrupted,
                        "n_flagged": exp.chosen_seed_result.n_flagged,
                        "tp": exp.chosen_seed_result.tp,
                        "precision": exp.chosen_seed_result.precision,
                        "recall": exp.chosen_seed_result.recall,
                        "f1": exp.chosen_seed_result.f1,
                    },
                    "mean_precision": exp.mean_precision,
                    "std_precision": exp.std_precision,
                    "mean_recall": exp.mean_recall,
                    "std_recall": exp.std_recall,
                }),
                file_name="labelaudit_experiment_SYNTHETIC.json",
                mime="application/json",
            )

    st.subheader("Configuration summary")
    st.json(config_summary)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _init_state()

    # Reserve the header position, then render it after upload handling so its
    # status reflects a newly uploaded dataset on the same run.
    header_slot = st.container()
    submitted_config = _render_sidebar()
    with header_slot:
        _render_header()

    # --- Run audit if form submitted ---
    if submitted_config is not None:
        load_result = st.session_state.load_result
        fingerprint = st.session_state.file_fingerprint
        audit_key = f"{fingerprint}-{submitted_config.config_hash()}"

        if st.session_state.audit_key != audit_key:
            progress_bar = st.progress(0, text="Preparing…")

            def _progress(step: int, msg: str) -> None:
                progress_bar.progress(step / 6, text=msg)

            with st.spinner("Running audit…"):
                try:
                    audit_result = run_audit(
                        load_result.df,
                        submitted_config,
                        fingerprint,
                        progress_callback=_progress,
                    )
                    st.session_state.audit_result = audit_result
                    st.session_state.audit_key = audit_key
                    st.session_state.corruption_result = None
                    progress_bar.empty()
                    st.success("Audit complete.")
                except Exception as exc:
                    progress_bar.empty()
                    st.error(f"Audit failed: {exc}")
                    raise

    # --- Render results ---
    audit_result: Optional[AuditResult] = st.session_state.audit_result
    if audit_result is None:
        if st.session_state.load_result is None:
            st.markdown(
                """
                ### Getting started
                1. Upload a CSV file using the sidebar.
                2. Select the text and label columns.
                3. Adjust settings and press **▶ Run Audit**.
                """
            )
        else:
            st.info("Choose your columns and settings in the Audit setup form above, then press **▶ Run Audit**.")
        return

    tabs = st.tabs([
        "📊 Overview",
        "🔁 Exact Duplicates",
        "🔍 Similar Texts",
        "🏷️ Suspicious Labels",
        "📈 Evaluation",
        "📤 Export",
    ])

    with tabs[0]:
        _tab_overview(audit_result)
    with tabs[1]:
        _tab_exact_dups(audit_result)
    with tabs[2]:
        _tab_near_dups(audit_result)
    with tabs[3]:
        _tab_suspicious_labels(audit_result)
    with tabs[4]:
        _tab_evaluation(audit_result)
    with tabs[5]:
        _tab_export(audit_result)


if __name__ == "__main__":
    main()
