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
# Custom CSS for a polished, professional UI
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Clean typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Header styling */
    .main-header {
        padding: 1.5rem 0 1rem 0;
        border-bottom: 2px solid #E2E8F0;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0 0 0.25rem 0;
        letter-spacing: -0.02em;
    }
    .main-header p {
        font-size: 0.95rem;
        color: #64748B;
        margin: 0;
    }

    /* Status badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.01em;
    }
    .status-ready {
        background: #F0FDF4;
        color: #166534;
        border: 1px solid #BBF7D0;
    }
    .status-loaded {
        background: #EFF6FF;
        color: #1E40AF;
        border: 1px solid #BFDBFE;
    }
    .status-empty {
        background: #F8FAFC;
        color: #64748B;
        border: 1px solid #E2E8F0;
    }
    .status-stale {
        background: #FFFBEB;
        color: #92400E;
        border: 1px solid #FDE68A;
    }

    /* Section headers */
    .section-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        margin-bottom: 0.5rem;
    }

    /* Column selector cards */
    .col-card {
        background: #FFFFFF;
        border: 2px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        transition: border-color 0.15s ease;
    }
    .col-card:hover {
        border-color: #93C5FD;
    }
    .col-card.selected {
        border-color: #2563EB;
        background: #EFF6FF;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    div[data-testid="stMetric"] label {
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748B !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: #0F172A !important;
    }

    /* Tab styling */
    button[data-baseweb="tab"] {
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 12px 20px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom-color: #2563EB !important;
        color: #2563EB !important;
    }

    /* Expander styling */
    details[data-testid="stExpander"] {
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        background: #FFFFFF !important;
    }

    /* Form submit button */
    div[data-testid="stFormSubmitButton"] > button {
        background: #2563EB !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.6rem 1.5rem !important;
        letter-spacing: 0.01em;
        transition: background 0.15s ease;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: #1D4ED8 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #F8FAFC !important;
        border-right: 1px solid #E2E8F0;
    }

    /* Download buttons */
    div[data-testid="stDownloadButton"] > button {
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        background: #FFFFFF !important;
        transition: all 0.15s ease;
    }
    div[data-testid="stDownloadButton"] > button:hover {
        border-color: #2563EB !important;
        background: #EFF6FF !important;
    }

    /* Dataframe */
    div[data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        overflow: hidden;
    }

    /* Progress bar */
    div[data-testid="stProgress"] > div {
        border-radius: 4px !important;
    }

    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Getting-started cards */
    .step-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.25rem;
        text-align: center;
        height: 100%;
    }
    .step-card .step-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: #EFF6FF;
        color: #2563EB;
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 0.75rem;
    }
    .step-card h4 {
        font-size: 0.9rem;
        font-weight: 600;
        color: #0F172A;
        margin: 0 0 0.4rem 0;
    }
    .step-card p {
        font-size: 0.82rem;
        color: #64748B;
        margin: 0;
        line-height: 1.45;
    }

    /* Multiselect row selector */
    .row-selector {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


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
        st.markdown("""
        <div class="main-header">
            <h1>🔍 LabelAudit</h1>
            <p>Inspect text datasets for duplicates and potential label issues</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.write("")
        if st.session_state.audit_result is not None:
            if _results_stale():
                st.markdown(
                    '<span class="status-badge status-stale">⚠ Results out of date</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span class="status-badge status-ready">✓ Audit complete</span>',
                    unsafe_allow_html=True,
                )
        elif st.session_state.load_result is not None:
            st.markdown(
                '<span class="status-badge status-loaded">📂 Dataset loaded</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-badge status-empty">⬆ Upload a dataset</span>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Sidebar upload + main-workspace settings form
# ---------------------------------------------------------------------------

def _render_sidebar() -> Optional[AuditConfig]:
    """
    Render the dataset uploader and main-workspace settings form.
    Returns an AuditConfig if the form was submitted, else None.
    """
    st.sidebar.markdown("### 📂 Dataset")

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

    st.markdown('<p class="section-label">Audit Setup</p>', unsafe_allow_html=True)

    with st.form("settings_form"):
        # Column selection with radio buttons
        st.markdown("##### Column Selection")
        col_left, col_right = st.columns(2)

        with col_left:
            text_col = st.radio(
                "Text column",
                all_cols,
                index=0,
                key="form_text_col",
                help="The column containing the text data to analyze.",
            )

        with col_right:
            label_col = st.radio(
                "Label column",
                all_cols,
                index=min(1, len(all_cols) - 1),
                key="form_label_col",
                help="The column containing the category/class labels.",
            )

        st.divider()

        # Row selection for audit
        st.markdown("##### Row Selection")
        row_selection_mode = st.radio(
            "Rows to audit",
            ["All rows", "Select specific rows", "Select row range"],
            index=0,
            key="form_row_mode",
            horizontal=True,
        )

        selected_rows = None
        csv_row_list = df["csv_row"].tolist()

        if row_selection_mode == "Select specific rows":
            def _format_row_label(r_num: int) -> str:
                match = df[df["csv_row"] == r_num]
                if not match.empty:
                    row_data = match.iloc[0]
                    lbl = str(row_data.get(label_col, ""))
                    txt = str(row_data.get(text_col, ""))[:35].strip()
                    return f"Row {r_num} [{lbl}]: {txt}…" if txt else f"Row {r_num} [{lbl}]"
                return f"Row {r_num}"

            selected_rows = st.multiselect(
                "Pick multiple rows to audit",
                options=csv_row_list,
                default=csv_row_list[:min(15, len(csv_row_list))],
                format_func=_format_row_label,
                help="Select multiple individual rows to include in this audit run.",
                key="form_selected_rows",
            )
            if selected_rows:
                st.caption(f"✓ {len(selected_rows)} of {len(csv_row_list)} rows selected for audit")
            else:
                st.warning("Please select at least 2 rows to run the audit.")

        elif row_selection_mode == "Select row range":
            min_r = int(min(csv_row_list))
            max_r = int(max(csv_row_list))
            range_col1, range_col2 = st.columns(2)
            with range_col1:
                start_row = st.number_input(
                    "Start csv_row",
                    min_value=min_r,
                    max_value=max_r,
                    value=min_r,
                    step=1,
                    key="form_range_start",
                )
            with range_col2:
                end_row = st.number_input(
                    "End csv_row",
                    min_value=min_r,
                    max_value=max_r,
                    value=max_r,
                    step=1,
                    key="form_range_end",
                )
            if start_row <= end_row:
                selected_rows = [r for r in csv_row_list if int(start_row) <= r <= int(end_row)]
                st.caption(f"✓ {len(selected_rows)} rows in range [{start_row}, {end_row}] selected for audit")
            else:
                st.error("Start row must be less than or equal to End row.")

        st.divider()

        # Similarity settings
        st.markdown("##### Similarity Settings")
        similarity_engine = st.radio(
            "Similarity Engine",
            [
                "Semantic Embeddings (Understands sentence meaning & paraphrases)",
                "Lexical TF-IDF (Surface word & character overlap)",
            ],
            index=0,
            key="form_similarity_engine",
            help=(
                "Semantic Embeddings uses all-MiniLM-L6-v2 neural embeddings to understand "
                "sentences that have the SAME meaning with DIFFERENT wording (synonyms, paraphrases). "
                "Lexical TF-IDF matches literal surface words and character overlaps."
            ),
        )
        is_semantic = "Semantic" in similarity_engine

        if is_semantic:
            st.caption(
                "🧠 **Semantic Neural Embeddings (`all-MiniLM-L6-v2`) active.** "
                "Understands sentence meaning, synonyms, and paraphrasing regardless of differing wording."
            )
        else:
            st.caption(
                "🔤 **Lexical TF-IDF active.** "
                "Word TF-IDF (1–2-grams) + optional character TF-IDF (3–5-grams)."
            )

        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            near_dup_thresh = st.slider(
                "Near-duplicate threshold",
                min_value=0.40 if is_semantic else 0.50,
                max_value=1.00,
                value=0.65 if is_semantic else 0.90,
                step=0.01,
                help="Pairs at or above this threshold with the same label are near-duplicates.",
                key="form_near_dup_thresh",
            )
        with sim_col2:
            conflict_thresh = st.slider(
                "Conflict threshold",
                min_value=0.40 if is_semantic else 0.50,
                max_value=1.00,
                value=0.60 if is_semantic else 0.80,
                step=0.01,
                help="Pairs at or above this threshold with different labels are conflicts.",
                key="form_conflict_thresh",
            )

        opt_col1, opt_col2 = st.columns(2)
        with opt_col1:
            use_char = st.toggle(
                "Character n-grams",
                value=True,
                disabled=is_semantic,
                help="Applicable to Lexical TF-IDF.",
            )
        with opt_col2:
            pairwise_limit = st.number_input(
                "Pairwise limit",
                min_value=100, max_value=20000, value=5000, step=100,
                help="Skip near-dup analysis if usable records exceed this count.",
            )

        st.divider()

        # Classifier settings
        st.markdown("##### Classifier Settings")
        clf_col1, clf_col2 = st.columns(2)
        with clf_col1:
            classifier = st.radio(
                "Classifier",
                ["LogisticRegression", "LinearSVC"],
                key="form_classifier",
                horizontal=True,
            )
            class_weight = st.toggle("Balanced class weights", value=True)
        with clf_col2:
            n_folds = st.slider("CV folds", min_value=2, max_value=10, value=5)
            random_seed = st.number_input("Random seed", min_value=0, value=42, step=1)

        min_class_size = st.number_input(
            "Min class size for CV",
            min_value=2, max_value=20, value=3, step=1,
            help="Classes with fewer records are excluded from label analysis.",
        )

        st.write("")
        run_clicked = st.form_submit_button(
            "▶ Run Audit",
            use_container_width=True,
            type="primary",
        )

    if run_clicked:
        # Handle row selection
        audit_df = df
        if selected_rows is not None and len(selected_rows) > 0:
            audit_df = df[df["csv_row"].isin(selected_rows)].reset_index(drop=True)
            if len(audit_df) == 0:
                st.error("No rows selected. Pick at least one row.")
                return None

        # Store the filtered df for use in main()
        st.session_state["_audit_df"] = audit_df

        val = validate_columns(audit_df, text_col, label_col)
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
            similarity_method="semantic" if is_semantic else "tfidf",
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

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total rows", excl.n_total_rows)
    m2.metric("Usable rows", excl.n_usable)
    m3.metric("Missing text", excl.n_missing_text)
    m4.metric("Missing label", excl.n_missing_label)

    st.write("")

    # Dataset preview
    st.markdown("##### Dataset Preview")
    preview_cols = [c for c in df.columns if c not in {"_norm_text"}]
    st.dataframe(
        df[preview_cols].head(20),
        column_config={
            cfg.text_col: st.column_config.TextColumn(width="large"),
        },
        use_container_width=True,
        height=320,
    )

    # Class distribution
    st.markdown("##### Class Distribution")
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
        color_discrete_sequence=["#2563EB", "#7C3AED", "#059669", "#DC2626", "#D97706",
                                 "#0891B2", "#4F46E5", "#DB2777"],
        labels={"label": cfg.label_col, "count": "Records"},
    )
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(t=10, b=40),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Exclusion summary
    with st.expander("Exclusion details"):
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
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No records were excluded.")


# ---------------------------------------------------------------------------
# Tab 2 — Exact duplicates
# ---------------------------------------------------------------------------

def _tab_exact_dups(result: AuditResult) -> None:
    ex = result.exact_dup
    cfg = result.config

    m1, m2, m3 = st.columns(3)
    m1.metric("Duplicate groups", ex.n_groups,
              help="Each group has ≥ 2 records with identical normalised text.")
    m2.metric("Affected records", ex.n_affected_records,
              help="Total records belonging to any duplicate group.")
    m3.metric("Redundant records", ex.n_redundant_records,
              help="Affected records minus one representative per group.")

    if ex.n_groups == 0:
        st.info("No exact duplicates found.")
        return

    n_conflict_groups = sum(1 for g in ex.groups if g.label_conflict)
    if n_conflict_groups:
        st.warning(
            f"**{n_conflict_groups} group(s)** contain records with conflicting labels — "
            "identical text but different supplied labels."
        )

    for i, group in enumerate(ex.groups):
        label = "⚠ Label conflict" if group.label_conflict else "✓ Labels match"
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
                hide_index=True,
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

    m1, m2, m3 = st.columns(3)
    m1.metric("Total pairs", nd.n_pairs_total)
    m2.metric("Near-duplicate pairs", nd.n_near_dup)
    m3.metric("Conflict pairs", nd.n_conflicts)

    engine_desc = (
        "🧠 Semantic Embeddings (all-MiniLM-L6-v2) — meaning-based"
        if getattr(nd, "similarity_method", "") == "semantic"
        else "🔤 Lexical TF-IDF"
    )
    st.caption(
        f"Engine: **{engine_desc}** · "
        f"Near-dup threshold: **{cfg.near_dup_threshold}** · "
        f"Conflict threshold: **{cfg.conflict_threshold}** · "
        f"Records analysed: **{nd.n_analyzed_records}**"
    )

    if nd.hit_pair_cap:
        st.warning("Pair storage cap reached — results shown are partial.")

    if nd.n_pairs_total == 0:
        st.info("No near-duplicate or conflicting pairs found at the current thresholds.")
        return

    pair_df = pairs_to_dataframe(nd.pairs)

    # Filters
    st.markdown("##### Filters")
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

    st.markdown(f"##### Pairs ({len(filtered)} shown)")
    st.dataframe(
        filtered.assign(similarity=filtered["similarity"].round(4)),
        column_config={
            "text_a": st.column_config.TextColumn(width="large"),
            "text_b": st.column_config.TextColumn(width="large"),
        },
        use_container_width=True,
        height=400,
        hide_index=True,
    )

    # Similarity histogram
    st.markdown("##### Similarity Distribution")
    fig = px.histogram(
        pair_df, x="similarity", color="finding_type",
        nbins=40,
        color_discrete_map={
            "near_duplicate": "#2563EB",
            "conflicting_label": "#DC2626",
            "near_dup_conflict": "#D97706",
        },
        labels={"similarity": "Cosine similarity", "count": "Pairs"},
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(t=10, b=40),
        yaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
        xaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Detail view
    if len(filtered) > 0:
        st.markdown("##### Pair Detail")
        row_options = list(filtered.index)
        if len(row_options) <= 12:
            selected_idx = st.radio(
                "Select pair to inspect",
                row_options,
                format_func=lambda idx: (
                    f"Pair #{idx}: Row {filtered.loc[idx, 'csv_row_a']} ↔ "
                    f"Row {filtered.loc[idx, 'csv_row_b']} "
                    f"(Sim: {filtered.loc[idx, 'similarity']:.3f})"
                ),
                key="near_dup_selected_pair",
                horizontal=len(row_options) <= 4,
            )
        else:
            page_size = 10
            n_pages = (len(row_options) - 1) // page_size + 1
            page = st.radio(
                "Page",
                list(range(1, n_pages + 1)),
                horizontal=True,
                key="near_dup_page_selector",
            )
            page_start = (page - 1) * page_size
            page_options = row_options[page_start : page_start + page_size]
            selected_idx = st.radio(
                "Select pair to inspect",
                page_options,
                format_func=lambda idx: (
                    f"Pair #{idx}: Row {filtered.loc[idx, 'csv_row_a']} ↔ "
                    f"Row {filtered.loc[idx, 'csv_row_b']} "
                    f"(Sim: {filtered.loc[idx, 'similarity']:.3f})"
                ),
                key="near_dup_selected_pair",
            )
        row = filtered.loc[selected_idx]
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.markdown(f"**csv_row {row['csv_row_a']}** — label: `{row['label_a']}`")
            st.text_area("Text A", row["text_a"], height=150, disabled=True, key=f"ta_{selected_idx}")
        with dcol2:
            st.markdown(f"**csv_row {row['csv_row_b']}** — label: `{row['label_b']}`")
            st.text_area("Text B", row["text_b"], height=150, disabled=True, key=f"tb_{selected_idx}")
        st.markdown(f"**Similarity:** `{row['similarity']:.4f}` · **Type:** `{row['finding_type']}`")
        st.markdown(f"**Reason:** {row['reason']}")


# ---------------------------------------------------------------------------
# Tab 4 — Suspicious labels
# ---------------------------------------------------------------------------

def _tab_suspicious_labels(result: AuditResult) -> None:
    la = result.label_analysis

    if la.cannot_run:
        st.error(f"Label analysis could not run: {la.cannot_run_reason}")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Flagged", la.n_flagged)
    m2.metric("Not flagged", la.n_not_flagged)
    m3.metric("Excluded", la.n_excluded)
    m4.metric(
        "OOF accuracy",
        f"{la.oof_accuracy:.2%}" if la.oof_accuracy is not None else "—",
    )

    splitter_note = f"Splitter: **{la.splitter_used}** ({la.folds_used} folds)"
    if la.splitter_fallback_reason:
        st.warning(f"⚠️ {la.splitter_fallback_reason}")
    st.caption(splitter_note)

    if la.low_accuracy_warning:
        st.warning(
            "OOF accuracy is below 0.6. Many flags likely reflect model weakness, "
            "not label errors. Review with care."
        )

    st.info(
        "**How to read flags:** A model that was not trained on each record predicted "
        "a different label. This is a review signal — the supplied label may still "
        "be correct."
    )

    if la.n_flagged == 0:
        st.success("No records flagged by the out-of-fold classifier.")

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

    st.markdown(f"##### Flagged Records ({len(filtered)} shown)")
    st.dataframe(
        filtered,
        column_config={"text": st.column_config.TextColumn(width="large")},
        use_container_width=True,
        height=400,
        hide_index=True,
    )

    # Flags per supplied class chart
    by_class = flagged_df.groupby("supplied_label").size().reset_index(name="flags")
    fig = px.bar(
        by_class, x="supplied_label", y="flags",
        color_discrete_sequence=["#DC2626"],
        labels={"supplied_label": "Supplied label", "flags": "Flagged records"},
    )
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(t=10, b=40),
        yaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
        xaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 5 — Classifier evaluation
# ---------------------------------------------------------------------------

def _tab_evaluation(result: AuditResult) -> None:
    metrics = result.baseline_metrics
    la = result.label_analysis

    st.markdown("##### Baseline Classifier Metrics")
    st.caption(
        "These metrics measure how well a simple TF-IDF classifier generalises on "
        "this dataset — not how accurate the audit flags are."
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
    )
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(t=10, b=40),
        yaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
        xaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Per-class table
    st.dataframe(class_df.set_index("class").round(3), use_container_width=True)

    # Confusion matrix
    st.markdown("##### Confusion Matrix")
    cm_df = pd.DataFrame(
        metrics.confusion_matrix,
        index=metrics.labels,
        columns=metrics.labels,
    )
    fig_cm = go.Figure(data=go.Heatmap(
        z=metrics.confusion_matrix,
        x=metrics.labels,
        y=metrics.labels,
        colorscale=[[0, "#EFF6FF"], [0.5, "#60A5FA"], [1, "#1E40AF"]],
        text=metrics.confusion_matrix,
        texttemplate="%{text}",
        showscale=True,
    ))
    fig_cm.update_layout(
        xaxis_title="Predicted",
        yaxis_title="True",
        height=350,
        yaxis_autorange="reversed",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    # --- Corruption experiment ---
    st.divider()
    st.markdown("##### Corruption Experiment")
    st.caption("Plants random label errors and measures detection. Synthetic only — not a real-world benchmark.")

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

        run_exp = st.button("Run experiment", key="run_experiment", type="secondary")

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
                f"Corrupted: **{sr.n_corrupted}** · "
                f"Flagged: **{sr.n_flagged}** · "
                f"TP: **{sr.tp}** · FP: **{sr.fp}** · FN: **{sr.fn}**"
            )
            st.markdown(
                f"**Random baseline precision** (= corruption rate): "
                f"`{exp_result.random_baseline_precision:.3f}`"
            )
            if exp_result.mean_precision is not None:
                st.markdown(
                    f"Multi-seed mean precision: `{exp_result.mean_precision:.3f}` "
                    f"±`{exp_result.std_precision:.3f}` · "
                    f"mean recall: `{exp_result.mean_recall:.3f}` "
                    f"±`{exp_result.std_recall:.3f}`"
                )


# ---------------------------------------------------------------------------
# Tab 6 — Export
# ---------------------------------------------------------------------------

def _tab_export(result: AuditResult, sanitize: bool = True) -> None:
    cfg = result.config

    st.markdown("##### Downloads")
    sanitize_toggle = st.toggle(
        "Guard against spreadsheet formula injection",
        value=True,
        help=(
            "Cells beginning with =, +, - or @ are prefixed with an apostrophe. "
            "Standard protection for opening CSVs in Excel or Google Sheets."
        ),
    )

    st.write("")
    col1, col2, col3 = st.columns(3)

    # Full row report
    with col1:
        full_report = build_row_report(result, sanitize=sanitize_toggle)
        st.download_button(
            "📥 Complete audit report",
            data=df_to_csv_bytes(full_report),
            file_name="labelaudit_full_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Flagged records
    with col2:
        flagged_report = build_flagged_report(result, sanitize=sanitize_toggle)
        st.download_button(
            "📥 Flagged records only",
            data=df_to_csv_bytes(flagged_report),
            file_name="labelaudit_flagged.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Pair report
    with col3:
        pair_report = build_pair_report(result, sanitize=sanitize_toggle)
        st.download_button(
            "📥 Near-dup / conflict pairs",
            data=df_to_csv_bytes(pair_report),
            file_name="labelaudit_pairs.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.write("")

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
                "📥 Corruption experiment results (JSON)",
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
                file_name="labelaudit_experiment.json",
                mime="application/json",
            )

    st.divider()
    st.markdown("##### Configuration Summary")
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
        # Use the filtered df if row selection was applied
        audit_df = st.session_state.get("_audit_df", st.session_state.load_result.df)
        fingerprint = st.session_state.file_fingerprint
        audit_key = f"{fingerprint}-{submitted_config.config_hash()}-{len(audit_df)}"

        if st.session_state.audit_key != audit_key:
            progress_bar = st.progress(0, text="Preparing…")

            def _progress(step: int, msg: str) -> None:
                progress_bar.progress(step / 6, text=msg)

            with st.spinner("Running audit…"):
                try:
                    audit_result = run_audit(
                        audit_df,
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
            # Getting-started screen
            st.write("")
            st.markdown("### Getting Started")
            st.write("")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("""
                <div class="step-card">
                    <div class="step-num">1</div>
                    <h4>Upload a CSV</h4>
                    <p>Use the sidebar uploader. Accepts comma, semicolon, or tab-delimited files up to 50 MB.</p>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown("""
                <div class="step-card">
                    <div class="step-num">2</div>
                    <h4>Select Columns</h4>
                    <p>Pick which column holds the text and which holds the category labels.</p>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown("""
                <div class="step-card">
                    <div class="step-num">3</div>
                    <h4>Run the Audit</h4>
                    <p>Adjust thresholds and classifier settings, then press Run Audit to start analysis.</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Dataset loaded. Select columns and settings in the Audit Setup form above, then press **▶ Run Audit**.")
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
