import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import io

from config import (
    APP_TITLE, APP_SUBTITLE,
    COLOR_LOW_RISK, COLOR_MED_RISK, COLOR_HIGH_RISK, COLOR_PRIMARY,
    MODEL_PATH, FEATURE_COLS_PATH,
)
from src.data_loader import load_data, load_from_kaggle
from src.validator import auto_map_columns, apply_column_mapping, validate_data, COLUMN_ALIASES
from src.predict import load_pipeline, predict
from src.explain import generate_local_explanation

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Claim Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.metric-card {
    background: white;
    border-radius: 10px;
    padding: 22px 18px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    border: 1px solid #e5e7eb;
    text-align: center;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #111827; line-height: 1.2; }
.metric-label { font-size: 0.78rem; color: #6b7280; text-transform: uppercase;
                letter-spacing: 0.06em; margin-bottom: 6px; }

.map-row {
    display: flex; align-items: center; gap: 10px;
    padding: 6px 0; border-bottom: 1px solid #f3f4f6;
}
.map-auto  { color: #16a34a; font-weight: 600; font-size: 0.85rem; }
.map-none  { color: #9ca3af; font-style: italic; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title(APP_TITLE)
st.markdown(f"**{APP_SUBTITLE}**")
st.divider()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Load Data")

    input_method = st.radio(
        "Input method",
        ["📁 Upload File", "🗂️ Kaggle Dataset"],
        label_visibility="collapsed",
    )

    uploaded_file   = None
    kaggle_id       = None
    load_kaggle_btn = False

    if input_method == "📁 Upload File":
        uploaded_file = st.file_uploader(
            "Upload CSV or XLSX", type=["csv", "xlsx"]
        )
    else:
        st.markdown("#### Kaggle Dataset ID")
        st.caption("Format: `owner/dataset-name`\nExample: `marcopesani/health-insurance-cross-sell-prediction`")
        kaggle_id = st.text_input("Dataset ID", placeholder="owner/dataset-name",
                                  label_visibility="collapsed")
        st.markdown("#### Kaggle API Credentials")
        kaggle_user = st.text_input("Username", placeholder="your_username")
        kaggle_key  = st.text_input("API Key", type="password", placeholder="xxxxxxxxxxxx")
        load_kaggle_btn = st.button("⬇️ Load from Kaggle", use_container_width=True, type="primary")
        st.info("💡 Store credentials in Streamlit Secrets to skip manual entry.", icon="ℹ️")

    st.markdown("---")
    st.markdown("### Instructions")
    st.markdown("1. Load data via file upload or Kaggle.\n2. Map columns if prompted.\n3. Run predictions.\n4. Download scored output.")

    st.markdown("---")
    sample_path = Path("data/sample/sample_insurance_data.csv")
    if sample_path.exists():
        with open(sample_path, "rb") as f:
            st.download_button("Download Sample CSV", f,
                               file_name="sample_insurance_data.csv", mime="text/csv")


# ── Credential helper ──────────────────────────────────────────────────────────
def _resolve_kaggle_creds(username: str, key: str) -> tuple[str, str]:
    try:
        username = username or st.secrets["kaggle"]["username"]
        key      = key      or st.secrets["kaggle"]["key"]
    except Exception:
        pass
    return username, key


# ── Load data ──────────────────────────────────────────────────────────────────
df          = None
source_label = ""

if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
        source_label = f"📁 {uploaded_file.name}"
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

elif load_kaggle_btn:
    if not kaggle_id or "/" not in kaggle_id:
        st.sidebar.error("Enter a valid Kaggle dataset ID (owner/dataset-name).")
        st.stop()
    ku, kk = _resolve_kaggle_creds(kaggle_user, kaggle_key)
    if not ku or not kk:
        st.sidebar.error("Kaggle credentials required. Enter them above or configure Streamlit Secrets.")
        st.stop()
    with st.spinner(f"Downloading **{kaggle_id}** from Kaggle…"):
        try:
            df, fname = load_from_kaggle(kaggle_id, ku, kk)
            source_label = f"🗂️ Kaggle · {kaggle_id} · `{fname}`"
        except Exception as e:
            st.error(f"Kaggle download failed: {e}")
            st.stop()


# ── Main pipeline (runs after data is loaded) ──────────────────────────────────
if df is not None:
    st.caption(
        f"**Data source:** {source_label} — "
        f"{len(df):,} rows × {len(df.columns)} columns"
    )

    # ── STEP 1: Data Preview ────────────────────────────────────────────────────
    st.subheader("1. Data Preview")
    with st.expander("View Raw Data (first 10 rows)", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

    # ── STEP 2: Column Mapping ──────────────────────────────────────────────────
    st.subheader("2. Column Mapping")

    # Auto-detect first pass
    auto_mapping = auto_map_columns(list(df.columns))
    unmatched    = [f for f, v in auto_mapping.items() if v is None]
    matched_auto = [f for f, v in auto_mapping.items() if v is not None]

    all_cols_option = ["— not available —"] + sorted(df.columns.tolist())

    if unmatched:
        st.warning(
            f"**{len(unmatched)} column(s)** could not be auto-detected: "
            f"`{'`, `'.join(unmatched)}`. "
            "Select the matching column from your dataset below, or choose "
            "**'— not available —'** to use a sensible default.",
            icon="⚠️",
        )
    else:
        st.success(
            f"✅ All {len(matched_auto)} required columns were auto-detected from your dataset.",
            icon="✅",
        )

    # Render mapping table — auto-matched shown in an expander, unmapped shown expanded
    with st.expander(
        f"✏️ Review / Edit column mapping ({len(matched_auto)}/{len(COLUMN_ALIASES)} auto-detected)",
        expanded=bool(unmatched),
    ):
        confirmed_mapping: dict[str, str | None] = {}
        feature_labels = {
            "policy_id":        "Policy / Record ID",
            "age":              "Customer Age",
            "vehicle_age":      "Vehicle Age",
            "vehicle_damage":   "Vehicle Damage Flag",
            "annual_premium":   "Annual Premium",
            "policy_tenure":    "Policy Tenure / Duration",
            "past_claims_count":"Past Claims Count",
            "credit_score":     "Credit Score",
        }

        cols_hdr = st.columns([2, 3, 1])
        cols_hdr[0].markdown("**Expected Feature**")
        cols_hdr[1].markdown("**Your Dataset Column**")
        cols_hdr[2].markdown("**Status**")

        for feature, label in feature_labels.items():
            auto_val = auto_mapping.get(feature)
            default_idx = (
                all_cols_option.index(auto_val)
                if auto_val and auto_val in all_cols_option
                else 0
            )

            cols = st.columns([2, 3, 1])
            cols[0].markdown(f"`{label}`")
            chosen = cols[1].selectbox(
                label=feature,
                options=all_cols_option,
                index=default_idx,
                key=f"map_{feature}",
                label_visibility="collapsed",
            )
            if chosen == "— not available —":
                cols[2].markdown('<span class="map-none">default</span>', unsafe_allow_html=True)
                confirmed_mapping[feature] = None
            else:
                cols[2].markdown('<span class="map-auto">✓ mapped</span>', unsafe_allow_html=True)
                confirmed_mapping[feature] = chosen

    # ── STEP 3: Apply mapping & validate ───────────────────────────────────────
    df_mapped = apply_column_mapping(df, confirmed_mapping)

    is_valid, errors = validate_data(df_mapped, FEATURE_COLS_PATH)
    if not is_valid:
        st.error("**Validation Failed** — please fix the issues below:")
        for e in errors:
            st.markdown(f"- {e}")
        st.stop()

    # ── STEP 4: Predict ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("3. Prediction & Risk Assessment")

    with st.spinner("Running inference pipeline…"):
        try:
            pipeline  = load_pipeline(MODEL_PATH)
            scored_df = predict(df_mapped, pipeline, FEATURE_COLS_PATH)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    # KPI cards
    total_records   = len(scored_df)
    high_risk_count = len(scored_df[scored_df["risk_band"] == "High Risk"])
    high_risk_pct   = (high_risk_count / total_records) * 100
    avg_prob        = scored_df["predicted_probability"].mean() * 100

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""<div class="metric-card">
        <div class="metric-label">Records Processed</div>
        <div class="metric-value">{total_records:,}</div></div>""",
        unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-card">
        <div class="metric-label">High Risk Policies</div>
        <div class="metric-value" style="color:{COLOR_HIGH_RISK}">
            {high_risk_count:,}<span style="font-size:1rem;font-weight:400"> ({high_risk_pct:.1f}%)</span>
        </div></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="metric-card">
        <div class="metric-label">Avg Claim Probability</div>
        <div class="metric-value">{avg_prob:.1f}%</div></div>""",
        unsafe_allow_html=True)

    st.write("")

    # ── STEP 5: Charts ─────────────────────────────────────────────────────────
    st.subheader("4. Risk Distribution")
    color_map = {
        "Low Risk":    COLOR_LOW_RISK,
        "Medium Risk": COLOR_MED_RISK,
        "High Risk":   COLOR_HIGH_RISK,
    }

    ch1, ch2 = st.columns(2)
    with ch1:
        fig = px.pie(scored_df, names="risk_band", color="risk_band",
                     color_discrete_map=color_map, hole=0.42,
                     title="Distribution by Risk Band")
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        fig2 = px.histogram(scored_df, x="predicted_probability",
                            color="risk_band", color_discrete_map=color_map,
                            nbins=30, title="Claim Probability Distribution",
                            labels={"predicted_probability": "Probability"})
        fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white", bargap=0.08)
        st.plotly_chart(fig2, use_container_width=True)

    # ── STEP 6: High-risk table & explanations ─────────────────────────────────
    st.divider()
    st.subheader("5. High-Risk Action Items")

    high_risk_df = scored_df[scored_df["risk_band"] == "High Risk"].sort_values(
        "predicted_probability", ascending=False
    )

    if len(high_risk_df) > 0:
        show_cols = [c for c in
                     ["policy_id", "predicted_probability", "risk_band",
                      "age", "vehicle_damage", "past_claims_count"]
                     if c in high_risk_df.columns]
        st.dataframe(high_risk_df[show_cols].head(15), use_container_width=True)

        st.markdown("#### Sample Risk Explanations")
        for i in range(min(3, len(high_risk_df))):
            row  = high_risk_df.iloc[[i]]
            expl = generate_local_explanation(pipeline, row)
            pid  = row["policy_id"].values[0] if "policy_id" in row.columns else f"#{i+1}"
            prob = row["predicted_probability"].values[0]
            st.info(f"**Policy {pid}** (Prob: {prob:.2f}): {expl}")
    else:
        st.info("No high-risk policies found in this batch.")

    # ── STEP 7: Download ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("6. Export Scored Data")

    buf = io.StringIO()
    scored_df.to_csv(buf, index=False)

    st.download_button(
        label="⬇️ Download Scored Results (CSV)",
        data=buf.getvalue(),
        file_name="scored_predictions.csv",
        mime="text/csv",
        type="primary",
    )

# ── Empty state ────────────────────────────────────────────────────────────────
else:
    st.info("👈 Use the sidebar to upload a file or load a Kaggle dataset to begin.")
    st.markdown("""
    ### Welcome to the Insurance Claim Predictor
    This application scores insurance policyholders by their likelihood of filing a claim.

    **Supports any dataset** — the app automatically maps your column names to the required
    features and lets you correct any mismatches before running predictions.

    **How it works:**
    1. Upload a CSV/XLSX **or** paste a Kaggle dataset ID.
    2. Review the auto-detected column mapping and adjust if needed.
    3. The ML pipeline scores each row with a claim probability and risk band.
    4. Explore the dashboard and download the scored output.
    """)
