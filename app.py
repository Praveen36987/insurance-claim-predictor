import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
import io

from config import (
    APP_TITLE, APP_SUBTITLE,
    COLOR_LOW_RISK, COLOR_MED_RISK, COLOR_HIGH_RISK, COLOR_PRIMARY,
    MODEL_PATH, FEATURE_COLS_PATH
)
from src.data_loader import load_data, load_from_kaggle
from src.validator import validate_data
from src.predict import load_pipeline, predict
from src.explain import generate_local_explanation

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Claim Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        color: #1f2937;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title(APP_TITLE)
st.markdown(f"**{APP_SUBTITLE}**")
st.divider()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Load Data")

    input_tab = st.radio(
        "Choose input method:",
        ["📁 Upload File", "🗂️ Kaggle Dataset"],
        label_visibility="collapsed"
    )

    uploaded_file = None
    kaggle_dataset_id = None
    load_kaggle_btn = False

    # ── Tab 1: File Upload ─────────────────────────────────────────────────────
    if input_tab == "📁 Upload File":
        uploaded_file = st.file_uploader(
            "Upload CSV or XLSX file", type=["csv", "xlsx"]
        )

    # ── Tab 2: Kaggle Dataset ──────────────────────────────────────────────────
    else:
        st.markdown("#### Kaggle Dataset ID")
        st.caption(
            "Paste the dataset identifier from the Kaggle URL.\n\n"
            "**Format:** `owner/dataset-name`\n\n"
            "**Example:** `marcopesani/health-insurance-cross-sell-prediction`"
        )
        kaggle_dataset_id = st.text_input(
            "Dataset ID",
            placeholder="owner/dataset-name",
            label_visibility="collapsed"
        )

        st.markdown("#### Kaggle API Credentials")
        st.caption(
            "Get these from [kaggle.com](https://www.kaggle.com/settings/account) → "
            "Account → API → Create New Token."
        )
        kaggle_username = st.text_input("Kaggle Username", placeholder="your_username")
        kaggle_key = st.text_input("Kaggle API Key", type="password", placeholder="xxxxxxxxxxxxxxxx")

        load_kaggle_btn = st.button("⬇️ Load from Kaggle", use_container_width=True, type="primary")

        st.info(
            "💡 **Streamlit Cloud users:** You can store your credentials "
            "in App Secrets instead of entering them here every time. "
            "See the README for setup instructions.",
            icon="ℹ️"
        )

    st.markdown("---")
    st.markdown("### Instructions")
    st.markdown("1. Choose an input method above.")
    st.markdown("2. Review validation results.")
    st.markdown("3. Analyze risk distributions.")
    st.markdown("4. Download the scored output.")

    st.markdown("---")
    st.markdown("### Sample Template")
    sample_path = Path("data/sample/sample_insurance_data.csv")
    if sample_path.exists():
        with open(sample_path, "rb") as f:
            st.download_button(
                label="Download Sample CSV",
                data=f,
                file_name="sample_insurance_data.csv",
                mime="text/csv"
            )

# ── Helper: Read Kaggle credentials from Streamlit secrets if not provided ─────
def _resolve_kaggle_creds(username: str, key: str) -> tuple[str, str]:
    """
    Falls back to st.secrets if the user left the sidebar fields empty.
    Secrets format in .streamlit/secrets.toml:
        [kaggle]
        username = "..."
        key = "..."
    """
    try:
        if not username:
            username = st.secrets["kaggle"]["username"]
        if not key:
            key = st.secrets["kaggle"]["key"]
    except (KeyError, FileNotFoundError):
        pass
    return username, key


# ── Resolve data source ────────────────────────────────────────────────────────
df = None
source_label = ""

if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
        source_label = f"📁 {uploaded_file.name}"
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

elif load_kaggle_btn:
    if not kaggle_dataset_id or "/" not in kaggle_dataset_id:
        st.sidebar.error("Please enter a valid Kaggle dataset ID (e.g. owner/dataset-name).")
        st.stop()

    # Resolve credentials (sidebar input → Streamlit secrets → fail)
    ku, kk = _resolve_kaggle_creds(kaggle_username, kaggle_key)
    if not ku or not kk:
        st.sidebar.error(
            "Kaggle credentials are required. Enter them in the sidebar or configure "
            "Streamlit secrets (see README)."
        )
        st.stop()

    with st.spinner(f"Downloading **{kaggle_dataset_id}** from Kaggle…"):
        try:
            df, filename = load_from_kaggle(kaggle_dataset_id, ku, kk)
            source_label = f"🗂️ Kaggle · {kaggle_dataset_id} · `{filename}`"
        except Exception as e:
            st.error(f"Kaggle download failed: {e}")
            st.stop()

# ── Main content (shared regardless of input source) ──────────────────────────
if df is not None:
    st.caption(f"**Data source:** {source_label} — {len(df):,} rows × {len(df.columns)} columns")

    # 1. Data Preview & Validation
    st.subheader("1. Data Preview & Validation")
    with st.expander("View Raw Data", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

    is_valid, errors = validate_data(df, FEATURE_COLS_PATH)
    if not is_valid:
        st.error("**Data Validation Failed** — Please fix the issues below before proceeding.")
        for err in errors:
            st.markdown(f"- {err}")
        st.stop()
    else:
        st.success("✅ Data structure validated successfully.")

    # 2. Model Inference
    st.divider()
    st.subheader("2. Prediction & Risk Assessment")

    with st.spinner("Running inference pipeline…"):
        try:
            pipeline = load_pipeline(MODEL_PATH)
            scored_df = predict(df, pipeline, FEATURE_COLS_PATH)
        except Exception as e:
            st.error(f"Prediction failed. Ensure the model has been trained. Error: {e}")
            st.stop()

    # KPIs
    total_records = len(scored_df)
    high_risk_count = len(scored_df[scored_df['risk_band'] == 'High Risk'])
    high_risk_pct = (high_risk_count / total_records) * 100
    avg_probability = scored_df['predicted_probability'].mean() * 100

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Records Processed</div>
            <div class="metric-value">{total_records:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">High Risk Policies</div>
            <div class="metric-value" style="color: {COLOR_HIGH_RISK};">{high_risk_count:,} ({high_risk_pct:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Claim Probability</div>
            <div class="metric-value">{avg_probability:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # 3. Risk Distribution charts
    st.subheader("3. Risk Distribution")
    col_chart1, col_chart2 = st.columns(2)

    color_map = {
        'Low Risk': COLOR_LOW_RISK,
        'Medium Risk': COLOR_MED_RISK,
        'High Risk': COLOR_HIGH_RISK
    }

    with col_chart1:
        fig_pie = px.pie(
            scored_df,
            names='risk_band',
            color='risk_band',
            color_discrete_map=color_map,
            hole=0.4,
            title="Distribution by Risk Band"
        )
        fig_pie.update_layout(plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_chart2:
        fig_hist = px.histogram(
            scored_df,
            x='predicted_probability',
            color='risk_band',
            color_discrete_map=color_map,
            nbins=30,
            title="Claim Probability Distribution",
            labels={'predicted_probability': 'Probability of Claim'}
        )
        fig_hist.update_layout(plot_bgcolor='white', paper_bgcolor='white', bargap=0.1)
        st.plotly_chart(fig_hist, use_container_width=True)

    # 4. High-Risk records & explanations
    st.divider()
    st.subheader("4. High-Risk Action Items")

    high_risk_df = scored_df[scored_df['risk_band'] == 'High Risk'].sort_values(
        by='predicted_probability', ascending=False
    )

    if len(high_risk_df) > 0:
        display_cols = [c for c in ['policy_id', 'predicted_probability', 'risk_band',
                                     'age', 'vehicle_damage', 'past_claims_count']
                        if c in high_risk_df.columns]
        st.dataframe(high_risk_df[display_cols].head(10), use_container_width=True)

        st.markdown("#### Sample Risk Explanations")
        for i in range(min(3, len(high_risk_df))):
            row = high_risk_df.iloc[[i]]
            explanation = generate_local_explanation(pipeline, row)
            pol_id = row['policy_id'].values[0] if 'policy_id' in row.columns else f"Record #{i+1}"
            prob = row['predicted_probability'].values[0]
            st.info(f"**Policy {pol_id}** (Prob: {prob:.2f}): {explanation}")
    else:
        st.info("No high-risk policies detected in this batch.")

    # 5. Export
    st.divider()
    st.subheader("5. Export Scored Data")

    csv_buffer = io.StringIO()
    scored_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    st.download_button(
        label="⬇️ Download Scored Results (CSV)",
        data=csv_data,
        file_name="scored_predictions.csv",
        mime="text/csv",
        type="primary"
    )

else:
    # Empty state
    st.info("👈 Use the sidebar to upload a file or load a Kaggle dataset to begin.")
    st.markdown("""
    ### Welcome to the Insurance Claim Predictor
    This application predicts the probability that an insurance policyholder will file a claim,
    allowing underwriters to proactively manage portfolio risk.

    **How it works:**
    1. **Upload a CSV/XLSX** file — or paste a **Kaggle dataset ID** to load directly from Kaggle.
    2. The app validates the schema and flags any issues.
    3. Our ML pipeline scores each policy with a claim probability and risk band (Low / Medium / High).
    4. Explore the dashboard, review high-risk records, and download the scored output.
    """)
