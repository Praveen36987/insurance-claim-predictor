import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import io
import json

from config import (
    APP_TITLE, APP_SUBTITLE,
    COLOR_LOW_RISK, COLOR_MED_RISK, COLOR_HIGH_RISK, COLOR_PRIMARY,
    MODEL_PATH, FEATURE_COLS_PATH,
)
from src.data_loader import load_data, load_from_kaggle
from src.validator import auto_map_columns, apply_column_mapping, validate_data, COLUMN_ALIASES
from src.predict import load_pipeline, predict
from src.explain import generate_local_explanation
from src.utils import get_baseline_data
from src.monitoring import detect_drift

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


UNAVAILABLE_OPTION = "-- not available --"

ANALYSIS_ALIASES: dict[str, list[str]] = {
    "policy_id": ["policy_id", "id", "policy_number", "customer_id"],
    "age": ["age", "customer_age", "insured_age", "policyholder_age"],
    "vehicle_age": ["vehicle_age", "vehicle age", "vehicleage", "car_age"],
    "vehicle_damage": ["vehicle_damage", "vehicle damage", "vehicledamage", "damage"],
    "annual_premium": ["annual_premium", "annual premium", "annualpremium", "premium"],
    "policy_tenure": ["policy_tenure", "policy tenure", "vintage", "tenure"],
    "claim_status": ["claim_status", "claim status", "response", "claim", "claimed"],
    "previously_insured": ["previously_insured", "previously insured", "previouslyinsured"],
}


def _norm_col(name: str) -> str:
    return str(name).lower().replace(" ", "").replace("_", "").replace("-", "")


def _detect_columns(df_columns: list[str], aliases: dict[str, list[str]]) -> dict[str, str | None]:
    uploaded = {_norm_col(col): col for col in df_columns}
    detected: dict[str, str | None] = {}
    for logical_name, candidates in aliases.items():
        detected[logical_name] = None
        for candidate in [logical_name] + candidates:
            match = uploaded.get(_norm_col(candidate))
            if match:
                detected[logical_name] = match
                break
    return detected


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce").dropna()


def _claim_series(df: pd.DataFrame, column: str) -> pd.Series:
    values = df[column]
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce")

    normalized = values.astype(str).str.strip().str.lower()
    return normalized.map({
        "1": 1, "0": 0,
        "yes": 1, "no": 0,
        "true": 1, "false": 0,
        "claimed": 1, "not claimed": 0,
        "claim": 1, "no claim": 0,
    })


def _metric_card(column, label: str, value: str, color: str = "#111827") -> None:
    column.markdown(
        f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color}">{value}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _no_visual(chart_name: str, missing: list[str]) -> None:
    st.info(
        f"No visualization available for {chart_name}. "
        f"Required column(s) not found: {', '.join(missing)}."
    )


def _render_direct_visualizations(df: pd.DataFrame, detected: dict[str, str | None]) -> None:
    st.subheader("2. Available Visualizations")

    rows, cols = len(df), len(df.columns)
    missing_cells = int(df.isna().sum().sum())
    missing_pct = (missing_cells / max(rows * cols, 1)) * 100
    c1, c2, c3 = st.columns(3)
    _metric_card(c1, "Rows", f"{rows:,}")
    _metric_card(c2, "Columns", f"{cols:,}")
    _metric_card(c3, "Missing Cells", f"{missing_pct:.1f}%")

    with st.expander("Column availability for analysis", expanded=False):
        st.dataframe(
            pd.DataFrame([
                {"Analysis field": field, "Detected dataset column": column or "Not found"}
                for field, column in detected.items()
            ]),
            use_container_width=True,
            hide_index=True,
        )

    left, right = st.columns(2)
    with left:
        age_col = detected.get("age")
        if age_col:
            values = _numeric_series(df, age_col)
            if values.empty:
                _no_visual("Age Distribution", [age_col])
            else:
                fig = px.histogram(
                    x=values, nbins=30, title="Age Distribution",
                    labels={"x": "Age", "y": "Records"},
                    color_discrete_sequence=["#2563eb"],
                )
                fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", bargap=0.08)
                st.plotly_chart(fig, use_container_width=True)
        else:
            _no_visual("Age Distribution", ["age"])

    with right:
        premium_col = detected.get("annual_premium")
        if premium_col:
            values = _numeric_series(df, premium_col)
            if values.empty:
                _no_visual("Annual Premium Distribution", [premium_col])
            else:
                fig = px.histogram(
                    x=values, nbins=35, title="Annual Premium Distribution",
                    labels={"x": "Annual Premium", "y": "Records"},
                    color_discrete_sequence=["#0f766e"],
                )
                fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", bargap=0.08)
                st.plotly_chart(fig, use_container_width=True)
        else:
            _no_visual("Annual Premium Distribution", ["annual_premium"])

    left, right = st.columns(2)
    with left:
        vehicle_age_col = detected.get("vehicle_age")
        if vehicle_age_col:
            counts = df[vehicle_age_col].fillna("Missing").astype(str).value_counts().reset_index()
            counts.columns = ["Vehicle Age", "Records"]
            fig = px.bar(
                counts, x="Vehicle Age", y="Records", title="Vehicle Age Breakdown",
                color_discrete_sequence=["#7c3aed"],
            )
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            _no_visual("Vehicle Age Breakdown", ["vehicle_age"])

    with right:
        damage_col = detected.get("vehicle_damage")
        if damage_col:
            counts = df[damage_col].fillna("Missing").astype(str).value_counts().reset_index()
            counts.columns = ["Vehicle Damage", "Records"]
            fig = px.pie(
                counts, names="Vehicle Damage", values="Records",
                title="Vehicle Damage Split", hole=0.42,
                color_discrete_sequence=["#dc2626", "#16a34a", "#94a3b8"],
            )
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            _no_visual("Vehicle Damage Split", ["vehicle_damage"])

    left, right = st.columns(2)
    claim_col = detected.get("claim_status")
    with left:
        if claim_col:
            valid_claims = _claim_series(df, claim_col).dropna()
            if valid_claims.empty:
                _no_visual("Claim Status Split", [claim_col])
            else:
                counts = valid_claims.map({0: "No Claim", 1: "Claim"}).value_counts().reset_index()
                counts.columns = ["Claim Status", "Records"]
                fig = px.pie(
                    counts, names="Claim Status", values="Records",
                    title="Claim Status Split", hole=0.42,
                    color_discrete_sequence=["#16a34a", "#dc2626"],
                )
                fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig, use_container_width=True)
        else:
            _no_visual("Claim Status Split", ["claim_status or response"])

    with right:
        tenure_col = detected.get("policy_tenure")
        if tenure_col:
            values = _numeric_series(df, tenure_col)
            if values.empty:
                _no_visual("Policy Tenure Distribution", [tenure_col])
            else:
                fig = px.histogram(
                    x=values, nbins=30, title="Policy Tenure / Vintage Distribution",
                    labels={"x": "Policy Tenure / Vintage", "y": "Records"},
                    color_discrete_sequence=["#d97706"],
                )
                fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", bargap=0.08)
                st.plotly_chart(fig, use_container_width=True)
        else:
            _no_visual("Policy Tenure Distribution", ["policy_tenure or vintage"])

    left, right = st.columns(2)
    with left:
        damage_col = detected.get("vehicle_damage")
        if claim_col and damage_col:
            chart_df = pd.DataFrame({
                "vehicle_damage": df[damage_col].fillna("Missing").astype(str),
                "claim": _claim_series(df, claim_col),
            }).dropna()
            if chart_df.empty:
                _no_visual("Claim Rate by Vehicle Damage", [claim_col, damage_col])
            else:
                grouped = chart_df.groupby("vehicle_damage", as_index=False)["claim"].mean()
                grouped["Claim Rate"] = grouped["claim"] * 100
                fig = px.bar(
                    grouped, x="vehicle_damage", y="Claim Rate",
                    title="Claim Rate by Vehicle Damage",
                    labels={"vehicle_damage": "Vehicle Damage"},
                    color_discrete_sequence=["#be123c"],
                )
                fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig, use_container_width=True)
        else:
            missing = []
            if not claim_col:
                missing.append("claim_status or response")
            if not damage_col:
                missing.append("vehicle_damage")
            _no_visual("Claim Rate by Vehicle Damage", missing)

    with right:
        vehicle_age_col = detected.get("vehicle_age")
        if claim_col and vehicle_age_col:
            chart_df = pd.DataFrame({
                "vehicle_age": df[vehicle_age_col].fillna("Missing").astype(str),
                "claim": _claim_series(df, claim_col),
            }).dropna()
            if chart_df.empty:
                _no_visual("Claim Rate by Vehicle Age", [claim_col, vehicle_age_col])
            else:
                grouped = chart_df.groupby("vehicle_age", as_index=False)["claim"].mean()
                grouped["Claim Rate"] = grouped["claim"] * 100
                fig = px.bar(
                    grouped, x="vehicle_age", y="Claim Rate",
                    title="Claim Rate by Vehicle Age",
                    labels={"vehicle_age": "Vehicle Age"},
                    color_discrete_sequence=["#4338ca"],
                )
                fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig, use_container_width=True)
        else:
            missing = []
            if not claim_col:
                missing.append("claim_status or response")
            if not vehicle_age_col:
                missing.append("vehicle_age")
            _no_visual("Claim Rate by Vehicle Age", missing)


def _load_model_config() -> dict:
    with open(FEATURE_COLS_PATH, "r") as f:
        return json.load(f)


def _render_prediction_readiness(df: pd.DataFrame) -> None:
    st.divider()
    st.subheader("3. Prediction Readiness")

    config = _load_model_config()
    model_features = config["categorical_columns"] + config["numeric_columns"]
    auto_mapping = auto_map_columns(list(df.columns))
    all_cols_option = [UNAVAILABLE_OPTION] + sorted(df.columns.tolist())
    confirmed_mapping: dict[str, str | None] = {}

    with st.expander("Review model column mapping", expanded=True):
        labels = {
            "policy_id": "Policy / Record ID",
            "age": "Customer Age",
            "vehicle_age": "Vehicle Age",
            "vehicle_damage": "Vehicle Damage",
            "annual_premium": "Annual Premium",
            "policy_tenure": "Policy Tenure / Vintage",
            "past_claims_count": "Past Claims Count",
            "credit_score": "Credit Score",
        }

        header = st.columns([2, 3, 1])
        header[0].markdown("**Model Feature**")
        header[1].markdown("**Dataset Column**")
        header[2].markdown("**Status**")

        for feature in ["policy_id"] + model_features:
            auto_val = auto_mapping.get(feature)
            default_idx = all_cols_option.index(auto_val) if auto_val in all_cols_option else 0
            row = st.columns([2, 3, 1])
            row[0].markdown(f"`{labels.get(feature, feature)}`")
            chosen = row[1].selectbox(
                label=feature,
                options=all_cols_option,
                index=default_idx,
                key=f"map_{feature}",
                label_visibility="collapsed",
            )
            if chosen == UNAVAILABLE_OPTION:
                confirmed_mapping[feature] = None
                row[2].markdown('<span class="map-none">missing</span>', unsafe_allow_html=True)
            else:
                confirmed_mapping[feature] = chosen
                row[2].markdown('<span class="map-auto">mapped</span>', unsafe_allow_html=True)

    missing_model_features = [feature for feature in model_features if not confirmed_mapping.get(feature)]
    if missing_model_features:
        st.warning(
            "Prediction is not available for this dataset because required model column(s) "
            f"are missing: {', '.join(missing_model_features)}. "
            "The visualizations above are still available because they use only existing data."
        )
        return

    df_mapped = apply_column_mapping(df, confirmed_mapping)
    is_valid, errors = validate_data(df_mapped, FEATURE_COLS_PATH)
    if not is_valid:
        st.error("Prediction validation failed.")
        for e in errors:
            st.markdown(f"- {e}")
        return

    st.success("All required model columns are available.")
    if not st.button("Run Prediction", use_container_width=True, type="primary"):
        return

    with st.spinner("Running inference pipeline..."):
        try:
            pipeline = load_pipeline(MODEL_PATH)
            scored_df = predict(df_mapped, pipeline, FEATURE_COLS_PATH)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

    st.divider()
    st.subheader("4. Prediction & Risk Assessment")

    total_records = len(scored_df)
    high_risk_count = len(scored_df[scored_df["risk_band"] == "High Risk"])
    high_risk_pct = (high_risk_count / max(total_records, 1)) * 100
    avg_prob = scored_df["predicted_probability"].mean() * 100

    c1, c2, c3 = st.columns(3)
    _metric_card(c1, "Records Processed", f"{total_records:,}")
    _metric_card(c2, "High Risk Policies", f"{high_risk_count:,} ({high_risk_pct:.1f}%)", COLOR_HIGH_RISK)
    _metric_card(c3, "Avg Claim Probability", f"{avg_prob:.1f}%")

    color_map = {
        "Low Risk": COLOR_LOW_RISK,
        "Medium Risk": COLOR_MED_RISK,
        "High Risk": COLOR_HIGH_RISK,
    }
    ch1, ch2 = st.columns(2)
    with ch1:
        fig = px.pie(
            scored_df, names="risk_band", color="risk_band",
            color_discrete_map=color_map, hole=0.42,
            title="Distribution by Risk Band",
        )
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    with ch2:
        fig = px.histogram(
            scored_df, x="predicted_probability", color="risk_band",
            color_discrete_map=color_map, nbins=30,
            title="Claim Probability Distribution",
            labels={"predicted_probability": "Probability"},
        )
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", bargap=0.08)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("5. High-Risk Action Items")
    high_risk_df = scored_df[scored_df["risk_band"] == "High Risk"].sort_values(
        "predicted_probability", ascending=False
    )

    if len(high_risk_df) > 0:
        show_cols = [c for c in [
            "policy_id", "predicted_probability", "risk_band",
            "age", "vehicle_damage", "past_claims_count",
        ] if c in high_risk_df.columns]
        st.dataframe(high_risk_df[show_cols].head(15), use_container_width=True)

        st.markdown("#### Sample Risk Explanations")
        for i in range(min(3, len(high_risk_df))):
            row = high_risk_df.iloc[[i]]
            expl = generate_local_explanation(pipeline, row)
            pid = row["policy_id"].values[0] if "policy_id" in row.columns else f"#{i + 1}"
            prob = row["predicted_probability"].values[0]
            st.info(f"Policy {pid} (Prob: {prob:.2f}): {expl}")
    else:
        st.info("No high-risk policies found in this batch.")

    # ── Model & Data Health Audit ──────────────────────────────────────────
    st.divider()
    st.subheader("6. Model & Data Health Audit")
    baseline_df = get_baseline_data()
    if not baseline_df.empty:
        drift_report = detect_drift(baseline_df, df_mapped)
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("#### Feature Distribution Audit")
            if drift_report["drift_detected"]:
                st.warning("⚠️ **Data Drift Detected!** The statistical distributions of features in the uploaded dataset have significantly shifted compared to the baseline training dataset. Model performance might degrade.")
            else:
                st.success("✅ **Data Health Check Passed.** The incoming features match the training distribution profile within statistical limits.")
                
            st.metric(
                label="Drifted / Tested Features",
                value=f"{drift_report['summary']['drifted_features_count']} / {drift_report['summary']['total_features_tested']}",
                delta="Drift Alert!" if drift_report["drift_detected"] else "Stable",
                delta_color="inverse" if drift_report["drift_detected"] else "normal"
            )
            
        with col_m2:
            st.markdown("#### Retraining Recommendation")
            if drift_report["retrain_recommended"]:
                st.info("💡 **Recommendation:** We recommend retraining the machine learning pipeline using this new batch of data to prevent concept/covariate drift from impacting underwriting accuracy.")
            else:
                st.info("💡 **Recommendation:** The current model pipeline is highly aligned with the incoming data. No retraining is required at this time.")
                
        # Detailed stats table
        with st.expander("🔍 View Statistical Test Details (KS-Test p-values)", expanded=False):
            drift_rows = []
            for feature, metrics in drift_report["metrics"].items():
                drift_rows.append({
                    "Feature": feature,
                    "KS Statistic": f"{metrics['statistic']:.4f}",
                    "p-value": f"{metrics['p_value']:.4e}",
                    "Drift Status": "🔴 DRIFTED" if metrics["drifted"] else "🟢 STABLE"
                })
            st.table(pd.DataFrame(drift_rows))
    else:
        st.info("Baseline training data is not available to run drift monitoring.")

    st.divider()
    st.subheader("7. Export Scored Data")
    buf = io.StringIO()
    scored_df.to_csv(buf, index=False)
    st.download_button(
        label="Download Scored Results (CSV)",
        data=buf.getvalue(),
        file_name="scored_predictions.csv",
        mime="text/csv",
        type="primary",
    )


def _render_upgraded_dashboard(df: pd.DataFrame, source_label: str) -> None:
    st.caption(
        f"**Data source:** {source_label} - "
        f"{len(df):,} rows x {len(df.columns)} columns"
    )

    st.subheader("1. Data Preview")
    with st.expander("View Raw Data", expanded=False):
        st.dataframe(df.head(25), use_container_width=True)

    detected = _detect_columns(list(df.columns), ANALYSIS_ALIASES)
    _render_direct_visualizations(df, detected)
    _render_prediction_readiness(df)


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
    _render_upgraded_dashboard(df, source_label)
    st.stop()

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
