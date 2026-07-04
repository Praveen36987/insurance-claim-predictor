import pandas as pd
import json
from pathlib import Path

# Common alternative names for each expected model feature.
# The app tries these aliases to auto-map uploaded column names.
COLUMN_ALIASES: dict[str, list[str]] = {
    "policy_id": [
        "id", "policy_number", "customer_id", "policyid", "policy_no",
        "policynumber", "customerid", "cust_id", "insured_id",
    ],
    "age": [
        "age", "customer_age", "insured_age", "age_of_customer",
        "policyholder_age", "age_years",
    ],
    "vehicle_age": [
        "vehicle_age", "car_age", "age_of_vehicle", "veh_age",
        "vehicle_age_band", "auto_age", "automobile_age",
    ],
    "vehicle_damage": [
        "vehicle_damage", "vehicle_damaged", "car_damage",
        "has_vehicle_damage", "auto_damage", "damage",
    ],
    "annual_premium": [
        "annual_premium", "premium", "annual_premium_amount",
        "policy_premium", "total_premium", "yearly_premium",
        "annualpremium",
    ],
    "policy_tenure": [
        "policy_tenure", "tenure", "policy_duration", "vintage",
        "days_since_policy", "policy_age", "months_active",
        "duration_months",
    ],
    "past_claims_count": [
        "past_claims_count", "claims_count", "num_claims",
        "claim_frequency", "previously_claimed", "number_of_claims",
        "total_claims", "claim_count", "no_of_claims",
    ],
    "credit_score": [
        "credit_score", "credit", "fico_score", "credit_rating",
        "creditworthiness", "credit_index",
    ],
}


def _normalise(col: str) -> str:
    """Lowercase + remove spaces/underscores for fuzzy matching."""
    return col.lower().replace(" ", "").replace("_", "").replace("-", "")


def auto_map_columns(df_columns: list[str]) -> dict[str, str | None]:
    """
    Attempts to automatically map the uploaded dataset's column names to the
    expected model features using the COLUMN_ALIASES lookup.

    Returns a dict:  { expected_feature: matched_uploaded_column | None }
    """
    norm_uploaded = {_normalise(c): c for c in df_columns}
    mapping: dict[str, str | None] = {}

    for feature, aliases in COLUMN_ALIASES.items():
        matched = None
        # Exact normalised match first
        for alias in [feature] + aliases:
            key = _normalise(alias)
            if key in norm_uploaded:
                matched = norm_uploaded[key]
                break
        mapping[feature] = matched

    return mapping


def apply_column_mapping(df: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    """
    Renames and selects columns in df according to a confirmed mapping.
    Missing optional columns are filled with sensible defaults.
    """
    rename_map = {v: k for k, v in mapping.items() if v is not None}
    df_out = df.rename(columns=rename_map).copy()

    # Fill completely absent columns with neutral defaults
    defaults = {
        "policy_id": [f"REC-{i+1:05d}" for i in range(len(df_out))],
        "age": 40,
        "vehicle_age": "1-2 Year",
        "vehicle_damage": "No",
        "annual_premium": df_out.get("annual_premium", pd.Series([30000] * len(df_out))).median()
            if "annual_premium" in df_out.columns else 30000,
        "policy_tenure": 24,
        "past_claims_count": 0,
        "credit_score": 650,
    }

    for col, default in defaults.items():
        if col not in df_out.columns:
            df_out[col] = default

    return df_out


def validate_data(df: pd.DataFrame, config_path: str) -> tuple[bool, list[str]]:
    """
    Light validation after column mapping has been applied.
    Checks only for excessive nulls — schema is guaranteed by apply_column_mapping.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    required_columns = config.get("required_columns", [])
    errors = []

    for col in required_columns:
        if col not in df.columns:
            errors.append(f"Column '{col}' is still missing after mapping.")
            continue
        null_pct = df[col].isnull().mean()
        if null_pct > 0.9:
            errors.append(
                f"Column '{col}' is {null_pct:.0%} empty — "
                "too few values to make reliable predictions."
            )

    return (len(errors) == 0), errors
