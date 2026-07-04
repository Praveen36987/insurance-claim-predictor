import pandas as pd
import json
from pathlib import Path
import numpy as np

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


def clean_mapped_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes data types and categories for the mapped columns
    to ensure compatibility with the trained model.
    """
    df_clean = df.copy()

    # 1. Clean Numeric Columns (convert and handle NaNs/outliers)
    numeric_cols = {
        "age": 40,
        "annual_premium": 30000,
        "policy_tenure": 24,
        "past_claims_count": 0,
        "credit_score": 650
    }

    for col, default in numeric_cols.items():
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            # Fill missing/NaNs
            median_val = df_clean[col].median()
            fill_val = median_val if not pd.isna(median_val) else default
            df_clean[col] = df_clean[col].fillna(fill_val)
            
            # Additional logical bounds
            if col == "age":
                df_clean[col] = df_clean[col].clip(18, 100)
            elif col == "past_claims_count":
                df_clean[col] = df_clean[col].clip(0, 50).round().astype(int)
            elif col == "credit_score":
                df_clean[col] = df_clean[col].clip(300, 850).round().astype(int)

    # 2. Clean Categorical: vehicle_damage
    if "vehicle_damage" in df_clean.columns:
        def standardize_damage(val):
            if pd.isna(val):
                return "No"
            s = str(val).strip().lower()
            if s in ['yes', 'y', '1', 'true', 'damaged']:
                return "Yes"
            return "No"
        df_clean["vehicle_damage"] = df_clean["vehicle_damage"].apply(standardize_damage)

    # 3. Clean Categorical: vehicle_age
    # Expected categories: "< 1 Year", "1-2 Year", "> 2 Years"
    if "vehicle_age" in df_clean.columns:
        def standardize_vehicle_age(val):
            if pd.isna(val):
                return "1-2 Year"
            
            # If it's numeric/float/int, map by value
            try:
                num = float(val)
                if num < 1:
                    return "< 1 Year"
                elif num <= 2:
                    return "1-2 Year"
                else:
                    return "> 2 Years"
            except ValueError:
                pass

            s = str(val).strip().lower()
            if any(x in s for x in ['<1', 'under 1', 'new', 'less than 1', '0 year']):
                return "< 1 Year"
            if any(x in s for x in ['1-2', '1 to 2', 'between 1', '1 year', '2 year']):
                return "1-2 Year"
            if any(x in s for x in ['>2', 'over 2', 'above 2', 'old', '2+', 'more than 2']):
                return "> 2 Years"
            return "1-2 Year"
        df_clean["vehicle_age"] = df_clean["vehicle_age"].apply(standardize_vehicle_age)

    return df_clean


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
        "annual_premium": 30000,
        "policy_tenure": 24,
        "past_claims_count": 0,
        "credit_score": 650,
    }

    for col, default in defaults.items():
        if col not in df_out.columns:
            if col == "policy_id":
                df_out[col] = default
            else:
                df_out[col] = default

    # Clean the values to ensure correct formats & bounds
    df_out = clean_mapped_values(df_out)

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
