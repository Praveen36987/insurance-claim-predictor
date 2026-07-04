import pandas as pd
import json
from pathlib import Path

def validate_data(df: pd.DataFrame, config_path: str) -> tuple[bool, list[str]]:
    """
    Validates the dataframe against the feature columns config.
    Returns (is_valid, list_of_errors)
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    required_columns = config.get("required_columns", [])
    
    errors = []
    
    # Check for missing columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}")
        
    if errors:
        return False, errors
        
    # Check for excessive nulls (e.g. > 50%) in required columns
    for col in required_columns:
        null_pct = df[col].isnull().mean()
        if null_pct > 0.5:
            errors.append(f"Column '{col}' has too many missing values ({null_pct:.1%}).")
            
    if errors:
        return False, errors
        
    return True, []
