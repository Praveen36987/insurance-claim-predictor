import pandas as pd
import joblib
import json
from pathlib import Path

def load_pipeline(model_path: str):
    """Loads the trained machine learning pipeline."""
    return joblib.load(model_path)

def predict(df: pd.DataFrame, pipeline, config_path: str) -> pd.DataFrame:
    """
    Applies the trained pipeline to generate predictions and risk bands.
    """
    # Load configs
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    threshold_path = Path(config_path).parent / "threshold_config.json"
    with open(threshold_path, 'r') as f:
        threshold_config = json.load(f)
        
    thresholds = threshold_config["thresholds"]
    labels = threshold_config["labels"]
    
    # We only pass the required features to the pipeline
    features = config["categorical_columns"] + config["numeric_columns"]
    X = df[features].copy()
    
    # Predict probabilities
    probabilities = pipeline.predict_proba(X)[:, 1]
    
    # Assign risk bands
    def assign_band(prob):
        if prob < thresholds["low"]:
            return labels["low"]
        elif prob < thresholds["medium"]:
            return labels["medium"]
        else:
            return labels["high"]
            
    risk_bands = [assign_band(p) for p in probabilities]
    
    # Attach predictions to a copy of the original dataframe
    results_df = df.copy()
    results_df['predicted_probability'] = probabilities
    results_df['risk_band'] = risk_bands
    
    return results_df
