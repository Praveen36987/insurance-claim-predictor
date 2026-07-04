import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import json
from pathlib import Path

def detect_drift(baseline_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05) -> dict:
    """
    Compares current inference data with baseline training data to detect feature drift.
    Uses the Kolmogorov-Smirnov (KS) test for numerical columns.
    
    Returns a dictionary with drift status, p-values, and recommendations.
    """
    drift_report = {
        "drift_detected": False,
        "metrics": {},
        "retrain_recommended": False,
        "summary": {
            "total_features_tested": 0,
            "drifted_features_count": 0
        }
    }
    
    # Identify numerical columns to test (exclude target and ID columns)
    exclude_cols = ['policy_id', 'claim_status', 'predicted_probability', 'risk_band']
    num_cols = [
        col for col in baseline_df.select_dtypes(include=[np.number]).columns 
        if col not in exclude_cols and col in current_df.columns
    ]
    
    drifted_features = []
    
    for col in num_cols:
        # Get clean non-null distributions
        baseline_dist = baseline_df[col].dropna()
        current_dist = current_df[col].dropna()
        
        if len(baseline_dist) < 10 or len(current_dist) < 10:
            continue
            
        # Run Kolmogorov-Smirnov test
        stat, p_val = ks_2samp(baseline_dist, current_dist)
        drifted = bool(p_val < threshold)
        
        drift_report["metrics"][col] = {
            "statistic": float(stat),
            "p_value": float(p_val),
            "drifted": drifted
        }
        
        drift_report["summary"]["total_features_tested"] += 1
        if drifted:
            drift_report["summary"]["drifted_features_count"] += 1
            drifted_features.append(col)
            
    # If 30% or more features are drifted, recommend retraining
    total_tested = drift_report["summary"]["total_features_tested"]
    drifted_count = drift_report["summary"]["drifted_features_count"]
    
    if total_tested > 0 and (drifted_count / total_tested) >= 0.3:
        drift_report["drift_detected"] = True
        drift_report["retrain_recommended"] = True
        
    return drift_report
