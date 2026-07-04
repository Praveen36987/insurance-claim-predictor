import shap
import pandas as pd
import numpy as np

def generate_global_explanations(pipeline, X: pd.DataFrame):
    """
    Generates global feature importance (we'll just extract from XGBoost).
    """
    try:
        classifier = pipeline.named_steps['classifier']
        preprocessor = pipeline.named_steps['preprocessor']
        
        # Get feature names after preprocessing
        numeric_features = preprocessor.transformers_[0][2]
        categorical_features = preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(preprocessor.transformers_[1][2])
        feature_names = list(numeric_features) + list(categorical_features)
        
        importance = classifier.feature_importances_
        
        df_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)
        
        return df_importance
    except Exception as e:
        print(f"Warning: Could not generate global explanations: {e}")
        return None

def generate_local_explanation(pipeline, row: pd.DataFrame):
    """
    Provides a simple business explanation for a single row prediction.
    For this version, we look at the engineered features and high risk markers 
    since standard SHAP with pipelines requires careful inverse transformations.
    """
    # Quick heuristic-based explanation for demo purposes
    reasons = []
    
    if row['past_claims_count'].values[0] > 0:
        reasons.append(f"Past claims count ({row['past_claims_count'].values[0]})")
        
    if row['vehicle_damage'].values[0] == 'Yes':
        reasons.append("Prior vehicle damage reported")
        
    if row['age'].values[0] < 25:
        reasons.append("Young driver age segment")
        
    if not reasons:
        return "Model did not find strong positive risk markers; prediction driven by baseline profile."
        
    return "This record's risk is mainly driven by: " + ", ".join(reasons) + "."
