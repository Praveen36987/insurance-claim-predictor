import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Applies business-driven feature engineering to the dataset.
    """
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_out = X.copy()
        
        # Example: Create Age Bands
        if 'age' in X_out.columns:
            X_out['is_young_driver'] = (X_out['age'] < 25).astype(int)
            X_out['is_senior_driver'] = (X_out['age'] > 65).astype(int)
            
        # Example: High Risk Indicator (Past Claims + Vehicle Damage)
        if 'past_claims_count' in X_out.columns and 'vehicle_damage' in X_out.columns:
            X_out['high_risk_profile'] = ((X_out['past_claims_count'] > 1) & (X_out['vehicle_damage'] == 'Yes')).astype(int)
            
        # Example: Credit Score relative to Premium
        if 'credit_score' in X_out.columns and 'annual_premium' in X_out.columns:
            X_out['premium_to_credit_ratio'] = X_out['annual_premium'] / (X_out['credit_score'] + 1)
            
        return X_out
