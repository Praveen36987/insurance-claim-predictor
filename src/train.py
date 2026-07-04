import pandas as pd
import json
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb

from src.feature_engineering import FeatureEngineer
from src.preprocessing import create_preprocessing_pipeline

def train_model():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "raw" / "insurance_data.csv"
    config_path = base_dir / "models" / "feature_columns.json"
    model_path = base_dir / "models" / "trained_pipeline.joblib"
    
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    target_col = config["target_column"]
    categorical_cols = config["categorical_columns"]
    numeric_cols = config["numeric_columns"]
    
    # After FeatureEngineer, we have new numeric columns
    extended_numeric_cols = numeric_cols + ['is_young_driver', 'is_senior_driver', 'high_risk_profile', 'premium_to_credit_ratio']
    
    print("Preparing features and target...")
    X = df.drop(columns=[target_col, config["id_column"]])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Building pipeline...")
    preprocessor = create_preprocessing_pipeline(extended_numeric_cols, categorical_cols)
    
    # We create a full pipeline including FeatureEngineer, Preprocessor, and Model
    pipeline = Pipeline([
        ('feature_engineer', FeatureEngineer()),
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=4, 
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        ))
    ])
    
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC-AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"Saving model to {model_path}...")
    joblib.dump(pipeline, model_path)
    print("Training completed successfully.")

if __name__ == "__main__":
    train_model()
