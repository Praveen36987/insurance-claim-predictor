from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.feature_engineering import FeatureEngineer

def create_preprocessing_pipeline(numeric_cols, categorical_cols):
    """
    Creates a scikit-learn preprocessing pipeline.
    """
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )
    
    # Complete pipeline with feature engineering step first
    # Note: FeatureEngineer might add new columns, so we apply it outside ColumnTransformer or adjust
    # Actually, it's easier to apply FeatureEngineer before the ColumnTransformer so that ColumnTransformer sees the new columns
    # We will need to update numeric_cols in the training script to include new engineered features.

    return preprocessor
