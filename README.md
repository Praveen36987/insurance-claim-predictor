# Insurance Claim Predictor

![Streamlit UI Placeholder](assets/screenshots/placeholder.png)

## Overview
The **Insurance Claim Predictor** is a production-style machine learning web application designed for insurance underwriters and portfolio managers. The app allows users to upload historical or current policyholder batches and predicts the probability that each policyholder will file a claim.

By predicting claim probabilities, insurance companies can:
- Adjust pricing dynamically based on risk profiles.
- Identify high-risk segments proactively.
- Better estimate capital reserve requirements.

## Key Features
1. **Bulk Ingestion & Validation**: Accepts `.csv` and `.xlsx` files and validates them against a predefined schema.
2. **End-to-End ML Pipeline**: Seamlessly runs data through a trained `scikit-learn` and `XGBoost` pipeline.
3. **Risk Banding**: Converts continuous probability predictions into business-friendly risk bands (Low/Medium/High) using configurable thresholds.
4. **Explainable AI (XAI)**: Highlights key risk drivers for high-risk policies directly in the UI.
5. **Interactive Dashboard**: Clean, recruiter-friendly `Streamlit` interface utilizing `plotly` for distribution visualization.
6. **Exportable Results**: Scored outputs are available for immediate download.

## Architecture & Tech Stack
- **Frontend / App**: Python, Streamlit
- **Machine Learning**: Scikit-Learn, XGBoost
- **Data Manipulation**: Pandas, NumPy
- **Visualizations**: Plotly
- **Serialization**: Joblib

### Directory Structure
```text
insurance-claim-predictor/
├── app.py                      # Main Streamlit application
├── config.py                   # Configuration and path settings
├── data/                       # Datasets (raw, processed, samples)
├── models/                     # Trained models and schemas (JSON)
├── notebooks/                  # EDA and data generation scripts
├── src/                        # Core ML modules
│   ├── data_loader.py          # File ingestion
│   ├── validator.py            # Schema validation
│   ├── preprocessing.py        # Scikit-learn preprocessing
│   ├── feature_engineering.py  # Business logic transformers
│   ├── train.py                # Model training script
│   ├── predict.py              # Inference script
│   └── explain.py              # Local/Global explainers
└── outputs/                    # Scored files location
```

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/insurance-claim-predictor.git
cd insurance-claim-predictor
```

### 2. Create a Virtual Environment & Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Generate Data and Train the Model (Optional)
If you wish to retrain the model on synthetic data:
```bash
python notebooks/generate_data.py
python src/train.py
```
This will train the XGBoost pipeline and save it to `models/trained_pipeline.joblib`.

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```

## Usage
1. Open the local Streamlit URL.
2. Download the **Sample CSV** from the sidebar.
3. Upload the sample file back into the app to see predictions, validation, and analytics.

## Future Improvements
- **Data Drift Monitoring**: Add hooks via evidently or simply track probability distributions over time.
- **Model Retraining Pipeline**: Implement CI/CD for retraining the model on new data.
- **Advanced SHAP Explanations**: Integrate interactive SHAP force plots for individual records.
- **Authentication**: Add standard login via Streamlit Authenticator for internal business use.
