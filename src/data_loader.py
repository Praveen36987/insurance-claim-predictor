import pandas as pd
from pathlib import Path
import os
import zipfile
import tempfile


def load_data(file_path_or_buffer) -> pd.DataFrame:
    """
    Loads data from a CSV or XLSX file path or Streamlit UploadedFile buffer.
    """
    try:
        if isinstance(file_path_or_buffer, str) or isinstance(file_path_or_buffer, Path):
            file_path_or_buffer = str(file_path_or_buffer)
            if file_path_or_buffer.endswith('.csv'):
                return pd.read_csv(file_path_or_buffer)
            elif file_path_or_buffer.endswith('.xlsx'):
                return pd.read_excel(file_path_or_buffer)
            else:
                raise ValueError("Unsupported file format. Please provide a CSV or XLSX file.")
        else:
            # Streamlit UploadedFile object
            name = file_path_or_buffer.name
            if name.endswith('.csv'):
                return pd.read_csv(file_path_or_buffer)
            elif name.endswith('.xlsx'):
                return pd.read_excel(file_path_or_buffer)
            else:
                raise ValueError("Unsupported file format. Please upload a CSV or XLSX file.")

    except Exception as e:
        raise RuntimeError(f"Error loading data: {str(e)}")


def load_from_kaggle(dataset_id: str, kaggle_username: str, kaggle_key: str) -> tuple[pd.DataFrame, str]:
    """
    Downloads a Kaggle dataset by ID and returns the first CSV found as a DataFrame.

    Args:
        dataset_id: Kaggle dataset identifier in 'owner/dataset-name' format.
                    e.g. 'marcopesani/health-insurance-cross-sell-prediction'
        kaggle_username: Kaggle account username (from API credentials).
        kaggle_key:      Kaggle API key (from API credentials).

    Returns:
        (DataFrame, filename) tuple with the loaded data and file used.
    """
    # Set Kaggle credentials as environment variables (no file needed)
    os.environ["KAGGLE_USERNAME"] = kaggle_username
    os.environ["KAGGLE_KEY"] = kaggle_key

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise RuntimeError(
            "The 'kaggle' package is not installed. Add 'kaggle' to requirements.txt."
        )

    api = KaggleApi()
    api.authenticate()

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Download dataset zip into the temp folder
        api.dataset_download_files(dataset_id, path=tmp_dir, unzip=True, quiet=True)

        # Find all CSV files in the downloaded folder
        csv_files = list(Path(tmp_dir).rglob("*.csv"))
        if not csv_files:
            raise RuntimeError(
                f"No CSV files found in the Kaggle dataset '{dataset_id}'. "
                "This dataset may use a different format."
            )

        # Load the first (or largest) CSV found
        csv_files.sort(key=lambda p: p.stat().st_size, reverse=True)
        chosen = csv_files[0]
        df = pd.read_csv(chosen)
        return df, chosen.name
