import pandas as pd
from pathlib import Path

def load_data(file_path_or_buffer) -> pd.DataFrame:
    """
    Loads data from a CSV or XLSX file path or stream buffer.
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
            # For Streamlit UploadedFile
            name = file_path_or_buffer.name
            if name.endswith('.csv'):
                return pd.read_csv(file_path_or_buffer)
            elif name.endswith('.xlsx'):
                return pd.read_excel(file_path_or_buffer)
            else:
                raise ValueError("Unsupported file format. Please upload a CSV or XLSX file.")
                
    except Exception as e:
        raise RuntimeError(f"Error loading data: {str(e)}")
