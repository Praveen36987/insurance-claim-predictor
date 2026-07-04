import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("insurance_claim_predictor")

def get_baseline_data() -> pd.DataFrame:
    """
    Safely retrieves the baseline training dataset.
    """
    base_dir = Path(__file__).resolve().parent.parent
    baseline_path = base_dir / "data" / "raw" / "insurance_data.csv"
    if baseline_path.exists():
        try:
            return pd.read_csv(baseline_path)
        except Exception as e:
            logger.error(f"Failed to load baseline data from {baseline_path}: {e}")
            return pd.DataFrame()
    logger.warn(f"Baseline data not found at {baseline_path}")
    return pd.DataFrame()
