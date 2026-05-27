"""
Data ingestion module for RiskLens.
Loads raw insurance data and tracks it with DVC.
"""

import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_raw_data(data_path: str = "data/raw/data.csv") -> pd.DataFrame:
    """
    Load raw vehicle insurance data from CSV.
    
    Args:
        data_path: Path to the raw CSV file
        
    Returns:
        DataFrame with raw insurance data
    """
    try:
        df = pd.read_csv(data_path)
        logger.info(f"✓ Loaded data from {data_path}")
        logger.info(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        logger.error(f"✗ File not found: {data_path}")
        raise
    except Exception as e:
        logger.error(f"✗ Error loading data: {str(e)}")
        raise


def save_processed_data(df: pd.DataFrame, output_path: str = "data/processed/data_processed.csv") -> None:
    """
    Save processed data to disk.
    
    Args:
        df: DataFrame to save
        output_path: Path where to save the processed data
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved processed data to {output_path}")


if __name__ == "__main__":
    # Load data
    df = load_raw_data()
    
    # Display basic info
    print("\n" + "="*60)
    print("DATA OVERVIEW")
    print("="*60)
    print(f"\nShape: {df.shape}")
    print(f"\nColumn Names and Types:")
    print(df.dtypes)
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nTarget Distribution:")
    print(df['Response'].value_counts())
    print(f"\nMissing Values:")
    print(df.isnull().sum())
