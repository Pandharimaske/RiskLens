"""
Data validation module for RiskLens.
Performs schema validation and data quality checks.
"""

import pandas as pd
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def validate_dataframe(df: pd.DataFrame) -> Dict:
    """
    Validate the insurance dataset for schema and data quality issues.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Dictionary with validation results
    """
    issues = []
    
    # 1. Schema Check - Required columns
    required_cols = [
        'id', 'Gender', 'Age', 'Driving_License', 'Region_Code',
        'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage',
        'Annual_Premium', 'Policy_Sales_Channel', 'Vintage', 'Response'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        issues.append(f"❌ Missing columns: {missing}")
    else:
        logger.info("✓ All required columns present")
    
    # 2. Range Checks
    if 'Age' in df.columns:
        if df['Age'].lt(0).any() or df['Age'].gt(120).any():
            issues.append("❌ Age contains out-of-range values (should be 0-120)")
    
    if 'Annual_Premium' in df.columns:
        if df['Annual_Premium'].lt(0).any():
            issues.append("❌ Annual_Premium contains negative values")
    
    if 'Vintage' in df.columns:
        if df['Vintage'].lt(0).any():
            issues.append("❌ Vintage contains negative values")
    
    # 3. Null Rate Check
    null_rates = df.isnull().mean()
    high_null_cols = null_rates[null_rates > 0.5].index.tolist()
    if high_null_cols:
        issues.append(f"❌ High null rate (>50%) in: {high_null_cols}")
    
    moderate_null = null_rates[(null_rates > 0.0) & (null_rates <= 0.5)].index.tolist()
    if moderate_null:
        null_info = {col: f"{null_rates[col]:.1%}" for col in moderate_null}
        logger.info(f"⚠️  Moderate null rates: {null_info}")
    
    # 4. Cardinality Check
    high_cardinality_cols = {col: df[col].nunique() for col in df.select_dtypes(include=['object']).columns}
    if high_cardinality_cols:
        logger.info(f"📊 Categorical cardinality: {high_cardinality_cols}")
    
    # 5. Target Distribution Check
    if 'Response' in df.columns:
        target_dist = df['Response'].value_counts(normalize=True)
        logger.info(f"✓ Target distribution: {dict(target_dist)}")
        imbalance_ratio = target_dist.min() / target_dist.max()
        if imbalance_ratio < 0.1:
            logger.warning(f"⚠️  Severe class imbalance detected (ratio: {imbalance_ratio:.2%})")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "row_count": len(df),
        "column_count": len(df.columns),
        "null_stats": null_rates.to_dict()
    }


if __name__ == "__main__":
    from ingestion import load_raw_data
    
    df = load_raw_data()
    result = validate_dataframe(df)
    
    print("\n" + "="*60)
    print("DATA VALIDATION REPORT")
    print("="*60)
    print(f"\n✓ Valid: {result['valid']}")
    print(f"✓ Rows: {result['row_count']}")
    print(f"✓ Columns: {result['column_count']}")
    
    if result['issues']:
        print("\n❌ Issues Found:")
        for issue in result['issues']:
            print(f"  {issue}")
    else:
        print("\n✓ No validation issues found!")
