"""
Feature engineering module for RiskLens.
Builds preprocessing pipeline and creates domain-engineered features.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Handles feature engineering and preprocessing for insurance claims data.
    Ensures no data leakage by fitting preprocessor only on training data.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.preprocessor = None
        self.feature_names = None
        
    def create_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain-specific features for insurance underwriting.
        
        Features engineered based on insurance domain knowledge:
        - Vehicle age bucketing (non-linear risk relationship)
        - Premium per vehicle year (exposure-adjusted risk)
        - High-value vehicle flag
        - Customer tenure segments
        
        Args:
            df: Original dataframe
            
        Returns:
            DataFrame with new features added
        """
        df_engineered = df.copy()
        
        # 1. Vehicle Age as numeric for calculations
        # Vehicle_Age: "< 1 Year", "1-2 Year", "> 2 Years"
        vehicle_age_mapping = {
            "< 1 Year": 0.5,
            "1-2 Year": 1.5,
            "> 2 Years": 3
        }
        df_engineered['Vehicle_Age_Numeric'] = df_engineered['Vehicle_Age'].map(vehicle_age_mapping)
        
        # 2. Premium per vehicle year - exposure-adjusted premium
        # Accounts for how long vehicle has been insured
        df_engineered['Premium_per_Vehicle_Year'] = (
            df_engineered['Annual_Premium'] / (df_engineered['Vehicle_Age_Numeric'] + 1)
        )
        
        # 3. High-value vehicle flag
        premium_75th = df_engineered['Annual_Premium'].quantile(0.75)
        df_engineered['High_Value_Vehicle'] = (
            df_engineered['Annual_Premium'] > premium_75th
        ).astype(int)
        
        # 4. Age risk bucket
        # Insurance risk is non-linear with age
        def age_risk_bucket(age):
            if age < 25:
                return 'very_high_risk'
            elif age < 35:
                return 'high_risk'
            elif age < 50:
                return 'medium_risk'
            elif age < 65:
                return 'low_risk'
            else:
                return 'very_low_risk'
        
        df_engineered['Age_Risk_Bucket'] = df_engineered['Age'].apply(age_risk_bucket)
        
        # 5. Customer tenure segments
        def tenure_segment(vintage):
            if vintage < 30:
                return 'new_customer'
            elif vintage < 90:
                return 'growing_customer'
            elif vintage < 365:
                return 'established_customer'
            else:
                return 'loyal_customer'
        
        df_engineered['Customer_Tenure_Segment'] = df_engineered['Vintage'].apply(tenure_segment)
        
        # 6. Premium bucket
        def premium_bucket(premium):
            if premium < df_engineered['Annual_Premium'].quantile(0.25):
                return 'low_premium'
            elif premium < df_engineered['Annual_Premium'].quantile(0.50):
                return 'medium_premium'
            elif premium < df_engineered['Annual_Premium'].quantile(0.75):
                return 'high_premium'
            else:
                return 'very_high_premium'
        
        df_engineered['Premium_Bucket'] = df_engineered['Annual_Premium'].apply(premium_bucket)
        
        # 7. Vehicle damage & previously insured interaction
        df_engineered['Damage_History_Risk'] = (
            df_engineered['Vehicle_Damage'].map({'Yes': 1, 'No': 0}) * 
            (1 - df_engineered['Previously_Insured'])
        )
        
        logger.info(f"✓ Created {len([c for c in df_engineered.columns if c not in df.columns])} domain features")
        
        return df_engineered
    
    def build_preprocessor(self, df: pd.DataFrame, target_col: str = 'Response'):
        """
        Build sklearn ColumnTransformer pipeline for preprocessing.
        
        Args:
            df: DataFrame to fit preprocessor on (should be training data)
            target_col: Name of target column to exclude
            
        Returns:
            Fitted preprocessor
        """
        # Separate numeric and categorical columns
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numeric_features = [col for col in numeric_features if col not in ['id', target_col]]
        
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        
        logger.info(f"Numeric features ({len(numeric_features)}): {numeric_features}")
        logger.info(f"Categorical features ({len(categorical_features)}): {categorical_features}")
        
        # Numeric pipeline
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        
        # Combine using ColumnTransformer
        self.preprocessor = ColumnTransformer([
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ])
        
        # Fit on provided data
        self.preprocessor.fit(df[numeric_features + categorical_features])
        self.feature_names = numeric_features + categorical_features
        
        logger.info(f"✓ Preprocessor built and fitted on {len(df)} samples")
        
        return self.preprocessor
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Apply preprocessing to dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Transformed feature matrix
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call build_preprocessor first.")
        
        return self.preprocessor.transform(df[self.feature_names])
    
    def save_preprocessor(self, path: str = 'artifacts/preprocessor.pkl'):
        """Save preprocessor to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.preprocessor, path)
        logger.info(f"✓ Preprocessor saved to {path}")
    
    def load_preprocessor(self, path: str = 'artifacts/preprocessor.pkl'):
        """Load preprocessor from disk."""
        self.preprocessor = joblib.load(path)
        logger.info(f"✓ Preprocessor loaded from {path}")


def prepare_data(input_path: str = 'data/raw/data.csv',
                 output_dir: str = 'data/processed',
                 test_size: float = 0.15,
                 val_size: float = 0.15,
                 random_state: int = 42):
    """
    Complete data preparation pipeline:
    1. Load raw data
    2. Create domain features
    3. Build preprocessor on train set only
    4. Split into train/val/test with stratification
    5. Save all splits
    
    Args:
        input_path: Path to raw CSV
        output_dir: Directory to save processed data
        test_size: Fraction for test set
        val_size: Fraction for validation (from remaining)
        random_state: Random seed
        
    Returns:
        Dictionary with train/val/test splits and preprocessor
    """
    try:
        from src.data.ingestion import load_raw_data
    except ImportError:
        from data.ingestion import load_raw_data
    
    # Load data
    df = load_raw_data(input_path)
    logger.info(f"Loaded data: {df.shape}")
    
    # Create domain features
    fe = FeatureEngineer(random_state=random_state)
    df = fe.create_domain_features(df)
    
    # Separate features and target
    X = df.drop(['id', 'Response'], axis=1)
    y = df['Response']
    
    # First split: separate test set (never touched until final evaluation)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: separate train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size/(1-test_size), 
        random_state=random_state, stratify=y_trainval
    )
    
    logger.info(f"Train set: {X_train.shape[0]} ({X_train.shape[0]/len(df)*100:.1f}%)")
    logger.info(f"Val set:   {X_val.shape[0]} ({X_val.shape[0]/len(df)*100:.1f}%)")
    logger.info(f"Test set:  {X_test.shape[0]} ({X_test.shape[0]/len(df)*100:.1f}%)")
    logger.info(f"Train positive rate: {y_train.mean():.2%}")
    logger.info(f"Val positive rate:   {y_val.mean():.2%}")
    logger.info(f"Test positive rate:  {y_test.mean():.2%}")
    
    # Build preprocessor ONLY on training data
    fe.build_preprocessor(X_train, target_col='Response')
    
    # Transform all splits
    X_train_transformed = fe.transform(X_train)
    X_val_transformed = fe.transform(X_val)
    X_test_transformed = fe.transform(X_test)
    
    # Save processed data
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save as numpy for compatibility
    np.save(f'{output_dir}/X_train.npy', X_train_transformed)
    np.save(f'{output_dir}/X_val.npy', X_val_transformed)
    np.save(f'{output_dir}/X_test.npy', X_test_transformed)
    np.save(f'{output_dir}/y_train.npy', y_train.values)
    np.save(f'{output_dir}/y_val.npy', y_val.values)
    np.save(f'{output_dir}/y_test.npy', y_test.values)
    
    # Also save as pickle for pandas compatibility
    pd.DataFrame(X_train_transformed).to_pickle(f'{output_dir}/X_train.pkl')
    pd.DataFrame(X_val_transformed).to_pickle(f'{output_dir}/X_val.pkl')
    pd.DataFrame(X_test_transformed).to_pickle(f'{output_dir}/X_test.pkl')
    y_train.to_pickle(f'{output_dir}/y_train.pkl')
    y_val.to_pickle(f'{output_dir}/y_val.pkl')
    y_test.to_pickle(f'{output_dir}/y_test.pkl')
    
    logger.info(f"✓ Processed data saved to {output_dir}")
    
    # Save preprocessor
    fe.save_preprocessor()
    
    return {
        'X_train': X_train_transformed,
        'X_val': X_val_transformed,
        'X_test': X_test_transformed,
        'y_train': y_train.values,
        'y_val': y_val.values,
        'y_test': y_test.values,
        'preprocessor': fe.preprocessor,
        'feature_names': fe.feature_names
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Prepare data
    result = prepare_data()
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"✓ Train: {result['X_train'].shape}")
    print(f"✓ Val:   {result['X_val'].shape}")
    print(f"✓ Test:  {result['X_test'].shape}")
    print(f"✓ Features: {len(result['feature_names'])}")
