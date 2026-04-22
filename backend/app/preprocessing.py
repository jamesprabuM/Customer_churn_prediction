import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    A robust data preprocessing pipeline for Churn Prediction.
    Handles missing values, categorical encoding, scaling, and data splits.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns: List[str] = []
        self.target_column: str = 'Churn'
        self.is_fitted = False
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Loads data from a CSV file."""
        logger.info(f"Loading dataset from {filepath}")
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Dataset loaded. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def clean_and_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies basic domain-specific cleaning rules."""
        logger.info("Cleaning and formatting dataset...")
        df_clean = df.copy()
        
        # Customer ID is not useful for modelling
        if 'customerID' in df_clean.columns:
            df_clean = df_clean.drop(columns=['customerID'])
            
        # TotalCharges often contains blank strings that need converting to numeric
        if 'TotalCharges' in df_clean.columns:
            df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
            
        return df_clean

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Fits transformers and transforms the training data."""
        logger.info("Fitting and transforming data...")
        df_processed = self.clean_and_format(df)
        
        # Impute numeric missing values (like those caught from TotalCharges)
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            df_processed[numeric_cols] = self.imputer.fit_transform(df_processed[numeric_cols])
            df_processed[numeric_cols] = self.scaler.fit_transform(df_processed[numeric_cols])

        # Separate target
        if self.target_column not in df_processed.columns:
            raise KeyError(f"Target column '{self.target_column}' not found in the dataset.")
            
        y = df_processed[self.target_column]
        df_processed = df_processed.drop(columns=[self.target_column])
        
        # Encode Target variables
        le_target = LabelEncoder()
        y = pd.Series(le_target.fit_transform(y), index=y.index, name=self.target_column)
        self.label_encoders[self.target_column] = le_target
        
        # Encode Categorical features
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            self.label_encoders[col] = le
            
        self.feature_columns = df_processed.columns.tolist()
        self.is_fitted = True
        logger.info("Data preprocessing completed successfully.")
        
        return df_processed, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms new/incoming data using fitted transformers."""
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted yet. Call fit_transform first.")
            
        logger.info("Transforming new data...")
        df_processed = self.clean_and_format(df)
        
        # Impute and Scale numeric features
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            df_processed[numeric_cols] = self.imputer.transform(df_processed[numeric_cols])
            df_processed[numeric_cols] = self.scaler.transform(df_processed[numeric_cols])
            
        # Encode Categorical features
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_cols:
            if col in self.label_encoders:
                # Handle new unseen values by mapping them to an 'Unknown' category or mapping -1 (simplification for robustness)
                known_classes = set(self.label_encoders[col].classes_)
                df_processed[col] = df_processed[col].apply(lambda x: x if x in known_classes else 'Unknown')
                
                # We need to temporarily add 'Unknown' class to encoders to avoid errors (in production, use OrdinalEncoder with handle_unknown)
                if 'Unknown' not in self.label_encoders[col].classes_:
                    self.label_encoders[col].classes_ = np.append(self.label_encoders[col].classes_, 'Unknown')
                    
                df_processed[col] = self.label_encoders[col].transform(df_processed[col].astype(str))
                
        # Ensure all columns exist and ordered identically
        # Add missing columns as 0
        for col in self.feature_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0

        # Drop extra columns
        df_processed = df_processed[self.feature_columns]
        
        return df_processed

def get_train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Wraps basic split to keep pipeline standard."""
    logger.info(f"Splitting dataset. Test size = {test_size}")
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
