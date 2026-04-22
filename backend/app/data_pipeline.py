import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)

def load_and_preprocess_data(csv_path: str):
    """
    Loads Telco churn dataset, handles missing values, applies OneHotEncoding 
    and StandardScaler via a scikit-learn Pipeline, and splits the data.
    """
    logger.info(f"Loading dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 1. Basic Cleaning
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        
    # TotalCharges is technically numeric but stored as strings with blank spaces
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
    # 2. Separate Features and Target
    target_col = 'Churn'
    if target_col not in df.columns:
        raise ValueError(f"Dataset is missing the target column: {target_col}")
        
    X = df.drop(target_col, axis=1)
    
    # Binarize the target variable (assuming Yes=1, No=0)
    y = df[target_col].map({'Yes': 1, 'No': 0})
    
    # Fill any missing target rows just in case (though typically dropped)
    if y.isnull().any():
        y = y.fillna(0)
    
    # 3. Identify categorical and numerical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 4. Define Scikit-Learn Pipelines
    # Pipeline for numerical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    # 5. Split train/test (80/20 split)
    logger.info("Splitting dataset into 80/20 train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 6. Fit & Transform
    logger.info("Transforming datasets via scikit-learn Pipeline...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Reconstruct DataFrames to keep column names (helpful for tree models & SHAP)
    # Extract feature names after OneHotEncoding
    try:
        # Requires Scikit-Learn 1.0+
        cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
        all_feature_names = numeric_features + list(cat_feature_names)
        
        X_train_processed = pd.DataFrame(X_train_transformed, columns=all_feature_names, index=X_train.index)
        X_test_processed = pd.DataFrame(X_test_transformed, columns=all_feature_names, index=X_test.index)
    except Exception as e:
        logger.warning(f"Could not extract feature names, reverting to numeric matrices: {e}")
        X_train_processed = X_train_transformed
        X_test_processed = X_test_transformed

    logger.info(f"Preprocessing completed. Train shape: {X_train_processed.shape}, Test shape: {X_test_processed.shape}")
    
    return X_train_processed, X_test_processed, y_train, y_test

if __name__ == "__main__":
    # Test block
    X_tr, X_te, y_tr, y_te = load_and_preprocess_data("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print(X_tr.head())
