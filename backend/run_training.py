"""
Quick training script: fits preprocessor on raw data, trains models, saves the best one.
Run from the backend/ directory with the venv activated.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from app.preprocessing import DataPreprocessor
from app.train_model import train_and_evaluate_models
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = "best_model.joblib"

def main():
    logger.info("Loading and preprocessing data...")
    preprocessor = DataPreprocessor()
    df_raw = preprocessor.load_data(DATA_PATH)
    X, y = preprocessor.fit_transform(df_raw)  # returns (features_df, labels_series)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    import xgboost as xgb
    import joblib
    
    logger.info("Training XGBoost model...")
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    logger.info(f"Done! Saving XGBoost model to {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)

if __name__ == "__main__":
    main()
