import logging
import joblib
import pandas as pd
from typing import Dict

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb

# Configure logger if not already configured
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_and_evaluate_models(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.Series, 
    y_test: pd.Series, 
    save_path: str = "best_model.joblib"
) -> Dict[str, Dict[str, float]]:
    """
    Trains candidate models, evaluates them, selects the best based on ROC-AUC,
    saves the best model via joblib, and returns all evaluation metrics.
    """
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1)
    }
    
    results = {}
    best_roc_auc = -1.0
    best_model_name = None
    best_model = None

    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        results[name] = {
            'Accuracy': float(acc),
            'Precision': float(prec),
            'Recall': float(rec),
            'ROC-AUC': float(roc_auc)
        }
        
        logger.info(f"[{name}] ROC-AUC: {roc_auc:.4f} | Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
        
        # Compare to find the best model
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model_name = name
            best_model = model
            
    logger.info(f"==> Best model selected: {best_model_name} with ROC-AUC {best_roc_auc:.4f}")
    
    # Save the best model
    logger.info(f"Saving best model ({best_model_name}) to: {save_path}")
    joblib.dump(best_model, save_path)
    
    return results
