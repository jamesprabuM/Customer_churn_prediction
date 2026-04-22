import optuna
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import pandas as pd
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

def tune_xgboost_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    n_trials: int = 30
) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    """
    Uses Optuna to tune an XGBoost classifier to maximize ROC-AUC.
    Tunes learning_rate, max_depth, and n_estimators.
    
    Returns:
        final_model: The trained XGBoost model using the best hyperparameters.
        best_params: A dictionary of the best hyperparameters.
    """
    logger.info(f"Starting Optuna tuning for XGBoost over {n_trials} trials...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        # Define hyperparameter search space
        param = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'random_state': 42
        }
        
        model = xgb.XGBClassifier(**param)
        model.fit(X_train, y_train)
        
        # Evaluate using ROC-AUC
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
        
        return roc_auc

    # Create & run study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    # Retrieve best results
    best_params = study.best_params
    logger.info(f"Tuning complete! Best validation ROC-AUC: {study.best_value:.4f}")
    logger.info(f"Best parameters identified: {best_params}")
    
    # Prepare params for final model (re-adding static params)
    final_params = best_params.copy()
    final_params.update({
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 42
    })
    
    # Train the final model with best params on training data
    logger.info("Training final XGBoost model using the optimal parameters...")
    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(X_train, y_train)
    
    return final_model, best_params
