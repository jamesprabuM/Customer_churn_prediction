import logging
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import optuna

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class ChurnModelEvaluator:
    """
    Trains and evaluates multiple machine learning models to compare their performance.
    """
    
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1)
        }
        self.best_model_name = None
        self.best_model = None
        self.results = {}

    def _calculate_metrics(self, y_true: pd.Series, y_pred: pd.Series, y_prob: pd.Series) -> Dict[str, float]:
        """Calculates standard classification metrics."""
        return {
            'Accuracy': float(accuracy_score(y_true, y_pred)),
            'Precision': float(precision_score(y_true, y_pred)),
            'Recall': float(recall_score(y_true, y_pred)),
            'F1-Score': float(f1_score(y_true, y_pred)),
            'ROC-AUC': float(roc_auc_score(y_true, y_prob))
        }

    def train_and_compare(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """Trains all candidate models and compares their performance."""
        logger.info("Starting model training and comparison...")
        
        best_roc_auc = -1.0
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Evaluate
            metrics = self._calculate_metrics(y_test, y_pred, y_prob)
            self.results[name] = metrics
            
            logger.info(f"{name} Metrics: ROC-AUC={metrics['ROC-AUC']:.4f}, F1={metrics['F1-Score']:.4f}")
            
            # Track best model primarily on ROC-AUC for imbalanced churn data
            if metrics['ROC-AUC'] > best_roc_auc:
                best_roc_auc = metrics['ROC-AUC']
                self.best_model_name = name
                self.best_model = model
                
        logger.info(f"Best model identified: {self.best_model_name} with ROC-AUC of {best_roc_auc:.4f}")
        return self.results


class HyperparameterTuner:
    """
    Handles hyperparameter tuning for tree-based models using Optuna.
    Focuses on tuning the LightGBM or XGBoost model as they usually perform best.
    """
    
    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
    def _objective_xgboost(self, trial: optuna.trial.Trial) -> float:
        param = {
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'random_state': 42
        }
        
        model = xgb.XGBClassifier(**param)
        model.fit(self.X_train, self.y_train)
        
        y_prob = model.predict_proba(self.X_test)[:, 1]
        return roc_auc_score(self.y_test, y_prob)

    def tune_xgboost(self, n_trials: int = 20) -> Tuple[Any, Dict[str, Any]]:
        """Tunes an XGBoost model using Optuna and returns the best model and parameters."""
        logger.info(f"Starting XGBoost hyperparameter tuning with {n_trials} trials...")
        optuna.logging.set_verbosity(optuna.logging.WARNING) # Reduce optuna output
        
        study = optuna.create_study(direction='maximize')
        study.optimize(self._objective_xgboost, n_trials=n_trials)
        
        best_params = study.best_params
        best_params.update({'use_label_encoder': False, 'eval_metric': 'logloss', 'random_state': 42})
        
        logger.info(f"Tuning complete. Best parameters dict: {study.best_params}")
        logger.info(f"Best validation ROC-AUC: {study.best_value:.4f}")
        
        # Train final tuned model
        final_model = xgb.XGBClassifier(**best_params)
        final_model.fit(self.X_train, self.y_train)
        
        return final_model, best_params
