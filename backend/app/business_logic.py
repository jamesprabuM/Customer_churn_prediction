import shap
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class BusinessExplainabilityLayer:
    """
    Applies SHAP for interpretability and computes domain-specific business cost logic (in INR).
    """

    # Assuming constants for generic Telecom churn
    CAC_INR = 5000  # Customer Acquisition Cost in INR ~ $60
    RETENTION_OFFER_COST_INR = 1500  # Cost to try to save a customer ~ $18
    # We calculate expected loss based on MonthlyCharges and tenure logic, but can default to fixed LTV
    LTV_AVG_INR = 20000  # Average Customer LTV in INR ~ $240

    def __init__(self, model, X_train: pd.DataFrame):
        self.model = model
        self.X_train = X_train
        logger.info("Initializing Explainer...")
        
        # Determine model type for SHAP
        # Use TreeExplainer for robust gradient boosted algorithms
        try:
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("Using TreeExplainer.")
        except Exception:
            logger.info("Falling back to KernelExplainer/LinearExplainer... Using ExactExplainer as fallback for generic use.")
            self.explainer = shap.Explainer(self.model, self.X_train)

    def explain_instance(self, X_instance: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates and formats SHAP values for a single customer.
        Returns the top driving features.
        """
        # Get shape values and base value
        try:
            shap_values = self.explainer.shap_values(X_instance)
            # Depending on model output, shap_values might be a list (multiclass/keras) or array
            if isinstance(shap_values, list):
                shap_values = shap_values[1] # Choose positive class if binary
                
            base_value = self.explainer.expected_value
            if isinstance(base_value, list) or isinstance(base_value, np.ndarray):
                base_value = float(base_value[1])
                
            # If shap output is just 2D
        except Exception as e:
            logger.warning(f"Error with standard SHAP API: {e}. Trying direct SHAP instance evaluation...")
            explanation = self.explainer(X_instance)
            shap_values = explanation.values
            if shap_values.ndim > 2: # [instances, features, classes]
                shap_values = shap_values[:, :, 1]
            base_value = float(explanation.base_values[0]) if isinstance(explanation.base_values, (list, np.ndarray)) else float(explanation.base_values)

        # Get features and values
        feature_names = X_instance.columns.tolist()
        feature_values = X_instance.iloc[0].values
        
        shap_array = shap_values[0] if shap_values.ndim > 1 else shap_values
        
        # Sort features by absolute impact
        feature_impacts = [
            {"feature": f_name, "value": f_val, "shap_value": float(s_val), "impact": abs(float(s_val))}
            for f_name, f_val, s_val in zip(feature_names, feature_values, shap_array)
        ]
        
        # Sort descending by impact
        feature_impacts.sort(key=lambda x: x["impact"], reverse=True)
        
        # Get top POSITIVE (pushing towards churn) and NEGATIVE (pushing towards retention)
        top_churn_drivers = [f for f in feature_impacts if f["shap_value"] > 0][:3]
        top_retention_drivers = [f for f in feature_impacts if f["shap_value"] < 0][:3]
        
        return {
            "base_value": base_value,
            "top_churn_drivers": top_churn_drivers,
            "top_retention_drivers": top_retention_drivers,
            "all_impacts": feature_impacts
        }

    def compute_business_impact(self, user_MonthlyCharges: float, churn_probability: float) -> Dict[str, float]:
        """
        Calculates expected value and ROI of a retention campaign in INR.
        
        Business Logic:
        Expected Loss if untouched = churn_probability * Estimated Future Value (e.g. 1 Year LTV approx MonthlyCharges * 12)
        Cost of Intervention = RETENTION_OFFER_COST_INR
        Expected Recovery assuming 50% success of intervention = (0.5 * Expected Loss) - RETENTION_OFFER_COST_INR
        """
        # Dynamic LTV mapping based on monthly charges (simplified assumption predicting next 12mo)
        if pd.isna(user_MonthlyCharges) or user_MonthlyCharges <= 0:
            expected_ltv = self.LTV_AVG_INR
        else:
            # Monthly charges derived from dataset are often USD. Let's convert to INR assumption rate (1 USD = 83 INR)
            monthly_inr = user_MonthlyCharges * 83
            expected_ltv = monthly_inr * 12 # predict 1 year
            
        expected_financial_loss = expected_ltv * churn_probability
        
        # Intervention strategy
        intervention_success_rate = 0.50 # Assume 50% response to retention promo
        expected_saved_revenue = expected_financial_loss * intervention_success_rate
        net_roi_of_intervention = expected_saved_revenue - self.RETENTION_OFFER_COST_INR
        
        recommend_intervention = bool(net_roi_of_intervention > 0 and churn_probability > 0.4)
        
        return {
            "estimated_financial_loss_inr": round(expected_financial_loss, 2),
            "retention_cost_inr": self.RETENTION_OFFER_COST_INR,
            "expected_net_roi_inr": round(net_roi_of_intervention, 2),
            "recommend_intervention": recommend_intervention,
            "cac_inr": self.CAC_INR
        }
