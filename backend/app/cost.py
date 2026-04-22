import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def calculate_business_cost(
    churn_probability: float, 
    potential_loss_default: float = 5000.0, 
    retention_cost_default: float = 500.0
) -> str:
    """
    Calculates business cost and expected savings in INR if a customer's churn
    probability exceeds the threshold of 0.7.
    
    Args:
        churn_probability: Model predicted probability of churning
        potential_loss_default: Base Customer LTV cost if they churn
        retention_cost_default: Marketing cost to retain them (promo code, etc)
        
    Returns:
        JSON string output containing potential_loss, retention_cost, and net_saving.
    """
    
    # Check if probability requires intervention
    is_high_risk = churn_probability > 0.60
    
    potential_loss = potential_loss_default * churn_probability
    
    if is_high_risk:
        net_saving = potential_loss - retention_cost_default
    else:
        net_saving = 0.0
        
    result = {
        "churn_probability": float(round(float(churn_probability), 4)),
        "potential_loss": float(potential_loss),
        "retention_cost": float(retention_cost_default),
        "net_savings": float(net_saving),
        "recommendation": "Targeted retention promotion is mathematically recommended to save revenue." if is_high_risk else "Do not intervene. Risk is low or ROI does not justify standardized retention costs."
    }
    
    return json.dumps(result, indent=4)
