import json
import shap
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_shap_explanation(model: Any, customer_data: pd.DataFrame) -> str:
    """
    Uses SHAP TreeExplainer on a trained XGBoost model to generate feature 
    impacts for a single customer. 
    
    Returns:
        JSON string containing the top 5 features with the highest absolute impact.
    """
    logger.info("Initializing SHAP TreeExplainer...")
    
    # Initialize the TreeExplainer
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values for the single customer row
    logger.info("Computing SHAP values...")
    shap_values = explainer.shap_values(customer_data)
    
    # SHAP output formats vary slightly depending on the exact model and version.
    # For binary classification in XGBoost, it typically returns a 2D array: (n_samples, n_features).
    # If it returns a list (e.g., [class_0_array, class_1_array]), safely extract the positive class.
    if isinstance(shap_values, list):
        vals = shap_values[1][0]
    else:
        if len(shap_values.shape) > 1:
            vals = shap_values[0]
        else:
            vals = shap_values

    feature_names = customer_data.columns.tolist()
    
    # Pair feature names with their corresponding SHAP value
    feature_impacts = [
        {
            "feature": name, 
            "shap_value": float(val), 
            "absolute_impact": abs(float(val))
        }
        for name, val in zip(feature_names, vals)
    ]
    
    # Sort by absolute impact, descending
    feature_impacts.sort(key=lambda x: x["absolute_impact"], reverse=True)
    
    # Get top 5 features
    top_5 = feature_impacts[:5]
    
    # Generate human-friendly explanation
    positive_drivers = [item["feature"] for item in top_5 if item["shap_value"] > 0][:2]
    negative_drivers = [item["feature"] for item in top_5 if item["shap_value"] < 0][:2]

    # Map raw feature names to human-friendly terms
    feature_map = {
        "tenure": "low tenure",
        "MonthlyCharges": "high monthly charges",
        "InternetService_Fiber optic": "Fiber optic internet service",
        "Contract_Month-to-month": "month-to-month contract",
        "PaymentMethod_Electronic check": "electronic check payment",
        "TechSupport_No": "lack of tech support",
        "OnlineSecurity_No": "lack of online security"
    }

    def get_friendly_name(f):
        return feature_map.get(f, f.replace("_", " ").lower())

    try:
        from google import genai
        client = genai.Client(api_key="AIzaSyCl9hbLYxkJJrVt_71Dr0q-W3tO5hhQOZo")
        
        friendly_pos = [get_friendly_name(f) for f in positive_drivers]
        friendly_neg = [get_friendly_name(f) for f in negative_drivers]
        
        prompt = f"""
        You are an expert telecom business analyst.
        A customer has a high churn risk due to: {', '.join(friendly_pos)}.
        Their top retention factors are: {', '.join(friendly_neg)}.
        Write exactly two professional, punchy sentences explaining their situation and suggesting a decisive action.
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        human_explanation = response.text.replace('*', '')
    except Exception as e:
        logger.error(f"Gemini API Error: {e}")
        if positive_drivers:
            reasons = " and ".join([get_friendly_name(f) for f in positive_drivers])
            human_explanation = f"Customer is likely to churn due to {reasons}."
        elif negative_drivers:
            reasons = " and ".join([get_friendly_name(f) for f in negative_drivers])
            human_explanation = f"Customer is likely to stay due to {reasons}."
        else:
            human_explanation = "Customer churn risk is neutral based on key factors."

    # Format the final dictionary
    result = {
        "top_features": [
            {
                "feature_name": item["feature"],
                "shap_value": item["shap_value"]
            }
            for item in top_5
        ],
        "human_explanation": human_explanation
    }
    
    # Return as JSON formatted string
    return json.dumps(result, indent=4)
