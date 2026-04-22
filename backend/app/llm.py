import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class LLMExplainer:
    """
    Translates SHAP outputs and Business Logic into human-readable explanations
    using an LLM API (placeholder or openAI/Azure if key is provided).
    """

    def __init__(self, api_key: str = None, provider: str = "openai"):
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.provider = provider
        
        if not self.api_key:
            logger.warning("No LLM API key provided. Using fallback rule-based template generation for customer explanations.")

    def explain(self, shap_results: Dict[str, Any], business_metrics: Dict[str, Any], persona: str) -> str:
        """
        Creates a natural language paragraph tailored for a Customer Success Manager.
        """
        
        # Real LLM integration could call out here via `openai.ChatCompletion.create` using a prompt template.
        # But we build a strong deterministic template generator as fallback.
        
        # Parse top drivers
        top_risk = [f"{i['feature']} ({i['value']})" for i in shap_results.get('top_churn_drivers', [])]
        top_safe = [f"{i['feature']} ({i['value']})" for i in shap_results.get('top_retention_drivers', [])]
        
        financial_risk = business_metrics.get("estimated_financial_loss_inr", 0)
        roi = business_metrics.get("expected_net_roi_inr", 0)
        action_flag = business_metrics.get("recommend_intervention", False)

        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key="AIzaSyCl9hbLYxkJJrVt_71Dr0q-W3tO5hhQOZo")
                model = genai.GenerativeModel('gemini-pro')
                
                # Format a concise prompt so it responds quickly for the UI
                prompt = f"""
                You are a senior business analyst in the telecom industry. 
                A '{persona}' customer is at risk of churning.
                Their highest churn risk factors (from ML SHAP values) are: {', '.join(top_risk) if top_risk else 'None'}.
                Their highest retention factors keeping them are: {', '.join(top_safe) if top_safe else 'None'}.
                
                Write exactly two short sentences analyzing their specific situation and offering a decisive professional opinion on retaining them.
                Do not include placeholders, asterisks, or markdown. Keep it punchy and analytical.
                """
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                logger.error(f"Gemini API Error: {e}")
                pass # fall through to the static template below
            
        # Fallback Template
        explanation = f"This customer belongs to the '{persona}' segment. "
        
        if len(top_risk) > 0:
            explanation += f"Their primary risk factors for churning are: {', '.join(top_risk)}. "
            
        if len(top_safe) > 0:
            explanation += f"However, retention elements working in our favor include: {', '.join(top_safe)}. "
            
        explanation += f"\n\nFinancial Impact: If lost, expected LTV loss is ~₹{financial_risk:,.2f}. "
        
        if action_flag:
            explanation += f"A targeted intervention is HIGHLY RECOMMENDED. Offering a retention promotion (est. cost ₹1500) yields a projected net positive ROI of ₹{roi:,.2f} based on typical save rates."
        else:
            explanation += "Intervention is NOT currently recommended via standard promotion, as ROI does not justify standardized retention costs at this probability."
            
        return explanation
