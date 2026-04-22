from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import pandas as pd
import joblib
import logging
import os

from app.preprocessing import DataPreprocessor
from app.explain import generate_shap_explanation
from app.cost import calculate_business_cost
from app.segmentation import ProfileSegmenter

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Churn Prediction Engine", version="1.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global app state
app_state = {
    "model": None,
    "preprocessor": None,
    "segmenter": None,
}

# --- Pydantic Schema --- 
class CustomerData(BaseModel):
    gender: str = "Female"
    SeniorCitizen: int = 0
    Partner: str = "Yes"
    Dependents: str = "No"
    tenure: int = 1
    PhoneService: str = "No"
    MultipleLines: str = "No phone service"
    InternetService: str = "DSL"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "Yes"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 29.85
    TotalCharges: str = "29.85"

@app.on_event("startup")
def load_artifacts():
    """Load model, preprocessor, and segmenter on startup."""
    try:
        # Assuming the model is saved during an offline training phase
        if os.path.exists("best_model.joblib"):
            app_state["model"] = joblib.load("best_model.joblib")
            logger.info("Trained model loaded successfully.")
            
        # Instantiate and fit auxiliary tools (Ideally these should also be loaded from disk)
        app_state["preprocessor"] = DataPreprocessor()
        app_state["segmenter"] = ProfileSegmenter(k_clusters=3)
        
        # We need raw data to quickly fit the preprocessor & segmenter for the demo
        # In a real app, preprocessor and segmenter state should be pickled like the model
        data_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
        if os.path.exists(data_path):
            df_raw = app_state["preprocessor"].load_data(data_path)
            app_state["segmenter"].fit_predict(df_raw)
            app_state["preprocessor"].fit_transform(df_raw)
            logger.info("Preprocessor and Segmenter initialized and fitted.")
        else:
            logger.warning(f"Dataset not found at {data_path}. Ensure data exists to fit preprocessors.")
            
    except Exception as e:
        logger.error(f"Error during startup: {e}")

@app.post("/predict")
def predict_churn(customer: CustomerData):
    if not app_state["model"] or not app_state["preprocessor"].is_fitted:
        raise HTTPException(status_code=500, detail="Model or preprocessor not loaded/fitted.")
        
    df_input = pd.DataFrame([customer.dict()])
    X_transformed = app_state["preprocessor"].transform(df_input)
    
    prob = float(app_state["model"].predict_proba(X_transformed)[0][1])
    return {"churn_probability": round(prob, 4)}

@app.post("/explain")
def explain_churn(customer: CustomerData):
    if not app_state["model"] or not app_state["preprocessor"].is_fitted:
        raise HTTPException(status_code=500, detail="Model or preprocessor not loaded/fitted.")
        
    df_input = pd.DataFrame([customer.dict()])
    X_transformed = app_state["preprocessor"].transform(df_input)
    
    explanation_json = generate_shap_explanation(app_state["model"], X_transformed)
    import json
    parsed_explanation = json.loads(explanation_json)
    
    # Convert list of dicts to key-value pairs for the React 'explanation' hook
    factors = {item["feature_name"]: item["shap_value"] for item in parsed_explanation["top_features"]}
    
    return {
        "top_factors": factors,
        "human_explanation": parsed_explanation.get("human_explanation")
    }

@app.post("/cost")
def cost_analysis(customer: CustomerData):
    # First get probability to run cost analysis
    df_input = pd.DataFrame([customer.dict()])
    X_transformed = app_state["preprocessor"].transform(df_input)
    prob = float(app_state["model"].predict_proba(X_transformed)[0][1])
    
    cost_json = calculate_business_cost(churn_probability=prob)
    import json
    return json.loads(cost_json)

@app.post("/segment")
def get_segment(customer: CustomerData):
    if not app_state["segmenter"] or not app_state["segmenter"].is_fitted:
        raise HTTPException(status_code=500, detail="Segmenter not initialized/fitted.")
        
    df_input = pd.DataFrame([customer.dict()])
    segment_info = app_state["segmenter"].get_segment(df_input)
    return segment_info

