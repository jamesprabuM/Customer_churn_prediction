# Customer Churn Prediction System

Full-stack machine learning application that predicts telecom customer churn and explains why a customer is likely to churn.

It combines:
- Real-time churn prediction API
- SHAP-based feature contribution analysis
- LLM-generated interpreter summary
- Customer segmentation and business cost estimation

## Preview

![Dashboard Preview](https://raw.githubusercontent.com/jamesprabuM/Customer_churn_prediction/main/docs/images/churn-dashboard.png)

## Quick Links

- [Features](#features)
- [Result Preview](#result-preview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup and Run](#setup-and-run)

## Features

### Backend
- FastAPI endpoints for prediction and analytics
- Preprocessing, model inference, and explanation pipeline
- Cost and ROI-oriented churn impact outputs

### Frontend
- React + Vite dashboard for interactive insights
- Visual SHAP contribution matrix and summary panels
- Human-readable churn reasoning from LLM interpreter

### ML and Analytics
- Optuna-tuned XGBoost model
- SHAP explainability for transparent predictions
- KMeans-based customer segmentation and action guidance

## Result Preview

### LLM Interpreter and SHAP Output

This panel highlights the model explanation in two parts:
- LLM Interpreter sentence for business readability
- SHAP feature contribution matrix for technical transparency

## Tech Stack

- Frontend: React, Vite, Axios, Chart.js
- Backend: Python, FastAPI, Uvicorn, Pydantic
- Data Science: Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, Optuna, SHAP, KMeans

## Project Structure

```text
churn_prediction/
|-- backend/
|   |-- app/
|   |   |-- main.py
|   |   |-- preprocessing.py
|   |   |-- model.py
|   |   |-- train_model.py
|   |   |-- tune_model.py
|   |   |-- explain.py
|   |   |-- segmentation.py
|   |   |-- cost.py
|   |   `-- llm.py
|   |-- data/
|   `-- requirements.txt
|-- frontend/
|   |-- src/
|   |   |-- App.jsx
|   |   |-- App.css
|   |   |-- main.jsx
|   |   `-- assets/
|   `-- package.json
`-- README.md
```

## Setup and Run

### 1) Run Backend

```bash
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Backend URL: `http://localhost:8000`  
Swagger Docs: `http://localhost:8000/docs`

### 2) Run Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend URL: `http://localhost:5173`