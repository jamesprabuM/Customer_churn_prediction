# Customer Churn Prediction System

This is a full-stack, production-ready machine learning application designed to proactively predict customer churn. Beyond just returning a probability score, this system provides personalized explanations using SHAP, recommends business actions based on behavioral clustering, and calculates the financial impact (ROI) for customer retention.

## Features & Highlights

- **FastAPI Backend Pipeline**: A robust API that serves real-time predictions, explanations, and segmentations.
- **React Dashboard**: An interactive, modern UI built with Vite summarizing customer traits into dynamic charts and figures.
- **Machine Learning Models**: End-to-end scikit-learn processing mapping through an Optuna-tuned **XGBoost** model for top-tier accuracy.
- **Explainable AI (SHAP)**: Translates opaque ML predictions into explicit, feature-by-feature impact graphs and plain-English driver descriptions.
- **Business Analytics**: Automatically translates churn probabilities into estimated potential loss, predicted retention cost, and net savings in INR (₹).
- **Customer Segmentation**: Automatic KMeans clustering flags risk brackets based purely on cohort similarity and recommends targeted interventions.

---

## 🛠️ Setup & Installation

### 1. Backend (Python + FastAPI)

Open a terminal and navigate to the backend directory:

```bash
cd backend
```

Create a virtual environment and install the required dependencies (assuming Windows):

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Launch the FastAPI development server:

```bash
uvicorn app.main:app --reload --port 8000
```
> Note: The FastAPI server runs on `http://localhost:8000`. You can access auto-generated API documentation at `http://localhost:8000/docs`.

### 2. Frontend (React + Vite + Axios)

Open a **new** terminal window and navigate to the frontend directory:

```bash
cd frontend
```

Install the Node.js packages and launch the client application:

```bash
npm install
npm run dev
```
> Note: The Vite React server usually runs on `http://localhost:5173`. Open this URL in your browser to interact with the dashboard.

---

## 📂 Project Structure

```
churn_prediction/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI Endpoints & Routing
│   │   ├── preprocessing.py     # Scikit-learn Imputers & Scalers
│   │   ├── model.py             # ML Model Registry
│   │   ├── train_model.py       # Model Benchmarking Pipeline
│   │   ├── tune_model.py        # Optuna Hyperparameter Optimization
│   │   ├── explain.py           # SHAP Context Generation
│   │   ├── segmentation.py      # KMeans Customer Clustering
│   │   ├── cost.py              # INR-based Financial Logic
│   │   └── llm.py               # LLM Output generation mapping
│   ├── data/                    # Raw Telco CSV Dataset
│   └── requirements.txt         # Python Package Definitions
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Core React Dashboard Component
│   │   ├── App.css              # Dashboard Styling
│   │   └── main.jsx             # React Virtual DOM bindings
│   └── package.json             # NPM Configuration
└── README.md                    # Project Documentation
```

## 🚀 Tech Stack

- **Frontend:** React, Vite, Chart.js, Axios
- **Backend:** Python 3, FastAPI, Uvicorn, Pydantic
- **Data Science:** Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, Optuna, SHAP, KMeans