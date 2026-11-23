# 🍲 EcoBite AI: Food Waste Prediction & Smart Preparation Intelligence

An enterprise-grade Food Waste Analytics and Machine Learning Demand Forecasting suite designed for commercial kitchens, restaurants, and cloud kitchens. Built with Python, Scikit-Learn, Plotly, and **Streamlit with a custom Dark Glassmorphism UI**.

---

## 🌟 Key Features & Dashboard Capabilities

1. **Executive KPI Overview (Power BI Style)**:
   - High-contrast Glassmorphic KPI cards tracking Total Prepared, Sold, Wasted, Waste Rate (%), Gross Revenue, and Financial Waste Loss.
   - Interactive Time-Series tracking Daily Sales vs Daily Waste.
   - Category-wise waste share donut chart and top 10 most wasted menu items.
   - Meal period waste distribution (Morning, Afternoon, Evening, Night, Midnight).

2. **Root-Cause Waste Analytics**:
   - Audit breakdown of waste reasons (Over-preparation, Leftovers, Expiration).
   - Category vs. Meal Period Heatmap Matrix.
   - Complete Menu Waste Audit table with financial loss ranking and risk grading.

3. **AI Demand Forecasting & Kitchen Preparation Plan**:
   - Random Forest Regressor demand model evaluation with MAE, RMSE, and Accuracy metrics.
   - Actual vs. Predicted sales validation curve.
   - Next-day kitchen preparation recommendations with customizable safety buffer margin (+5%).
   - **One-click CSV Export** of tomorrow's kitchen preparation dispatch sheet.

4. **Live "What-If" Demand Predictor**:
   - Real-time scenario simulator allowing chefs and managers to test any menu item, price point, day of the week, month, and previous day sales volume.
   - Instant machine learning inference with dynamic portion recommendation.

5. **Inventory & Stock Health Monitor**:
   - Reorder level threshold tracker.
   - Urgent reorder warnings and supplier management.

6. **ROI & ESG Sustainability Simulator**:
   - Customizable waste reduction targets (10% to 80%).
   - Calculates annual money saved (₹), portions rescued from landfill, avoided carbon emissions (kg CO2e), and conserved water footprint (Liters).

---

## 🚀 Quick Start Guide

### 1. Launch the Streamlit Dashboard
You can run the dashboard with one command:
```bash
streamlit run app.py
```
*(Or run `streamlit run dashboard/app.py`)*

On Windows, you can also simply double-click:
```cmd
run_dashboard.bat
```
The dashboard will open automatically in your browser at:
`http://localhost:8501`

---

### 2. Run the Full Data Pipeline & Retrain Models
To run ingestion, data cleaning, analysis, demand prediction, evaluation, and recommendation generation:
```bash
python main.py
```

---

## 📁 Project Architecture

```
Food-Waste-Prediction/
├── app.py                             # Root Streamlit app launcher
├── run_dashboard.bat                  # One-click Windows starter
├── requirements.txt                   # Project dependencies (Streamlit, Plotly, Scikit-learn, etc.)
├── main.py                            # End-to-end data pipeline runner
├── .streamlit/
│   └── config.toml                    # Dark theme configuration
├── dashboard/
│   └── app.py                         # Full Streamlit application with custom Dark CSS
├── models/
│   └── demand_model.joblib            # Trained Random Forest demand model
├── src/
│   ├── data_ingestion.py              # Ingestion from JSON & HTML
│   ├── data_cleaning.py               # Data cleaning & standardization
│   ├── data_analysis.py               # Statistical analysis & charts
│   ├── demand_prediction.py           # Demand forecasting logic
│   ├── model_evaluation.py            # Model error metrics & testing
│   ├── prep_recommendation.py         # Kitchen preparation recommendations
│   └── model_saver.py                 # Pipeline model persistence helper
└── data/
    ├── raw/                           # Raw input datasets
    ├── processed/                     # Cleaned & processed datasets
    └── charts/                        # Generated visual artifacts
```

---

## 🛠️ Tech Stack
- **Dashboard & UI**: Streamlit 1.45+, Plotly, HTML5, Custom Glassmorphic Dark CSS
- **Machine Learning**: Scikit-learn (RandomForestRegressor, OneHotEncoder, Pipeline)
- **Data Engineering**: Pandas, NumPy
- **Visuals**: Plotly Express, Plotly Graph Objects, Matplotlib

