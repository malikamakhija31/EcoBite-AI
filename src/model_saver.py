import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

PROJECT_DIR = Path(__file__).resolve().parent.parent
data_file = PROJECT_DIR / "data" / "processed" / "food_sales_cleaned.csv"
model_dir = PROJECT_DIR / "models"
model_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_file)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
for col in ["quantity", "item_price"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["date", "item_name", "food_category", "quantity", "item_price"])

daily = (
    df.groupby(["date", "item_name", "food_category"], as_index=False)
    .agg(quantity=("quantity", "sum"), item_price=("item_price", "mean"))
    .sort_values(["item_name", "date"])
)
daily["previous_sales"] = daily.groupby("item_name")["quantity"].shift(1)
daily["day_of_week"] = daily["date"].dt.dayofweek
daily["month"] = daily["date"].dt.month
daily = daily.dropna(subset=["previous_sales"]).sort_values("date").reset_index(drop=True)

features = ["item_name", "food_category", "item_price", "day_of_week", "month", "previous_sales"]
preprocessor = ColumnTransformer(
    transformers=[("categories", OneHotEncoder(handle_unknown="ignore"), ["item_name", "food_category"])],
    remainder="passthrough"
)
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])
model.fit(daily[features], daily["quantity"])
joblib.dump(model, model_dir / "demand_model.joblib")
print(f"Model saved successfully to {model_dir / 'demand_model.joblib'}")

