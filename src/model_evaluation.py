import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_DIR / "data" / "processed"
    / "food_sales_cleaned.csv"
)

CHART_DIR = PROJECT_DIR / "data" / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("\n--- MODEL EVALUATION STARTED ---")

    # 1. Load cleaned dataset
    df = pd.read_csv(DATA_FILE)

    df["date"] = pd.to_datetime(
        df["date"], errors="coerce"
    )

    for column in ["quantity", "item_price"]:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    df = df.dropna(
        subset=[
            "date", "item_name",
            "food_category", "quantity", "item_price"
        ]
    )

    # 2. Aggregate daily sales
    daily = (
        df.groupby(
            ["date", "item_name", "food_category"],
            as_index=False
        )
        .agg(
            quantity=("quantity", "sum"),
            item_price=("item_price", "mean")
        )
        .sort_values(["item_name", "date"])
    )

    # 3. Create features
    daily["previous_sales"] = (
        daily.groupby("item_name")["quantity"].shift(1)
    )

    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month

    daily = daily.dropna(
        subset=["previous_sales"]
    ).sort_values("date").reset_index(drop=True)

    features = [
        "item_name",
        "food_category",
        "item_price",
        "day_of_week",
        "month",
        "previous_sales"
    ]

    # 4. Chronological train-test split
    split_index = int(len(daily) * 0.8)
    train = daily.iloc[:split_index]
    test = daily.iloc[split_index:]

    if train.empty or test.empty:
        raise ValueError(
            "Not enough records for model evaluation."
        )

    # 5. Prepare model
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categories",
                OneHotEncoder(handle_unknown="ignore"),
                ["item_name", "food_category"]
            )
        ],
        remainder="passthrough"
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        )
    ])

    # 6. Train and predict
    model.fit(train[features], train["quantity"])

    predictions = np.maximum(
        0, model.predict(test[features])
    )

    # 7. Calculate evaluation metrics
    mae = mean_absolute_error(
        test["quantity"], predictions
    )

    rmse = np.sqrt(
        mean_squared_error(test["quantity"], predictions)
    )

    print("\n--- MODEL PERFORMANCE ---")
    print("Training records:", len(train))
    print("Testing records:", len(test))
    print("Mean Absolute Error (MAE):", round(mae, 2))
    print("Root Mean Squared Error (RMSE):", round(rmse, 2))

    # 8. Save actual vs predicted comparison
    comparison = test[
        ["date", "item_name", "quantity"]
    ].copy()

    comparison["predicted_quantity"] = predictions

    comparison.to_csv(
        PROJECT_DIR / "data" / "processed"
        / "model_evaluation_results.csv",
        index=False
    )

    # 9. Create graph using first 50 test records
    comparison = (
        comparison.sort_values("date").head(50)
    )

    plt.figure(figsize=(12, 6))

    plt.plot(
        range(len(comparison)),
        comparison["quantity"].to_numpy(),
        label="Actual Demand"
    )

    plt.plot(
        range(len(comparison)),
        comparison["predicted_quantity"].to_numpy(),
        label="Predicted Demand"
    )

    plt.title("Actual vs Predicted Food Demand")
    plt.xlabel("Test Records (Sample of 50)")
    plt.ylabel("Quantity Sold")
    plt.legend()
    plt.tight_layout()

    chart_file = CHART_DIR / "actual_vs_predicted.png"
    plt.savefig(chart_file, dpi=150)
    plt.close()

    print("\nEvaluation results saved.")
    print("Graph saved to:", chart_file)
    print("--- MODEL EVALUATION COMPLETED ---")


if __name__ == "__main__":
    main()
