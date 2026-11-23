
import pandas as pd
import numpy as np
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

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("\n--- FOOD DEMAND PREDICTION STARTED ---")

    # 1. Load cleaned sales dataset
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

    if df.empty:
        raise ValueError("No valid sales records available.")

    # 2. Summarize daily sales by item
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

    # 3. Add previous sales and calendar features
    daily["previous_sales"] = (
        daily.groupby("item_name")["quantity"].shift(1)
    )

    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month

    daily = daily.dropna(
        subset=["previous_sales"]
    ).sort_values("date").reset_index(drop=True)

    if len(daily) < 5:
        raise ValueError(
            "Not enough historical records for demand prediction."
        )

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
        raise ValueError("Training or testing dataset is empty.")

    # 5. Encode categorical columns
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

    # 6. Create Random Forest model
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

    # 7. Train and evaluate model
    model.fit(train[features], train["quantity"])

    predictions = np.maximum(
        0, model.predict(test[features])
    )

    mae = mean_absolute_error(
        test["quantity"], predictions
    )

    rmse = np.sqrt(
        mean_squared_error(test["quantity"], predictions)
    )

    print("\n--- MODEL PERFORMANCE ---")
    print("Training records:", len(train))
    print("Testing records:", len(test))
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))

    # 8. Predict demand for the next date in the dataset
    latest_date = daily["date"].max()
    next_date = latest_date + pd.Timedelta(days=1)

    latest_items = (
        daily.sort_values("date")
        .groupby("item_name")
        .tail(1)
        .copy()
    )

    future = latest_items[
        [
            "item_name",
            "food_category",
            "item_price",
            "quantity"
        ]
    ].copy()

    future["previous_sales"] = future["quantity"]
    future["day_of_week"] = next_date.dayofweek
    future["month"] = next_date.month

    future["predicted_demand"] = np.maximum(
        0, model.predict(future[features])
    ).round().astype(int)

    # 5% safety buffer
    future["recommended_preparation"] = np.ceil(
        future["predicted_demand"] * 1.05
    ).astype(int)

    future.insert(
        0,
        "forecast_date",
        next_date.strftime("%Y-%m-%d")
    )

    future = future.sort_values(
        "predicted_demand",
        ascending=False
    )

    # 9. Save predictions
    output_file = OUTPUT_DIR / "demand_predictions.csv"
    future.to_csv(output_file, index=False)

    print("\n--- NEXT-DAY DEMAND PREDICTIONS ---")
    print(future.head(10).to_string(index=False))
    print("\nPredictions saved to:", output_file)

    print(
        "\nNote: Forecast date is based on the latest date "
        "available in the dataset."
    )


if __name__ == "__main__":
    main()
