
import pandas as pd
import numpy as np
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR / "data" / "processed"
    / "demand_predictions.csv"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Safety buffer: 5%
SAFETY_BUFFER = 0.05


def main():
    print("\n--- PREPARATION RECOMMENDATION STARTED ---")

    # 1. Load demand predictions
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {INPUT_FILE}. "
            "Run demand_prediction.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    # 2. Check required columns
    required_columns = [
        "forecast_date",
        "item_name",
        "food_category",
        "predicted_demand"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in prediction file: {missing_columns}"
        )

    # 3. Clean predicted demand
    df["predicted_demand"] = pd.to_numeric(
        df["predicted_demand"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["predicted_demand"]
    )

    df = df[
        df["predicted_demand"] >= 0
    ].copy()

    # 4. Calculate recommended preparation quantity
    df["recommended_preparation"] = np.ceil(
        df["predicted_demand"] * (1 + SAFETY_BUFFER)
    ).astype(int)

    # 5. Calculate extra quantity
    df["extra_buffer_quantity"] = (
        df["recommended_preparation"]
        - df["predicted_demand"]
    )

    # 6. Prepare final report
    report = df[
        [
            "forecast_date",
            "item_name",
            "food_category",
            "predicted_demand",
            "recommended_preparation",
            "extra_buffer_quantity"
        ]
    ].sort_values(
        "predicted_demand",
        ascending=False
    )

    # 7. Save report
    output_file = (
        OUTPUT_DIR / "preparation_recommendations.csv"
    )

    report.to_csv(output_file, index=False)

    # 8. Display results
    print("\n--- PREPARATION RECOMMENDATIONS ---")
    print(report.head(15).to_string(index=False))

    print(
        "\nTotal predicted demand:",
        int(report["predicted_demand"].sum())
    )

    print(
        "Total recommended preparation:",
        int(report["recommended_preparation"].sum())
    )

    print(
        "\nRecommendation report saved to:",
        output_file
    )

    print("--- RECOMMENDATION COMPLETED ---")


if __name__ == "__main__":
    main()
