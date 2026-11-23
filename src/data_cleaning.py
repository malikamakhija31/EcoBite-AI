
import pandas as pd
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Input files
SALES_INPUT = DATA_DIR / "food_waste_processed.csv"
INVENTORY_INPUT = DATA_DIR / "inventory_ingested.csv"
WASTE_INPUT = DATA_DIR / "waste_records_ingested.csv"


def clean_sales_data():
    print("\n--- CLEANING SALES DATA ---")

    df = pd.read_csv(SALES_INPUT)
    print("Rows before cleaning:", len(df))
    print("Duplicate rows:", df.duplicated().sum())

    # Handle missing transaction type
    if "transaction_type" in df.columns:
        df["transaction_type"] = (
            df["transaction_type"].fillna("Unknown")
        )

    # Clean dates
    if "date" in df.columns:
        raw_dates = df["date"].astype(str).str.strip()

        try:
            dates = pd.to_datetime(
                raw_dates,
                format="mixed",
                errors="coerce"
            )
        except (TypeError, ValueError):
            dates = pd.to_datetime(
                raw_dates,
                errors="coerce"
            )

        missing = dates.isna()

        if missing.any():
            dates.loc[missing] = pd.to_datetime(
                raw_dates.loc[missing],
                format="%m/%d/%Y",
                errors="coerce"
            )

        df["date"] = dates

    # Convert numeric columns
    numeric_columns = [
        "quantity",
        "prepared_qty",
        "wasted_qty",
        "item_price",
        "transaction_amount"
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # Remove duplicates and rows missing important data
    df = df.drop_duplicates()

    required_columns = [
        "date",
        "item_name",
        "quantity",
        "prepared_qty",
        "wasted_qty",
        "item_price"
    ]

    required_columns = [
        col for col in required_columns
        if col in df.columns
    ]

    df = df.dropna(subset=required_columns)

    # Remove negative values
    for column in [
        "quantity",
        "prepared_qty",
        "wasted_qty",
        "item_price"
    ]:
        if column in df.columns:
            df = df[df[column] >= 0]

    # Calculate waste percentage
    if {"prepared_qty", "wasted_qty"}.issubset(df.columns):
        df["waste_percentage"] = 0.0

        valid = df["prepared_qty"] > 0

        df.loc[valid, "waste_percentage"] = (
            df.loc[valid, "wasted_qty"]
            / df.loc[valid, "prepared_qty"]
            * 100
        )

    if "date" in df.columns:
        df = df.sort_values("date")

    output = DATA_DIR / "food_sales_cleaned.csv"
    df.to_csv(output, index=False)

    print("Rows after cleaning:", len(df))
    print("Saved:", output)


def clean_inventory_data():
    print("\n--- CLEANING INVENTORY DATA ---")

    df = pd.read_csv(INVENTORY_INPUT)

    if "item_name" in df.columns:
        df["item_name"] = (
            df["item_name"].astype(str).str.strip()
        )

    for column in ["stock_quantity", "reorder_level"]:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )
            df = df.dropna(subset=[column])
            df = df[df[column] >= 0]

    if "record_date" in df.columns:
        df["record_date"] = pd.to_datetime(
            df["record_date"], errors="coerce"
        )

    df = df.drop_duplicates()

    output = DATA_DIR / "inventory_cleaned.csv"
    df.to_csv(output, index=False)

    print("Inventory records after cleaning:", len(df))
    print("Saved:", output)


def clean_waste_records():
    print("\n--- CLEANING HTML WASTE RECORDS ---")

    df = pd.read_csv(WASTE_INPUT)

    # Standardize column names
    df.columns = [
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        for column in df.columns
    ]

    # Rename HTML table quantity column
    if "waste_quantity_servings" in df.columns:
        df = df.rename(
            columns={
                "waste_quantity_servings": "waste_quantity"
            }
        )

    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"], errors="coerce"
        )

    if "waste_quantity" in df.columns:
        df["waste_quantity"] = pd.to_numeric(
            df["waste_quantity"], errors="coerce"
        )

        df = df.dropna(subset=["waste_quantity"])
        df = df[df["waste_quantity"] >= 0]

    df = df.drop_duplicates()

    output = DATA_DIR / "waste_records_cleaned.csv"
    df.to_csv(output, index=False)

    print("Waste records after cleaning:", len(df))
    print("Saved:", output)


def main():
    # Check whether ingestion outputs exist
    required_files = [
        SALES_INPUT,
        INVENTORY_INPUT,
        WASTE_INPUT
    ]

    for file in required_files:
        if not file.exists():
            raise FileNotFoundError(
                f"File not found: {file}. "
                "Run data_ingestion.py first."
            )

    clean_sales_data()
    clean_inventory_data()
    clean_waste_records()

    print("\nAll datasets cleaned successfully!")


if __name__ == "__main__":
    main()
