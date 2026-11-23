import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
CHART_DIR = PROJECT_DIR / "data" / "charts"

CHART_DIR.mkdir(parents=True, exist_ok=True)


def analyze_sales_data():
    print("\n--- SALES DATA ANALYSIS ---")

    file_path = DATA_DIR / "food_sales_cleaned.csv"
    df = pd.read_csv(file_path)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    total_prepared = df["prepared_qty"].sum()
    total_sold = df["quantity"].sum()
    total_wasted = df["wasted_qty"].sum()

    waste_percentage = (
        total_wasted / total_prepared * 100
        if total_prepared > 0 else 0
    )

    print("Total Food Prepared:", total_prepared)
    print("Total Food Sold:", total_sold)
    print("Total Food Wasted:", total_wasted)
    print("Overall Waste Percentage:", round(waste_percentage, 2), "%")

    # Waste by food item
    item_summary = (
        df.groupby("item_name", as_index=False)
        .agg(
            Total_Prepared=("prepared_qty", "sum"),
            Total_Sold=("quantity", "sum"),
            Total_Wasted=("wasted_qty", "sum")
        )
    )

    item_summary["Waste_Percentage"] = (
        item_summary["Total_Wasted"]
        / item_summary["Total_Prepared"].replace(0, float("nan"))
        * 100
    ).fillna(0)

    item_summary = item_summary.sort_values(
        "Total_Wasted", ascending=False
    )

    item_summary.to_csv(
        DATA_DIR / "waste_by_food_item.csv",
        index=False
    )

    print("\n--- TOP 10 ITEMS BY WASTE ---")
    print(item_summary.head(10).to_string(index=False))

    # Chart: top 10 items
    top_items = item_summary.head(10).sort_values("Total_Wasted")

    plt.figure(figsize=(10, 6))
    plt.barh(top_items["item_name"], top_items["Total_Wasted"])
    plt.title("Top 10 Food Items by Total Waste")
    plt.xlabel("Quantity Wasted")
    plt.ylabel("Food Item")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "top_food_waste.png", dpi=150)
    plt.close()

    # Waste by category
    if "food_category" in df.columns:
        category_summary = (
            df.groupby("food_category", as_index=False)
            .agg(
                Total_Prepared=("prepared_qty", "sum"),
                Total_Sold=("quantity", "sum"),
                Total_Wasted=("wasted_qty", "sum")
            )
        )

        category_summary["Waste_Percentage"] = (
            category_summary["Total_Wasted"]
            / category_summary["Total_Prepared"].replace(
                0, float("nan")
            )
            * 100
        ).fillna(0)

        category_summary.to_csv(
            DATA_DIR / "waste_by_category.csv",
            index=False
        )

        plt.figure(figsize=(10, 6))
        plt.bar(
            category_summary["food_category"],
            category_summary["Total_Wasted"]
        )
        plt.title("Total Food Waste by Category")
        plt.xlabel("Food Category")
        plt.ylabel("Quantity Wasted")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(CHART_DIR / "waste_by_category.png", dpi=150)
        plt.close()

        print("\n--- WASTE BY CATEGORY ---")
        print(category_summary.to_string(index=False))


def analyze_inventory_data():
    print("\n--- INVENTORY ANALYSIS ---")

    file_path = DATA_DIR / "inventory_cleaned.csv"
    df = pd.read_csv(file_path)

    if "stock_quantity" in df.columns:
        print("Total inventory stock:", df["stock_quantity"].sum())

    if {"item_name", "stock_quantity", "reorder_level"}.issubset(df.columns):
        latest = df.copy()

        if "record_date" in latest.columns:
            latest["record_date"] = pd.to_datetime(
                latest["record_date"], errors="coerce"
            )
            latest = latest.sort_values("record_date")
            latest = latest.groupby(
                "item_name", as_index=False
            ).tail(1)

        low_stock = latest[
            latest["stock_quantity"] <= latest["reorder_level"]
        ]

        print("\nItems at or below reorder level:")
        if low_stock.empty:
            print("No low-stock items found.")
        else:
            print(low_stock.to_string(index=False))

        low_stock.to_csv(
            DATA_DIR / "low_inventory_items.csv",
            index=False
        )


def analyze_waste_records():
    print("\n--- HTML WASTE RECORDS ANALYSIS ---")

    file_path = DATA_DIR / "waste_records_cleaned.csv"
    df = pd.read_csv(file_path)

    print("Total waste-log records:", len(df))

    if "waste_quantity" in df.columns:
        df["waste_quantity"] = pd.to_numeric(
            df["waste_quantity"], errors="coerce"
        )
        print("Total waste in log dataset:", df["waste_quantity"].sum())

    if {"waste_reason", "waste_quantity"}.issubset(df.columns):
        reason_summary = (
            df.groupby("waste_reason", as_index=False)
            .agg(
                Total_Waste=("waste_quantity", "sum"),
                Total_Records=("waste_quantity", "count")
            )
            .sort_values("Total_Waste", ascending=False)
        )

        reason_summary.to_csv(
            DATA_DIR / "waste_by_reason.csv",
            index=False
        )

        print("\nWaste by reason:")
        print(reason_summary.to_string(index=False))

        plt.figure(figsize=(9, 5))
        plt.bar(
            reason_summary["waste_reason"],
            reason_summary["Total_Waste"]
        )
        plt.title("Food Waste by Reason")
        plt.xlabel("Waste Reason")
        plt.ylabel("Waste Quantity")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(CHART_DIR / "waste_by_reason.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    analyze_sales_data()
    analyze_inventory_data()
    analyze_waste_records()

    print("\nAnalysis completed successfully!")
    print("Summary files saved in data/processed/")
    print("Charts saved in data/charts/")
