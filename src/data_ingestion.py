
import json
import pandas as pd
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Input files
CSV_FILE = RAW_DIR / "food_waste_dataset.csv"
JSON_FILE = RAW_DIR / "food_inventory.json"
HTML_FILE = RAW_DIR / "food_waste_records.html"


def ingest_data():
    print("\n--- DATA INGESTION STARTED ---")

    # 1. Read CSV sales dataset
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_FILE}")

    sales_df = pd.read_csv(CSV_FILE)

    if sales_df.empty:
        raise ValueError("CSV dataset is empty.")

    sales_output = OUTPUT_DIR / "food_waste_processed.csv"
    sales_df.to_csv(sales_output, index=False)

    print("\nCSV DATASET")
    print("Rows:", len(sales_df))
    print("Columns:", sales_df.columns.tolist())
    print("Saved:", sales_output)

    # 2. Read JSON inventory dataset
    if not JSON_FILE.exists():
        raise FileNotFoundError(f"JSON file not found: {JSON_FILE}")

    with open(JSON_FILE, "r", encoding="utf-8") as file:
        json_data = json.load(file)

    if isinstance(json_data, dict):
        inventory_records = json_data.get("records", [])
    elif isinstance(json_data, list):
        inventory_records = json_data
    else:
        raise ValueError("Invalid JSON dataset format.")

    inventory_df = pd.DataFrame(inventory_records)

    if inventory_df.empty:
        raise ValueError("JSON inventory dataset is empty.")

    inventory_output = OUTPUT_DIR / "inventory_ingested.csv"
    inventory_df.to_csv(inventory_output, index=False)

    print("\nJSON DATASET")
    print("Rows:", len(inventory_df))
    print("Columns:", inventory_df.columns.tolist())
    print("Saved:", inventory_output)

    # 3. Read HTML food waste dataset
    if not HTML_FILE.exists():
        raise FileNotFoundError(f"HTML file not found: {HTML_FILE}")

    html_tables = pd.read_html(HTML_FILE)

    if not html_tables or html_tables[0].empty:
        raise ValueError("No data found in HTML table.")

    waste_df = html_tables[0]

    waste_output = OUTPUT_DIR / "waste_records_ingested.csv"
    waste_df.to_csv(waste_output, index=False)

    print("\nHTML DATASET")
    print("Rows:", len(waste_df))
    print("Columns:", waste_df.columns.tolist())
    print("Saved:", waste_output)

    print("\nAll three datasets ingested successfully!")
    print("--- DATA INGESTION COMPLETED ---")


if __name__ == "__main__":
    ingest_data()
