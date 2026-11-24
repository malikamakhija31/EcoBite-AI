
import pandas as pd
import json
from bs4 import BeautifulSoup
from sqlalchemy import text
from database.db_connection import engine

# 1. Import CSV - 2392 records
csv_path = "data/raw/food_waste_dataset.csv"
df = pd.read_csv(csv_path)

df["date"] = pd.to_datetime(
    df["date"], format="mixed", dayfirst=True, errors="coerce"
).dt.date

# 2. Import JSON - 15 records
json_path = "data/raw/food_inventory.json"

with open(json_path, "r", encoding="utf-8") as file:
    inventory_data = json.load(file)

inventory_df = pd.DataFrame(inventory_data["records"])

inventory_df["record_date"] = pd.to_datetime(
    inventory_df["record_date"], errors="coerce"
).dt.date

# 3. Import HTML - 15 records
html_path = "data/raw/food_waste_records.html"

with open(html_path, "r", encoding="utf-8") as file:
    soup = BeautifulSoup(file, "html.parser")

html_records = []

for row in soup.select("table tbody tr"):
    cells = [cell.get_text(strip=True) for cell in row.find_all("td")]

    if len(cells) == 6:
        html_records.append({
            "record_id": cells[0],
            "record_date": cells[1],
            "food_category": cells[2],
            "waste_quantity": cells[3],
            "waste_reason": cells[4],
            "meal_period": cells[5]
        })

html_df = pd.DataFrame(html_records)
html_df["record_date"] = pd.to_datetime(
    html_df["record_date"], errors="coerce"
).dt.date

# 4. Insert all data into PostgreSQL
with engine.begin() as connection:

    connection.execute(text("TRUNCATE TABLE sales RESTART IDENTITY"))
    connection.execute(text("TRUNCATE TABLE inventory"))
    connection.execute(text("TRUNCATE TABLE html_waste_records"))

    df.to_sql("sales", connection, if_exists="append", index=False)
    inventory_df.to_sql("inventory", connection, if_exists="append", index=False)
    html_df.to_sql("html_waste_records", connection, if_exists="append", index=False)

    sales_count = connection.execute(
        text("SELECT COUNT(*) FROM sales")
    ).scalar()

    inventory_count = connection.execute(
        text("SELECT COUNT(*) FROM inventory")
    ).scalar()

    html_count = connection.execute(
        text("SELECT COUNT(*) FROM html_waste_records")
    ).scalar()

print("All data imported successfully!")
print("CSV records:", sales_count)
print("JSON records:", inventory_count)
print("HTML records:", html_count)
