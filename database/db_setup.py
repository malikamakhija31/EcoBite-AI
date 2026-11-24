
from sqlalchemy import text
from database.db_connection import engine


def create_tables():
    queries = [
        """
        CREATE TABLE IF NOT EXISTS sales (
            id SERIAL PRIMARY KEY,
            order_id VARCHAR(100),
            date DATE,
            item_name VARCHAR(150),
            item_type VARCHAR(100),
            item_price NUMERIC(10,2),
            quantity INTEGER,
            transaction_amount NUMERIC(12,2),
            transaction_type VARCHAR(100),
            received_by VARCHAR(150),
            time_of_sale VARCHAR(50),
            data_source VARCHAR(100),
            food_category VARCHAR(100),
            prepared_qty INTEGER,
            wasted_qty INTEGER
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS inventory (
            inventory_id INTEGER PRIMARY KEY,
            item_name VARCHAR(150),
            stock_quantity NUMERIC(12,2),
            unit VARCHAR(50),
            reorder_level NUMERIC(12,2),
            supplier VARCHAR(150),
            record_date DATE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS html_waste_records (
            record_id VARCHAR(20) PRIMARY KEY,
            record_date DATE,
            food_category VARCHAR(100),
            waste_quantity NUMERIC(10,2),
            waste_reason VARCHAR(150),
            meal_period VARCHAR(50)
        )
        """
    ]

    with engine.begin() as connection:
        for query in queries:
            connection.execute(text(query))

    print("All 3 tables created successfully!")


if __name__ == "__main__":
    create_tables()
