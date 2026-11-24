
from sqlalchemy import text
from database.db_connection import engine

try:
    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT current_database(), current_user")
        )
        print("PostgreSQL Connected Successfully!")
        print(result.fetchone())
except Exception as e:
    print("Database connection failed:", e)
