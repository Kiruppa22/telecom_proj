import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "telecom_ops.db"


def inspect_schema():
    connection = sqlite3.connect(DB_PATH)

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT name, sql
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)

        tables = cursor.fetchall()

        print("DATABASE SCHEMA")
        print("=" * 80)

        for table_name, create_sql in tables:
            print(f"\nTABLE: {table_name}")
            print("-" * 80)
            print(create_sql)

    finally:
        connection.close()


if __name__ == "__main__":
    inspect_schema()