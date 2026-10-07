from pathlib import Path
import sqlite3


BASE_DIR = Path(__file__).resolve().parent.parent

SCHEMA_FILE = BASE_DIR / "sql" / "01_schema.sql"
SEED_FILE = BASE_DIR / "sql" / "02_seed_data.sql"

DATA_DIR = BASE_DIR / "data"
DATABASE_FILE = DATA_DIR / "telecom_ops.db"


def execute_sql_file(connection, file_path):
    print(f"Running {file_path.name}...")

    sql = file_path.read_text(encoding="utf-8")
    connection.executescript(sql)

    print(f"{file_path.name} completed.")


def main():
    DATA_DIR.mkdir(exist_ok=True)

    if DATABASE_FILE.exists():
        print(f"Removing existing database: {DATABASE_FILE}")
        DATABASE_FILE.unlink()

    connection = sqlite3.connect(DATABASE_FILE)

    try:
        execute_sql_file(connection, SCHEMA_FILE)
        execute_sql_file(connection, SEED_FILE)

        connection.commit()

        print()
        print("Database created successfully.")
        print(f"Location: {DATABASE_FILE}")

    except Exception as error:
        connection.rollback()
        print()
        print("Database creation failed.")
        print(error)
        raise

    finally:
        connection.close()


if __name__ == "__main__":
    main()