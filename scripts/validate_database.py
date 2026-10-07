from pathlib import Path
import sqlite3


BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_FILE = BASE_DIR / "data" / "telecom_ops.db"


def main():
    connection = sqlite3.connect(DATABASE_FILE)
    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    print("=" * 60)
    print("DATABASE VALIDATION")
    print("=" * 60)

    cursor.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        AND name NOT LIKE 'sqlite_%'
        ORDER BY name
    """)

    tables = [row["name"] for row in cursor.fetchall()]

    print("\nTables found:")

    for table in tables:
        print(f"  - {table}")

    print("\nRow counts:")

    for table in tables:
        cursor.execute(f'SELECT COUNT(*) AS count FROM "{table}"')
        count = cursor.fetchone()["count"]

        print(f"  {table}: {count}")

    print("\n" + "=" * 60)
    print("NETWORK TEST")
    print("=" * 60)

    cursor.execute("""
        SELECT *
        FROM network_towers
        WHERE tower_id = ?
    """, ("TX-512",))

    tower = cursor.fetchone()

    if tower:
        for key in tower.keys():
            print(f"{key}: {tower[key]}")
    else:
        print("TX-512 not found")

    print("\n" + "=" * 60)
    print("BILLING TEST")
    print("=" * 60)

    cursor.execute("""
        SELECT *
        FROM billing_accounts
        WHERE customer_id = ?
    """, ("CUST-10002",))

    account = cursor.fetchone()

    if account:
        for key in account.keys():
            print(f"{key}: {account[key]}")
    else:
        print("CUST-10002 not found")

    print("\n" + "=" * 60)
    print("CHARGES FOR CUST-10002")
    print("=" * 60)

    cursor.execute("""
        SELECT *
        FROM billing_charges
        WHERE customer_id = ?
    """, ("CUST-10002",))

    charges = cursor.fetchall()

    for charge in charges:
        print(dict(charge))

    print("\n" + "=" * 60)
    print("EXISTING BILLING CREDITS")
    print("=" * 60)

    cursor.execute("""
        SELECT *
        FROM billing_credits
        WHERE customer_id = ?
    """, ("CUST-10002",))

    credits = cursor.fetchall()

    if credits:
        for credit in credits:
            print(dict(credit))
    else:
        print("No existing credits for CUST-10002")

    connection.close()

    


if __name__ == "__main__":
    main()