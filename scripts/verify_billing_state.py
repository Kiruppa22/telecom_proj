import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "telecom_ops.db"


connection = sqlite3.connect(DB_PATH)
cursor = connection.cursor()


balance = cursor.execute(
    """
    SELECT current_balance
    FROM billing_accounts
    WHERE customer_id = ?
    """,
    ("CUST-10002",),
).fetchone()


credits = cursor.execute(
    """
    SELECT
        credit_id,
        amount,
        status,
        related_charge_id
    FROM billing_credits
    WHERE customer_id = ?
    ORDER BY credit_id
    """,
    ("CUST-10002",),
).fetchall()


print("=" * 60)
print("CUST-10002 BILLING VERIFICATION")
print("=" * 60)

print()
print("Current Balance:")
print(balance[0] if balance else "Account not found")

print()
print("Credits:")

for credit in credits:
    print(credit)

print()
print("=" * 60)

connection.close()