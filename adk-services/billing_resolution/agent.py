import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.a2a.utils.agent_to_a2a import to_a2a


BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "data" / "telecom_ops.db"

AUTO_APPROVAL_LIMIT = 50.00

load_dotenv(BASE_DIR / ".env")


def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


async def lookup_billing_account(customer_id: str) -> dict:
    """
    Look up a customer's billing account, current balance,
    recent charges, credits, and billing disputes from SQLite.
    """

    if not DB_PATH.exists():
        return {
            "success": False,
            "error": f"Database not found: {DB_PATH}",
        }

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                customer_id,
                customer_name,
                account_type,
                current_balance,
                currency,
                service_region,
                city,
                state,
                billing_cycle,
                auto_pay_enabled,
                account_status,
                last_updated
            FROM billing_accounts
            WHERE customer_id = ?
            """,
            (customer_id,),
        )

        account = cursor.fetchone()

        if account is None:
            return {
                "success": False,
                "customer_id": customer_id,
                "error": "Billing account not found.",
            }

        cursor.execute(
            """
            SELECT
                charge_id,
                description,
                amount,
                billing_period,
                charge_date,
                charge_type,
                is_duplicate_flag,
                invoice_status
            FROM billing_charges
            WHERE customer_id = ?
            ORDER BY charge_date DESC, charge_id DESC
            """,
            (customer_id,),
        )

        charge_rows = cursor.fetchall()

        charges = []

        for row in charge_rows:
            charges.append(
                {
                    "charge_id": row["charge_id"],
                    "description": row["description"],
                    "amount": row["amount"],
                    "billing_period": row["billing_period"],
                    "charge_date": row["charge_date"],
                    "charge_type": row["charge_type"],
                    "is_duplicate_flag": bool(
                        row["is_duplicate_flag"]
                    ),
                    "invoice_status": row["invoice_status"],
                }
            )

        cursor.execute(
            """
            SELECT
                credit_id,
                amount,
                reason,
                status,
                created_at,
                related_charge_id
            FROM billing_credits
            WHERE customer_id = ?
            ORDER BY created_at DESC, credit_id DESC
            """,
            (customer_id,),
        )

        credit_rows = cursor.fetchall()

        credits = []

        for row in credit_rows:
            credits.append(
                {
                    "credit_id": row["credit_id"],
                    "amount": row["amount"],
                    "reason": row["reason"],
                    "status": row["status"],
                    "created_at": row["created_at"],
                    "related_charge_id":
                        row["related_charge_id"],
                }
            )

        cursor.execute(
            """
            SELECT
                dispute_id,
                charge_id,
                reason,
                status,
                opened_at,
                resolved_at,
                resolution_notes
            FROM billing_disputes
            WHERE customer_id = ?
            ORDER BY opened_at DESC
            """,
            (customer_id,),
        )

        dispute_rows = cursor.fetchall()

        disputes = []

        for row in dispute_rows:
            disputes.append(
                {
                    "dispute_id": row["dispute_id"],
                    "charge_id": row["charge_id"],
                    "reason": row["reason"],
                    "status": row["status"],
                    "opened_at": row["opened_at"],
                    "resolved_at": row["resolved_at"],
                    "resolution_notes":
                        row["resolution_notes"],
                }
            )

        open_charge_total = round(
            sum(
                row["amount"]
                for row in charge_rows
                if row["invoice_status"] == "OPEN"
            ),
            2,
        )

        return {
            "success": True,
            "account": {
                "customer_id": account["customer_id"],
                "customer_name": account["customer_name"],
                "account_type": account["account_type"],
                "current_balance": account["current_balance"],
                "currency": account["currency"],
                "service_region": account["service_region"],
                "city": account["city"],
                "state": account["state"],
                "billing_cycle": account["billing_cycle"],
                "auto_pay_enabled": bool(
                    account["auto_pay_enabled"]
                ),
                "account_status": account["account_status"],
                "last_updated": account["last_updated"],
            },
            "open_charge_total": open_charge_total,
            "charges": charges,
            "credits": credits,
            "disputes": disputes,
        }

    finally:
        connection.close()


async def check_duplicate_charges(customer_id: str) -> dict:
    """
    Find actual duplicate OPEN billing charges for a customer.

    A charge is treated as the duplicate when:
    1. is_duplicate_flag = 1, or
    2. the charge is explicitly referenced by an OPEN billing
       dispute describing a duplicate/double charge.

    Matching recurring charges in different billing periods
    are not duplicates.
    """

    if not DB_PATH.exists():
        return {
            "success": False,
            "error": f"Database not found: {DB_PATH}",
        }

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                customer_id,
                customer_name,
                current_balance,
                currency,
                account_status
            FROM billing_accounts
            WHERE customer_id = ?
            """,
            (customer_id,),
        )

        account = cursor.fetchone()

        if account is None:
            return {
                "success": False,
                "customer_id": customer_id,
                "error": "Billing account not found.",
            }

        cursor.execute(
            """
            SELECT
                c.charge_id,
                c.description,
                c.amount,
                c.billing_period,
                c.charge_date,
                c.charge_type,
                c.is_duplicate_flag,
                c.invoice_status,

                (
                    SELECT COUNT(*)
                    FROM billing_charges AS c2
                    WHERE c2.customer_id = c.customer_id
                      AND c2.description = c.description
                      AND c2.billing_period = c.billing_period
                      AND c2.invoice_status = 'OPEN'
                ) AS matching_open_count,

                EXISTS (
                    SELECT 1
                    FROM billing_disputes AS d
                    WHERE d.customer_id = c.customer_id
                      AND d.charge_id = c.charge_id
                      AND d.status = 'OPEN'
                      AND (
                          LOWER(d.reason) LIKE '%duplicate%'
                          OR LOWER(d.reason) LIKE '%twice%'
                          OR LOWER(d.reason) LIKE '%double%'
                      )
                ) AS duplicate_dispute

            FROM billing_charges AS c

            WHERE c.customer_id = ?
              AND c.invoice_status = 'OPEN'
              AND (
                    c.is_duplicate_flag = 1
                    OR EXISTS (
                        SELECT 1
                        FROM billing_disputes AS d
                        WHERE d.customer_id = c.customer_id
                          AND d.charge_id = c.charge_id
                          AND d.status = 'OPEN'
                          AND (
                              LOWER(d.reason) LIKE '%duplicate%'
                              OR LOWER(d.reason) LIKE '%twice%'
                              OR LOWER(d.reason) LIKE '%double%'
                          )
                    )
                  )

            ORDER BY
                c.billing_period DESC,
                c.charge_date DESC,
                c.charge_id DESC
            """,
            (customer_id,),
        )

        rows = cursor.fetchall()

        duplicate_charges = []

        for row in rows:
            cursor.execute(
                """
                SELECT
                    credit_id,
                    amount,
                    reason,
                    status,
                    created_at
                FROM billing_credits
                WHERE customer_id = ?
                  AND related_charge_id = ?
                ORDER BY credit_id DESC
                """,
                (
                    customer_id,
                    row["charge_id"],
                ),
            )

            existing_credits = cursor.fetchall()

            already_applied = any(
                credit["status"] == "APPLIED"
                for credit in existing_credits
            )

            pending_approval = any(
                credit["status"] == "PENDING_APPROVAL"
                for credit in existing_credits
            )

            if row["is_duplicate_flag"] == 1:
                duplicate_reason = "Explicit duplicate flag"
            else:
                duplicate_reason = (
                    "OPEN billing dispute identifies this "
                    "charge as a duplicate"
                )

            duplicate_charges.append(
                {
                    "charge_id": row["charge_id"],
                    "description": row["description"],
                    "amount": row["amount"],
                    "billing_period": row["billing_period"],
                    "charge_date": row["charge_date"],
                    "charge_type": row["charge_type"],
                    "invoice_status": row["invoice_status"],
                    "is_duplicate_flag": bool(
                        row["is_duplicate_flag"]
                    ),
                    "matching_open_count":
                        row["matching_open_count"],
                    "duplicate_reason": duplicate_reason,
                    "already_applied_credit":
                        already_applied,
                    "pending_credit_approval":
                        pending_approval,
                }
            )

        actionable_duplicates = [
            item
            for item in duplicate_charges
            if not item["already_applied_credit"]
            and not item["pending_credit_approval"]
        ]

        return {
            "success": True,
            "customer_id": customer_id,
            "customer_name": account["customer_name"],
            "current_balance": account["current_balance"],
            "currency": account["currency"],
            "duplicate_count": len(
                duplicate_charges
            ),
            "actionable_duplicate_count": len(
                actionable_duplicates
            ),
            "duplicate_charges": duplicate_charges,
            "actionable_duplicates":
                actionable_duplicates,
        }

    finally:
        connection.close()

async def apply_billing_credit(
    customer_id: str,
    amount: float,
    reason: str,
) -> dict:
    """
    Apply or request approval for a billing credit.

    Credits of $50 or less are APPLIED and reduce the
    customer's current balance.

    Credits above $50 are inserted as PENDING_APPROVAL and
    do not change the customer's current balance.

    The function identifies the duplicate charge from SQL
    and prevents an already credited duplicate charge from
    being credited again.
    """

    if not DB_PATH.exists():
        return {
            "success": False,
            "error": f"Database not found: {DB_PATH}",
        }

    try:
        amount = round(float(amount), 2)
    except (TypeError, ValueError):
        return {
            "success": False,
            "error": "Credit amount must be numeric.",
        }

    if amount <= 0:
        return {
            "success": False,
            "error": (
                "Credit amount must be greater than zero."
            ),
        }

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("BEGIN IMMEDIATE")

        cursor.execute(
            """
            SELECT
                customer_id,
                customer_name,
                current_balance,
                currency,
                account_status
            FROM billing_accounts
            WHERE customer_id = ?
            """,
            (customer_id,),
        )

        account = cursor.fetchone()

        if account is None:
            connection.rollback()

            return {
                "success": False,
                "customer_id": customer_id,
                "error": "Billing account not found.",
            }

        if account["account_status"] == "CLOSED":
            connection.rollback()

            return {
                "success": False,
                "customer_id": customer_id,
                "error": (
                    "Credit cannot be created for a "
                    "closed billing account."
                ),
            }

        cursor.execute(
            """
            SELECT
                c.charge_id,
                c.description,
                c.amount,
                c.billing_period,
                c.charge_date,
                c.is_duplicate_flag

            FROM billing_charges AS c

            WHERE c.customer_id = ?
              AND c.invoice_status = 'OPEN'
              AND ABS(c.amount - ?) < 0.005
              AND (
                    c.is_duplicate_flag = 1

                    OR EXISTS (
                        SELECT 1
                        FROM billing_disputes AS d
                        WHERE d.customer_id = c.customer_id
                          AND d.charge_id = c.charge_id
                          AND d.status = 'OPEN'
                          AND (
                              LOWER(d.reason)
                                  LIKE '%duplicate%'
                              OR LOWER(d.reason)
                                  LIKE '%twice%'
                              OR LOWER(d.reason)
                                  LIKE '%double%'
                          )
                    )
                  )

            ORDER BY
                c.is_duplicate_flag DESC,
                c.charge_date DESC,
                c.charge_id DESC
            """,
            (
                customer_id,
                amount,
            ),
        )

        candidate_charges = cursor.fetchall()

        related_charge = None

        for charge in candidate_charges:
            cursor.execute(
                """
                SELECT
                    credit_id,
                    status
                FROM billing_credits
                WHERE customer_id = ?
                  AND related_charge_id = ?
                  AND status IN (
                      'APPLIED',
                      'PENDING_APPROVAL'
                  )
                ORDER BY credit_id DESC
                LIMIT 1
                """,
                (
                    customer_id,
                    charge["charge_id"],
                ),
            )

            existing_credit = cursor.fetchone()

            if existing_credit is None:
                related_charge = charge
                break

        if related_charge is None:
            connection.rollback()

            if candidate_charges:
                return {
                    "success": False,
                    "customer_id": customer_id,
                    "amount": amount,
                    "error": (
                        "The matching duplicate charge "
                        "already has an APPLIED or "
                        "PENDING_APPROVAL credit."
                    ),
                }

            return {
                "success": False,
                "customer_id": customer_id,
                "amount": amount,
                "error": (
                    "No eligible OPEN duplicate charge "
                    "matching this credit amount was found."
                ),
            }

        old_balance = round(
            float(account["current_balance"]),
            2,
        )

        if amount <= AUTO_APPROVAL_LIMIT:
            status = "APPLIED"

            new_balance = round(
                max(0.0, old_balance - amount),
                2,
            )
        else:
            status = "PENDING_APPROVAL"
            new_balance = old_balance

        cursor.execute(
            """
            INSERT INTO billing_credits (
                customer_id,
                amount,
                reason,
                status,
                related_charge_id
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                customer_id,
                amount,
                reason,
                status,
                related_charge["charge_id"],
            ),
        )

        credit_id = cursor.lastrowid

        if status == "APPLIED":
            cursor.execute(
                """
                UPDATE billing_accounts
                SET
                    current_balance = ?,
                    last_updated = datetime('now')
                WHERE customer_id = ?
                """,
                (
                    new_balance,
                    customer_id,
                ),
            )

        connection.commit()

        return {
            "success": True,
            "credit_id": credit_id,
            "customer_id": customer_id,
            "customer_name": account["customer_name"],
            "related_charge": {
                "charge_id":
                    related_charge["charge_id"],
                "description":
                    related_charge["description"],
                "amount":
                    related_charge["amount"],
                "billing_period":
                    related_charge["billing_period"],
                "charge_date":
                    related_charge["charge_date"],
            },
            "credit": {
                "amount": amount,
                "reason": reason,
                "status": status,
                "auto_approval_limit":
                    AUTO_APPROVAL_LIMIT,
            },
            "previous_balance": old_balance,
            "new_balance": new_balance,
            "balance_changed":
                status == "APPLIED",
            "requires_approval":
                status == "PENDING_APPROVAL",
        }

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


ANTHROPIC_API_KEY = os.getenv(
    "ANTHROPIC_API_KEY"
)

ANTHROPIC_MODEL = os.getenv(
    "ANTHROPIC_MODEL",
    "claude-haiku-4-5-20251001",
)

if not ANTHROPIC_API_KEY:
    raise ValueError(
        "ANTHROPIC_API_KEY was not found in .env"
    )


claude_model = LiteLlm(
    model=f"anthropic/{ANTHROPIC_MODEL}"
)


root_agent = Agent(
    name="billing_resolution_agent",

    model=claude_model,

    description=(
        "Billing Resolution specialist that investigates "
        "customer billing accounts, detects duplicate "
        "charges, and processes eligible billing credits."
    ),

    instruction="""
You are the Billing Resolution specialist for a telecom
operations center.

Your responsibility is to investigate customer billing
disputes using the SQL-backed tools and take appropriate
billing-credit actions.

You have three tools:

1. lookup_billing_account

Use this tool to inspect a customer's account, current
balance, charges, existing credits, and disputes.

OPEN charges represent the customer's current billing
activity.

PAID charges are historical charges from previous bills.
Do not add PAID historical charges to the current balance.

2. check_duplicate_charges

Use this tool when the customer believes they were charged
twice.

A duplicate is identified using billing data from SQLite.

The same description in an earlier billing period is not,
by itself, a duplicate.

Do not treat normal historical recurring plan charges as
duplicates.

Do not create another credit when a charge already has an
APPLIED credit.

Do not create another credit when a charge already has a
PENDING_APPROVAL credit.

3. apply_billing_credit

Use this tool only after the billing data supports a
duplicate-charge credit.

Credits of 50 dollars or less are automatically APPLIED.

An APPLIED credit reduces the customer's current balance.

Credits above 50 dollars must be PENDING_APPROVAL.

A PENDING_APPROVAL credit must not reduce the customer's
current balance.

Never claim that a pending credit has already been applied.

Never invent customer balances, charge amounts, charge IDs,
credit IDs, billing periods, disputes, or credit statuses.

SQLite tool results are the source of truth.

When resolving a duplicate-charge request:

- inspect the billing account;
- check for duplicate charges;
- identify the actual duplicate charge;
- apply the appropriate credit only when supported by the
  SQL data;
- clearly state whether the credit was APPLIED or
  PENDING_APPROVAL;
- clearly state whether the balance changed.

For amounts above 50 dollars, explain that approval is
required and the balance remains unchanged until approval.

Keep the response factual and concise.
""",

    tools=[
        lookup_billing_account,
        check_duplicate_charges,
        apply_billing_credit,
    ],
)


a2a_app = to_a2a(
    root_agent,
    port=8002,
)