from orchestration.crew_nodes import (
    run_customer_comms,
)


def main():
    query = (
        "Customer CUST-10002 says they were charged "
        "twice for Unlimited Plus in September 2026. "
        "What happened and has the issue been resolved?"
    )

    agent_context = """
Billing Resolution findings:

Customer:
Alex Romero
Customer ID: CUST-10002

Current balance:
131.98 USD

Duplicate charge:
Charge ID: CHG-50022
Description: Unlimited Plus
Amount: 65.99 USD
Billing period: 2026-09
Invoice status: OPEN
Duplicate flag: True

A legitimate Unlimited Plus charge also exists:
Charge ID: CHG-50021
Amount: 65.99 USD
Billing period: 2026-09

Credit:
Credit ID: 7
Amount: 65.99 USD
Status: PENDING_APPROVAL
Related charge: CHG-50022

The credit exceeds the 50 USD automatic approval
threshold.

Because the credit is PENDING_APPROVAL, it has not yet
reduced the customer's balance.

Current balance remains 131.98 USD.

No additional credit should be created because a pending
credit already exists for CHG-50022.
"""

    response = run_customer_comms(
        query,
        agent_context,
    )

    print()
    print("=" * 70)
    print("FINAL CUSTOMER RESPONSE")
    print("=" * 70)
    print()
    print(response)


if __name__ == "__main__":
    main()