
from orchestration.adk_remote_client import (
    ask_network_diagnostics,
    ask_billing_resolution,
)


def main():
    print("=" * 60)
    print("TEST 1: NETWORK DIAGNOSTICS")
    print("=" * 60)

    network_result = ask_network_diagnostics(
        "What is the current status of tower TX-512? "
        "Include the latest performance metrics and "
        "any open incidents."
    )

    print(network_result)

    print()
    print("=" * 60)
    print("TEST 2: BILLING RESOLUTION")
    print("=" * 60)

    billing_result = ask_billing_resolution(
        "Customer CUST-10002 says they were charged "
        "twice for Unlimited Plus in September 2026. "
        "Investigate the duplicate and report the "
        "current credit status and balance. "
        "Do not create another credit if one is "
        "already pending."
    )

    print(billing_result)


if __name__ == "__main__":
    main()
