from orchestration.graph import (
    run_ai_operations_center,
)


SCENARIOS = [
    {
        "name": "Policy RAG - Japan Roaming",
        "query": (
            "What are the roaming charges in Japan?"
        ),
        "expected": [
            "policy_rag",
            "customer_comms_crew",
        ],
    },
    {
        "name": "Network Analytics - Critical Outages",
        "query": (
            "Which region had the most CRITICAL "
            "network outages?"
        ),
        "expected": [
            "network_analytics",
            "customer_comms_crew",
        ],
    },
    {
        "name": "Network Diagnostics - TX-512",
        "query": (
            "What is the current status of tower "
            "TX-512? Include its latest performance "
            "metrics and any open incidents."
        ),
        "expected": [
            "network_diagnostics_adk",
            "customer_comms_crew",
        ],
    },
    {
        "name": "Billing Resolution - CUST-10002",
        "query": (
            "Customer CUST-10002 says they were "
            "charged twice for Unlimited Plus in "
            "September 2026. Investigate the duplicate "
            "charge and resolve it according to the "
            "billing rules."
        ),
        "expected": [
            "billing_resolution_adk",
            "customer_comms_crew",
        ],
    },
    {
        "name": "Multi-Agent - Midwest SLA",
        "query": (
            "A CRITICAL network outage in the Midwest "
            "on September 12, 2026 lasted 6 hours and "
            "affected 8,500 customers. Analyze the "
            "outage and determine whether the duration "
            "met the applicable SLA requirements."
        ),
        "expected": [
            "network_analytics",
            "policy_rag",
            "customer_comms_crew",
        ],
    },
]


def print_result(
    scenario,
    result,
):
    print()
    print("=" * 80)
    print(scenario["name"])
    print("=" * 80)

    print()
    print("QUERY:")
    print(scenario["query"])

    print()
    print("EXPECTED ROUTE:")
    print(
        " -> ".join(
            scenario["expected"]
        )
    )

    actual = result.get(
        "completed_workers",
        [],
    )

    print()
    print("ACTUAL ROUTE:")
    print(
        " -> ".join(actual)
    )

    print()
    print(
        "ROUTING RESULT:",
        "PASS"
        if actual == scenario["expected"]
        else "FAIL",
    )

    print()
    print("EXECUTION TRACE:")

    for index, item in enumerate(
        result.get(
            "execution_trace",
            [],
        ),
        start=1,
    ):
        print(
            f"{index}. {item['worker']}"
        )

    print()
    print("FINAL RESPONSE:")
    print(
        result.get(
            "final_response",
            "",
        )
    )

    if result.get("error"):
        print()
        print("ERROR:")
        print(result["error"])


def main():
    passed = 0

    for scenario in SCENARIOS:
        result = run_ai_operations_center(
            scenario["query"]
        )

        print_result(
            scenario,
            result,
        )

        if (
            result.get(
                "completed_workers",
                [],
            )
            == scenario["expected"]
        ):
            passed += 1

    print()
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(
        f"Passed: {passed}/{len(SCENARIOS)}"
    )


if __name__ == "__main__":
    main()