from orchestration.graph import (
    run_ai_operations_center,
)


def main():
    query = (
        "What are the roaming charges in Japan?"
    )

    print("=" * 70)
    print("LANGGRAPH TEST")
    print("=" * 70)

    print()
    print("QUERY:")
    print(query)

    result = run_ai_operations_center(
        query
    )

    print()
    print("=" * 70)
    print("COMPLETED WORKERS")
    print("=" * 70)

    for worker in result[
        "completed_workers"
    ]:
        print(worker)

    print()
    print("=" * 70)
    print("EXECUTION TRACE")
    print("=" * 70)

    for index, item in enumerate(
        result["execution_trace"],
        start=1,
    ):
        print(
            f"{index}. {item['worker']}"
        )
        print(item["output"])
        print()

    print()
    print("=" * 70)
    print("FINAL RESPONSE")
    print("=" * 70)
    print()
    print(
        result["final_response"]
    )

    if result["error"]:
        print()
        print("=" * 70)
        print("ERROR")
        print("=" * 70)
        print(result["error"])


if __name__ == "__main__":
    main()