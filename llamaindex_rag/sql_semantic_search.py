import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from llama_index.core import (
    SQLDatabase,
    Settings,
    VectorStoreIndex,
)
from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)
from llama_index.core.indices.struct_store import (
    SQLTableRetrieverQueryEngine,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "telecom_ops.db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# How many tables the retriever picks for each question
TOP_K_TABLES = 2

# Maximum length of the LLM's answer
LLM_MAX_TOKENS = 1000


# Plain-language description of each table. The retriever compares the
# user's question to these descriptions to decide which tables to query.
TABLE_DESCRIPTIONS = {
    "network_towers": (
        "Current tower inventory. Contains tower_id, tower_name, geographic "
        "region, city, state, network technology, and current operational "
        "status. Use with tower_performance when a question requires both "
        "tower identity/location and performance metrics."
    ),

    "tower_performance": (
        "Time-series network performance measurements for towers. Contains "
        "tower_id, recorded_at timestamps, latency, packet loss percentage, "
        "download and upload throughput, signal strength, and active "
        "connections. For questions asking about current, latest, or "
        "right-now performance, use only the row with the maximum recorded_at "
        "for each tower. Do not average historical samples for current "
        "performance questions."
    ),

    "network_outages": (
        "Historical and ongoing network outages. Contains region, severity, "
        "start and end times, duration, affected customer count, root cause, "
        "status, and outage description. Use for outage counts, outage "
        "severity, outage duration, affected customers, and regional outage "
        "analytics."
    ),

    "open_incidents": (
        "Currently active network incidents associated with towers. Contains "
        "tower_id, severity, incident status, classification, description, "
        "assigned team, and opened timestamp."
    ),

    "customer_subscriptions": (
        "Customer subscription and service-plan information. Contains "
        "customer_id, account type, plan name, monthly fee, service region, "
        "location, subscription status, line count, and start date."
    ),

    "billing_accounts": (
        "Current customer billing account information. Contains customer_id, "
        "customer name, account type, current balance, service location, "
        "billing cycle, auto-pay status, and account status."
    ),

    "billing_charges": (
        "Customer billing charge lines. Contains customer_id, charge "
        "description, amount, billing period, charge date, charge type, "
        "duplicate indicator, and invoice status."
    ),

    "billing_credits": (
        "Billing credits issued or requested for customers. Contains "
        "customer_id, credit amount, reason, status, creation timestamp, "
        "and related charge."
    ),

    "billing_disputes": (
        "Billing disputes opened against customer charges. Contains "
        "customer_id, charge_id, dispute reason, status, opening and "
        "resolution timestamps, and resolution notes."
    ),
}


# Holds the query engine after the first request so the embedding model
# and table index are not rebuilt for every question.
_query_engine = None


def configure_models():
    """Loads the API key and sets the embedding model and LLM."""
    load_dotenv(BASE_DIR / ".env")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    model_name = os.getenv("ANTHROPIC_MODEL")

    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY was not found in .env")

    if not model_name:
        raise ValueError("ANTHROPIC_MODEL was not found in .env")

    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)

    Settings.llm = Anthropic(
        model=model_name,
        api_key=api_key,
        max_tokens=LLM_MAX_TOKENS,
    )


def create_sql_database():
    """Connects to the SQLite database and wraps the allowed tables."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"SQLite database not found: {DB_PATH}")

    database_url = f"sqlite:///{DB_PATH.as_posix()}"

    engine = create_engine(database_url)

    sql_database = SQLDatabase(
        engine,
        include_tables=list(TABLE_DESCRIPTIONS.keys()),
    )

    return engine, sql_database


def create_table_retriever(sql_database):
    """Builds a retriever that picks the most relevant tables for a question."""
    table_node_mapping = SQLTableNodeMapping(sql_database)

    table_schema_objects = []

    for table_name, description in TABLE_DESCRIPTIONS.items():
        table_schema_objects.append(
            SQLTableSchema(
                table_name=table_name,
                context_str=description,
            )
        )

    object_index = ObjectIndex.from_objects(
        table_schema_objects,
        table_node_mapping,
        VectorStoreIndex,
    )

    return object_index.as_retriever(similarity_top_k=TOP_K_TABLES)


def create_sql_query_engine():
    """Creates the text-to-SQL query engine."""
    configure_models()

    _, sql_database = create_sql_database()

    table_retriever = create_table_retriever(sql_database)

    return SQLTableRetrieverQueryEngine(
        sql_database=sql_database,
        table_retriever=table_retriever,
    )


def get_query_engine():
    """Returns the shared query engine, creating it on first use."""
    global _query_engine

    if _query_engine is None:
        _query_engine = create_sql_query_engine()

    return _query_engine


def ask_network_analytics(question):
    """Answers a network analytics question using the database."""
    response = get_query_engine().query(question)
    return str(response)


# ---------------------------------------------------------------
# Validation helpers (run directly against the database)
# ---------------------------------------------------------------

def test_latest_packet_loss(engine):
    """Lists the five towers with the highest latest packet loss."""
    sql = """
    WITH latest_samples AS (
        SELECT
            tower_id,
            MAX(recorded_at) AS latest_recorded_at
        FROM tower_performance
        GROUP BY tower_id
    )
    SELECT
        t.tower_id,
        t.tower_name,
        t.region,
        t.status,
        p.recorded_at,
        p.packet_loss_pct
    FROM tower_performance AS p
    JOIN latest_samples AS latest
        ON p.tower_id = latest.tower_id
        AND p.recorded_at = latest.latest_recorded_at
    JOIN network_towers AS t
        ON p.tower_id = t.tower_id
    ORDER BY p.packet_loss_pct DESC
    LIMIT 5
    """

    print("\n" + "=" * 80)
    print("VALIDATION QUERY")
    print("Which towers have the highest packet loss right now?")
    print("=" * 80)

    with engine.connect() as connection:
        result = connection.execute(text(sql))

        for row in result:
            print(
                f"{row.tower_id:8} | "
                f"{row.tower_name:25} | "
                f"{row.region:10} | "
                f"{row.status:12} | "
                f"{row.recorded_at} | "
                f"{row.packet_loss_pct:.1f}%"
            )


def test_critical_outages_by_region(engine):
    """Counts CRITICAL outages per region."""
    sql = """
    SELECT
        region,
        COUNT(*) AS critical_outage_count,
        SUM(affected_customers) AS total_affected_customers,
        ROUND(SUM(duration_hours), 2) AS total_duration_hours
    FROM network_outages
    WHERE severity = 'CRITICAL'
    GROUP BY region
    ORDER BY critical_outage_count DESC, region
    """

    print("\n" + "=" * 80)
    print("VALIDATION QUERY")
    print("Which region had the most CRITICAL outages?")
    print("=" * 80)

    with engine.connect() as connection:
        result = connection.execute(text(sql))

        for row in result:
            print(
                f"{row.region:10} | "
                f"Critical outages: {row.critical_outage_count:2} | "
                f"Affected customers: {row.total_affected_customers:6} | "
                f"Duration: {row.total_duration_hours:.2f} hours"
            )


def test_semantic_table_selection(table_retriever):
    """Prints which tables the retriever picks for sample questions."""
    test_queries = [
        "Which towers have the highest packet loss right now?",
        "Which region had the most CRITICAL outages?",
        "Show me the current status of tower TX-512.",
        "Does customer CUST-10002 have duplicate charges?",
        "What credits have been issued to customers?",
        "Which customers have open billing disputes?",
    ]

    print("\n" + "=" * 80)
    print("SEMANTIC TABLE SELECTION")
    print("=" * 80)

    for query in test_queries:
        print(f"\nQuestion: {query}")

        retrieved_tables = table_retriever.retrieve(query)

        print("Selected tables:")

        for table in retrieved_tables:
            print(f"  - {table.table_name}")


def main():
    """Command-line mode: ask questions until the user types 'exit'."""
    print("NETWORK ANALYTICS - LLAMAINDEX SEMANTIC SQL")
    print("=" * 80)
    print("Type 'exit' to quit.")

    while True:
        question = input("\nEnter network analytics question: ").strip()

        if question.lower() in {"exit", "quit", ""}:
            break

        answer = ask_network_analytics(question)

        print("\nANSWER")
        print("-" * 80)
        print(answer)


if __name__ == "__main__":
    main()