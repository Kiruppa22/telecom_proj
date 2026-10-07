from pathlib import Path

from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = BASE_DIR / "data" / "vector_index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_policy_index():
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL
    )

    Settings.llm = None

    storage_context = StorageContext.from_defaults(
        persist_dir=str(INDEX_DIR)
    )

    return load_index_from_storage(
        storage_context
    )


def test_query(index, query):
    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    retriever = index.as_retriever(
        similarity_top_k=3
    )

    nodes = retriever.retrieve(query)

    for position, node in enumerate(nodes, start=1):
        file_name = node.metadata.get(
            "file_name",
            "Unknown"
        )

        print(f"\nRESULT {position}")
        print(f"Source: {file_name}")
        print(f"Similarity Score: {node.score:.4f}")
        print("-" * 80)
        print(node.text[:1200])


def main():
    print("Loading persisted policy index...")

    index = load_policy_index()

    print("Policy index loaded successfully.")

    test_queries = [
        "What are the roaming charges in Japan?",
        "What happens when a customer disputes a billing charge?",
        "What should be done during a network outage?",
        "What is the SLA for service restoration?",
        "What should I know about upgrading my device?",
        "What information is available about 5G service?",
    ]

    for query in test_queries:
        test_query(index, query)


if __name__ == "__main__":
    main()