import os
from pathlib import Path

from dotenv import load_dotenv

from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic


BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = BASE_DIR / "data" / "vector_index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def configure_models():
    load_dotenv(BASE_DIR / ".env")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    model_name = os.getenv("ANTHROPIC_MODEL")

    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY was not found in .env")

    if not model_name:
        raise ValueError("ANTHROPIC_MODEL was not found in .env")

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL
    )

    Settings.llm = Anthropic(
        model=model_name,
        api_key=api_key,
        max_tokens=1000,
    )

def load_policy_index():
    if not INDEX_DIR.exists():
        raise FileNotFoundError(
            f"Vector index not found: {INDEX_DIR}"
        )

    storage_context = StorageContext.from_defaults(
        persist_dir=str(INDEX_DIR)
    )

    return load_index_from_storage(
        storage_context
    )


def create_policy_query_engine():
    configure_models()

    index = load_policy_index()

    query_engine = index.as_query_engine(
        similarity_top_k=3,
    )

    return query_engine


def ask_policy_question(query):
    query_engine = create_policy_query_engine()

    response = query_engine.query(query)

    sources = []

    for source_node in response.source_nodes:
        file_name = source_node.metadata.get(
            "file_name",
            "Unknown"
        )

        sources.append(
            {
                "file_name": file_name,
                "score": source_node.score,
            }
        )

    return {
        "answer": str(response),
        "sources": sources,
    }


def main():
    print("POLICY RAG WORKER")
    print("=" * 80)

    query = input("\nEnter policy question: ")

    result = ask_policy_question(query)

    print("\nANSWER")
    print("-" * 80)
    print(result["answer"])

    print("\nSOURCES")
    print("-" * 80)

    for source in result["sources"]:
        score = source["score"]

        if score is None:
            score_text = "N/A"
        else:
            score_text = f"{score:.4f}"

        print(
            f"{source['file_name']} "
            f"(similarity: {score_text})"
        )


if __name__ == "__main__":
    main()