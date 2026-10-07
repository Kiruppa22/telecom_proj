from pathlib import Path

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
INDEX_DIR = BASE_DIR / "data" / "vector_index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def configure_settings():
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL
    )

    Settings.text_splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50,
    )

    Settings.llm = None


def load_documents():
    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {DOCUMENTS_DIR}"
        )

    documents = SimpleDirectoryReader(
        input_dir=str(DOCUMENTS_DIR),
        required_exts=[".txt"],
    ).load_data()

    if not documents:
        raise ValueError("No policy documents were found.")

    return documents


def build_index():
    configure_settings()

    documents = load_documents()

    print(f"Loaded {len(documents)} documents.")

    for document in documents:
        file_name = document.metadata.get("file_name", "Unknown")
        print(f"  - {file_name}")

    print("\nBuilding vector index...")

    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True,
    )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    index.storage_context.persist(
        persist_dir=str(INDEX_DIR)
    )

    print("\nVector index created successfully.")
    print(f"Index location: {INDEX_DIR}")


if __name__ == "__main__":
    build_index()