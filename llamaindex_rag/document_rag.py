from pathlib import Path

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


BASE_DIR = Path(__file__).resolve().parent.parent

# Where the policy documents are read from and the index is saved
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
INDEX_DIR = BASE_DIR / "data" / "vector_index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Documents are split into chunks of this size before indexing
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def configure_settings():
    """Sets the embedding model and chunking rules for LlamaIndex."""
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)

    Settings.text_splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    # No LLM is needed here: this step only builds the index
    Settings.llm = None


def load_documents():
    """Reads every .txt policy document from the documents folder."""
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
    """Builds the vector index from the documents and saves it to disk."""
    configure_settings()

    documents = load_documents()

    print(f"Loaded {len(documents)} documents:")

    for document in documents:
        file_name = document.metadata.get("file_name", "Unknown")
        print(f"  - {file_name}")

    print("\nBuilding vector index...")

    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True,
    )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    index.storage_context.persist(persist_dir=str(INDEX_DIR))

    print("\nVector index created successfully.")
    print(f"Index location: {INDEX_DIR}")


if __name__ == "__main__":
    build_index()