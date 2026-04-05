import os  # <--- MAKE SURE THIS LINE IS HERE
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

# 1. API key: optional .env next to this script; always respect existing os.environ first.
#    If there is no .env file, use OPENAI_API_KEY exported in the shell or set by the OS.
env_path = Path(__file__).resolve().parent / ".env"
if env_path.is_file():
    load_dotenv(dotenv_path=env_path, override=False)

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("WISDOM_PDF_DIR", ROOT / "Data"))
PERSIST_DIR = Path(os.environ.get("WISDOM_STORAGE_DIR", ROOT / "storage"))


def _storage_looks_valid(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def run_engine(force_reindex: bool = False) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "Set OPENAI_API_KEY in your environment or in a .env file next to app.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not DATA_DIR.is_dir():
        print(f"Error: Put your PDFs in '{DATA_DIR}' first.", file=sys.stderr)
        sys.exit(1)

    if force_reindex and PERSIST_DIR.exists():
        print("Cleaning out old index to re-scan Data folder...")
        shutil.rmtree(PERSIST_DIR)

    persist_path = str(PERSIST_DIR)
    data_path = str(DATA_DIR)

    if not _storage_looks_valid(PERSIST_DIR):
        print("Indexing PDFs (first run or after reindex; this can take a while)...")
        documents = SimpleDirectoryReader(data_path).load_data()
        if not documents:
            print(
                f"Error: No documents loaded from '{DATA_DIR}'. Add PDFs and try again.",
                file=sys.stderr,
            )
            sys.exit(1)

        index = VectorStoreIndex.from_documents(documents)
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=persist_path)
        print("Index created and saved to disk.")
    else:
        print("Loading existing index from storage...")
        try:
            storage_context = StorageContext.from_defaults(persist_dir=persist_path)
            index = load_index_from_storage(storage_context)
        except Exception as exc:
            print(
                f"Saved index could not be loaded ({exc!s}). Rebuilding from PDFs...",
                file=sys.stderr,
            )
            documents = SimpleDirectoryReader(data_path).load_data()
            if not documents:
                print(f"No documents loaded from {DATA_DIR}.", file=sys.stderr)
                sys.exit(1)
            index = VectorStoreIndex.from_documents(documents)
            index.storage_context.persist(persist_dir=persist_path)
            print(f"Rebuilt and saved to {PERSIST_DIR}")

    query_engine = index.as_query_engine(similarity_top_k=3)

    print("\n--- WISDOM ENGINE READY ---")
    while True:
        query = input("\nQuery: ").strip()
        if query.lower() in ("exit", "quit", "q"):
            break
        if not query:
            continue
        response = query_engine.query(query)
        print(f"\nRESULT:\n{response}\n")


if __name__ == "__main__":
    reindex_flag = "--reindex" in sys.argv
    run_engine(force_reindex=reindex_flag)
