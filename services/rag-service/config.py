from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent
PERSIST_DIRECTORY = str(SERVICE_DIR / "chroma_db")
COLLECTION_NAME = "listings"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
