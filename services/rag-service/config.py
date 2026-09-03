import os
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent
PERSIST_DIRECTORY = str(SERVICE_DIR / "chroma_db")
COLLECTION_NAME = "listings"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# GGUF model for insight generation. Override with RAG_LLM_MODEL_PATH,
# or place the file at this default path (see README).
LLM_MODEL_PATH = os.environ.get(
    "RAG_LLM_MODEL_PATH",
    str(SERVICE_DIR / "models" / "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
)
LLM_N_CTX = 4096
LLM_N_THREADS = os.cpu_count() or 4
LLM_MAX_TOKENS = 300
LLM_TEMPERATURE = 0.1  # low temperature: this is a grounded-summary task, not creative writing
