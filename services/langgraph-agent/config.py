import os

# Defaults to compose network hostnames; override for standalone/local use.
RAG_SERVICE_URL = os.environ.get("RAG_SERVICE_URL", "http://rag-service:8001")
IMAGE_ANALYSER_URL = os.environ.get("IMAGE_ANALYSER_URL", "http://image-analyser:8003")
TOOL_HTTP_TIMEOUT_SECONDS = 30  # rag-service's LLM-generated insight can take ~10-15s

GEMINI_MODEL_NAME = "gemini-3.5-flash-lite"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_TEMPERATURE = 0.1  # planning/synthesis, not creative writing
