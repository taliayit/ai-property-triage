import os
from pathlib import Path

# /check/input: not_empty_or_malformed thresholds
MIN_TEXT_LENGTH = 10
MIN_ALPHA_RATIO = 0.5
MAX_REPEATED_CHAR_RUN = 6

# /check/input: spam thresholds
SPAM_MAX_EXCLAMATIONS = 3
SPAM_MAX_CAPS_RATIO = 0.6
SPAM_CAPS_MIN_LENGTH = 15

# NeMo Guardrails (topic + policy rails, both sides)
NEMO_CONFIG_PATH = str(Path(__file__).resolve().parent / "nemo_config")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
