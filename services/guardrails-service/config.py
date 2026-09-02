# --- /check/input: not_empty_or_malformed thresholds ---
MIN_TEXT_LENGTH = 10
MIN_ALPHA_RATIO = 0.5
MAX_REPEATED_CHAR_RUN = 6

# --- /check/input: spam thresholds ---
SPAM_MAX_EXCLAMATIONS = 3
SPAM_MAX_CAPS_RATIO = 0.6
SPAM_CAPS_MIN_LENGTH = 15

# --- /check/output: placeholder risk-language check ---
# Deterministic substring match standing in for a real NeMo output audit.
# Flags absolute claims a generated report shouldn't be making unverified.
OUTPUT_RISK_PHRASES = [
    "guaranteed",
    "100% legal",
    "no inspection needed",
    "risk-free",
    "zero risk",
    "no risk",
    "flawless condition",
    "guaranteed to sell",
    "guaranteed return",
    "guaranteed appreciation",
    "certain to increase in value",
]
