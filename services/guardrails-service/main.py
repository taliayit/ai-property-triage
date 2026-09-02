"""Guardrails Service - deterministic structural checks.

POST /check/input validates submission text (length, format, spam
patterns). POST /check/output is a placeholder that stands in for a real
NeMo output-audit until an LLM Chain is generating reports - it only
flags a fixed list of absolute-claim phrases. Semantic judgment (language,
offensive content, on-topic relevance, real fabrication detection) is out
of scope for rule-based logic and is deferred to a future NeMo Guardrails
integration - see README.
"""

import re

from fastapi import FastAPI
from pydantic import BaseModel

from config import (
    MAX_REPEATED_CHAR_RUN,
    MIN_ALPHA_RATIO,
    MIN_TEXT_LENGTH,
    OUTPUT_RISK_PHRASES,
    SPAM_CAPS_MIN_LENGTH,
    SPAM_MAX_CAPS_RATIO,
    SPAM_MAX_EXCLAMATIONS,
)

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{" + str(MAX_REPEATED_CHAR_RUN - 1) + ",}")

app = FastAPI(title="Guardrails Service")


class CheckRequest(BaseModel):
    text: str


def make_check(passed: bool, reason: str) -> dict:
    return {"passed": passed, "reason": reason}


def aggregate(checks: list[dict]) -> tuple[bool, str | None]:
    passed = all(c["passed"] for c in checks)
    if passed:
        return True, None
    reason = "; ".join(c["reason"] for c in checks if not c["passed"])
    return False, reason


def check_not_empty_or_malformed(text: str) -> dict:
    stripped = text.strip()
    if not stripped:
        return make_check(False, "submission is empty")

    if len(stripped) < MIN_TEXT_LENGTH:
        return make_check(
            False, f"too short to be genuine content (minimum {MIN_TEXT_LENGTH} characters)"
        )

    non_space = [c for c in stripped if not c.isspace()]
    alpha = [c for c in non_space if c.isalpha()]
    alpha_ratio = len(alpha) / len(non_space)
    if alpha_ratio < MIN_ALPHA_RATIO:
        return make_check(False, "too many non-alphabetic characters to be readable text")

    if REPEATED_CHAR_PATTERN.search(stripped):
        return make_check(False, "contains long runs of repeated characters")

    return make_check(True, "text is well-formed, non-empty content")


def check_spam(text: str) -> dict:
    reasons = []

    if URL_PATTERN.search(text):
        reasons.append("contains a URL/link")

    exclamation_count = text.count("!")
    if exclamation_count >= SPAM_MAX_EXCLAMATIONS:
        reasons.append(f"excessive exclamation marks ({exclamation_count})")

    if len(text) > SPAM_CAPS_MIN_LENGTH:
        letters = [c for c in text if c.isalpha()]
        if letters:
            caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            if caps_ratio > SPAM_MAX_CAPS_RATIO:
                reasons.append(f"excessive use of capital letters ({caps_ratio:.0%})")

    if reasons:
        return make_check(False, "; ".join(reasons))
    return make_check(True, "no spam signals detected")


def check_output_risk_language(text: str) -> dict:
    lowered = text.lower()
    matched = [phrase for phrase in OUTPUT_RISK_PHRASES if phrase in lowered]
    if matched:
        return make_check(False, f"contains unverifiable absolute claim(s): {', '.join(matched)}")
    return make_check(True, "no risk language detected")


@app.post("/check/input")
def check_input(request: CheckRequest):
    passed, reason = aggregate(
        [check_not_empty_or_malformed(request.text), check_spam(request.text)]
    )
    return {"pass": passed, "reason": reason}


@app.post("/check/output")
def check_output(request: CheckRequest):
    passed, reason = aggregate([check_output_risk_language(request.text)])
    return {"pass": passed, "reason": reason, "safe_text": request.text}
