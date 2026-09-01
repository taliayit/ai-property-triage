"""Guardrails Service - deterministic structural checks (length, format,
spam patterns). Semantic checks (language, offensive content, relevance)
are deferred to a future NeMo Guardrails integration - see README.
"""

import re

from fastapi import FastAPI
from pydantic import BaseModel

from config import (
    MAX_REPEATED_CHAR_RUN,
    MIN_ALPHA_RATIO,
    MIN_DESCRIPTION_LENGTH,
    SPAM_CAPS_MIN_LENGTH,
    SPAM_MAX_CAPS_RATIO,
    SPAM_MAX_EXCLAMATIONS,
)

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{" + str(MAX_REPEATED_CHAR_RUN - 1) + ",}")

app = FastAPI(title="Guardrails Service")


class CheckRequest(BaseModel):
    description: str


def make_check(name: str, passed: bool, reason: str) -> dict:
    return {"name": name, "passed": passed, "reason": reason}


def check_not_empty_or_malformed(description: str) -> dict:
    stripped = description.strip()
    if not stripped:
        return make_check("not_empty_or_malformed", False, "submission is empty")

    if len(stripped) < MIN_DESCRIPTION_LENGTH:
        return make_check(
            "not_empty_or_malformed",
            False,
            f"too short to be a genuine listing (minimum {MIN_DESCRIPTION_LENGTH} characters)",
        )

    non_space = [c for c in stripped if not c.isspace()]
    alpha = [c for c in non_space if c.isalpha()]
    alpha_ratio = len(alpha) / len(non_space)
    if alpha_ratio < MIN_ALPHA_RATIO:
        return make_check(
            "not_empty_or_malformed",
            False,
            "too many non-alphabetic characters to be readable text",
        )

    if REPEATED_CHAR_PATTERN.search(stripped):
        return make_check(
            "not_empty_or_malformed",
            False,
            "contains long runs of repeated characters",
        )

    return make_check("not_empty_or_malformed", True, "text is well-formed, non-empty content")


def check_spam(description: str) -> dict:
    reasons = []

    if URL_PATTERN.search(description):
        reasons.append("contains a URL/link")

    exclamation_count = description.count("!")
    if exclamation_count >= SPAM_MAX_EXCLAMATIONS:
        reasons.append(f"excessive exclamation marks ({exclamation_count})")

    if len(description) > SPAM_CAPS_MIN_LENGTH:
        letters = [c for c in description if c.isalpha()]
        if letters:
            caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            if caps_ratio > SPAM_MAX_CAPS_RATIO:
                reasons.append(f"excessive use of capital letters ({caps_ratio:.0%})")

    if reasons:
        return make_check("spam", False, "; ".join(reasons))
    return make_check("spam", True, "no spam signals detected")


@app.post("/check")
def check(request: CheckRequest):
    checks = [
        check_not_empty_or_malformed(request.description),
        check_spam(request.description),
    ]
    return {
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
    }
