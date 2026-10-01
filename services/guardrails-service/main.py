"""Guardrails Service - deterministic pre-filter, then NeMo Guardrails.

Deterministic checks (empty/malformed, spam formatting) run first and
short-circuit before any LLM call - see README for why this pre-filter
(not itself a spec requirement) is kept as a deliberate design choice.

Semantic judgment runs via two NeMo Guardrails rails, both backed by
Gemini: topic detection (genuine, on-topic, English-language input /
coherent, on-topic output) and policy enforcement (offensive content on
input; false legal claims, invented prices, fabricated certifications on
output). See nemo_rails.py and docs/prompt-engineering-log.md.
"""

import logging
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from config import (
    MAX_REPEATED_CHAR_RUN,
    MIN_ALPHA_RATIO,
    MIN_TEXT_LENGTH,
    SPAM_CAPS_MIN_LENGTH,
    SPAM_MAX_CAPS_RATIO,
    SPAM_MAX_EXCLAMATIONS,
)
from nemo_rails import build_rails, check_input_rails, check_output_rails

logger = logging.getLogger(__name__)

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{" + str(MAX_REPEATED_CHAR_RUN - 1) + ",}")

rails = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rails
    rails = build_rails()
    yield


app = FastAPI(title="Guardrails Service", lifespan=lifespan)


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


@app.post("/check/input")
async def check_input(request: CheckRequest):
    passed, reason = aggregate(
        [check_not_empty_or_malformed(request.text), check_spam(request.text)]
    )
    if not passed:
        return {"pass": False, "reason": reason}

    nemo_passed, nemo_reason = await check_input_rails(rails, request.text)
    return {"pass": nemo_passed, "reason": nemo_reason}


@app.post("/check/output")
async def check_output(request: CheckRequest):
    nemo_passed, nemo_reason = await check_output_rails(rails, request.text)
    return {"pass": nemo_passed, "reason": nemo_reason, "safe_text": request.text}
