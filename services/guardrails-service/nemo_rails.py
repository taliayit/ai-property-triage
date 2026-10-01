"""NeMo Guardrails integration: builds an LLMRails instance backed by
Gemini, and runs the topic/policy check actions (defined in
nemo_config/actions.py) directly via the action dispatcher.

Two real integration gotchas, if this breaks after a
nemoguardrails/langchain-google-genai upgrade:
- NEMOGUARDRAILS_LLM_FRAMEWORK=langchain must be set for a
  custom-registered LangChain provider to be wrapped in NeMo's LLMModel
  adapter; without it, NeMo's newer native framework doesn't recognize
  "google_genai"/custom providers and raises "No default base_url".
- nemoguardrails' internal llm_call() always passes max_tokens (an
  OpenAI-convention name) via .bind(); this langchain-google-genai
  version's GenerateContentConfig rejects it (wants max_output_tokens).
  GeminiChatModel below translates it.
"""

import os

os.environ.setdefault("NEMOGUARDRAILS_LLM_FRAMEWORK", "langchain")

from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402
from nemoguardrails import LLMRails, RailsConfig  # noqa: E402
from nemoguardrails.llm.providers import register_chat_provider  # noqa: E402

from config import GEMINI_API_KEY, NEMO_CONFIG_PATH  # noqa: E402

# google-genai SDK reads GOOGLE_API_KEY from the environment instead of GEMINI_API_KEY
if GEMINI_API_KEY:
    os.environ.setdefault("GOOGLE_API_KEY", GEMINI_API_KEY)


class GeminiChatModel(ChatGoogleGenerativeAI):
    """nemoguardrails always passes max_tokens (OpenAI convention) via
    .bind(); this Gemini integration only accepts max_output_tokens and
    rejects unrecognized fields. Translate on the way in."""

    def bind(self, **kwargs):
        if "max_tokens" in kwargs:
            kwargs["max_output_tokens"] = kwargs.pop("max_tokens")
        return super().bind(**kwargs)


register_chat_provider("gemini_shim", GeminiChatModel)


def build_rails() -> LLMRails:
    config = RailsConfig.from_path(NEMO_CONFIG_PATH)
    return LLMRails(config)


async def _run_action(rails: LLMRails, action_name: str, context: dict) -> str:
    result, status = await rails.runtime.action_dispatcher.execute_action(
        action_name, {"context": context, "llm": rails.llm}
    )
    if status != "success":
        # Fail closed: if the check itself errored (e.g. a transient API error), 
        # treat it as a block rather than silently letting unchecked content through.
        return "the safety check could not be completed"
    return result or ""


async def check_input_rails(rails: LLMRails, text: str) -> tuple[bool, str | None]:
    """Runs topic then policy (policy is checked only if topic is passed, to save an LLM call)."""
    reason = await _run_action(rails, "topic_check_input", {"user_message": text})
    if reason:
        return False, reason

    reason = await _run_action(rails, "policy_check_input", {"user_message": text})
    if reason:
        return False, reason

    return True, None


async def check_output_rails(rails: LLMRails, text: str) -> tuple[bool, str | None]:
    reason = await _run_action(rails, "topic_check_output", {"bot_message": text})
    if reason:
        return False, reason

    reason = await _run_action(rails, "policy_check_output", {"bot_message": text})
    if reason:
        return False, reason

    return True, None
