# WebUI

Gradio app with two tabs, run locally (not Dockerized - alongside a local
Ollama server per the spec).

```bash
cd webui
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
python app.py
```

Opens at `http://localhost:7860`. Requires:
- [Ollama](https://ollama.com) running locally with `llama3.1` pulled
  (`ollama pull llama3.1`) - for Tab 1.
- The n8n workflow published and reachable at
  `http://localhost:5678/webhook/submit-property` - for Tab 2.

Both URLs are configurable via `OLLAMA_URL` and `N8N_WEBHOOK_URL`
environment variables (`config.py`).

## Tab 1 - Conversational Assistant

A chat interface (`gr.ChatInterface`) calling Ollama's `POST /api/chat`
with `llama3.1`. The system prompt (`config.py`, `CHAT_SYSTEM_PROMPT`)
grounds it as a real estate assistant: redirects off-topic questions,
never states a price/valuation the user didn't provide, declines legal
advice in favor of recommending a licensed professional, and treats its
own rules as non-overridable.

This is a prompt-engineering surface with real, logged failures - see
`docs/prompt-engineering-log.md` ("WebUI Chat Assistant System Prompt")
for iteration history and test results.

Response time is CPU-bound and scales with reply length, not just prompt
length - a single substantive answer measured 394 output tokens in 84.5s
on this hardware (~209ms/token). `OLLAMA_TIMEOUT_SECONDS` (`config.py`) is
set to 180s, giving roughly 2x headroom over that measurement.

## Tab 2 - Listing Submission

A form (description, comma-separated image URLs, agent name) that POSTs
`{"description", "image_urls", "agent_name"}` to the n8n webhook and
renders the result. The flow runs several LLM calls in sequence, so a
submission can take up to a few minutes - the request timeout
(`N8N_TIMEOUT_SECONDS` in `config.py`) is set to 180s.

The response is one of three documented shapes, all handled:
- `{"brief_markdown": ..., "routed_to": ...}` - rendered as the report,
  with the routing shown above it.
- `{"status": "flagged_for_review", "reason": ...}` - shown as a
  flagged-for-review notice.
- `{"pass": false, "reason": ...}` - shown as a rejected-submission
  notice.

A fourth case is handled defensively: if the response is valid JSON but
matches none of the three shapes above, the raw JSON is shown rather than
failing silently. All three documented shapes are correctly produced by
the workflow today (verified live, including the input-rejection path),
so this isn't currently covering a known gap - it's kept as a general
safety net against a future workflow change introducing an
unrecognized shape, rather than showing a blank result.
