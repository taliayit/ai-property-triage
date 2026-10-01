# Guardrails Service

Two endpoints, same response shape as before — only the internal logic
changed. Semantic judgment (is this genuinely a property listing? is it
offensive? does this claim look fabricated?) now runs via
[NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails), backed by
Gemini, instead of fixed phrase lists.

| Endpoint              | Purpose                                                       |
|-----------------------|---------------------------------------------------------------|
| `POST /check/input`   | Validates a property submission before it enters the pipeline |
| `POST /check/output`  | Audits an AI-generated report before it's shown to a user     |

Both accept `{"text": "<text to check>"}` and return
`{"pass": bool, "reason": str | null}` (`/check/output` also echoes
`safe_text`). Either endpoint always returns `200 OK` — a failed
guardrail is a normal result, not a request error.

## Architecture: deterministic pre-filter, then two NeMo rails

`/check/input` runs a deterministic pre-filter first
(`not_empty_or_malformed`, `spam` — unchanged logic, thresholds in
[config.py](config.py)) and short-circuits before any LLM call if it
fails. `/check/output` has no pre-filter — there's no cheap syntactic
signal for "is this brief fabricated," so every output goes straight to
NeMo.

**Design choice, not a spec requirement:** keeping the deterministic
pre-filter in front of the input-side NeMo rails. Two reasons: (1) it
catches a genuinely different failure class than semantic
judgment — malformed/spam-formatted text doesn't need an LLM's opinion,
a regex already knows; (2) Gemini's free tier is rate-limited
(15 req/min observed), and every trivially-invalid submission
short-circuited here is one that doesn't burn quota that a real listing
might need. If this ever moves off the Gemini free tier, revisit whether
the pre-filter is still worth the added code path versus letting NeMo's
topic rail catch the same cases on its own.

If it passes (or for any `/check/output` call), two independent rails
run, both Gemini-backed:

| Rail   | Input side checks                                      | Output side checks                                                      |
|--------|--------------------------------------------------------|-------------------------------------------------------------------------|
| Topic  | Genuinely describes a real property, in English        | Coherent, on-topic for a listing brief                                  |
| Policy | No hateful/discriminatory/threatening/abusive language | No unverified legal claims, invented prices, or unhedged certifications |

`/check/input` runs topic then policy, fail-fast (skips policy if topic
already blocks, to save an LLM call). `/check/output` runs topic then
policy the same way. Both rails' prompts live in
[nemo_config/actions.py](nemo_config/actions.py); their iteration
history is in
[docs/prompt-engineering-log.md](../../docs/prompt-engineering-log.md).

## Why this bypasses NeMo's Colang/conversational flow

[nemo_rails.py](nemo_rails.py) builds an `LLMRails` instance from
[nemo_config/config.yml](nemo_config/config.yml) (just the model
declaration — there's no `rails.co` flow file) and calls each rail's
action directly via `rails.runtime.action_dispatcher.execute_action()`,
never NeMo's `generate_async()`/`check_async()` entrypoint.

`check_async()`/`generate_async()` are built to audit a live, multi-turn
conversation, not to validate a single arbitrary string — feeding them a
`[user, assistant]` pair to check an existing output routes through
NeMo's dialog-generation machinery instead of just evaluating the text.
Calling the registered action directly skips that entirely: it's the
same dispatcher Colang flows themselves use, just invoked without one.
The tradeoff is no Colang flow logic, which this service doesn't need —
each rail is a single independent LLM call, not a multi-turn policy.

Each action returns a plain string (`""` = pass, non-empty = block
reason) rather than a richer object — simple enough for this
direct-call pattern, with no Colang expression evaluation involved.

## Gemini integration notes

- `GEMINI_API_KEY` (this project's convention) is translated to
  `GOOGLE_API_KEY` at startup in `nemo_rails.py`, since
  `langchain-google-genai` reads the latter and NeMo's `config.yml`
  model construction has no field for passing it through directly.
- `NEMOGUARDRAILS_LLM_FRAMEWORK=langchain` is set in `nemo_rails.py` so
  NeMo wraps the custom-registered `gemini_shim` provider in its
  LangChain model adapter — NeMo's native framework doesn't support
  custom providers.
- `GeminiChatModel` in `nemo_rails.py` subclasses
  `ChatGoogleGenerativeAI` to translate NeMo's `max_tokens` kwarg (an
  OpenAI convention NeMo always passes) to `max_output_tokens`, which is
  what this Gemini integration actually accepts.

## Run without Docker

```bash
cd services/guardrails-service
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
set GEMINI_API_KEY=your-key-here
uvicorn main:app --reload --port 8002
```

## Run via Docker

Not yet updated for this NeMo-based version — the existing
[Dockerfile](Dockerfile) still reflects the old deterministic-only
implementation. `nemoguardrails` 0.24.1 (the version this pulls in) has
no `annoy` dependency, so the current single-stage pattern may work
as-is without needing rag-service's multi-stage/`build-essential`
approach — that should be verified, not assumed, when Docker
integration is picked back up.

## Examples

```bash
curl -X POST http://localhost:8002/check/input \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Bright 2-bedroom apartment with a renovated kitchen and hardwood floors.\"}"
```

```json
{"pass":true,"reason":null}
```

```bash
curl -X POST http://localhost:8002/check/input \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"BUY NOW!!! CLICK HERE!!! www.spam.com !!!\"}"
```

```json
{"pass":false,"reason":"contains a URL/link; excessive exclamation marks (9)"}
```

(Blocked by the deterministic pre-filter — no LLM call made.)

```bash
curl -X POST http://localhost:8002/check/input \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Great house but honestly avoid this area, it's full of disgusting people from certain ethnic backgrounds.\"}"
```

```json
{"pass":false,"reason":"The submission contains hate speech and discriminatory remarks regarding ethnic backgrounds."}
```

```bash
curl -X POST http://localhost:8002/check/output \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"This property is 100% code compliant with zero legal risk.\"}"
```

```json
{"pass":false,"reason":"The claim of 100% code compliance and zero legal risk is an absolute, unverifiable guarantee.","safe_text":"This property is 100% code compliant with zero legal risk."}
```
