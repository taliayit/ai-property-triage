# Guardrails Service

Deterministic, structural validation — checks the *shape* of text, not
its meaning. Two endpoints:

| Endpoint         | Purpose                                                        |
|-------------------|------------------------------------------------------------------|
| `POST /check/input`  | Validates a property submission before it enters the pipeline |
| `POST /check/output` | Audits an AI-generated report before it's shown to a user      |

Both accept `{"text": "<text to check>"}`.

## `/check/input`

Runs two checks against the submission text:

| Check                     | Validates                                                          |
|----------------------------|---------------------------------------------------------------------|
| `not_empty_or_malformed`  | Non-empty, minimum length, mostly alphabetic, no long repeated-character runs |
| `spam`                     | No links, not excessive capitalization, not excessive `!` usage     |

Both checks always run, so a rejection can combine reasons from more than
one. Thresholds live in [config.py](config.py).

```json
{"pass": true, "reason": null}
```

```json
{"pass": false, "reason": "too many non-alphabetic characters to be readable text; excessive exclamation marks (3)"}
```

## `/check/output`

**Placeholder.** There's no LLM Chain generating reports yet, so this
can't do a real fabrication/risk audit — it flags a fixed list of
absolute-claim phrases (`"guaranteed"`, `"100% legal"`,
`"no inspection needed"`, etc., see `OUTPUT_RISK_PHRASES` in
[config.py](config.py)) that a generated listing report shouldn't state
unverified. It also echoes the text back as `safe_text`, unmodified —
that field exists now so callers don't need to change once this actually
sanitizes content.

```json
{"pass": false, "reason": "contains unverifiable absolute claim(s): guaranteed, 100% legal", "safe_text": "This property is guaranteed and 100% legal."}
```

## Deferred to NeMo Guardrails

Both endpoints only check text *shape* — length, formatting, fixed
phrase lists. Actual semantic judgment (is this really about real estate,
is it offensive, is this claim actually fabricated vs. just confident)
needs language understanding a rule-based check can't provide. That's
planned for a future NeMo Guardrails integration, once there's a real
Information Extractor / report-generation chain to audit. The response
shape (`{"pass": bool, "reason": str | null}`, plus `safe_text` for
output) is meant to stay stable across that swap.

Either endpoint always returns `200 OK` — a failed guardrail is a normal
result, not a request error.

## Run via docker compose (recommended)

From the project root:

```bash
docker compose up -d --build
```

See the [root README](../../README.md) for the full stack (n8n +
rag-service too). This service alone is reachable at
`http://localhost:8002`, and from other compose services at
`http://guardrails-service:8002`. It's stateless, so the
[Dockerfile](Dockerfile) is a single stage - no compiled dependencies
here, unlike rag-service's `llama-cpp-python`.

## Run without Docker

```bash
cd services/guardrails-service
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
uvicorn main:app --reload --port 8002
```

## Examples

```bash
curl -X POST http://localhost:8002/check/input \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"!!!###@@@$$$%%%^^^&&&***(((\"}"
```

```json
{"pass":false,"reason":"too many non-alphabetic characters to be readable text; excessive exclamation marks (3)"}
```

```bash
curl -X POST http://localhost:8002/check/output \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"This property is guaranteed to sell within a week and is 100% legal, no inspection needed.\"}"
```

```json
{"pass":false,"reason":"contains unverifiable absolute claim(s): guaranteed, 100% legal, no inspection needed, guaranteed to sell","safe_text":"This property is guaranteed to sell within a week and is 100% legal, no inspection needed."}
```
