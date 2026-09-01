# Guardrails Service

Deterministic, structural validation for property submissions — checks
the *shape* of the text, not its meaning. `POST /check` runs two checks:

| Check name               | Validates                                                          |
|---------------------------|---------------------------------------------------------------------|
| `not_empty_or_malformed`  | Non-empty, minimum length, mostly alphabetic, no long repeated-character runs |
| `spam`                    | No links, not excessive capitalization, not excessive `!` usage     |

Both checks always run, so a single request can surface multiple problems
at once. Thresholds live in [config.py](config.py).

Semantic judgment — language, offensive content, on-topic relevance — is
intentionally out of scope here; those require actual language
understanding and are planned for a future NeMo Guardrails integration.

### Response shape

```json
{
  "passed": true,
  "checks": [
    {"name": "not_empty_or_malformed", "passed": true, "reason": "..."},
    {"name": "spam", "passed": true, "reason": "..."}
  ]
}
```

`passed` is `true` only if every check passed. Each check reports its own
`reason`, so a rejection is diagnosable from the response alone. The
service always returns `200 OK` — a failed guardrail is a normal result,
not an error.

## Setup

```bash
cd services/guardrails-service
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Run the service

```bash
uvicorn main:app --reload --port 8002
```

## Example

```bash
curl -X POST http://localhost:8002/check \
  -H "Content-Type: application/json" \
  -d "{\"description\": \"!!!###@@@$$$%%%^^^&&&***(((\"}"
```

```json
{"passed":false,"checks":[{"name":"not_empty_or_malformed","passed":false,"reason":"too many non-alphabetic characters to be readable text"},{"name":"spam","passed":false,"reason":"excessive exclamation marks (3)"}]}
```
