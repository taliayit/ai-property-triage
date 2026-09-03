## RAG Insight Generation

### v1 - LangChain + Llama.cpp baseline
Model: Llama-3.2-1B-Instruct (Q4_K_M GGUF), served locally via
llama-cpp-python + LangChain's ChatLlamaCpp (chat_format="llama-3"),
temperature 0.1, max_tokens 300.

Prompt design:
- System message states 5 rules: use only retrieved facts, cite every
  claim with [Listing N], only cite genuinely relevant listings, say so
  plainly when nothing matches well, keep to 2-4 sentences.
- One fixed few-shot example (a human/ai turn pair) anchors the citation
  format - it's always a "good match" example; v1 has no negative
  ("no good match") example, which turned out to matter (see failures
  below).
- Context injection: each retrieved listing is formatted as
  `[Listing N] (property_type, $price, rooms rooms): "description"` and
  the top-3 ChromaDB results are inserted verbatim into the final human
  turn alongside the new submission text.

Mechanical guard: after generation, regex-extract every `[Listing N]` tag
and verify N is within the retrieved set's range. This only catches
out-of-range citations - it turned out to miss every real failure found
below.

Design principle: retrieval (embeddings/ChromaDB) stays deterministic and
untouched; only the natural-language insight is LLM-generated, and only
when there's at least one retrieved listing to ground it in (zero results
short-circuits to a canned message without calling the LLM at all).

Test input: "Bright 2-bedroom apartment on the 4th floor with a renovated
kitchen, hardwood floors, and a small balcony overlooking the courtyard.
Walking distance to the metro station." (near-exact match to a seed
listing)
Result: Correctly identified [Listing 1] as a strong match and cited real
shared features. citation_check: valid. BUT: ignored the "2-4 sentences"
rule (3 paragraphs), and the model drifted off-task midway - instead of
comparing the submission to Listing 1, it started comparing Listing 1 to
Listing 2 ("there are some key differences between the two listings...").
It also repeated itself, emitting a second "Insight:" line with a
near-duplicate summary - the model re-triggered on its own completion
cue. No fabricated facts, but the stop sequence (`\nNew submission:`)
wasn't sufficient to stop a self-triggered restart; `\nInsight:` should
be added as a stop sequence too.

Test input: "Newly built 5-bedroom family home with an open-concept
living area, modern kitchen island, and a two-car garage. Energy
efficient with solar panels installed." (near-exact match, house)
Result: Correctly matched and cited [Listing 1] with accurate shared
features (open-concept layout, kitchen island, two-car garage, solar
panels - all genuinely in the source text). Correctly excluded the
unrelated apartment listing (Listing 3) entirely, per the "only cite if
relevant" rule. citation_check: valid. Same drift/length issues as
above (compared Listing 1 to Listing 2 instead of the submission, ran to
3 paragraphs).

Test input: "Ground-floor retail unit on a busy high street, large
storefront windows, storage room in the back, and existing HVAC system.
Previously operated as a boutique clothing store." (near-exact match,
commercial)
Result: FAILURE (hard requirement #2, must cite). Described the
retrieved listing's real features accurately but never used a single
`[Listing N]` tag anywhere in the response. citation_check reported
valid: true, which is a blind spot in that check - it only flags
citations that reference a listing outside the retrieved set; it has no
way to detect the absence of citation altogether.

Test input: "Vintage 32-foot sailboat for sale, teak deck, recently
repainted hull, fully equipped galley and two berths." (off-domain,
no genuinely similar listing exists in the KB)
Result: FAILURE (hard requirement #4, say so plainly if no good match).
Retrieved distances were noticeably higher than the on-domain queries
above (1.44-1.66 vs. 0-0.96), correctly signaling weak matches - but the
model never surfaced that. Instead it evaluated the sailboat itself as
if it were a property listing ("this vintage sailboat is a unique and
potentially valuable property... an attractive option for buyers looking
for a rare find") and cited all three unrelated listings without ever
stating that none of them are a good match. The per-listing facts cited
were individually accurate, but the overall response is exactly the
"forced comparison" rule 4 exists to prevent.

Test input: "Small studio, good location" (sparse/minimal input)
Result: FAILURE (hard requirement #1, no fabrication) - the clearest one
found. The response opens: "This new submission is a small studio
apartment, priced at $165,000." The submission never mentioned a price;
$165,000 is [Listing 1]'s price, misattributed to the submission itself
as a stated fact rather than clearly attributed to the retrieved
listing. citation_check still reported valid: true, since "$165,000"
isn't a citation tag - this class of fabrication (importing a retrieved
number into the description of the query itself) is entirely invisible
to the mechanical check.

Test input: "2-bedroom apartment for rent downtown, good condition"
(generic, multiple plausible candidates)
Result: FAILURE (hard requirement #1, no fabrication). "Both listings
feature a renovated kitchen, hardwood floors, and a small balcony
overlooking the courtyard" - none of those features were in the
submission text; they're [Listing 1]'s, misattributed to the submission
via "both listings" phrasing. Worse: "a similar layout with two bedrooms
and one bathroom" - "one bathroom" appears in neither the submission nor
any retrieved listing; it's invented from nothing. citation_check:
valid (no out-of-range tags), completely missing both issues.

Assessment: retrieval is solid (ChromaDB + embeddings correctly rank
near-exact matches at distance ~0 and the off-domain query far higher).
Generation is the weak link - across 6 queries, 4 violated at least one
hard requirement (missing citations, forced comparison on an irrelevant
query, and twice fabricating/misattributing specific facts - a price and
a room detail - to the submission). The mechanical citation_check never
caught any of these; it only verifies citation *numbers* are in range,
which turned out to be nearly orthogonal to the actual failure modes
observed. A 1B-parameter model appears to struggle with the compound
instruction (cite everything + never fabricate + stay under 4 sentences
+ know when to decline) simultaneously, even with a one-shot example and
low temperature.

Recommendations for v2:
- Add "\nInsight:" (and similar) to the LLM stop sequences to prevent
  the self-triggered restart/duplication seen in query 1.
- Add a distance threshold: when the best retrieved match exceeds it,
  skip the LLM call and return a deterministic "no similar listings"
  message instead of trusting the model to recognize a poor match itself
  (this would have fully prevented the sailboat failure).
- Add a second, negative few-shot example (no good match -> plain
  decline) - the current baked-in example is always a positive match,
  so the model has no in-context template for what a decline should
  look like.
- Strengthen (or replace) citation_check: it should at minimum flag
  zero-citation responses as suspicious when listings were retrieved,
  and ideally cross-check any dollar amounts / room counts stated near
  "this submission"/"new submission" phrasing against the actual
  submission text, not just against the retrieved listings.
- The guardrails-service /check/output endpoint (deterministic
  risk-phrase check, see services/guardrails-service) is a natural place
  to add a second, independent audit pass on this insight text before it
  reaches a user - it wasn't wired in for this v1 test, but should be
  before this goes further than a demo.
- Consider whether a larger instruct model handles the compound
  instruction more reliably before assuming this is purely a prompting
  problem - 1B models are known to struggle with multi-constraint
  instructions regardless of how the prompt is worded.
