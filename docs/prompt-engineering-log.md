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

Note on versioning below: v1 surfaced three distinct hard-requirement
failures (fabrication, missing citation, forced comparison) but the
required structure only allots two "targeted iteration" slots (v2-v3)
before "refinement" (v4-v5). Prioritized by severity - fabrication is the
most dangerous to ship (false price/feature info reaching an agent), so
it's tackled first, then forced comparison (also a hard requirement,
and the more embarrassing failure mode), with missing-citation and a
refinement pass following under "refinement."

### v2 - fix: submission/listing fact conflation
Failure (one sentence): the model stated a retrieved listing's price and
features as if they belonged to the new submission itself (v1's "Small
studio, good location" and "2-bedroom apartment for rent downtown"
queries).

Fix attempted: added rule 6 to `INSIGHT_SYSTEM_PROMPT` - the submission's
own price/room count/features are only what's in its own text; a fact
from a listing must be attributed explicitly to that listing, never
implied about the submission.

Retest:
- "Small studio, good location" -> "This submission is a good match for
  [Listing 1], which also features a cozy studio apartment. The new
  submission's description highlights the ideal location and amenities,
  making it a strong contender to be compared to [Listing 1]." No price
  stated for the submission (was: "priced at $165,000" in v1). Fixed for
  this case.
- "2-bedroom apartment for rent downtown, good condition" -> prices now
  correctly attributed per-listing ("[Listing 1] priced at $750,000
  compared to [Listing 3] which is priced at $310,000"), and the
  invented "one bathroom" from v1 is gone. But: "which shares the
  luxurious features such as a private elevator entrance, floor-to-
  ceiling windows, and access to a rooftop pool" still implies the
  submission has those features via "shares," without ever literally
  stating them as submission facts - the rule doesn't catch implication,
  only direct assertion.

Result: Partial fix. Eliminated the direct-assertion form of fabrication
(stating a specific number/feature as the submission's own), but left an
implication-shaped loophole open - carried into v5.

### v3 - fix: forced comparison on off-domain queries
Failure (one sentence): an off-domain query (a sailboat) got evaluated as
if it were a property listing instead of triggering "no good match,"
even though retrieval correctly ranked it far below genuine matches.

Fix attempted: not a prompt change - a deterministic code-level gate.
Added `MAX_RELEVANT_DISTANCE = 1.2` to config.py (chosen from v1 data:
genuine matches scored 0-0.96, the sailboat scored 1.44-1.66) and a check
in `generate_insight`: if the closest retrieved listing's distance
exceeds this, skip the LLM call entirely and return a canned "No
sufficiently similar listings were found for this submission." This
doesn't rely on the model recognizing a bad match - retrieval already
signals it reliably, so the code acts on that signal directly.

Retest:
- Sailboat query -> "No sufficiently similar listings were found for
  this submission." in 0.5s (LLM never invoked). Fixed.
- Regression check (near-exact apartment match) -> LLM still invoked
  normally, correct citation, no fabrication. No regression.

Result: Complete, robust fix for this failure mode - deterministic, so
not subject to model-generation variance the way a prompt-only fix would
be.

### v4 - fix: missing citations
Failure (one sentence): a response accurately described a retrieved
listing's real features but never used a single `[Listing N]` citation
tag anywhere, failing the "must cite" hard requirement outright despite
having no fabricated content.

Fix attempted: reinforced the citation instruction by moving it to right
before the generation cue (recency bias - small models tend to follow
instructions closer to where generation starts more reliably than ones
stated once, earlier, in a longer system prompt). `TASK_TEMPLATE` now
ends with `"(Remember: cite every listing you reference using its exact
tag, e.g. [Listing 1].)\nInsight:"` instead of going straight to
`"Insight:"`.

Retest:
- The exact query that failed to cite in v1 (ground-floor retail unit)
  now produces "[Listing 1]" and "[Listing 2]" tags correctly.
- Spot check on "Small studio, good location" also cites [Listing 1]
  and [Listing 3] correctly (and price attribution from the v2 fix still
  holds alongside this change).

Result: Fixed in every case retested. Not proven to be 100% reliable
(still a prompt-level fix, so subject to the same model variance as any
instruction), but no missing-citation case reappeared in later testing
either.

### v5 - refinement: implicit feature-sharing loophole
Failure (one sentence): even after v2, the model could still imply the
submission has a listing's feature via "both listings share/have/
feature" phrasing without ever directly asserting a fact "for the
submission" - functionally the same fabrication, worded around the
letter of rule 6.

Fix attempted: added rule 7 - never say the submission and a listing
"share," "both have," or "both feature" something unless that exact
thing is written in the submission text; attribute listing-only features
to that listing by name instead.

Retest:
- "2-bedroom apartment for rent downtown, good condition" (previously
  had this issue) -> now says "[Listing 1], which offers..." and "While
  [Listing 3] is also a duplex apartment, it lacks the prestige..." -
  correctly scoped to the listings, no implied submission features.
  Fixed for this case.
- "Small studio, good location" (previously had this issue) -> "Both
  listings highlight the benefits of living in a converted industrial
  space, with exposed brick walls and high ceilings." Still implies the
  submission has brick walls/high ceilings - the model substituted
  "highlight the benefits of" for the banned "share"/"both have"/"both
  feature" phrases and reproduced the same fabrication through a phrase
  the rule didn't name.

Result: Partial fix - one of two retest cases cleared, the other found a
different wording that reintroduces the same problem. This is the
clearest evidence in this log that narrow phrase-banning has diminishing
returns: patching specific wordings doesn't close the underlying
tendency to generalize "both X" across submission and listing. A
structural fix (e.g. forcing the model to name its subject explicitly
per clause, or a two-pass generate-then-verify approach) would likely be
needed to close this fully - out of scope for this baseline.

## Final design and evaluation

Final design = v5's prompt/code as tested above: a 7-rule system prompt
(grounding, citation, relevance, decline-when-no-match, length,
no-submission-fact-invention, no-implied-sharing), one fixed positive
few-shot example, a deterministic distance-threshold gate that skips the
LLM entirely below `MAX_RELEVANT_DISTANCE`, and a citation-range check
logged (not returned) as a diagnostic signal.

Justification for shipping this as the baseline rather than iterating
further: the two failure modes with deterministic, code-level fixes
(off-domain forced comparison, and by extension anything gated on
retrieval distance) are now fully resolved and not subject to model
variance. The two failure modes that are pure prompt-engineering targets
(missing citations, fact conflation) are substantially improved but not
airtight - and the v5 result shows that further prompt patching hits
diminishing returns with this 1B model. Rather than keep adding rules
indefinitely, this is the point to stop, measure honestly, and record
what's actually shipped: a system that is reliable on the mechanically-
checkable failure mode and meaningfully better (not perfect) on the two
that require the model to reason about what it was and wasn't told.

### Test suite (12 cases) and pass criteria

A case **passes** if: (a) it doesn't state or imply any price, room
count, or feature for the submission that isn't in the submission's own
text, and (b) it declines (via the distance gate) when nothing is a good
match, or cites at least one genuinely relevant listing when something
is. Verbosity and the "compares Listing A to Listing B instead of the
submission" drift noted since v1 are **not** gating - they're a real,
open quality issue (see Known limitations below) but not a hard-
requirement violation per the spec's original three musts (grounded,
cited, non-fabricated).

| # | Query | Category | Result |
|---|-------|----------|--------|
| 1 | Near-exact apartment match | exact match | **PASS** - 1 sentence, correct citation, no fabrication |
| 2 | Near-exact house match | exact match | **PASS** - same |
| 3 | Near-exact commercial match | exact match | **PASS** - citation used, facts genuinely in submission text |
| 4 | Sailboat (off-domain) | decline | **PASS** - distance gate declined correctly |
| 5 | Used sedan (off-domain) | decline | **PASS** - distance gate declined correctly |
| 6 | "Small studio, good location" | sparse/generic | **FAIL** - "Both listings highlight... exposed brick walls and high ceilings" not in submission (the v5 loophole) |
| 7 | "2-bedroom apartment for rent downtown, good condition" | generic | **FAIL** - "offers similar amenities, including a renovated kitchen, hardwood floors, and a small balcony" - none in submission, uncited source |
| 8 | "Family house with a garden, needs some work, good schools nearby" | generic | **PASS** (borderline) - "both listings highlight the importance of having a private outdoor space" is a fair generalization over the submission's own "garden," not an invented concrete feature |
| 9 | "Office space downtown available for lease" | generic | **FAIL** - "both cases" use of meeting rooms and a kitchenette; neither is in the submission |
| 10 | "House for sale" (minimal, on-domain) | sparse | **PASS** - explicitly noted the submission "does not mention any specific price range, room count, or features" instead of inventing any |
| 11 | "4-bedroom house under $300,000 in a quiet neighborhood" | numeric constraint | **FAIL** - claims the submission "highlights its... mid-century modern design and recently listed after light renovation," which are a *different*, uncited listing's facts, plus an invented "close to good schools" |
| 12 | "Looking for a small retail space or a compact apartment near downtown" | mixed-intent | **FAIL** - "also features a renovated kitchen and hardwood floors" and "proximity to the metro station," neither in the submission |

**Pass rate: 7/12 (58%).**

The failure pattern is clean and consistent: **6/6 (100%)** on near-exact
matches and off-domain declines (cases 1-5, 10) - the cases where the
model can largely echo back what's already there, or where the
distance gate acts before the LLM runs at all. **1/5 (20%)** on
generic/ambiguous queries where the model has to actively synthesize a
comparison across multiple plausible listings (cases 6, 7, 9, 11, 12) -
this is where it reliably reaches for a plausible-sounding shared detail
that isn't actually grounded in the submission. Case 11 is the most
concerning single result: it attributes facts from a listing it doesn't
even cite by number directly to the submission's own description.

### Known limitations carried into this baseline
- Fact conflation (v2/v5's target) is fixed for direct assertions but
  not for generalized/implied sharing language, and specifically fails
  on generic multi-candidate queries - the most common real-world input
  shape, not an edge case.
- Verbosity and "compares Listing A to Listing B instead of the
  submission" drift (noted since v1) is still present and unaddressed -
  it was deliberately left out of scope for this iteration to focus on
  the hard-requirement failures, but should be next if this continues.
- The logged citation-range check still only catches out-of-range
  listing numbers; it caught none of the 5 failures above, all of which
  were fluent, plausible-sounding, and involved either valid citations
  or none at all. It should not be read as a fabrication safety net.
- Per v1's recommendation, wiring `guardrails-service`'s `/check/output`
  in as a second, independent audit pass has still not been done - given
  the 42% failure rate measured here on generic queries, that's a higher
  priority than it looked after v1's smaller test set.
- This remains a single 1B-parameter model; a larger instruct model
  might resolve the synthesis-time fabrication pattern directly rather
  than needing further prompt engineering.

## Information Extractor

### v1 - Baseline
Model: gemini-3.5-flash-lite
Fields: property_type (required), price/rooms/location/key_features/
certifications (optional, extract only if explicit in text)

Initial test across 3 varied descriptions (commercial retail, house
with solar panels, 8-room office) surfaced one failure: property_type
returned free-form values ("residential", "office space") instead of
the required enum (apartment/house/commercial).

Fix: tightened the property_type field description to require exactly
one of three words, with explicit mapping examples (condo/flat/studio
-> apartment; villa/cottage -> house; office/retail/warehouse ->
commercial).

Retest: all 3 cases returned correct enum values. Other fields
(rooms, key_features, certifications) performed correctly on first
attempt - rooms correctly totals bedroom+1 when only bedrooms stated,
correctly omitted when no room info present, and correctly took an
explicit total (8-room office) as-is.

No further iteration needed at this stage - single targeted fix
resolved the only identified failure mode.

## LangGraph Agent Tool Selection

### v1 - Baseline
Model: gemini-3.5-flash-lite, via the planner node's native
function-calling (`bind_tools` - see `services/langgraph-agent/graph.py`).

Tools offered: `search_similar_listings` (rag-service, text-only
property comparison) and `analyse_room_photo` (image-analyser, single
photo room type + condition). Full tool descriptions - the actual prompt
surface being tested - are in `services/langgraph-agent/tools.py`.

8 test queries: the spec's two named examples, plus rag-only, image-only,
both-tools, no-tools-needed, and two cases designed to tempt the wrong
tool.

| # | Query (abridged) | Expected | Tools called | Correct? |
|---|---|---|---|---|
| 1 | Photo of a bathroom - what's needed to reach condition 5? (spec example) | image only | `analyse_room_photo` | Yes |
| 2 | 3 photos - which room needs the most attention? (spec example) | image x3 | `analyse_room_photo` x3 | Yes |
| 3 | 3BR apartment, renovated kitchen, $300k - how does it compare? | rag only | `search_similar_listings` | Yes |
| 4 | Is $850k reasonable for a 5BR house with solar panels? | rag only | `search_similar_listings` | Yes |
| 5 | Listing description + kitchen photo - compare it AND assess the kitchen | both | `search_similar_listings`, `analyse_room_photo` | Yes |
| 6 | 3 tips for staging a living room before an open house | none | *(none)* | Yes |
| 7 | "Tell me about this property." (no description, no photo) | none | *(none)*, asked for clarification | Yes |
| 8 | Photo of a living room - better or worse than average condition? | image only | `analyse_room_photo` | Yes |

Result: 8/8 correct.

Query 8's phrasing ("better or worse than *average*") reads like a
market-comparison question, which could plausibly trigger
`search_similar_listings` - but that tool has no condition-score data to
offer, and the planner called only `analyse_room_photo`.

Query 7 has no description or photo. The planner called no tools; the
synthesiser asked the user for more detail rather than calling
`search_similar_listings` with an empty/junk description.

Queries 1 and 8 hit images the trained classifier (image-analyser)
reports `"uncertain"` for (confidence below its 0.4 threshold - see its
own README). Tool selection was correct in both cases regardless; the
synthesiser reported the uncertainty rather than fabricating a condition
score. This is image-analyser's own known limitation surfacing through
the pipeline, not a tool-selection failure.

No iteration needed - 8/8 on the first test pass. A genuine misselection
surfacing in real usage would be the v2 entry for this surface.

## Node 5 (AI Agent) System Prompt

### v1 - Baseline
Role: senior property analyst. 3 tools defined (rag/image/langgraph).
JSON output format: summary, market_comparison, photo_analysis, recommendations.

The AI Agent's prompt combines the Information Extractor's 
output with image_urls from the Webhook node's payload.

Test 1 (commercial listing, no photo): correct tool selection
(search_similar_listings only), valid JSON, photo_analysis correctly null.

Test 2 (same listing + image_urls via webhook): correct tool selection
for both tools, photo_analysis populated correctly, JSON structure held.

Known issue (not a Node 5 failure): rag-service's insight fabricated a
comparison price not present in any retrieved listing - already
documented in rag-service's own log (58% pass rate). Surfaces here but
isn't specific to this surface.

No further iteration needed at this baseline - core mechanics solid.

## Node 6 (LLM Chain)

### v1 - Baseline
System Message defines role (property brief writer), explicit input
shape (summary/market_comparison/photo_analysis/recommendations from
Node 5), required JSON output structure (title/brief_markdown) with an
example response as a template, and an explicit instruction against
extra wrapper keys or code fences.

Structured Output Parser configured with schema: title (string),
brief_markdown (string), both required.

Known Gemini inconsistency: output format varies between runs even with
identical prompts and schema - sometimes returns {"title",
"brief_markdown"} directly, sometimes wraps it as {"output": {...}}, and
section coverage in brief_markdown isn't always identical. A Code node
receiving both the LLM Chain's success and error outputs normalizes this
- unwraps an "output" key if present, strips markdown code fences, and
parses the JSON - so Respond to Webhook always receives the same clean
{title, brief_markdown} shape regardless of which format Gemini chose
that run.

Verified across multiple runs: valid Markdown brief generated each time,
integrating all available Node 5 fields into a coherent document, no
fabricated content, consistent title/brief_markdown structure after
normalization.

No further iteration planned - the safety-net design handles Gemini's
inherent output variability without needing further prompt tuning.
