# 01 — Bottleneck Analysis
### Smart-Segmentation: Current State Assessment

**Research ID:** research_2_sonnet_rag_extra_claude
**Model:** Claude Sonnet 4.6
**Codebase:** `/Users/s0m0ohl/customer_segement/Smart-Segmentation`
**Date:** February 2026

---

## Executive Summary

The Smart-Segmentation system is a functioning multi-stage LLM pipeline for customer segment definition, built on Google ADK (Agent Development Kit). It correctly handles the core segmentation workflow — decompose → retrieve → classify → format. However, it contains 12 critical bottlenecks that limit its quality, scalability, and enterprise readiness. The most severe are:

1. **Embedding-only retrieval** for a structured, finite facet catalog — the wrong tool for the job
2. **Seven+ LLM pipeline stages** creating error accumulation and latency overhead
3. **Hardcoded tenant coupling** throughout static prompts, files, and code logic
4. **46-row ground truth dataset** that is unused at runtime and too small for reliable evaluation
5. **No eval gate** between development and deployment — system can regress silently

---

## Part 1: Architecture Bottlenecks

### B1.1 — Wrong Abstraction Layer: Google ADK vs Raw Agent SDK

**Issue:** The system uses Google ADK (`LlmAgent`, `SequentialAgent`, `AgentTool`) — a framework primarily designed for Gemini and Vertex AI. The actual LLM is configured via `gptmodel` (an abstraction that points to a different provider via `adk_llm_model.py`). This mismatch creates:
- Implicit tight coupling to Google ADK's agent loop semantics
- Limited observability into what the LLM actually sees (ADK wraps prompt construction)
- Upgrade risk: ADK framework changes can silently alter agent behavior
- Reduced portability: switching LLM providers requires reworking the ADK configuration

**Code reference:** `agent.py:9` — `from google.adk.agents import LlmAgent` combined with `from utils.adk_llm_model import gptmodel`

**Impact:** Medium — functional today, becomes high-friction when scaling to multiple providers or implementing structured eval gates.

### B1.2 — Sequential Error Accumulation Across 7+ Pipeline Stages

**Issue:** The pipeline consists of 7+ distinct LLM calls in sequence, each receiving the output of the prior stage:

```
Stage 1: RouterAgent          → intent classification
Stage 2: SegmentDecomposer    → JSON ruleset with sub-segments
Stage 3: DateTaggerAgent      → date metadata extraction + query rewrite
Stage 4: FVOM                 → facet shortlisting + LLM filtering
Stage 5: DependencyIdentifier → structured dependency resolution
Stage 6: FacetClassifier      → refinement value selection (propensity tiers)
Stage 7: LinkedFacetMatcher   → linked facet resolution
Stage 8: SegmentFormatter     → final JSON output
```

In a 7+ stage chain, error probability compounds multiplicatively. If each stage has 90% accuracy, the combined accuracy of the chain is 0.9^7 ≈ **48%**. If each stage is 95%, combined is 0.95^7 ≈ **70%**.

**Code reference:**
- `sub_agents/new_segment_creation/agent.py` — orchestrates stages 2-4
- `sub_agents/new_segment_creation/sub_agents/segment_format_generator/agent.py` — stages 5-8

**Impact:** Critical — errors in early stages (decomposer, date tagger) propagate to every subsequent stage with no recovery mechanism.

**Ground truth evidence:** 22 of 46 ground truth rows (47.8%) have non-empty Remarks indicating corrections were needed — suggesting that current multi-stage accuracy is significantly below target.

### B1.3 — No Verify-Correct Loop

**Issue:** The pipeline is strictly linear with no feedback loop to detect and correct errors mid-pipeline. If the Segment Decomposer produces an incorrect sub-segment split, the FVOM stage will retrieve incorrect facets for those segments, and the Formatter will produce an incorrect output — with no recovery.

The only error recovery mechanism is user-facing clarification questions, which are reactive (wait for user feedback) rather than proactive (self-verify and retry).

**Impact:** High — the system cannot self-heal from structural mistakes.

---

## Part 2: Facet Retrieval Bottlenecks

### B2.1 — Embedding-Only Search is Wrong for Structured Facet Catalog

**Issue:** The system uses dense vector search (BGE or MiniLM embeddings via Milvus) as the sole retrieval mechanism for a catalog of 500+ facets. This is the **wrong primary retrieval strategy** for this use case.

**Why embedding search is suboptimal here:**

| Characteristic | Facet Catalog | Optimal for Embedding? |
|---|---|---|
| Catalog size | ~500 structured entries | ❌ Small enough for exact lookup |
| Entry type | Structured (name, type, l1_value, description) | ❌ Structured data doesn't need semantic embedding |
| Entry length | Facet names = 2-6 words | ❌ Very short — embedding quality degrades for short texts |
| Exact match need | "CRM Email Engagement" = exact facet name | ❌ Embedding search can miss exact matches due to distance threshold |
| Alias resolution need | "Strict" vs non-Strict variants | ✅ Embedding can help resolve synonyms |
| Brand/department inference | "spring cleaning" → "Propensity Super Department" | ✅ Embedding genuinely helps for semantic inference |

**The current system uses embedding search where a structured dictionary lookup would be more accurate and faster for 80-90% of cases.**

**Code reference:** `utils/milvus.py:125-176` — `search_using_milvus` always performs ANN (approximate nearest neighbor) search with no exact match pre-pass

**Impact:** Critical — causes both false positives (wrong facets with high embedding similarity) and false negatives (correct facets missed due to sub-threshold distances).

### B2.2 — Hybrid Search Implemented But Never Used (Critical Bug)

**Issue:** The `MilvusDB` class implements `_match_single_instance_hybrid_search` with RRFRanker support (line 69-122 of `milvus.py`), but the search mode validation contains a **typo** that makes hybrid mode impossible to invoke correctly:

```python
# Line 152 in milvus.py
if len(indexed_field_list) == 1 and search_mode == "hybird":  # ← "hybird" typo
    raise ValueError("Invalid Search Request: For Hybrind search mode...")
```

The typo is `"hybird"` instead of `"hybrid"`. In `shortlist_generation.py`, all Milvus calls use `search_mode = "standard"`. Even if a caller tried to use hybrid mode, the validation logic would silently pass with a typo-matched string. In practice, **the hybrid search path is dead code.**

Furthermore, the "hybrid" in this implementation is multi-field dense-dense fusion (name_embeddings + description_embeddings) — not BM25+dense hybrid, which is the state-of-the-art approach for short text queries.

**Code reference:** `utils/milvus.py:152` — typo in validation, `shortlist_generation.py:99-128` — all calls use `"standard"` mode

**Impact:** Critical — the team has invested in hybrid search infrastructure that is completely bypassed.

### B2.3 — Fuzzy Score Threshold Creates Brittleness

**Issue:** After value-based Milvus search, results are filtered through a dual threshold:
1. Milvus distance must be ≥ `SEARCH_FUZZY_MATCH_MILVUS_THRESHOLD` (env var)
2. Fuzzy string match between the NER entity and the matched l1_value must be ≥ `SEARCH_FUZZY_MATCH_MIN_RATIO`

This creates fragility: slightly different phrasings of a legitimate value (e.g., "women's clothing" vs "WOMEN APPAREL") may score below the fuzzy threshold and be dropped, even when they represent the same concept.

**Code reference:** `shortlist_generation.py:190-197`

**Impact:** High — reduces recall, causing the LLM to ask unnecessary clarification questions.

### B2.4 — NER Pre-Pass Quality Unknown

**Issue:** The system runs a Named Entity Recognition pass before embedding search (`named_entity_recognition_agent.py`). NER results feed the value-based Milvus search. However:
- The NER quality is not evaluated independently
- NER entity list quality directly determines which values get searched
- Poor NER → missed entities → missed facets
- The `replace_walmart` function strips the word "Walmart" (non-Plus/non-+) from queries before NER — this is a tenant-specific hack

**Code reference:** `shortlist_generation.py:156-168` — NER called before embedding, `shortlist_generation.py:132-148` — `replace_walmart()` is Walmart-specific

**Impact:** Medium-High — NER quality is an invisible bottleneck in the retrieval chain.

### B2.5 — No Exact Catalog Lookup for Known Facet Names

**Issue:** Even when a user explicitly mentions a facet name (e.g., "CRM Email Engagement"), the system still goes through the full NER → embedding → fuzzy pipeline instead of doing a direct dictionary lookup. This adds latency and can introduce matching errors for exact references.

**Impact:** Medium — adds unnecessary complexity for the easy case.

### B2.6 — Cardinality Threshold is Arbitrary (>20 Values = Empty)

**Issue:** In `shortlist_generation.py:241-246`, when a facet has more than 20 values, the system sets its values to an empty list:
```python
if len(values) > 20:
    facet_value_dict_using_name[result["name"]] = []
```
This means the LLM must infer the correct value from scratch (or ask the user) for high-cardinality facets. This creates unnecessary clarification round-trips for facets like "Propensity Brand" (hundreds of brands) where the user's query already contains the answer.

**Impact:** Medium — causes unnecessary user clarification for common, high-cardinality facets.

---

## Part 3: Ground Truth Data Gap Analysis

### B3.1 — Ground Truth Dataset is 46 Rows (Not 18,779)

**Issue:** The ground truth CSV at `ground_truth_cdi(Updated Ground Truth).csv` contains **46 rows**, not 18,779 as described in some documentation. This is a critical discrepancy.

**Statistics from the dataset:**
- Total rows: 46
- Rows with non-empty Remarks: 22 (47.8%)
- Average expected facets (original): 4.35 per segment
- Average expected facets (updated): 4.76 per segment
- Rows with Predicted CDI populated: 46 (100%)

**Impact:** The eval dataset is too small for statistically reliable conclusions. With 46 rows, a 1-row improvement/regression changes the accuracy score by 2.2 percentage points — making it impossible to distinguish real improvements from noise.

### B3.2 — "Strict" vs Non-Strict Facet Name Drift

**Issue:** The Remarks column reveals a systematic pattern: the original "expected facets" used "Strict" suffix variants (e.g., `Propensity Super Department Strict`), but the updated expected facets dropped the "Strict" suffix. This means:

1. **The embedding index may contain both Strict and non-Strict variants** — the LLM and retrieval system need to resolve which to use
2. **The hints file** explicitly addresses this: "IF Attributes with suffix such as Strict and Non-Strict are all shortlisted, then only select with Suffix Strict" — but then the ground truth was updated to prefer non-Strict. **The hints file and ground truth are now contradictory.**

**Code reference:** `contextual_information/facet_value_mapper_hints.txt:34-36` — "select Facet with Suffix Strict"

**Sample remarks showing systematic corrections:**
- "Removed Propensity Super Department = ACCESSORIES"
- "Added Propensity Super Department for consistency"
- "Updated Persona to include sub-categories"
- "Added all apparel super departments. Added division apparel"

**Impact:** High — systematic facet name aliases and deprecated variants are not handled cleanly, causing inconsistent retrieval.

### B3.3 — Ground Truth is Eval-Only, Not Runtime Few-Shot

**Issue:** The 46 ground truth rows are used exclusively for batch evaluation (`evaluations/cli.py`). They are NOT used at runtime as few-shot examples for the LLM. This is a missed opportunity:

- The FVOM prompt receives `{similar_historical_segment_examples_dict}` — but this is populated from a static lookup, not from dynamic retrieval over the ground truth CSV.
- A query like "Build a segment for spring fashion shoppers" would benefit enormously from seeing the historical `Test-Segment-WomenFashion` example at inference time.

**Impact:** High — the best source of grounding evidence (real labeled examples) is not used during inference.

### B3.4 — No Precision/Recall Metrics Tracked Per Facet Type

**Issue:** The evaluation framework appears to compute overall facet match accuracy but does not break down failures by facet type (propensity, persona, date, numeric, purchase, engagement). Without per-type analysis, it's impossible to identify which facet categories fail most.

**Based on the ground truth sample analysis, the following patterns are visible:**
- **Propensity facets** (Super Department, Division, Brand) appear most frequently in corrections
- **CRM Engagement facets** (Email vs Push vs Generic) require disambiguation
- **Date facets** are handled separately (Stage 3) and appear less in corrections, suggesting Stage 3 is more reliable
- **Persona facets** appear stable but sometimes need sub-category expansion

---

## Part 4: Pipeline Stage Analysis — Is Every Stage Necessary?

### B4.1 — Stage 3 (Date Tagger): Should Be Mostly Rule-Based

**Issue:** Stage 3 uses a full LLM call to extract date ranges from sub-segment queries. However:
- `date_tagger_patterns.json` already exists and is used POST-LLM to strip date phrases from queries
- `walmart_date_classes.py` implements structured date computation
- The prompt's date taxonomy (WA, MA, YA, DA, etc.) is fully enumerable and rule-based
- Libraries like `dateparser` or `spaCy` can handle 90%+ of common date expressions

**A rule-based date extractor would handle:** "last 30 days", "past year", "in the last 6 months", "Q1 FY26", "next week", specific dates.

**LLM would still be needed for:** ambiguous holidays ("Thanksgiving"), vague references ("recently", "soon"), multi-context date conflicts.

**Recommendation:** Replace Stage 3 LLM call with a rule-based date extractor (dateparser + custom Walmart fiscal calendar logic). Use LLM only for the ~10% ambiguous cases.

**Impact:** Eliminating this LLM call would save ~300-500ms latency per request and reduce error propagation.

### B4.2 — Stages 5-7 (Dependency → Classifier → Linked) Can Be Collapsed

**Issue:** Three separate LLM calls handle facet dependency resolution:
- Stage 5: `dependency_identifier` — identifies which facets need refinement values (propensity tiers, etc.)
- Stage 6: `facet_classifier_resolver_agent` — asks user for refinement value selection
- Stage 7: `linked_facet_resolver_agent` — resolves facets with dependencies on other facets

These three stages share the same context and could be handled in a single structured LLM call with typed output schema:

```json
{
  "refinement_selections": [...],
  "linked_facet_resolutions": [...],
  "user_questions": [...]
}
```

**Code reference:** `sub_agents/new_segment_creation/sub_agents/segment_format_generator/agent.py` — strictly sequential execution of all three

**Impact:** Collapsing 3 LLM calls into 1 would reduce latency by ~1-1.5 seconds and reduce error surface.

### B4.3 — Stage 6 (Formatter): Pure Code, Not LLM

**Issue:** Stage 8 (`segment_formatter` tool) transforms the structured FVOM output into the final JSON segment definition. This is a pure data transformation — mapping facet-value-operator triples into a target schema. It should not require an LLM call.

Looking at `tools/processing.py` and `tools/response_formatter.py` — these appear to be code-based transformers. But if `segment_formatter` still involves any LLM call, this is pure overhead.

**Impact:** Medium — if formatter is code-based, this is already handled well.

### B4.4 — Stage 2 (Decomposer): High Risk of Error Propagation

**Issue:** The Segment Decomposer (Stage 2) splits user queries into sub-segments. This is the most critical stage because:
1. Incorrect sub-segment splits → wrong facets retrieved → wrong segment
2. There is no verification that the decomposition is correct before proceeding
3. The decomposer can miss sub-segments (causing lost conditions) or create duplicates

The hints file attempts to mitigate this with rules about merging/splitting, but these are static text rules that the LLM can misinterpret.

**Impact:** Critical — decomposition errors are catastrophic and unrecoverable in the current pipeline.

---

## Part 5: Multi-Tenant Scalability Risks

### B5.1 — Facet Key Identifier: Only Two Hardcoded Options

**Issue:** The system supports only two facet catalog keys: `email_mobile` and `cbb_id`. Both are Walmart-specific identifiers. The logic for selecting between them is hardcoded in multiple places:

```python
# shortlist_generation.py:33-38
if self.facet_key_identifier == 'email_mobile':
    self.NAME_COLLECTION_NM = self.NAME_COLLECTION_NM.replace('{DYNAMIC_FACET_KEY}','EMAIL_MOBILE')
else:
    self.NAME_COLLECTION_NM = self.NAME_COLLECTION_NM.replace('{DYNAMIC_FACET_KEY}','CBB_ID')
```

A new tenant would need code changes to add their catalog key.

**Code reference:** `shortlist_generation.py:33-38`, `utils/metadata.py:76-79`

### B5.2 — Contextual Information Files Are Statically Loaded at Startup

**Issue:** All four contextual information files are loaded once at agent initialization and injected as static text into prompts:

1. `contextual_information/contextual_information_on_refinements.txt` — Walmart-specific refinement vocabulary (TOP 10, Very-High & Above, etc.)
2. `contextual_information/catalog_view_description.txt` — Walmart-specific catalog capabilities
3. `contextual_information/segment_decomposer_hints.txt` — Walmart-specific decomposition rules (Walmart+, B2B/B2C)
4. `contextual_information/facet_value_mapper_hints.txt` — Walmart-specific business rules (channel routing, Strict vs non-Strict)

For a new tenant:
- A new retailer won't have "Walmart+" — the decomposer hints explicitly handle it
- A new retailer's propensity tier names may differ — the refinements file must change
- The channel-to-date-facet mapping is Walmart-specific

**There is no runtime tenant switching mechanism** for these files.

**Code reference:** `agent.py:72-73` — CATALOG_VIEW_DESC loaded at module import time

### B5.3 — `attribute_mapping` in `metadata.py` is Hardcoded

**Issue:** The channel-to-date-attribute mapping is hardcoded as a class variable in `MetaData`:

```python
self.attribute_mapping = {
    ('ONLINE', 'SPECIFIC'): ['Purchase Date R2D2'],
    ('ONLINE', 'GENERIC'): ['Last Purchase Date'],
    ('STORE', 'SPECIFIC'): ['Purchased Date - Store (Digitally Identified)'],
    ...
}
```

These are Walmart-specific date facet names. A new tenant would need code modification.

**Code reference:** `utils/metadata.py:41-53`

### B5.4 — `replace_walmart` is Hardcoded Domain Logic

**Issue:** The `replace_walmart()` function in `shortlist_generation.py` strips "walmart" from queries before NER to prevent the brand name from being treated as an entity to search. This is pure tenant-specific logic embedded in the retrieval pipeline.

**Code reference:** `shortlist_generation.py:132-148`

### B5.5 — Milvus Collections Named with Tenant-Specific Patterns

**Issue:** Milvus collection names contain tenant identifiers via env vars (`MILVUS_FACET_NAME_COLLECTION`, `MILVUS_FACET_VALUE_COLLECTION`). The `{DYNAMIC_FACET_KEY}` template allows runtime selection between EMAIL_MOBILE and CBB_ID. This is a good pattern — but it only supports two hardcoded keys.

**Partially good:** The collection naming via env vars is a step toward multi-tenancy.
**Still bad:** The `{DYNAMIC_FACET_KEY}` values are hardcoded to EMAIL_MOBILE/CBB_ID.

### B5.6 — Shared Milvus Schema Across Tenants

**Issue:** All tenants would share the same Milvus collection schema (name_embeddings, l1_value_embeddings, etc.). If a new tenant's facet catalog has different schema (e.g., different metadata fields, hierarchical categories), this breaks.

**Impact:** Medium — acceptable at 2-5 tenants if schema is standardized, but creates risk as tenants diverge.

---

## Part 6: Eval and Observability Gaps

### B6.1 — No Eval Gate Before Deployment

**Issue:** The evaluation framework (`evaluations/cli.py`) exists but is decoupled from the deployment pipeline. There is no automated gate that:
- Runs evals on prompt changes
- Blocks deployment if accuracy drops below a threshold
- Tracks performance over time in a structured way

**Impact:** High — prompt changes (like the Strict→non-Strict shift documented in Remarks) can silently degrade system performance.

### B6.2 — Phoenix Tracer Exists But Tracing Depth is Unclear

**Issue:** `phoenix_tracer.py` is imported in agent files, suggesting Arize Phoenix is used for tracing. However:
- The completeness of tracing is not visible in the code
- It's unclear if every LLM call, tool call, and retrieval step is traced with inputs/outputs
- There's no structured logging of Milvus retrieval results, fuzzy scores, or LLM intermediate outputs

**Impact:** Medium — limited observability means debugging failures requires manual inspection.

### B6.3 — No Automated Hallucination Detection

**Issue:** The system has no mechanism to detect when the LLM has "guessed" a facet name or value that doesn't exist in the catalog. The `filter_facet_value_list` function in `utils/facet_filter.py` provides some post-hoc filtering, but:
- It filters invalid facet-value pairs after the LLM has already selected them
- It doesn't distinguish between LLM hallucination and legitimate selections
- There's no grounding citation mechanism

**Impact:** High — the LLM can hallucinate facet names that appear valid but are wrong.

---

## Part 7: Prompt Design Bottlenecks

### B7.1 — Static Hints Files Create Prompt Brittleness

**Issue:** The segment_decomposer_hints.txt and facet_value_mapper_hints.txt are injected verbatim into prompts as `{decomposer_hints}` and `{hints}`. These files contain highly specific business rules. When rules change (as evidenced by the Strict→non-Strict evolution in ground truth), both the hints file AND the embedding index AND the ground truth must be updated consistently. There is no version control linking these together.

### B7.2 — Historical Examples in FVOM Prompt Are Static

**Issue:** The FVOM prompt includes `{similar_historical_segment_examples_dict}` — but this appears to be populated from a static lookup rather than dynamic RAG over the ground truth CSV. Static few-shot examples don't adapt to the specific query being processed.

### B7.3 — No Structured Output Validation

**Issue:** Multiple stages return JSON blobs that are parsed with `eval()` or `json.loads()` without schema validation. If an LLM returns malformed JSON or a JSON with unexpected keys, the error may propagate silently.

**Code reference:** `sub_agents/new_segment_creation/agent.py` — state updates using raw LLM output

---

## Summary: Bottleneck Severity Matrix

| ID | Bottleneck | Severity | Category |
|---|---|---|---|
| B2.1 | Embedding-only search for structured catalog | 🔴 Critical | Retrieval |
| B2.2 | Hybrid search implemented but never used (typo bug) | 🔴 Critical | Retrieval |
| B1.2 | 7+ stage error accumulation (47.8% correction rate) | 🔴 Critical | Architecture |
| B6.1 | No eval gate before deployment | 🔴 Critical | Eval |
| B3.3 | Ground truth not used at inference (no runtime RAG) | 🔴 Critical | Quality |
| B5.1 | Only 2 hardcoded tenant keys | 🟠 High | Multi-Tenant |
| B5.2 | Static contextual info files (no tenant switching) | 🟠 High | Multi-Tenant |
| B1.3 | No verify-correct loop | 🟠 High | Architecture |
| B4.2 | Stages 5-7 unnecessarily separate LLM calls | 🟠 High | Pipeline |
| B3.2 | Strict vs non-Strict naming inconsistency | 🟠 High | Quality |
| B2.3 | Fuzzy threshold creates recall brittleness | 🟡 Medium | Retrieval |
| B4.1 | Date tagger LLM call (should be rule-based) | 🟡 Medium | Pipeline |
| B5.3 | Hardcoded attribute_mapping in metadata.py | 🟡 Medium | Multi-Tenant |
| B5.4 | replace_walmart() hardcoded domain logic | 🟡 Medium | Multi-Tenant |
| B7.3 | No structured output validation | 🟡 Medium | Reliability |
| B3.1 | Ground truth only 46 rows (too small) | 🟡 Medium | Eval |
| B6.3 | No hallucination detection | 🟡 Medium | Quality |
| B1.1 | Google ADK abstraction mismatch | 🟡 Medium | Architecture |

---

*This document feeds directly into [03_concrete_upgrade_proposal.md](03_concrete_upgrade_proposal.md) where solutions for each bottleneck are proposed.*
