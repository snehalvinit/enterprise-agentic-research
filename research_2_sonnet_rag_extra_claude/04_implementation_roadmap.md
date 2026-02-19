# 04 — Implementation Roadmap
### Smart-Segmentation → Enterprise Agentic System: Prioritized Execution Plan

**Research ID:** research_2_sonnet_rag_extra_claude
**Date:** February 2026

---

## Guiding Principles

1. **Fix the retrieval foundation first** — everything else builds on facet retrieval quality
2. **Eval before shipping** — no change reaches production without an eval gate
3. **Parallelize where safe** — independent tracks can run concurrently
4. **Tenant-agnostic by design** — every new feature must be configurable, not hardcoded
5. **Collapse before adding** — reduce the 7-stage pipeline before adding new capabilities

---

## Phase 0: Foundation (Week 1-2) — Unblock Everything

> **Hypothesis:** Several quick wins can be unlocked immediately without architectural changes. These create the measurement baseline and fix critical bugs.

### P0.1 — Fix the Hybrid Search Typo (1 day)

**File:** `utils/milvus.py:152`
**Change:** Fix `"hybird"` → `"hybrid"` in the validation logic
**Risk:** Zero — the code path currently never executes, so fixing the typo only enables a new path
**Parallel:** Run eval comparison: standard vs hybrid search on the 46 ground truth rows

### P0.2 — Establish Eval Baseline (2-3 days)

**Action:** Run `evaluations/cli.py` on all 46 ground truth rows, capture baseline metrics:
- Per-row facet recall (expected facets correctly retrieved)
- Per-row facet precision (no extra incorrect facets)
- Per-type breakdown (propensity, persona, date, numeric, engagement)
- P50/P95 latency per pipeline stage

**Why first:** Cannot measure improvement without a baseline.

### P0.3 — Add Structured Output Validation (2 days)

**Action:** Add Pydantic schema validation after each LLM stage output. Any invalid output triggers a retry with error feedback, not silent propagation.
**Files:** `data_models/` — already has some Pydantic models, extend to all stages

### P0.4 — Enable Phoenix Tracing on All LLM Calls (1 day)

**Action:** Ensure every LLM call, Milvus query, and tool call is traced with full input/output logging.
**Why:** Required for debugging Phase 1 improvements.

**Phase 0 Deliverables:**
- Hybrid search enabled (typo fixed)
- Eval baseline metrics documented
- All outputs schema-validated
- Full tracing coverage

---

## Phase 1: Retrieval Overhaul (Week 3-6) — Highest Impact

> **Hypothesis:** Switching from pure embedding search to a cascade approach (exact lookup → structured filter → semantic embedding → LLM rerank) will be the single largest quality improvement.

### P1.1 — Implement Exact Catalog Lookup Pre-Pass (3-4 days)

**Action:** Before any Milvus call, check if the query term (or NER entity) exactly matches a facet name or known alias in the facet catalog.

```python
# Pseudo-code: catalog_lookup.py
class CatalogLookup:
    def __init__(self, facet_catalog):
        self.name_index = {row['name'].lower(): row for _, row in facet_catalog.iterrows()}
        self.alias_index = {}  # populated from alias table

    def lookup(self, query: str) -> Optional[dict]:
        normalized = query.lower().strip()
        return self.name_index.get(normalized) or self.alias_index.get(normalized)
```

**Expected impact:** Eliminates embedding noise for the 30-40% of queries that contain explicit facet names.

### P1.2 — Implement Alias/Synonym Table (3-4 days)

**Action:** Create a structured alias table that maps:
- "Strict" variants → non-Strict (e.g., `Propensity Super Department Strict` → `Propensity Super Department`)
- Common synonyms (e.g., "email engagement" → `CRM Email Engagement`)
- Historical names to current names

**Stored in:** A JSON/YAML config file (tenant-specific, loaded at runtime).

### P1.3 — Replace BM25-Absent Hybrid with True BM25+Dense Hybrid (1 week)

**Current situation:** The "hybrid" search in `milvus.py` merges two dense indexes (name + description embeddings). This is NOT BM25+dense hybrid.

**Action:**
- Add BM25 sparse embedding support via Milvus sparse index (Milvus 2.4+ supports sparse vectors via BGE-M3 BM25 integration)
- OR: Implement BM25 search separately using `rank_bm25` Python library against facet names, then merge with dense results via RRF

**Expected impact:** BM25 excels at exact keyword matching for short queries (facet names are 2-6 words). Dense excels at semantic similarity. Combining both covers the full spectrum.

**Evidence from research:** Hybrid BM25+dense consistently outperforms single-index dense on short domain-specific queries by 5-15% MRR in enterprise benchmarks.

### P1.4 — Implement Tiered Cascade Retrieval (1 week)

Replace the flat embedding search with a waterfall:

```
Tier 1: Exact name match (catalog dictionary)     → 0ms
Tier 2: Alias/synonym resolution                   → <1ms
Tier 3: Structured filter (type, category)         → <50ms via Milvus metadata filter
Tier 4: BM25 keyword search                        → <100ms
Tier 5: Dense embedding semantic search            → <200ms
Tier 6: LLM rerank top-20 candidates              → +300ms only when needed
```

Only fall through to next tier when upper tiers return no results.

### P1.5 — Add Typed Tool Retrieval (Replace "One Big Search")

**Action:** Replace the single `FacetValueShortlister` with typed retrieval tools:
- `search_propensity_facets(query)` — searches only propensity catalog section
- `search_engagement_facets(query)` — searches CRM engagement section
- `search_purchase_facets(query)` — searches purchase history section
- `search_persona_facets(query)` — searches persona section
- `search_date_facets(query)` — searches date facets section
- `search_demographic_facets(query)` — searches demographic section

**Benefit:** Type-aware retrieval dramatically reduces the search space per query, improving both precision and speed.

**Phase 1 Deliverables:**
- Cascade retrieval implemented
- Alias table covering all known Strict/non-Strict variants
- BM25+dense hybrid enabled
- Typed retrieval tools for 6 facet categories
- Eval comparison: Phase 0 baseline vs Phase 1 cascade (target: >10% recall improvement)

---

## Phase 2: Pipeline Compression (Week 5-8) — Latency & Reliability

> **Hypothesis:** Collapsing 7+ stages to 4 stages while keeping output quality eliminates cascaded errors and reduces end-to-end latency by 40-50%.

### P2.1 — Replace Stage 3 (Date Tagger) with Rule-Based Parser (4-5 days)

**Action:**
1. Use `dateparser` Python library for common date patterns
2. Add Walmart fiscal calendar logic (FY, quarter definitions) as a standalone module
3. Reserve LLM call only for ambiguous cases (<10% of queries)

**Files to create:** `tools/date_parser_rules.py` (expanding existing `tools/date_parser.py`)

**Expected impact:** -300ms per request for the 90% of queries with standard date formats. Zero LLM cost for date extraction.

### P2.2 — Collapse Stages 5+6+7 (Dependency+Classifier+Linked) into One LLM Call (1 week)

**Current:** 3 sequential LLM calls with overlapping context
**Target:** 1 structured LLM call with output schema:

```json
{
  "refinement_selections": [
    {"facet_name": "...", "selected_value": "...", "ask_user": false}
  ],
  "linked_facet_pairs": [
    {"primary": "...", "linked": "...", "relationship": "..."}
  ],
  "user_questions": []
}
```

**Expected impact:** -600-900ms per request, reduced error surface.

### P2.3 — Establish Verify Step Between Decompose and FVOM (3-4 days)

**Action:** Add a lightweight verification after Stage 2 (Decomposer):
- Does the sub-segment count match the logical operators in the ruleset?
- Do all sub-segment IDs appear in the ruleset?
- Are there duplicate sub-segments?

If verification fails → retry Decomposer with error feedback (max 2 retries).

**This is deterministic code — no LLM needed for verification.**

### P2.4 — Add ReAct Loop at NSC Orchestrator Level (1 week)

**Action:** Give the `NewSegmentCreationAgent` a Plan-Act-Verify loop instead of a fixed sequential execution:

```
Plan:   Determine which stages are needed for this query
Act:    Execute stages
Verify: Check output quality (schema valid? All sub-segments covered?)
Retry:  If verify fails, regenerate with feedback context
```

**Phase 2 Deliverables:**
- Rule-based date parser (90%+ coverage)
- 3 LLM calls → 1 for dependency/classifier/linked
- Verify step after decomposition
- ReAct loop at orchestrator level
- Target: 40% latency reduction, 20% accuracy improvement vs Phase 1 baseline

---

## Phase 3: Ground Truth as Runtime RAG (Week 7-10)

> **Hypothesis:** Using historical labeled segment definitions as dynamic few-shot context at inference time will improve LLM output quality by grounding the model in proven examples.

### P3.1 — Build Ground Truth Vector Store (3-4 days)

**Action:**
- Embed all 46 ground truth segment descriptions using the same embedding model
- Store in a separate Milvus collection: `GT_SEGMENTS_<TENANT_ID>`
- At inference: query this collection with the user's segment description to retrieve top-3 most similar historical examples

### P3.2 — Inject Similar Historical Examples at FVOM (2-3 days)

**Action:** The FVOM prompt already has `{similar_historical_segment_examples_dict}`. Make this dynamic:

```python
# In facet_value_mapper_agent.py
similar_examples = gt_retriever.get_similar(
    query=user_description,
    tenant_id=tool_context.state[TENANT_ID],
    top_k=3
)
```

**Expected impact:** 10-15% improvement on segments similar to historical examples.

### P3.3 — Expand Ground Truth to 200+ Rows (Ongoing)

**Action:**
- Review all eval run outputs in `evaluations/data/raw_outputs/`
- Extract high-confidence correct predictions as additional ground truth rows
- Target: 200 rows minimum for statistically reliable evals (±5% at 95% CI)

---

## Phase 4: Multi-Tenant Architecture (Week 8-12)

> **Hypothesis:** Tenant isolation via runtime config loading (not code changes) enables onboarding new tenants in hours, not weeks.

### P4.1 — Tenant Config Manifest (3-4 days)

**Action:** Create `tenants/<tenant_id>/config.yaml`:

```yaml
tenant_id: "walmart_email_mobile"
facet_catalog_key: "email_mobile"
milvus_name_collection: "SEGMENT_AI_EMAIL_MOBILE_FACET_NAME_..."
milvus_value_collection: "SEGMENT_AI_EMAIL_MOBILE_FACET_VALUE_..."
contextual_info:
  refinements: "tenants/walmart_email_mobile/refinements.txt"
  catalog_description: "tenants/walmart_email_mobile/catalog.txt"
  decomposer_hints: "tenants/walmart_email_mobile/decomposer_hints.txt"
  fvom_hints: "tenants/walmart_email_mobile/fvom_hints.txt"
  channel_date_map: "tenants/walmart_email_mobile/channel_date_map.json"
  attribute_mapping: "tenants/walmart_email_mobile/attribute_mapping.json"
ground_truth:
  path: "tenants/walmart_email_mobile/ground_truth.csv"
embedding_model: "BGE"
alias_table: "tenants/walmart_email_mobile/aliases.json"
```

### P4.2 — Runtime Tenant Loader (3-4 days)

**Action:**
- Replace all static file loads (`open(os.getcwd()+'/.../refinements.txt')`) with `TenantLoader.get(tenant_id, 'refinements')`
- `TenantLoader` reads the tenant config at request start (or caches it)
- No code changes needed to add a new tenant

### P4.3 — Tenant-Scoped Milvus Partitions (1 week)

**Decision:** Use partitions within shared collections (not separate collections per tenant) until >5 tenants.

**Rationale:**
- Shared collections: lower operational cost, easy to manage at 2-5 tenants
- Per-tenant partition: provides data isolation with metadata filtering (`tenant_id == "xxx"`)
- Per-tenant collection: use only when tenant catalog size >1000 facets or compliance requires isolation

### P4.4 — Tenant Onboarding Playbook (1 week)

**New tenant checklist:**
1. Create `tenants/<new_tenant_id>/config.yaml`
2. Export facet catalog to standard pickle format
3. Build Milvus partition with BGE embeddings for facet names + values
4. Create contextual info files (LLM-generated from catalog + 10 labeled examples)
5. Add alias table for any name variants
6. Import minimum 50 labeled ground truth rows
7. Run eval baseline — must achieve >60% recall before going live

---

## Phase 5: Auto-Improvement (Week 12-16)

> **Hypothesis:** Closing the feedback loop from production predictions to eval dataset enables continuous quality improvement without manual intervention.

### P5.1 — Production Feedback Collection (1 week)

**Action:**
- Log every segment definition output with the user's original description
- Track user edits (via DirectSegmentEditorAgent calls) as implicit feedback
- If a user immediately edits a freshly created segment → that's a quality signal

### P5.2 — Automated Eval on Every Prompt Change (3-4 days)

**Action:**
- Git hook or CI pipeline trigger: on any prompt file change, run full eval suite
- Block merge if eval score drops >5% vs baseline
- Alert on regressions with specific failing test cases

### P5.3 — Prompt Optimization Loop (2 weeks)

**Action:**
- Use DSPy or similar prompt optimization framework
- Optimize on the ground truth dataset
- Generate new prompt variants, eval each, accept only if improvement confirmed

---

## Parallel Execution Tracks

These tracks can run simultaneously after Phase 0:

| Track | Phases | Owner Role | Dependency |
|---|---|---|---|
| **Retrieval** | P1.1→P1.5 | ML Engineer | P0 baseline |
| **Pipeline Compression** | P2.1→P2.4 | Backend Engineer | P0 baseline |
| **Ground Truth Expansion** | P3.3 | Data/Eval Engineer | P0 complete |
| **Tenant Config** | P4.1→P4.2 | Platform Engineer | P0 complete |
| **Observability** | Phoenix depth | DevOps | P0.4 |

---

## Risk Registry

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Hybrid search reduces quality for some query types | Medium | High | A/B test on 20% traffic with easy rollback |
| Collapsing stages 5-7 misses edge cases | Medium | Medium | Run comprehensive eval comparison before cutover |
| Rule-based date parser misses Walmart-specific patterns | Low | Medium | Keep LLM fallback; monitor cases where rules fail |
| Multi-tenant data leakage via shared Milvus collection | Low | Critical | Strict `tenant_id` metadata filter + integration test |
| Ground truth too small for reliable Phase 1 measurement | High | High | Parallelize ground truth expansion (P3.3) with P1 |

---

## Success Metrics

| Phase | Target Metric | Baseline | Target |
|---|---|---|---|
| Phase 0 | Eval baseline established | Unknown | 46-row dataset fully evaluated |
| Phase 1 | Facet recall improvement | Current (measure at P0) | +15% recall |
| Phase 2 | Latency reduction | Measure at P0 | -40% P95 latency |
| Phase 3 | Few-shot similarity accuracy | Measure at P2 | +10% on similar queries |
| Phase 4 | Tenant onboarding time | Days-weeks | <4 hours |
| Phase 5 | Regression detection | Manual only | 100% automated |
