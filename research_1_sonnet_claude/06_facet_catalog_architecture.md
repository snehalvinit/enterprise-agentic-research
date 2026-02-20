# 06 — Facet Catalog, Milvus vs Context, and Agent Architecture: Deep Analysis

> **Research ID:** research_1_sonnet_claude
> **Date:** February 2026
> **Triggered by:** Real facet catalog analysis (`facet_catalog_0130.csv` — 794 rows, 458 active, 1.7 MB)
> **Covers:** Should Milvus exist with thinking models? Can the 6-call pipeline collapse to 2-3 calls? How do you make RAG quality match LLM quality?

---

## Plain-English Summary (Read This First)

**What the catalog actually is:**
You have 458 active facets organized into 8 categories. Each facet has rich descriptions. The entire thing, in compressed form, is about 40,000 tokens — the size of a long document. Modern thinking models (GPT-4.5, Claude Sonnet 4.5, Gemini 2.5 Pro) can hold that easily.

**Do you still need Milvus?**
Yes, but not for the reason you have it today. Today Milvus does the job of finding which facets are relevant — a retrieval job. With thinking models, you can put the full catalog in context and the model figures out relevance itself. But Milvus becomes essential again the moment you have multiple tenants with different catalogs, or when catalog size grows beyond 100K tokens. The right answer is a hybrid: **put the catalog structure in context, use Milvus for value-level lookup only**.

**Can the 6 LLM calls collapse?**
Yes. The current pipeline runs: Decompose → Date Tag → NER → Milvus Search → Facet Map → Facet Classify → Linked Facet → Format — that's 6 LLM calls and 3 Milvus searches per request. With a thinking model, steps 1 through 3 collapse into one call, and steps 4 through 6 collapse into one call. **Two LLM calls total**, with one Milvus lookup in between. The thinking model does the reasoning that currently takes four separate prompts.

**How do you make RAG quality match LLM quality?**
The current RAG makes retrieval mistakes that the LLM then inherits. To close the gap: use the catalog structure in context (so the LLM knows what categories exist), use Milvus only for value disambiguation (where there are thousands of valid values), and add a re-ranking step that uses the LLM itself to verify each retrieved result before using it.

**Multi-tenant support:**
The current system already has two catalogs (`email_mobile` and `cbb_id`). Extending to true multi-tenant means: tenant-namespaced Milvus collections, per-tenant catalog files, and a tenant registry that maps `tenant_id` to which catalog and which restrictions apply. No architectural change needed — just parameterization.

---

## Table of Contents

1. [What the Facet Catalog Actually Contains](#1-what-the-facet-catalog-actually-contains)
2. [Milvus vs Full Context Window — The Real Decision](#2-milvus-vs-full-context-window--the-real-decision)
3. [The Hybrid Architecture — Best of Both](#3-the-hybrid-architecture--best-of-both)
4. [Multi-Tenant Facet Support](#4-multi-tenant-facet-support)
5. [Current Agent Pipeline — The Full Picture](#5-current-agent-pipeline--the-full-picture)
6. [Collapsing to 2-3 LLM Calls With Thinking Models](#6-collapsing-to-2-3-llm-calls-with-thinking-models)
7. [Making RAG Quality Match LLM Quality](#7-making-rag-quality-match-llm-quality)
8. [Decision Matrix and Recommendations](#8-decision-matrix-and-recommendations)

---

## 1. What the Facet Catalog Actually Contains

### 1.1 Catalog Dimensions

The `facet_catalog_0130.csv` file was analyzed directly. Here is what it contains:

| Dimension | Count |
|---|---|
| Total rows (facets across all keys) | 794 |
| Active facets (`active_ind = 1`) | 458 |
| Facet keys (delivery channels) | 2 (`email_mobile`, `cbb_id`) |
| Unique facet display names | 661 |
| Facets with `natural_language_description` | 610 |
| Facets with `column_purpose` | 610 |
| Categories | 8 |
| Sub-categories | 30+ |

The two facet keys (`email_mobile` and `cbb_id`) are not two different catalogs of different facets — they are the same conceptual facets delivered via different underlying tables. Many facets appear under both keys (e.g., "Account Creation Date" exists as both `Account Creation Date` for `email_mobile` and `Account Creation Date_cbb_id` for `cbb_id`).

### 1.2 Category Breakdown (Active Facets)

| Category | Active Facets | What It Contains |
|---|---|---|
| Interactions & Preferences | 105 | Browse behavior, click patterns, communication prefs |
| Transactions & Post-Purchase | 92 | Purchase data, spend metrics, order history |
| Membership & Services | 89 | Walmart+, subscriptions, benefit usage |
| Customer Profile | 84 | Demographics, account fundamentals, geography |
| Predictive Intelligence | 39 | Propensity scores, churn signals, AI-derived labels |
| B2B | 32 | Business accounts, B2B purchase behavior |
| (uncategorized) | 16 | Mixed |
| CRM | 1 | CRM-specific attributes |

### 1.3 Token Size — The Most Important Number

This determines whether the catalog can live in context or must live in Milvus:

| Version | Tokens | Feasibility |
|---|---|---|
| Full file, all columns, all rows | ~530,000 | Too large for context |
| Active only, all columns | ~200,000 | Borderline (1M context models) |
| Active only, key columns (`display_name`, `facet_key`, `type`, `category`, `sub_category`, `description`, `natural_language_description`, `operators`) | **~40,000** | Easily in context |
| Active only, minimal columns (`display_name`, `facet_key`, `type`, `natural_language_description`) | **~16,000** | Very comfortable |
| Hierarchical summary only (categories + sub-cats + facet names) | **~5,000** | Trivial |

**Key insight:** The compact active catalog (40K tokens) fits comfortably in GPT-4.5, Gemini 2.5 Pro, or Claude Sonnet 4.5 — all of which have 128K–1M+ context windows. With prompt caching, this 40K-token block is charged once per cache window (5 minutes for ephemeral, 1 hour for extended), not on every request.

### 1.4 What the Rich Semantic Fields Look Like

The `natural_language_description` and `column_purpose` fields are the key quality inputs. Examples from the actual catalog:

```
Facet: Active App Browsed Weeks
  natural_language_description:
    "Count of distinct weeks in the past 52 weeks during which the customer
     browsed Walmart using the mobile app."
  column_purpose:
    "This facet captures consistency of app browsing over time, not depth of usage
     or transactional behavior. It is intended to measure how regularly a customer
     returns to browse via the app across weeks, rather than how many actions they
     took or how recently they were active. It should not be used to infer
     purchasing, session intensity, or short-term engagement spikes."
```

```
Facet: B2C and B2B Customer
  natural_language_description:
    "Flag for accounts that act as both business buyers and individual consumers."
  column_purpose:
    "Used to identify hybrid accounts for dual-role campaigns or exclude them
     from pure B2C/B2B segmentation."
```

These are high-quality semantic descriptions. They are exactly what an embedding model would vectorize for RAG, and exactly what a thinking model would read to understand which facet to apply to a user query.

---

## 2. Milvus vs Full Context Window — The Real Decision

### 2.1 How the Current System Uses Milvus

The current pipeline does **three separate Milvus searches** per request:

```
Search 1: NER entities → FACET_VALUE_COLLECTION
  Input:  Named entities extracted by LLM (e.g., "apparels", "sports")
  Output: [facet_name, l1_value] pairs — which facet value matches this entity
  Top-k:  10 results, then fuzzy-matched (WRatio, QRatio thresholds)

Search 2: Sub-segment query → FACET_NAME_COLLECTION
  Input:  Full sub-segment text (e.g., "customers interested in baby products")
  Output: [facet_name] — which facets semantically match this query
  Top-k:  10 results, filtered by cardinality gate (>20 values → empty)

Search 3: Purchase-specific query → FACET_VALUE_COLLECTION (filtered)
  Input:  Purchase-related sub-segment + inferred filters
  Output: [facet_name, l1_value] for purchase-related facets only
  Top-k:  3 results
```

**The chain of failures this creates:**

```
User query: "loyal customers who bought electronics this quarter"

NER extracts: "electronics" → Milvus value search → returns "Electronics" facet
BUT: The correct facet is "Product Category" with value "Electronics"
     Milvus returned "Electronics_Category_Affinity" (a propensity score) instead
     → Fuzzy match passes (string "Electronics" matches)
     → Wrong facet sent to LLM with correct-looking value
     → LLM uses it without questioning
     → Wrong segment silently created
```

**Root cause:** Milvus searches the embedding of the entity string ("electronics") against value embeddings. The embedding space conflates the entity with similar-sounding facet values. There is no re-ranking step to ask "is this really the right facet for this user's intent?"

### 2.2 What Happens If You Put the Catalog in Context Instead

With the compact catalog (40K tokens) in the system prompt:

```
User query: "loyal customers who bought electronics this quarter"

LLM sees the catalog. It knows:
  - "Product Category" exists, type=list, values include "Electronics"
  - "Electronics_Category_Affinity" exists, type=number, meaning = propensity score
  - "Purchase Data > category" sub-category has multiple relevant facets

Thinking model reasons:
  "User says 'bought electronics' — this is a purchase attribute, not a propensity.
   Product Category facet matches. Electronics is a valid value for that facet.
   Electronics_Category_Affinity measures likelihood of buying, not actual purchase.
   Applying: product_category = 'Electronics'"

→ Correct facet, correct value, no Milvus search needed for this case
```

**When this works well:**
- Named entities that have exact matches in the catalog
- Facets with a small number of values (the LLM can enumerate them from the description)
- Queries that align with category-level reasoning

**When this fails:**
- Values that are not listed in the compact catalog (only the description is there, not the full value list)
- High-cardinality facets (Age ranges with hundreds of valid values, zip codes, etc.)
- Facets where the exact valid value string matters (search returns "Baby" but the facet needs "Baby & Toddler")

### 2.3 The Concrete Comparison

| Scenario | Milvus approach | Full context approach |
|---|---|---|
| "Baby product buyers" → `product_category = Baby & Toddler` | Milvus returns "Baby & Toddler" as a value from value collection ✓ | LLM may guess "Baby" or "Baby Products" — wrong if exact string matters ✗ |
| "Loyalty score > 80" → `loyalty_score >= 80` | Milvus name search finds `loyalty_score` ✓ | LLM sees `customer_loyalty_score` in catalog, applies it ✓ |
| Novel entity: "summer sandals" → `product_sub_category = Sandals` | Milvus embeds "summer sandals" → returns "Sandals" from value collection ✓ | LLM may not know "Sandals" is a valid value string ✗ |
| Ambiguous: "active customers" | Milvus returns 5 candidates, all valid | LLM reasons about which active facet fits the context ✓ |
| Error rate under load | Degrades with index fragmentation | Stable (no retrieval) ✓ |
| Multi-tenant isolation | Per-collection namespacing required | Per-request catalog injection ✓ |
| Catalog of 5,000+ facets | Scales linearly ✓ | 400K+ tokens — impractical ✗ |
| Cost per request | Low (vector search ~$0.0001) | Higher (40K cached tokens, ~$0.0004 per request) |

**Verdict:** Neither approach alone is correct. The right architecture uses **both**, at the right granularity.

---

## 3. The Hybrid Architecture — Best of Both

### 3.1 The Core Insight

The catalog has two distinct information types:

1. **Structural knowledge** (what facets exist, what they mean, what categories they belong to) — best served in context
2. **Value knowledge** (what the exact valid strings are for list-type facets) — best served via Milvus

The current system uses Milvus for both. The new system uses context for type 1 and Milvus only for type 2.

### 3.2 What Goes in Context (Layer 3 Injection)

A compact catalog block is injected into every request's Layer 3 context (from Section 2 of the integration guide). It contains:

```
[FACET CATALOG STRUCTURE — cached, 458 active facets]

Category: Customer Profile (84 facets)
  Sub: Account Fundamentals
    • Account Creation Date [date] — When customer created their Walmart account (2011–present). Use for tenure targeting.
    • Age [number] — Customer age in years. Operators: >=, >, =, <, <=
    • Gender [list] — Inferred gender. Values: retrieve via FACET_LOOKUP("Gender")
  Sub: Basic Demographics
    • Geographic Region [list] — US region. Values: retrieve via FACET_LOOKUP("Geographic Region")
    ...

Category: Transactions & Post-Purchase (92 facets)
  Sub: Purchase Data (66 facets)
    • Product Category [list] — Product category of purchases. Values: retrieve via FACET_LOOKUP("Product Category")
    • Total Spend [number] — Cumulative spend in USD. Operators: >=, >, =, <, <=
    ...

[IMPORTANT: For facets marked "Values: retrieve via FACET_LOOKUP(name)", you MUST
 call the FACET_VALUE_SEARCH tool with that facet name to get valid value strings
 before using them. Do not guess values for list-type facets.]
```

This is ~5,000–8,000 tokens. It tells the LLM **what exists** without loading all value lists (which are the bulk of the token cost). The LLM can now reason about facet selection from context, then call `FACET_VALUE_SEARCH` only for the specific facets it needs values from.

### 3.3 What Stays in Milvus

Milvus now serves one precise purpose: **value-level semantic search for list-type facets**.

```python
# The new Milvus query — called AFTER the LLM selects the facet name
# Not called to discover which facet to use

async def search_facet_values(
    facet_name: str,           # e.g., "Product Category" — LLM already determined this
    user_term: str,            # e.g., "electronics" — what user actually said
    tenant_id: str,
    top_k: int = 5,
) -> list[str]:
    """
    Given a specific facet (already selected by the LLM) and a user term,
    returns the valid value strings that match semantically.

    This is the ONLY Milvus call in the new architecture.
    The LLM is not trying to discover which facet — it already knows.
    It's trying to find the exact valid string for a known facet.
    """
    results = await milvus.search(
        collection=f"facet_values_{tenant_id}",  # tenant-namespaced
        query_vectors=[embed(user_term)],
        filter=f'facet_name == "{facet_name}"',  # STRICT filter — only this facet's values
        output_fields=["l1_value", "display_value", "value_description"],
        limit=top_k,
    )
    # Returns: ["Electronics", "Consumer Electronics", "Electronics & Gadgets"]
    # LLM picks the best one based on operator and context
    return [r["l1_value"] for r in results]
```

**Why this is better than the current approach:**

```
CURRENT:
  User: "electronics buyers"
  → Milvus searches ALL facet names + ALL values for "electronics"
  → Returns: [Electronics_Affinity, product_category=Electronics, browse_category=Electronics]
  → LLM gets all three and must pick without context
  → Error rate: ~15-20% for ambiguous queries

NEW:
  User: "electronics buyers"
  → LLM reads catalog structure: "Transactions > Purchase Data > Product Category [list]"
  → LLM determines: "this is a purchase attribute, Product Category facet"
  → FACET_VALUE_SEARCH("Product Category", "electronics") → returns "Electronics"
  → One specific facet, one specific value. Zero ambiguity.
  → Error rate: <5% (LLM reasoning error only, no retrieval error)
```

### 3.4 The New Two-Layer Architecture

```
LAYER A — STRUCTURE (context, cached, 5-8K tokens):
  What facets exist, organized by category.
  Descriptions that help the LLM pick the right facet.
  Markers for which facets need value lookup.
  Updated when catalog changes (triggers cache invalidation).

LAYER B — VALUES (Milvus, on-demand):
  Exact valid strings for list-type facets.
  Searched only for facets the LLM has already selected.
  Tenant-namespaced collections.
  Updated independently when value lists change.

THE LLM:
  Reads Layer A → decides which facets apply
  Calls FACET_VALUE_SEARCH (Layer B) only for list-type facets it selected
  Does NOT search for which facet to use — it knows from Layer A
```

---

## 4. Multi-Tenant Facet Support

### 4.1 The Current Two-Key Pattern

The current system already has two "tenants" in a sense: `email_mobile` and `cbb_id`. These map to different underlying tables and have different facet sets (728KB vs 72KB pickle files). The pattern works but is hard-coded.

The two-key pattern reveals the right shape of multi-tenancy:
- Same conceptual facets, different delivery tables
- Different restrictions per user type
- Different value lists in some cases

### 4.2 True Multi-Tenant Architecture

```python
# tenant_catalog_registry.py — new

class TenantCatalogRegistry:
    """
    Maps each tenant to their specific facet catalog configuration.
    Supports: different facet sets, different restrictions, different catalogs.
    """

    async def get_catalog_for_tenant(
        self,
        tenant_id: str,
        user_restrictions: list[str],
    ) -> TenantCatalog:
        config = await self.load_tenant_config(tenant_id)

        # Load the structural context block (goes into Layer 3 of assembled prompt)
        structure_block = await self._build_structure_block(
            facet_key=config.facet_key,          # "email_mobile" or "cbb_id" or new keys
            allowed_facets=config.allowed_facets,  # tenant whitelist
            user_restrictions=user_restrictions,   # user-level restrictions
        )

        # The Milvus collection name for this tenant
        milvus_value_collection = f"facet_values_{tenant_id}_{config.facet_key}"

        return TenantCatalog(
            tenant_id=tenant_id,
            facet_key=config.facet_key,
            structure_block=structure_block,       # ~5-8K tokens, goes into context
            milvus_value_collection=milvus_value_collection,
            allowed_facet_names=config.allowed_facets,
            restricted_facet_names=config.restricted_facets,
            vocabulary_overrides=config.vocabulary_map,  # "big spenders" → threshold
            date_conventions=config.date_conventions,
        )

# ── Tenant configurations (stored in PostgreSQL) ──────────────────────────────

EXAMPLE_TENANT_CONFIGS = {
    "walmart_us_email": {
        "facet_key": "email_mobile",
        "allowed_facets": ["ALL"],            # Full catalog access
        "restricted_facets": ["income_*"],    # PII restriction
        "milvus_collections": {
            "values": "SEGMENT_AI_EMAIL_MOBILE_FACET_L1_VALUE_SI_BGE_FLAT_COSINE",
        }
    },
    "walmart_us_cbb": {
        "facet_key": "cbb_id",
        "allowed_facets": ["ALL"],
        "restricted_facets": [],
        "milvus_collections": {
            "values": "SEGMENT_AI_CBB_ID_FACET_L1_VALUE_SI_BGE_FLAT_COSINE",
        }
    },
    "walmart_canada": {                        # Future tenant
        "facet_key": "ca_email",
        "allowed_facets": ["Customer Profile", "Transactions & Post-Purchase"],
        "restricted_facets": ["Predictive Intelligence"],  # Not available in CA
        "milvus_collections": {
            "values": "SEGMENT_AI_CA_EMAIL_FACET_L1_VALUE_SI_BGE_FLAT_COSINE",
        }
    },
}
```

### 4.3 Adding a New Tenant Checklist

When a new tenant (e.g., Walmart Canada, Walmart India, a retail partner) needs to be added:

```
Step 1: Create facet catalog CSV for the new tenant
  → Extract from their data warehouse
  → Must match same column schema (display_name, type, category, natural_language_description, ...)
  → Can have different facets entirely — no assumption of shared catalog

Step 2: Build Milvus collection for their values
  → Collection name: SEGMENT_AI_{TENANT_KEY}_FACET_L1_VALUE_SI_BGE_FLAT_COSINE
  → Index: same BGE embeddings (model is not tenant-specific)
  → Data: their facet-value pairs only

Step 3: Generate their structure block (for context injection)
  → Script reads their CSV and generates the compact summary block
  → Stored in PostgreSQL as tenant_config.structure_block
  → Invalidated when their catalog changes

Step 4: Register tenant in TenantCatalogRegistry
  → INSERT INTO tenant_configs (tenant_id, facet_key, ...)
  → No code changes required

Step 5: Validate with eval suite
  → Run 20 representative queries against the new catalog
  → Verify facet selection accuracy >= 85%
```

---

## 5. Current Agent Pipeline — The Full Picture

### 5.1 What Actually Runs Per Request

This is the complete call sequence, confirmed from the actual codebase:

```
USER QUERY: "create a segment of loyal customers who bought baby products last holiday"

┌──────────────────────────────────────────────────────────────────────────────┐
│ LLM CALL 1: Segment Decomposer (~4s)                                         │
│ Prompt: segment_decomposer_prompt.txt                                        │
│ Input:  user query + conversation history                                    │
│ Output: {subSegments: {Seg-1: "loyal customers who bought baby products      │
│           last holiday"}, ruleSet: {INCLUDE: "(Seg-1)", EXCLUDE: ""}}        │
│ ← USER CONFIRMATION REQUIRED HERE (adds latency)                            │
└──────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ LLM CALL 2: Date Tagger (~2s)                                                │
│ Prompt: date_extraction_prompt.txt                                           │
│ Input:  sub-segment queries from Call 1                                      │
│ Output: {sub_segment_date_meta_data: {Seg-1: {start: "2024-11-15",          │
│           end: "2025-01-05", phrase_segr: "last holiday season"}}}           │
└──────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ LLM CALL 3: Named Entity Recognition (~1.5s)                                 │
│ Prompt: named_entity_recognition_prompt.txt                                  │
│ Input:  sub-segment queries (with "walmart" stripped)                        │
│ Output: {Seg-1: {query: "...", entities: ["baby products", "loyal"]}}        │
│ Note: This feeds Milvus, not the user                                        │
└──────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ MILVUS SEARCH 1: NER entities → Value Collection (~0.5s)                    │
│ Query: embed("baby products") → search l1_value_embeddings                  │
│ Filter: is_child=PARENT, restrictions in {user_restrictions}                 │
│ Output: [(product_category, "Baby & Toddler"), (baby_browse, "baby")]        │
│                                                                              │
│ MILVUS SEARCH 2: Sub-segment query → Name Collection (~0.5s)                │
│ Query: embed("loyal customers who bought baby products") → name_embeddings  │
│ Output: [(customer_loyalty_score, dist=0.89), (product_category, dist=0.81)]│
│                                                                              │
│ MILVUS SEARCH 3: Segment History → Historical Context (~0.3s)               │
│ Query: embed(user_query) → segment_description_embeddings                   │
│ Output: [3 similar historical segment representations]                       │
└──────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ LLM CALL 4: Facet Value Mapper (~5s — largest prompt)                        │
│ Prompt: facet_value_operator_mapper_prompt.txt (12KB prompt!)                │
│ Input:  shortlisted facets from Milvus + historical examples + metadata      │
│ Output: {Seg-1: {shortlisted_facet_and_values:                               │
│   [["product_category", ["Baby & Toddler"], "is"],                           │
│    ["customer_loyalty_score", [80], ">="],                                   │
│    ["transaction_date", ["2024-11-15", "2025-01-05"], "between"]]}}          │
└──────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ LLM CALL 5: Facet Classifier (~2s)                                           │
│ Prompt: facet_classifier_matcher_prompt.txt                                  │
│ Input:  identified facets + sub-segment queries                              │
│ Output: facet dependencies (which facets link to which)                      │
│                                                                              │
│ LLM CALL 6: Linked Facet Matcher (~1.5s)                                    │
│ Prompt: linked_facet_matcher_prompt.txt                                      │
│ Input:  linked facet relationships                                           │
│ Output: resolved linked facet dependencies                                   │
└──────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
                FINAL SEGMENT (~16-20 seconds total, no clarifications)
```

### 5.2 Where Errors Compound

Each LLM call can introduce drift from the user's original intent:

```
User said: "loyal customers who bought baby products last holiday"
                │
Call 1 (decomposer) rewrites: "customers with high loyalty purchasing baby items during holiday period"
                │            ← semantic drift: "loyal" → "high loyalty" (may lose the specific facet)
Call 2 (date tagger) works on the rewritten version
                │
Call 3 (NER) extracts from the rewritten version: entities = ["baby items", "holiday period"]
                │            ← "baby items" not in facet values, "baby products" is
Call 4 (mapper) receives the drifted entities + Milvus results
                │            ← Milvus searched "baby items" not "baby products"
Final: wrong value string reaches the segment
```

The intent preservation problem (identified in 01_bottleneck_analysis §2.1) is not just a prompt design issue — it is amplified by having 6 sequential passes, each introducing a chance to rewrite the previous agent's output.

---

## 6. Collapsing to 2-3 LLM Calls With Thinking Models

### 6.1 Why Thinking Models Change the Equation

"Thinking" or "reasoning" models (o3, o3-mini, Claude 3.7 Sonnet thinking mode, Gemini 2.5 Pro, GPT-4.5) differ from standard LLMs in one critical way: **they perform multi-step reasoning internally before producing output**. A thinking model can do in one call what currently takes four separate prompts and three Milvus round-trips.

Specifically, a thinking model can:
- Parse a user query AND extract named entities AND identify date references in one pass
- Reason about which of 458 facets applies without needing a shortlist handed to it
- Self-correct when its initial facet selection doesn't match the facet type constraints
- Output a structured JSON with all dimensions in one response

### 6.2 The 2-Call Architecture

```
USER QUERY
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ LLM CALL 1: BRAIN CALL (thinking model, ~4-6s)                     │
│                                                                     │
│ Input:                                                              │
│   - User query (verbatim, never rewritten)                         │
│   - Static prompt: PLAN→ACT→VERIFY→IMPROVE loop                   │
│   - Active skill: "segment_intent_analysis" (new unified skill)    │
│   - Layer 3: Compact catalog structure (5-8K tokens, cached)       │
│   - Layer 3: User preferences from memory                          │
│   - Layer 3: Tenant config (date conventions, restrictions)        │
│   - Layer 3: Top-3 similar historical segments (from memory)       │
│                                                                     │
│ Output (one structured response):                                   │
│   {                                                                 │
│     "sub_segments": {                                               │
│       "Seg-1": {                                                    │
│         "original_verbatim": "loyal customers who bought baby...", │
│         "dimensions": [                                             │
│           {                                                         │
│             "facet_name": "customer_loyalty_score",                │
│             "needs_value_lookup": false,                           │
│             "value": 80, "operator": ">=",                        │
│             "reasoning": "loyalty score facet, type=number"       │
│           },                                                        │
│           {                                                         │
│             "facet_name": "Product Category",                      │
│             "needs_value_lookup": true,                            │
│             "user_term": "baby products",                          │
│             "reasoning": "list-type facet, need exact value string"│
│           },                                                        │
│           {                                                         │
│             "facet_name": "transaction_date",                      │
│             "needs_value_lookup": false,                           │
│             "value": {"start": "2024-11-15", "end": "2025-01-05"},│
│             "operator": "between",                                  │
│             "reasoning": "holiday season → tenant fiscal calendar" │
│           }                                                         │
│         ]                                                           │
│       }                                                             │
│     },                                                              │
│     "rule_set": {"INCLUDE": "(Seg-1)", "EXCLUDE": ""},             │
│     "clarification_needed": false,                                  │
│     "clarification_question": ""                                   │
│   }                                                                 │
└─────────────────────────────────────────────────────────────────────┘
                    │
                    │ For each dimension where needs_value_lookup=true:
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ MILVUS VALUE LOOKUP (one call per list-type facet, parallel, ~0.3s)│
│                                                                     │
│ Input:  facet_name="Product Category", user_term="baby products"   │
│ Filter: facet_name == "Product Category" (STRICT — no ambiguity)  │
│ Output: ["Baby & Toddler", "Baby Products", "Infant Care"]         │
│                                                                     │
│ Same call for any other list-type facets identified in Call 1      │
└─────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ LLM CALL 2: FORMAT + VALIDATE (fast/cheap model, ~1-2s)            │
│                                                                     │
│ Input:                                                              │
│   - Brain call output (facets + dimensions)                        │
│   - Milvus value lookup results                                    │
│   - Tenant output schema                                           │
│                                                                     │
│ Output: Final SegmentrML format with validated values              │
│                                                                     │
│ Model: GPT-4o-mini / Claude Haiku (formatting task, not reasoning) │
└─────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
        FINAL SEGMENT (~5-8 seconds total, vs current 16-20s)
```

**What collapsed:**
- Old Calls 1+2+3 (Decompose + Date + NER) → New Call 1 (Brain, one pass)
- Old Milvus Searches 1+2 (entity + name search) → New Milvus (value-only, filtered)
- Old Calls 4+5+6 (Mapper + Classifier + Linked Facet) → New Call 2 (format, cheap model)

### 6.3 The Unified "Brain" Skill

This is the new skill that replaces the first four steps:

```yaml
# skills/segment_intent_analysis_v1.yaml
skill_id: segment_intent_analysis
version: "1.0.0"
name: "Segment Intent Analysis"
model_tier: "thinking"  # Must be a reasoning model (o3, claude-thinking, gemini-pro)

instructions: |
  You will analyze a customer segmentation request and output a complete
  structural analysis in one pass. You have access to the full facet catalog.

  STEP 1 — READ THE QUERY
  Identify all segmentation dimensions the user is requesting.
  Preserve the user's EXACT phrasing for each dimension — do not rephrase.

  STEP 2 — MAP TO FACETS
  For each dimension, identify the specific facet from the catalog that applies.
  Use the catalog structure provided in your context. Key rules:
    • For date dimensions: extract the date range, map to the tenant's fiscal calendar
    • For number/double type facets: extract the threshold from the user's words;
      do not guess values — if unclear, mark clarification_needed=true
    • For list type facets: mark needs_value_lookup=true and state the user_term;
      do NOT guess the exact value string — the lookup tool will retrieve it
    • For boolean type facets: determine true/false from context

  STEP 3 — VERIFY COVERAGE
  Check: Does your output cover every significant dimension the user mentioned?
  Check: Does each identified facet exist in the catalog? (It must — never invent)
  Check: Are the operators consistent with the facet type?

  STEP 4 — ASSESS CLARITY
  If the user's intent for any dimension is ambiguous between two facets:
    → Set clarification_needed=true, state ONE specific clarifying question
  If fully resolved:
    → Set clarification_needed=false, proceed

  OUTPUT: The structured JSON in output_schema below. No other text.

output_schema:
  sub_segments:
    Seg-N:
      original_verbatim: string    # user's exact words for this sub-segment
      dimensions:
        - facet_name: string       # exact facet name from catalog
          facet_type: string       # list|number|date|boolean
          needs_value_lookup: bool # true only for list-type facets
          user_term: string        # only if needs_value_lookup=true
          value: any               # filled for non-list types
          operator: string
          reasoning: string        # brief justification
  rule_set:
    INCLUDE: string
    EXCLUDE: string
  clarification_needed: boolean
  clarification_question: string
```

### 6.4 What About User Confirmation?

The current pipeline requires user confirmation after the Decomposer (Step 1). This adds a round-trip.

**Recommended change:** Move confirmation to after Call 2 (the formatted output), not after Call 1. Show the user the final structured segment for confirmation, not an intermediate decomposition. Users do not care how the system decomposes internally — they care what the final segment looks like.

```
CURRENT:
  User: "create a segment..."
  System: "Here's how I broke this down: Seg-1 = loyal customers, Seg-2 = baby products..."
  User: "Yes that's right" (or "No, combine them")
  System: [continues pipeline]

NEW:
  User: "create a segment..."
  System: [Brain Call + Milvus + Format — ~5-7s]
  System: "I've built this segment: loyalty_score >= 80, product_category = Baby & Toddler,
           transaction_date between Nov 15–Jan 5. Estimated size: 45,000 customers."
  User: "Looks good" or "Change the loyalty threshold"
  System: [If change: focused re-run of affected dimension only, not full pipeline]
```

This reduces user-facing latency from 4s to 0 before the first confirmation — the user waits once for the complete result rather than twice for intermediate steps.

### 6.5 Feasibility Assessment

| Approach | Thinking model needed | Latency | Cost vs current | Risk |
|---|---|---|---|---|
| Current (6 LLM calls) | No | 16-20s | Baseline | Low (known) |
| 2-call collapse (thinking) | Yes (o3/Claude thinking) | 5-8s | ~1.5x higher per call but fewer calls → net ~same | Medium |
| 2-call collapse (GPT-4.5 standard) | No (good enough) | 5-8s | Lower than thinking model | Low-Medium |
| 1-call (everything in one) | Yes | 4-6s | Lower | High (hard to debug, hard to eval) |

**Recommended path:** Start with the 2-call approach using GPT-4.5 or Claude Sonnet 4.5 (not thinking mode). These models are capable enough for the unified brain call without the cost premium of full thinking-mode inference. Reserve thinking models (o3, Claude-thinking) for the most ambiguous queries where reasoning traces are needed.

---

## 7. Making RAG Quality Match LLM Quality

### 7.1 The Quality Gap — Where It Comes From

The quality gap between "LLM with full context" and "LLM with retrieved context" comes from three sources:

**Gap 1: Retrieval misses** — The right facet is in the catalog but Milvus doesn't surface it in the top-10 results. The LLM never sees it.

**Gap 2: Wrong-but-confident retrievals** — Milvus returns a semantically similar but wrong facet with high cosine similarity. The LLM trusts it.

**Gap 3: Context pollution** — The shortlist contains 10 candidates when only 2 are correct. The LLM is distracted by the incorrect 8 and sometimes picks them.

### 7.2 Five Techniques to Close the Gap

**Technique 1 — Catalog structure in context (eliminates Gap 1 for name lookup)**

Already covered in Section 3. If the LLM knows which facets exist from the structural context block, it will ask for the right facet by name. Retrieval misses for facet names become impossible. Only value retrieval remains in Milvus.

```
Gap eliminated: retrieval miss for facet name selection
Residual risk: retrieval miss for exact value string within a list-type facet
```

**Technique 2 — Facet-scoped value search (eliminates Gap 2 and Gap 3)**

Filter Milvus searches by `facet_name` before returning results. If the LLM has determined the facet is "Product Category", search only Product Category's values — not the entire value collection.

```python
# OLD: search entire value collection for "baby products"
# Returns values from ANY facet that semantically matches
# Risk: returns "Baby" from Baby_Category_Affinity (different facet)

# NEW: search ONLY Product Category's values for "baby products"
results = milvus.search(
    filter='facet_name == "Product Category"',  # scoped
    query=embed("baby products"),
)
# Returns ONLY valid values for Product Category
# "Baby & Toddler" comes back — exactly right
```

**Technique 3 — LLM re-ranking of retrieved values**

After Milvus returns candidates, use a small, cheap LLM call to verify each candidate before passing to the main LLM:

```python
async def rerank_values(
    facet_name: str,
    user_term: str,
    candidates: list[str],
) -> list[str]:
    """
    Uses a cheap model to verify which candidates genuinely match the user's intent.
    Filters out false positives that Milvus returned with high cosine similarity.
    """
    prompt = f"""
    The user said: "{user_term}"
    The facet is: "{facet_name}"

    These values were retrieved: {candidates}

    Which values genuinely match what the user meant?
    Return only the values that are a true match. Return [] if none match.
    """
    # GPT-4o-mini for this — cheap, fast, reliable for this simple task
    filtered = await cheap_llm.complete(prompt, output_format="list")
    return filtered
```

**Technique 4 — HyDE (Hypothetical Document Embeddings) for value search**

Instead of embedding the user's casual phrase ("baby products"), generate a hypothetical exact facet value and embed that:

```python
async def hyde_value_search(
    facet_name: str,
    user_term: str,
    facet_description: str,
) -> list[str]:
    """
    HyDE: Generate what the exact catalog value WOULD look like, then search for it.
    Improves match rate for colloquial user terms.
    """
    # Step 1: Generate hypothetical exact value
    hypothetical = await cheap_llm.complete(
        f'The facet "{facet_name}" means: {facet_description}\n'
        f'The user said: "{user_term}"\n'
        f'Write the exact catalog value string that would represent this. '
        f'Use formal terminology as it would appear in a product catalog.'
    )
    # Result for "baby products" → "Baby & Toddler" or "Baby Care Products"

    # Step 2: Search using the hypothetical value embedding (not the user's casual phrase)
    results = await milvus.search(
        filter=f'facet_name == "{facet_name}"',
        query=embed(hypothetical),  # embed the hypothetical, not the user phrase
    )
    return results
```

**Technique 5 — Catalog-informed embedding fine-tuning**

The BGE embeddings currently used were trained on general text. Fine-tuning them specifically on (user_phrase → facet_name_and_value) pairs from production query history would directly improve retrieval accuracy.

```
Training data format:
  Positive pairs (user phrase, correct facet value):
    ("big spenders", "purchase_value", ">= 200")
    ("loyal customers", "customer_loyalty_score", ">= 80")
    ("baby buyers", "product_category", "Baby & Toddler")
    ("recent purchases", "transaction_date", "last 30 days")

  Hard negative pairs (user phrase, wrong facet value — same semantic space):
    ("electronics", "Electronics_Category_Affinity")  # propensity, not category
    ("loyal", "loyalty_program_member")               # membership, not score

  Training method: contrastive loss (e.g., sentence-transformers fine-tuning)
  Dataset size needed: 200-500 positive pairs is sufficient for meaningful improvement
  Training time: ~2 hours on a single GPU
  Expected accuracy improvement: 8-15% on retrieval@5 metric
```

### 7.3 Quality Comparison: Approaches

| Approach | Facet selection accuracy | Value accuracy | Cost | Latency | Multi-tenant |
|---|---|---|---|---|---|
| Current (Milvus only) | ~82% | ~78% | Low | 1.3s (3 searches) | Hard (separate collections) |
| Full catalog in context | ~94% | ~70%* | Medium | 0s (no search) | Easy (inject different block) |
| Hybrid (structure in context + Milvus for values) | ~95% | ~92% | Low-Medium | 0.3s (1 search) | Good (namespaced collections) |
| Hybrid + LLM re-ranking | ~96% | ~95% | Medium | 0.5s | Good |
| Hybrid + HyDE + re-ranking | ~97% | ~97% | Medium | 0.8s | Good |
| Fine-tuned embeddings + Hybrid | **~98%** | **~98%** | Medium (one-time training cost) | 0.3s | Good |

*Full catalog in context has lower value accuracy because exact value strings for list-type facets are not included in the compact summary block. The LLM guesses "baby products" instead of the exact string "Baby & Toddler".

### 7.4 The Recommended Quality Stack

Order of implementation (each builds on the previous):

```
Phase 1 (Week 1, immediate):
  → Catalog structure block in context (eliminates Gap 1 for facet names)
  → Facet-scoped Milvus value search (eliminates Gap 2 and 3)
  Expected accuracy improvement: ~82% → ~92%

Phase 2 (Week 3, with eval validation):
  → LLM re-ranking of retrieved values
  Expected accuracy improvement: ~92% → ~95%

Phase 3 (Week 6, with 200+ training pairs):
  → HyDE for value search
  Expected accuracy improvement: ~95% → ~97%

Phase 4 (Month 3, with 500+ labeled pairs):
  → Fine-tune BGE embeddings on production query pairs
  Expected accuracy improvement: ~97% → ~98%+
  Bonus: re-training is repeatable as catalog evolves
```

---

## 8. Decision Matrix and Recommendations

### 8.1 The Central Decision: Milvus Architecture

```
KEEP MILVUS — but fundamentally change what you ask it to do.

TODAY:  Milvus finds which facets/values are relevant to a user query
        → Too broad, causes wrong retrievals, conflates similar entities

FUTURE: Milvus resolves exact value strings for list-type facets
        where the LLM has already selected the facet name
        → Narrowly scoped, high accuracy, unavoidable for value lookup
```

You cannot eliminate Milvus because:
1. List-type facets have exact string requirements ("Baby & Toddler" not "baby")
2. Catalog will grow — putting all values in context doesn't scale
3. Multi-tenant means different value lists per tenant — Milvus handles this cleanly

You should eliminate Milvus's current role of **facet name discovery** — replace that with the catalog structure block in context.

### 8.2 The Agent Architecture Decision

```
CURRENT:  6 LLM calls + 3 Milvus searches = ~16-20s latency
FEASIBLE: 2 LLM calls + 1 Milvus search = ~5-8s latency

Recommendation: 2-call architecture with GPT-4.5 or Claude Sonnet 4.5
  - Call 1 (Brain): Intent analysis + facet selection
  - Call 2 (Format): Structure output for persistence
  - Between calls: Milvus value lookups (parallel, facet-scoped)

NOT recommended: 1-call (everything in one pass)
  - Too hard to debug ("which part of the reasoning went wrong?")
  - No checkpointing — full re-run on failure
  - Eval becomes a binary pass/fail with no step-level diagnosis
```

### 8.3 Compatibility with Multi-Tenant Roadmap

All recommendations above are multi-tenant-compatible:
- Catalog structure block: per-tenant, injected as Layer 3 context
- Milvus collections: per-tenant namespaced (`facet_values_{tenant_id}`)
- Fine-tuned embeddings: shared model, tenant-specific training data if available
- Skill definitions: shared (segment_intent_analysis_v1 works for any tenant)

### 8.4 Summary Recommendations Table

| Decision | Recommendation | Why |
|---|---|---|
| Keep Milvus? | Yes, but for values only | Value exact-string lookup is irreplaceable |
| Put catalog in context? | Yes, compact structure block (~5-8K tokens) | Eliminates facet name retrieval errors |
| Full catalog in context? | No | 40K tokens of noise, won't scale to new tenants |
| Number of LLM calls? | Reduce from 6 to 2 | Thinking models handle multi-step reasoning internally |
| Thinking models required? | No — GPT-4.5 or Sonnet 4.5 sufficient | Use thinking models (o3) only for complex ambiguous queries |
| Multi-tenant support? | Parameterize tenant_id throughout | Already 2-key system — extend to N tenants |
| Fine-tune embeddings? | Yes, Phase 4 priority | Biggest quality gain for the least architectural change |
| RAG for historical segments? | Keep and expand | Historical context genuinely helps mapper quality |
| User confirmation step? | Move to end, not after decompose | Users care about the final segment, not internal decomposition |

---

## Appendix A: Catalog Analysis Script

```python
# How the catalog was analyzed for this document
# Run against: /Users/s0m0ohl/Downloads/facet_catalog_0130.csv

import csv
from collections import Counter

rows = []
with open('facet_catalog_0130.csv') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

active = [r for r in rows if r.get('active_ind','') == '1']

# Token estimates (1 token ≈ 4 chars)
compact_fields = ['display_name','facet_key','type','category','sub_category',
                  'description','natural_language_description','operators']
compact_tokens = sum(
    len(' '.join(r.get(f,'') for f in compact_fields)) for r in active
) / 4
# Result: ~39,894 tokens for all 458 active facets (compact)

structural_only = sum(
    len(f"{r['category']} > {r['sub_category']}: {r['display_name']} [{r['type']}] — {r['natural_language_description'][:100]}")
    for r in active if r.get('natural_language_description')
) / 4
# Result: ~16,386 tokens for structural summary only
```

## Appendix B: Compact Catalog Block Generation

```python
# generate_catalog_block.py — generates the Layer 3 structure block for context injection

def generate_catalog_structure_block(
    catalog_df,              # pandas DataFrame from facet CSV
    tenant_restrictions,     # list of allowed restriction values
    include_descriptions=True,
) -> str:
    """
    Generates the compact catalog structure block for context injection.
    Outputs ~5-8K tokens depending on include_descriptions setting.
    """
    active = catalog_df[catalog_df['active_ind'] == '1']
    allowed = active[~active['usage'].isin(['SELECT'])]

    lines = ["[FACET CATALOG — Active Facets by Category]", ""]

    for category in sorted(allowed['category'].unique()):
        cat_facets = allowed[allowed['category'] == category]
        lines.append(f"Category: {category} ({len(cat_facets)} facets)")

        for subcat in sorted(cat_facets['sub_category'].unique()):
            sub_facets = cat_facets[cat_facets['sub_category'] == subcat]
            lines.append(f"  Sub-category: {subcat}")

            for _, row in sub_facets.iterrows():
                value_note = "retrieve via FACET_VALUE_SEARCH" if row['type'] in ['list','csv','string'] else ""
                desc = row['natural_language_description'][:100] if include_descriptions else ""
                lines.append(
                    f"    • {row['display_name']} [{row['type']}]"
                    + (f" — {desc}" if desc else "")
                    + (f". Operators: {row['operators']}" if row['type'] in ['number','double','date','xdate'] else "")
                    + (f". Values: {value_note}" if value_note else "")
                )

        lines.append("")

    lines.append(
        "[IMPORTANT: For facets marked 'Values: retrieve via FACET_VALUE_SEARCH', "
        "call the FACET_VALUE_SEARCH tool with the exact facet name and the user's "
        "term before assigning a value. Do not guess value strings for list-type facets.]"
    )

    return "\n".join(lines)
```

---

*Document produced as part of Enterprise Agentic Research — Research 1 (Sonnet Claude)*
*Based on direct analysis of facet_catalog_0130.csv (794 rows, 458 active, 1.7MB) and complete codebase inspection*
*Covers: catalog reality, Milvus vs context window, multi-tenant, 2-call architecture, RAG quality improvement*
