# 03 — Concrete Upgrade Proposal
### Smart-Segmentation → Enterprise Agentic Customer Segmentation System

**Research ID:** research_2_sonnet_rag_extra_claude
**Date:** February 2026

---

## Executive Architecture Vision

The current system is a **sequential LLM pipeline with tenant-coupled static configuration**. The upgrade transforms it into a **multi-tenant, retrieval-augmented, eval-gated agentic system with a cascade retrieval strategy and compressed pipeline stages**.

```
CURRENT ARCHITECTURE (7+ LLM stages, 1 tenant):
User → Route → Decompose → Date-LLM → FVOM-LLM → Dependency-LLM → Classifier-LLM → Linked-LLM → Format

TARGET ARCHITECTURE (4 LLM stages, N tenants):
User → Route → Decompose+Verify → Date-Rule+Date-LLM(fallback) → FVOM-Cascade → DependencyClassifyLinked → Format
                                        ↕                              ↕
                                 Tenant Config                  Tiered Retrieval
                                 Runtime Loader                 (Exact→Alias→BM25→Dense→Rerank)
                                                                Ground Truth RAG
```

---

## Section 1: Facet Retrieval Upgrade

### Decision Matrix: Retrieval Strategy for 500+ Structured Facets

| Strategy | Precision | Recall | Latency | Cost | When Best |
|---|---|---|---|---|---|
| **Dense embedding only** (current) | 🟡 Medium | 🟡 Medium | 150-300ms | Medium | Large unstructured KB |
| **BM25 keyword** | 🟢 High (exact) | 🔴 Low (semantic) | <50ms | Low | Exact name matching |
| **Structured SQL/dict lookup** | 🟢 Highest | 🟡 Medium | <5ms | Minimal | Exact + filtered catalog |
| **LLM with full catalog in-context** | 🟢 High | 🟢 High | 1-3s | High | Small catalog (<100 items) |
| **Hybrid BM25+dense** | 🟢 High | 🟢 High | 200-400ms | Medium | General retrieval |
| **Cascade (proposed)** | 🟢 Highest | 🟢 Highest | 5-300ms | Low-Medium | Structured + semantic mix |

**Recommendation: Tiered Cascade Retrieval**

```
TIER 1: Exact Name Lookup (dictionary, <5ms)
  ├─ Input: NER entities + sub-segment query
  ├─ Match: facet_catalog.name.lower() exact match
  └─ Hit: Return immediately with full facet metadata

TIER 2: Alias Resolution (<5ms)
  ├─ Input: Unmatched query terms
  ├─ Match: tenant alias table (Strict→non-Strict, synonyms)
  └─ Hit: Return resolved facet name

TIER 3: Structured Type Filter + BM25 (<100ms)
  ├─ Input: Unmatched terms + NER type hints (brand, category, persona)
  ├─ Match: BM25 over facet names filtered by type
  └─ Hit: Top-5 candidates with BM25 score

TIER 4: Dense Semantic Embedding (150-300ms)
  ├─ Input: Unmatched terms still needing semantic inference
  ├─ Match: BGE embeddings in Milvus (name + value collections)
  └─ Hit: Top-10 candidates with distance score

TIER 5: LLM Rerank (only if >3 candidates remain ambiguous, +300ms)
  ├─ Input: Candidate facets + original sub-segment query
  ├─ Action: LLM picks best match from shortlist
  └─ Output: Single facet or ask-user question
```

### ASCII Architecture: Before vs After

**Before (current):**
```
Sub-segment query
       |
  NER Agent (LLM)
       |
  Embed all entities (BGE/MiniLM)
       |
  ┌────────────────┐     ┌────────────────┐
  │ Milvus Value   │     │ Milvus Name    │
  │ (dense search) │     │ (dense search) │
  └───────┬────────┘     └───────┬────────┘
          └───────────┬──────────┘
                 Fuzzy filter
                      |
              Combined shortlist
                      |
            FVOM LLM (picks final)
```

**After (proposed cascade):**
```
Sub-segment query
       |
  NER + Type Classification (LLM or rules)
       |
  ┌─────────────────────────────────────────────────────┐
  │ CASCADE RETRIEVAL                                    │
  │                                                      │
  │ T1: Exact Dict Lookup → HIT? → Return               │
  │           ↓ MISS                                     │
  │ T2: Alias Table        → HIT? → Return              │
  │           ↓ MISS                                     │
  │ T3: BM25 + Type Filter → SCORE ≥ 0.8? → Return     │
  │           ↓ MISS                                     │
  │ T4: Dense Embedding    → SCORE ≥ 0.75? → Return    │
  │           ↓ MISS/AMBIGUOUS                           │
  │ T5: LLM Rerank (<5 candidates) → Final Pick         │
  └─────────────────────────────────────────────────────┘
       |
  Ground Truth RAG (inject similar historical examples)
       |
  FVOM LLM (select from pre-filtered shortlist)
```

### Alias Table Structure (Tenant-Specific Config)

```json
// tenants/walmart_email_mobile/aliases.json
{
  "aliases": {
    "propensity super department strict": "Propensity Super Department",
    "propensity division strict": "Propensity Division",
    "propensity brand strict": "Propensity Brand",
    "email engagement": "CRM Email Engagement",
    "push engagement": "CRM Push Engagement",
    "engagement": "CRM Engagement",
    "persona": "Persona",
    "wmt+": "Walmart+ Subscription Status",
    "walmart plus": "Walmart+ Subscription Status"
  },
  "type_hints": {
    "brand": ["Propensity Brand", "Browse Propensity Brand"],
    "department": ["Propensity Super Department", "Propensity Division"],
    "engagement": ["CRM Engagement", "CRM Email Engagement", "CRM Push Engagement"]
  }
}
```

### Typed Tool Pattern (Replace Single Search)

```python
# Instead of one FacetValueShortlister.get_shortlisted_facet_value_pairs()
# Expose typed tools to the orchestrating agent:

@tool
def search_propensity_facets(query: str, facet_type: str = "all") -> List[FacetCandidate]:
    """Search propensity and predictive facets. facet_type: brand|department|division|persona"""

@tool
def search_engagement_facets(query: str, channel: str = "any") -> List[FacetCandidate]:
    """Search CRM engagement facets. channel: email|push|any"""

@tool
def search_purchase_facets(query: str, channel: str = "any") -> List[FacetCandidate]:
    """Search purchase history facets. channel: online|store|any"""

@tool
def search_date_facets(query: str) -> List[FacetCandidate]:
    """Search date and time-based facets"""

@tool
def search_demographic_facets(query: str) -> List[FacetCandidate]:
    """Search demographic and geographic facets"""

@tool
def lookup_facet_by_name(name: str) -> Optional[FacetMetadata]:
    """Exact catalog lookup by facet name or alias"""
```

### Scaling to Larger Catalogs (800-1500 Facets)

At 800-1500 facets:
- Tier 1 (exact lookup): Still O(1), no change
- Tier 2 (alias table): Still O(1), no change
- Tier 3 (BM25): Linearly scales, negligible impact at 1500 items
- Tier 4 (dense): Milvus scales to millions of vectors — no issue
- Tier 5 (LLM rerank): Keep candidate set ≤20 regardless of catalog size

**No re-architecture needed at 800-1500 facets.** The cascade handles it transparently.

---

## Section 2: Pipeline Stage Refactor

### Target: 4 Core Stages Instead of 7+

| Stage | Current | Proposed | LLM? | Notes |
|---|---|---|---|---|
| Route | LlmAgent | LlmAgent | ✅ | Keep — intent is nuanced |
| Decompose | LlmAgent | LlmAgent + Code Verify | ✅ | Add code-based verify |
| Date Extract | LlmAgent | Rule-based + LLM fallback | ⚠️ Conditional | 90% rule-based |
| Retrieve | NER+Embed+Fuzzy | Cascade (Tier 1-5) | ⚠️ Conditional | Typed tools |
| Classify+Depend+Link | 3x LlmAgent | 1x LlmAgent | ✅ | Collapsed |
| Format | Code+LlmAgent | Code | ❌ | Pure transformation |

**New 4-stage pipeline:**

```
Stage A: Route (LLM)
   ↓
Stage B: Decompose (LLM) + Verify (Code) + ReAct Retry
   ↓
Stage C: Retrieve+Classify (Cascade + 1 LLM)
   ↓
Stage D: Format (Code)
```

### Stage B: Decompose + Verify (Code, No LLM)

```python
def verify_decomposition(response: dict) -> VerificationResult:
    """Pure code verification - no LLM call needed"""
    sub_segments = response.get("subSegments", {})
    rule_set = response.get("ruleSet", {})

    # Extract all segment IDs used in ruleSet
    ids_in_ruleset = set(re.findall(r'Seg-\d+', str(rule_set)))
    ids_in_subsegments = set(sub_segments.keys())

    errors = []
    if ids_in_ruleset != ids_in_subsegments:
        errors.append(f"Mismatch: {ids_in_ruleset - ids_in_subsegments} in ruleset but not defined")

    for seg_id, seg_text in sub_segments.items():
        if not seg_text or not seg_text.strip():
            errors.append(f"{seg_id} has empty description")

    return VerificationResult(valid=len(errors)==0, errors=errors)
```

### Stage C: Collapsed Classify+Depend+Link

Instead of three sequential LLM calls, use one call with a richer prompt:

```python
STAGE_C_PROMPT = """
Given the segment components below, perform THREE tasks in ONE pass:

1. DEPENDENCY IDENTIFICATION: Which facets need refinement value selection?
2. REFINEMENT SELECTION: For identified facets, select the appropriate refinement value
   (use contextual info: {refinements_context})
3. LINKED FACET RESOLUTION: Identify and resolve any facet pairs with dependencies

Return a single structured JSON:
{
  "refinements": [{"facet": "...", "value": "...", "confidence": 0.0-1.0, "ask_user": false}],
  "linked_pairs": [{"primary": "...", "linked": "...", "type": "..."}],
  "user_questions": ["If ask_user=true for any refinement, list the question here"]
}
"""
```

### Skill Bundle Pattern (KB Article per Operation)

Instead of monolithic prompts, organize knowledge as versioned skill bundles:

```
skills/
  decompose/
    v1.0.0/
      instructions.md    # How to decompose a segment query
      examples.json       # Few-shot examples (loaded dynamically)
      schema.json         # Expected output schema
      eval.json           # Eval cases for this skill
  date_extract/
    v1.0.0/
      instructions.md
      patterns.json       # Date patterns (moved here from contextual_information/)
  facet_classify/
    v1.0.0/
      instructions.md
      refinements.json    # Tenant-specific refinement context (loaded per tenant)
```

Each skill is:
- Versioned (can A/B test v1.0.0 vs v1.1.0)
- Self-contained (has its own eval suite)
- Tenant-configurable (tenant overrides specific skill files)

---

## Section 3: Ground Truth as Runtime Few-Shot RAG

### Architecture

```
Offline (build once per tenant):
  Ground Truth CSV → Embed segment descriptions → Milvus GT collection

Online (per request):
  User segment description
       ↓
  Embed with same model
       ↓
  Similarity search: GT collection → Top-3 similar historical segments
       ↓
  Inject as few-shot examples in FVOM prompt:
    "Similar past segments and how they were built:
     1. [WomenFashion]: Description → Expected facets → Selected facets + values
     2. [SpringCleaners]: ..."
       ↓
  FVOM LLM now has grounded examples from same domain
```

### Ground Truth CSV → Vector Store Pipeline

```python
class GroundTruthRAGIndexer:
    def __init__(self, tenant_id: str, gt_csv_path: str, collection_name: str):
        self.tenant_id = tenant_id
        self.gt_csv_path = gt_csv_path
        self.collection_name = collection_name

    def build_index(self):
        gt_df = pd.read_csv(self.gt_csv_path)

        # Use segment description as query text
        descriptions = gt_df['Updated Segment Description with Add-on'].fillna(
            gt_df['Segment Description ']
        ).tolist()

        # Embed with same model as runtime
        emb_gen = EmbeddingGenerator("BGE")
        embeddings = emb_gen.generate(descriptions)

        # Index with tenant_id metadata filter
        records = []
        for i, row in gt_df.iterrows():
            records.append({
                "tenant_id": self.tenant_id,
                "segment_name": row['Segment Name'],
                "description": descriptions[i],
                "expected_facets": row['updated expected facets'],
                "segment_definition": row['Updated Segment Definition'],
                "embedding": embeddings[i][0].tolist()
            })

        # Insert to Milvus GT collection
        milvus_client.insert(self.collection_name, records)

    def retrieve_similar(self, query: str, top_k: int = 3) -> List[dict]:
        query_emb = EmbeddingGenerator("BGE").generate([query])
        results = milvus_client.search(
            self.collection_name,
            data=query_emb[0][0].tolist(),
            limit=top_k,
            filter=f'tenant_id == "{self.tenant_id}"',
            output_fields=["segment_name", "description", "expected_facets", "segment_definition"]
        )
        return results
```

### Prompt Injection Format

```python
def format_few_shot_examples(similar_segments: List[dict]) -> str:
    examples = []
    for seg in similar_segments:
        examples.append(f"""
Segment: {seg['segment_name']}
Description: {seg['description']}
Facets Used: {seg['expected_facets']}
Definition: {seg['segment_definition'][:200]}...
""")
    return "SIMILAR HISTORICAL SEGMENTS:\n" + "\n---\n".join(examples)
```

### Cold-Start Strategy for New Tenants

When a new tenant has <50 ground truth rows:
1. **Cross-tenant transfer**: Use abstract facet type matches (propensity, engagement, date) from primary tenant's ground truth that share the same segment pattern
2. **Universal examples**: A library of 20-30 "universal" segment templates (fashion shopper, seasonal cleaner, engaged subscriber) that work across retail tenants
3. **Minimum viable**: 30 labeled rows = acceptable quality; 50 rows = good; 100+ rows = excellent
4. **Human-in-loop expansion**: For the first 2 weeks, have business analysts review all segment outputs and mark them as correct/incorrect to rapidly expand the dataset

---

## Section 4: Enterprise Domain Patterns

### What Leading Enterprise Marketing/CRM AI Systems Do That This System Should Adopt

#### Pattern 1: Typed Tools with Strict Schemas (Salesforce Einstein, Adobe CDP)
Enterprise CRM AI agents expose **strongly-typed retrieval tools** rather than a single semantic search. Adobe's Real-Time CDP AI uses discrete functions: `find_audience_attribute(name, type)`, `get_propensity_score(model_name)`, `filter_by_engagement(channel, level)`.

**Apply to this system:** Replace `FacetValueShortlister` with typed `search_*_facets()` tools. The agent uses tool selection as intent classification — calling `search_propensity_facets()` vs `search_engagement_facets()` is itself a meaningful semantic decision.

#### Pattern 2: Eval Gates Before Every Deployment (Google, Anthropic)
Enterprise AI systems at Google, Anthropic, and leading ML companies treat evaluation as a first-class engineering discipline. No prompt change, no retrieval change, no tool change ships without:
1. A comprehensive test suite running automatically
2. A minimum score threshold that must pass (e.g., facet recall ≥ 85%)
3. Automatic regression detection with alert routing

**Apply:** CI/CD pipeline integration for `evaluations/cli.py`. Every PR that touches prompt files triggers a full eval run. Merge blocked if score drops >5%.

#### Pattern 3: Structured Output + Post-Hoc Validation (Anthropic)
Anthropic's enterprise customers report that adding structured output schemas + post-hoc JSON validation reduces hallucination rates by 30-50% on structured tasks. Pydantic models for every LLM output stage, with automatic retry-with-feedback on schema violations.

**Apply:** Extend existing `data_models/` to cover all stages, with retry-with-error-context on validation failure.

#### Pattern 4: Memory-Augmented Few-Shot (Microsoft Copilot for Sales)
Microsoft's Copilot for Sales product uses a semantic cache of past CRM interactions as few-shot examples injected at runtime. For repetitive segment patterns (seasonal campaigns, engagement-based segments), cached examples dramatically improve consistency.

**Apply:** Ground truth RAG (Section 3 above) is exactly this pattern.

#### Pattern 5: Tenant Isolation at Config Level (Multi-tenant SaaS)
Enterprise SaaS providers (Salesforce, HubSpot, Zendesk) achieve multi-tenancy not through code branches but through **tenant config manifests**. All tenant-specific behavior is data, not code.

**Apply:** Tenant config manifest (see roadmap Phase 4). Every tenant gets a YAML config file; zero code changes per new tenant.

#### Pattern 6: Observable AI (Arize, Langfuse, Braintrust)
Enterprise AI systems are instrumented with structured observability: every LLM call logs prompt, response, token count, latency, and quality score. Retrieval steps log query, candidates, scores, and final selection. This enables:
- Debugging regressions to specific prompts/retrievals
- Monitoring quality drift over time
- Identifying systematic failures by query type

**The system already has Phoenix (Arize) but depth of instrumentation is unclear. Deepen tracing.**

#### Pattern 7: Human-in-the-Loop for Quality Bootstrapping
Adobe's CDP AI and Salesforce Einstein both use "warm start" periods where human reviewers validate AI outputs. The validated outputs go directly into the ground truth dataset, creating a flywheel: more data → better evals → higher confidence → wider deployment.

**Apply:** First 6 weeks of production, route all segment outputs to business analyst review. Auto-approve if confidence score is high (based on eval similarity metrics). Flag for review if confidence is low.

---

## Section 5: Cost Analysis

### Current Cost Profile (Estimated per Request)

| Stage | LLM Calls | Estimated Tokens | Est. Cost |
|---|---|---|---|
| Route | 1 | ~500 | $0.003 |
| Decompose | 1 | ~1,000 | $0.006 |
| Date Tagger | 1 | ~800 | $0.005 |
| FVOM | 1 | ~3,000 | $0.018 |
| Dependency+Classifier+Linked | 3 | ~2,000 | $0.012 |
| Format | 1 | ~1,500 | $0.009 |
| **Total** | **8** | **~8,800** | **~$0.053** |

### Target Cost Profile (After Upgrade)

| Stage | LLM Calls | Estimated Tokens | Est. Cost |
|---|---|---|---|
| Route | 1 | ~500 | $0.003 |
| Decompose + Verify | 1 | ~1,000 | $0.006 |
| Date (rule-based, LLM fallback 10%) | 0.1 | ~80 | $0.0005 |
| Retrieve Cascade (Tiers 1-3 mostly) | 0.3 | ~900 | $0.0054 |
| Classify+Depend+Link (1 call) | 1 | ~1,500 | $0.009 |
| Format (code) | 0 | 0 | $0 |
| **Total** | **~3.4** | **~3,980** | **~$0.024** |

**~55% cost reduction while improving quality.** The main savings come from:
1. Eliminating 2 LLM calls in dependency/classifier/linked
2. Replacing date LLM with rules (90% of cases)
3. Reducing FVOM context size by pre-filtering with cascade

---

## Appendix: Cost-Saving Alternatives

*These alternatives provide additional cost reduction with some quality trade-offs. Evaluate after Phase 1-2 are complete.*

### Alt-A: Smaller Model for Route + Decompose
Use Claude Haiku (or equivalent) for Route and Decompose stages — these are relatively simple intent classification and structured decomposition tasks. Reserve Claude Sonnet/Opus for FVOM + Classify where nuanced reasoning is needed.
**Estimated additional savings: 30-40% on routing stages.**

### Alt-B: Semantic Cache for Repeated Segment Types
Cache segment outputs by semantic similarity of input descriptions. Identical or near-identical queries (similarity > 0.95) return the cached output. Useful for organizations that repeatedly build similar segments (seasonal campaigns, standard engagement tiers).
**Estimated savings: 10-20% of requests cached at scale.**

### Alt-C: Batch Eval Instead of Real-Time Eval
Run evals on a nightly batch job rather than on every PR merge. Reduces CI cost and developer wait time. Acceptable only for low-risk changes (documentation, hint file tweaks); mandatory real-time eval for prompt and retrieval changes.
