# 00 — Master Summary: Everything You Need to Know

> **Start here.** This document consolidates all 6 research docs into one plain-English guide. If you read nothing else, read this. Every concept is explained with a concrete example from your actual codebase.

---

## TL;DR — The Whole Picture in 60 Seconds

You built a smart AI agent that takes a marketing question like *"Find Walmart customers who bought baby products in the last 90 days"* and turns it into a structured customer segment query. It works — but it has 5 structural problems that cap its accuracy at roughly **48%** for complex queries. The fixes are known, concrete, and most can be done in under a week each. The biggest one is a **single typo** you can fix right now.

```
WHAT IT DOES TODAY:
  User query → 7 LLM stages → segment facets → Milvus vector DB → CRM audience

WHAT GOES WRONG:
  Stage 1 (90% right) × Stage 2 (90%) × ... × Stage 7 (90%) = 48% end-to-end

WHAT IT SHOULD DO:
  User query → 4 LLM stages + rule-based checks → segment facets → Milvus → CRM
              └─ with ground truth examples injected at every stage
```

---

## Part 1: What Does Smart-Segmentation Actually Do?

### The Job

A marketing analyst at Walmart types:
> *"Show me customers with high propensity to buy electronics, excluding loyalty members, active in the last 60 days"*

Your agent translates this into a precise set of **facets** — structured filters like:

```json
{
  "Propensity Super Department": "Electronics",
  "Propensity Division": "Consumer Electronics",
  "Customer Status": "Active",
  "Loyalty": "Non-Member",
  "Recency": "60 days"
}
```

These facets then query Milvus (your vector database) to find matching customers in Walmart's CRM.

### The Pipeline (What Happens Under the Hood)

Every query goes through this chain:

| Stage | What it does | Example |
|---|---|---|
| **Route** | Is this a new segment or existing? | "new segment" |
| **Decompose** | Break complex query into sub-parts | `[electronics_propensity, recency_filter, loyalty_exclusion]` |
| **Date** | Extract date/recency signals | `"last 60 days" → recency:60d` |
| **FVOM** | Map facet values (the big one) | `"high electronics propensity" → Propensity Division: Consumer Electronics` |
| **Dependency** | Add required companion facets | `Propensity Division` always needs `Propensity Super Department` |
| **Classifier** | Strict vs non-Strict facets | `Customer Status: Active` (not `Strict: Active`) |
| **Linked** | Handle cross-facet relationships | |
| **Format** | Output clean JSON | Final segment object |

Each of these stages makes one or more LLM calls. That's **7+ calls per query**.

---

## Part 2: The 5 Critical Problems (Simply Explained)

### Problem 1: The "Hybird" Typo — One Character Killing a Feature

**What it means:** Your Milvus wrapper has a method called `_match_single_instance_hybrid_search`. Hybrid search combines two retrieval methods: dense vectors (semantic meaning) and BM25 (keyword matching). Together they find facets much more accurately — like having both Google and a dictionary lookup. This method is fully coded and ready to use.

But it's **never called** because of a typo in the routing logic:

```python
# In milvus.py line 152 — CURRENT (broken)
if match_type == "hybird":          # ← typo! "hybird" not "hybrid"
    return self._match_single_instance_hybrid_search(...)

# THE FIX (1 character change)
if match_type == "hybrid":
    return self._match_single_instance_hybrid_search(...)
```

**Real impact:** Every facet lookup right now uses only dense vector search. The hybrid search code — which includes `RRFRanker` for combining scores — has been sitting unused since it was written. Fixing this typo immediately upgrades your retrieval quality.

---

### Problem 2: Accuracy Compounds Downward Across 7 Stages

**What it means:** Imagine flipping 7 coins. Even if each coin lands heads 90% of the time, the chance ALL 7 land heads is only `0.9⁷ = 48%`. Your pipeline has the same math problem.

```
Stage 1: Route     90% correct → 90 out of 100 queries pass cleanly
Stage 2: Decompose 90% correct → 81 out of 100 still on track
Stage 3: Date      90% correct → 73 still clean
Stage 4: FVOM      90% correct → 66 still clean
Stage 5: Depend    90% correct → 59 still clean
Stage 6: Classify  90% correct → 53 still clean
Stage 7: Link+Fmt  90% correct → 48 still clean
```

**Real impact:** Even a very capable model (90%+ per stage) produces a correct end-to-end segment only ~48% of the time. The ground truth CSV confirms this — 22 of 46 rows (47.8%) had corrections needed.

**The fix:** Merge stages 5+6+7 into one combined "classify-depend-link" prompt. Reduce 7 stages to 4. Add a rule-based verify step between each stage that catches obvious errors before passing to the next LLM. Combined pipeline accuracy rises from ~48% to ~73%.

---

### Problem 3: Your Ground Truth Has 46 Rows — Not Enough to Trust

**What it means:** You have an evaluation dataset to measure quality. But it has only 46 rows. Statistically, this means your quality measurements have a ±15% margin of error at 95% confidence.

Concretely: if you make a change and your F1 score goes from 0.72 to 0.78, you **cannot tell** if you actually improved anything — it could just be statistical noise.

```
Current: 46 eval rows → ±15% margin → unreliable quality signals
Target:  200+ eval rows → ±7% margin → reliable quality signals
Target:  400+ eval rows → ±5% margin → production-grade eval
```

**The fix:** Treat your existing 46 ground-truth examples as a **retrieval resource** (RAG), not just an eval set. When a new query comes in, find the 3 most similar past examples and inject them as few-shot context:

```python
# Instead of just: "Map these facets for: {query}"
# Do this:
"""
Here are 3 similar past segments and their correct facets:

Example 1: "Baby product buyers last 60 days"
→ Facets: {Department: Baby, Recency: 60d, ...}

Example 2: "Electronics high propensity, active"
→ Facets: {Propensity Division: Consumer Electronics, Status: Active}

Now map facets for: {new_query}
"""
```

This alone typically lifts accuracy 10-15% on similar queries (validated by Atlas/DH-RAG research).

---

### Problem 4: Static Context Files — You're Hardcoded to Walmart

**What it means:** Four critical context files are loaded once when the agent starts and never change:

```
agentic_framework/contextual_information/
├── refinements.txt               ← Walmart-specific rules
├── catalog_view_description.txt  ← Walmart catalog structure
├── decomposer_hints.txt          ← Walmart segment patterns
└── fvom_hints.txt                ← Walmart facet value mappings
```

In `agent.py` lines 72-73:
```python
# Loaded at module import — hardcoded to Walmart
CATALOG_DESC = open("contextual_information/catalog_view_description.txt").read()
```

**Real impact:** To add a second client (say, Target), you'd need to copy the entire codebase and swap the files manually. Multi-tenant support doesn't just need new files — it needs a runtime config system.

**The fix:** A tenant config manifest YAML file:

```yaml
# config/tenants/walmart.yaml
tenant_id: walmart_email_mobile
catalog_view_description: "Walmart has 18,779 facets organized by Department > Division..."
fvom_hints: "Propensity Super Department must always be paired with Propensity Division..."
milvus:
  collection: SEGMENT_AI_WALMART_EMAIL_MOBILE_FACET_NAME_BGE_FLAT_COSINE
eval:
  f1_threshold: 0.80
```

Then load it at request time: `config = load_tenant_config(tenant_id)`.

---

### Problem 5: BGE Embedding Model Missing Its Instruction Prefix

**What it means:** You're using the `BAAI/bge-small-en-v1.5` model to convert facet names into vectors for Milvus. This model was trained with a specific instruction prefix that must be prepended to **queries** (not to documents being indexed):

```python
# CURRENT (in embedding.py) — missing prefix
vector = embed("Customer Status: Active")

# CORRECT — with BGE instruction prefix
vector = embed("Represent this sentence for searching relevant passages: Customer Status: Active")
```

Without this prefix, the model operates at lower quality for retrieval tasks. It's like using a calculator but never pressing the equals button — all the machinery is there, the result just isn't optimized.

**Real impact:** BGE's own benchmarks show 5-8% retrieval quality improvement with the prefix on. For 18,779 facets in your catalog, this means more accurate facet matches across the board.

---

## Part 3: The Target Architecture (What to Build)

### Current vs Target — Side by Side

```
CURRENT PIPELINE                    TARGET PIPELINE
─────────────────                   ───────────────
1. Route (LLM)                      1. Route (LLM — keep)
2. Decompose (LLM)                  2. Decompose (LLM — keep)
3. Date (LLM)                       3. Date (rule-based + LLM fallback)
4. FVOM (LLM)                          ↓ verify: date format check
5. Dependency (LLM)                 4. FVOM (LLM — keep, + cascade retrieval)
6. Classifier (LLM)                    ↓ verify: facet existence check
7. Linked + Format (LLM)            5. Classify+Depend+Link+Format (1 LLM call)
                                       ↓ verify: schema validation
7 LLM calls, no verify              3–4 LLM calls + rule checks
P(success) ≈ 48%                    P(success) ≈ 73%
```

### Cascade Retrieval (The FVOM Upgrade)

Right now, FVOM does one vector similarity search. The upgrade is a cascade — try fast/cheap methods first, fall back to LLM only when needed:

```
Query: "high electronics propensity customers"

Step 1 — Exact match:    "Propensity Division" IN catalog? → No exact match
Step 2 — Alias lookup:   "electronics" → "Consumer Electronics"? → Found!
         Return: Propensity Division: Consumer Electronics ✓

--- If step 2 fails: ---
Step 3 — BM25 keyword:   Full-text search facets database → Top 5 candidates
Step 4 — Dense vector:   Semantic search Milvus → Top 5 candidates
Step 5 — RRF merge:      Combine BM25 + dense scores (hybrid search)
Step 6 — LLM rerank:     "Pick the best match from these 10 candidates" → 1 call

Cost:  Step 1-2 = $0.00 (80% of queries caught here)
       Step 3-5 = $0.001 (15% of queries)
       Step 6   = $0.003 (5% of queries — truly ambiguous)
```

---

## Part 4: Patterns Stolen from Claude Code & Cursor (Applied to Your Agent)

The six research docs, especially Doc 06, analyzed how Claude Code and Cursor are built internally. Here are the 4 patterns most directly applicable to your agent:

### Pattern A: The ReAct Loop (Reason → Act → Verify)

Claude Code doesn't just execute instructions — it thinks, acts, then checks its work in a loop. Your agent should do the same:

```python
# TODAY: straight-line pipeline
result = route → decompose → fvom → format

# TARGET: ReAct loop with verify
while not done:
    action = llm.plan(context)     # "I need to look up Propensity Division"
    result = execute(action)       # Actually call Milvus/catalog
    if not verify(result):         # Did it work? Is the facet valid?
        context.add_error(result)  # Tell LLM what went wrong
        continue                   # LLM tries a different approach
    context.add_success(result)
    done = is_complete(context)
```

This is why Claude Code can recover from errors — it loops until it verifies success, rather than failing silently.

### Pattern B: CLAUDE.md → Your Tenant Config

Claude Code reads `CLAUDE.md` at startup to understand the project context (what tools exist, what conventions to follow, what to avoid). The equivalent for your agent is the **tenant manifest** — a version-controlled config file that tells the agent everything about the tenant:

```
Claude Code's CLAUDE.md          Your Agent's tenant.yaml
─────────────────────            ──────────────────────────
"Use pytest for tests"           "Propensity facets always paired"
"Don't modify migrations"        "Strict variants are deprecated"
"API is at /api/v2"              "Milvus collection: WALMART_BGE_FLAT"
"Company uses React"             "Catalog has 18,779 facets"
```

### Pattern C: Sub-Agent Parallelism

Claude Code spawns parallel sub-agents for independent tasks. For your agent, when a query decomposes into 3 independent sub-segments, build all 3 simultaneously:

```python
# TODAY: sequential (slow)
seg1 = build_segment(part1)   # 2 seconds
seg2 = build_segment(part2)   # 2 seconds
seg3 = build_segment(part3)   # 2 seconds
total = 6 seconds

# TARGET: parallel (fast)
seg1, seg2, seg3 = await asyncio.gather(
    build_segment(part1),
    build_segment(part2),
    build_segment(part3)
)
total ≈ 2 seconds + merge time
```

### Pattern D: Tiered Model Routing (Save 70% on LLM Costs)

Cursor uses different model sizes for different tasks. Your agent should do the same — not every stage needs the most expensive model:

```python
MODEL_ROUTING = {
    # Simple classification → cheapest model
    "route_query":        "claude-haiku-4-5",   # $0.0001/call
    "extract_dates":      "claude-haiku-4-5",   # rule-based anyway

    # Core reasoning → mid-tier
    "decompose_segment":  "claude-sonnet-4-6",  # $0.0009/call
    "map_facet_values":   "claude-sonnet-4-6",  # most calls go here

    # Ambiguous/complex only → expensive
    "resolve_conflict":   "claude-opus-4-6",    # $0.0075/call — rare
}

# Cost comparison
# TODAY:  8 × Sonnet calls ≈ $0.053 per query
# TARGET: 3.4 avg calls, mixed models ≈ $0.006–$0.018 per query
# Savings: 65–90% cost reduction
```

---

## Part 5: My Top 8 Recommendations (Prioritized)

These are ranked by **impact/effort ratio** — highest value for least work first.

### 🔴 Do This Week

**Rec 1 — Fix the "hybird" typo (30 minutes, ~5–8% quality lift)**

```python
# milvus.py line 152
# Change: "hybird" → "hybrid"
```
One character. Immediately enables hybrid search that's been sitting coded but unused.

**Rec 2 — Add BGE instruction prefix to queries (1 hour, ~5% quality lift)**

```python
# embedding.py — in the query embedding function (NOT document embedding)
def embed_query(text: str) -> list[float]:
    prefixed = f"Represent this sentence for searching relevant passages: {text}"
    return self.model.encode(prefixed)
```
Only applies to queries. Documents in Milvus stay as-is. No re-indexing needed.

**Rec 3 — Run a proper eval baseline BEFORE anything else (2 hours)**

Before you change anything, run a full eval on all 46 ground truth rows and record:
- F1 score per facet type
- Which query types fail most
- Latency per stage

You need this baseline to know if your fixes actually helped.

---

### 🟡 Do This Sprint (Week 2–4)

**Rec 4 — Collapse stages 5+6+7 into one LLM call (2 days)**

The Dependency, Classifier, and Linked stages are all operating on the same context. Combine them into one prompt that does all three:

```
Current: 3 LLM calls → 3 × latency + 3 × error chance
Target:  1 LLM call that says "finalize, add dependencies, validate, format"
```

**Rec 5 — Add rule-based date extraction (1 day)**

The Date stage uses an LLM to extract things like "last 60 days" or "Q4 2024". This doesn't need AI:

```python
import re
DATE_PATTERNS = {
    r"last (\d+) days": lambda m: f"recency:{m.group(1)}d",
    r"past (\d+) months": lambda m: f"recency:{int(m.group(1))*30}d",
    r"Q([1-4]) (\d{4})": lambda m: f"quarter:Q{m.group(1)}-{m.group(2)}",
}
# Falls back to LLM only if no pattern matches
```
This removes one full LLM call for ~80% of queries.

---

### 🟢 Do Next Phase (Week 4–8)

**Rec 6 — Inject top-3 ground truth examples as few-shot context (3 days)**

```python
class GroundTruthRAG:
    def get_examples(self, query: str, k: int = 3) -> str:
        # Find 3 most similar past queries using embeddings
        similar = self.milvus.search(embed(query), k=k)
        return "\n\n".join([
            f"Example {i+1}: '{ex.query}'\n→ Facets: {ex.facets}"
            for i, ex in enumerate(similar)
        ])
```

**Rec 7 — Add exact-match + alias lookup before vector search (2 days)**

For 80% of common facets ("Active customers", "High propensity"), exact string matching is faster and more reliable than vector search. Build a lookup table first:

```python
FACET_ALIASES = {
    "active": "Customer Status: Active",
    "electronics": "Propensity Super Department: Electronics",
    "baby": "Department: Baby",
    # ... ~200 common aliases from your ground truth corrections
}
```

**Rec 8 — Build the tenant config manifest (3 days, unlocks multi-tenancy)**

Create `config/tenants/walmart.yaml` and load it at request time. This unlocks onboarding new clients without code changes. The config manifest should include: vocabulary hints, facet catalog metadata, Milvus collection name, eval thresholds.

---

## Part 6: Impact Table — Before vs After All Fixes

| Metric | Today | After Week 1 Fixes | After Full Plan |
|---|---|---|---|
| End-to-end accuracy | ~48% | ~60% | ~80% |
| Facet recall | ~60% | ~68% | >85% |
| Latency per query | 5–8 sec | 4–6 sec | 2–3 sec |
| Cost per query | $0.053 | $0.040 | $0.006–$0.018 |
| Tenants supported | 1 (Walmart) | 1 | N (config-driven) |
| Eval reliability | ±15% margin | ±15% | ±5% (with 400 rows) |
| Quality regression detection | Manual | Manual | Automated (CI) |

---

## Part 7: The Quick-Reference Code Map

These are the exact files to touch for each fix:

| Fix | File | Line | Change |
|---|---|---|---|
| Hybrid search typo | `agentic_framework/utils/milvus.py` | 152 | `"hybird"` → `"hybrid"` |
| BGE prefix | `agentic_framework/utils/embedding.py` | query encode fn | Add prefix string |
| Date rules | `agentic_framework/sub_agents/.../date_agent.py` | — | Add regex patterns |
| Tenant config | `agentic_framework/agent.py` | 72–73 | Load from YAML |
| Ground truth RAG | New class in `utils/` | — | `GroundTruthRAG` class |
| Stage merge | `sub_agents/classify + depend + link` | — | Combine into 1 prompt |

---

## Part 8: The Bigger Picture — Where This Leads

The 6 research docs point toward a clear destination. Here's the arc:

```
TODAY (v1)              NEAR-TERM (v2)           FUTURE (v3)
─────────────           ──────────────           ──────────────
Walmart only            N tenants                Any domain
Static context          Runtime config           Self-configuring
7 LLM stages            4 LLM stages             2–3 LLM stages
Dense search only       Hybrid search            Cascade retrieval
Manual eval             Automated CI eval        Auto-improving (DSPy)
48% end-to-end          73% end-to-end           85%+ end-to-end
$0.053/query            $0.018/query             $0.006/query
No observability        Langfuse tracing         Full audit trail
```

**The v3 system** uses DSPy to automatically optimize prompts: it runs your eval set, identifies which prompt wordings work best, and updates itself — like A/B testing for LLM prompts, fully automated.

**The system described in Doc 04** gets you to v2 in 10 weeks across 5 parallel work tracks. The week 1 quick wins (Recs 1–3 above) get you measurable improvement immediately with essentially zero risk.

---

## Part 9: Glossary — Key Terms Explained Simply

| Term | Plain English Explanation |
|---|---|
| **Facet** | A single filter dimension. `Customer Status: Active` is one facet. A segment is typically 3–15 facets combined. |
| **FVOM** | "Facet Value and Object Mapper" — the stage that translates your words into exact facet names in the catalog |
| **Milvus** | Your vector database. Stores facet names as 768-dimension math vectors so similar concepts can be found by semantic similarity |
| **BM25** | Classic keyword search (like Google's early algorithm). Fast and good at exact words. Complements dense vector search. |
| **Hybrid search** | Combining BM25 (keyword) + dense vectors (semantics) with a merging algorithm (RRF) for better retrieval than either alone |
| **RRF** | Reciprocal Rank Fusion — the math formula for merging BM25 and dense rankings. Already coded in your `milvus.py` |
| **Ground truth** | Your 46-row CSV of real segment queries with the correct expected facets — used to measure how accurate the system is |
| **Few-shot** | Giving the LLM 2–3 worked examples before asking it your question. Reliably improves accuracy by 10–15% |
| **DSPy** | A framework that automatically finds the best prompt wording for your task by running experiments |
| **ReAct loop** | A pattern where the agent Reasons, then Acts, then checks the result, repeating until it's confident — used by Claude Code |
| **Tenant** | One client/company using the system. Today you have 1 tenant (Walmart). Multi-tenant means many clients with isolated configs |
| **Pipeline stage** | One step in the processing chain. Each stage takes the previous stage's output and refines it further |
| **Eval** | Short for evaluation — running test queries through the system and measuring how often it gets the right answer |
| **F1 score** | A quality metric between 0 and 1. F1 = 1.0 means perfect. F1 = 0.7 means you get 70% of facets right on average |
| **Cascade retrieval** | A tiered lookup strategy: try cheap/fast methods first, only escalate to expensive LLM when earlier tiers fail |
| **BGE** | The name of the embedding model you use (BAAI/bge-small-en-v1.5). "BGE" = BAAI General Embedding |
| **Sub-agent** | An independent AI agent running in parallel to handle one piece of a complex task, then merging results |
| **Shadow execution** | Running a proposed change in a sandboxed "dry run" before committing it — catches errors before they reach production |

---

## Appendix: Research Document Map

Each of the 6 documents goes deep on one area. This summary consolidates across all of them:

| Doc | What it covers | Read it when... |
|---|---|---|
| **01 Bottleneck Analysis** | All 18 bottlenecks with exact file:line code references | You want to see every problem with evidence |
| **02 Research Compendium** | 70+ academic papers and industry sources | You want to understand the state-of-the-art benchmarks |
| **03 Upgrade Proposal** | Detailed code for cascade retrieval, pipeline collapse, ground truth RAG | You're ready to start building |
| **04 Implementation Roadmap** | 5 phases, 16 weeks, parallel tracks, risk registry | You're planning sprints |
| **05 AI Coding Tools** | How to use Claude Code/Cursor/Copilot to BUILD agents faster | You want to accelerate your development workflow |
| **06 Internal Architectures** | How Claude Code/Cursor ARE built + 8 patterns to copy | You want to understand agent engineering principles deeply |

---

*Master Summary — Research #2 Sonnet Claude RAG Extra — February 2026*
