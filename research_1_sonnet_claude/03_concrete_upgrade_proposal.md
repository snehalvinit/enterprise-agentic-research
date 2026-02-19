# 03 — Concrete Upgrade Proposal: Enterprise-Grade Agentic Segmentation System

> **Research ID:** research_1_sonnet_claude
> **Document Purpose:** Detailed reformation and upgrade plan for Smart-Segmentation
> **Audience:** Engineering leads, architects, and product stakeholders
> **Date:** February 2026

---

## Executive Summary

This document proposes a concrete, phased upgrade of the Smart-Segmentation system into an enterprise-grade agentic customer segmentation platform. The upgrade preserves all working components while systematically replacing brittle patterns with production-grade alternatives.

The core philosophy is **static control + dynamic content**:
- The agent's *identity, loop, and safety rules* are static and version-controlled
- The agent's *skills, knowledge, tools, and tenant configs* are dynamic and loaded at runtime
- Every change to any dynamic component passes through an **eval gate** before reaching production

The result is a system that:
1. Gets smarter with every interaction (memory + auto-improvement)
2. Can be customized per tenant without code changes
3. Maintains strict quality through continuous evaluation
4. Scales to thousands of concurrent users
5. Fails gracefully with appropriate fallbacks

---

## Table of Contents

1. [High-Level Architecture Transformation](#1-high-level-architecture-transformation)
2. [Component-Level Upgrade Details](#2-component-level-upgrade-details)
3. [Static Prompt Architecture](#3-static-prompt-architecture)
4. [Skill System Implementation](#4-skill-system-implementation)
5. [Knowledge and RAG Layer](#5-knowledge-and-rag-layer)
6. [Memory System Design](#6-memory-system-design)
7. [Evaluation-First Infrastructure](#7-evaluation-first-infrastructure)
8. [Auto-Improvement Pipeline](#8-auto-improvement-pipeline)
9. [Multi-Tenant Architecture](#9-multi-tenant-architecture)
10. [Observability and Tracing](#10-observability-and-tracing)
11. [Cost Optimization Strategy](#11-cost-optimization-strategy)
12. [Concrete Transformation Examples](#12-concrete-transformation-examples)
13. [Appendix: Cost-Saving Alternatives](#appendix-cost-saving-alternatives)

---

## 1. High-Level Architecture Transformation

### Current Architecture (As-Is)

```
┌─────────────────────────────────────────────────────────┐
│                    USER REQUEST                          │
└───────────────────────────┬─────────────────────────────┘
                            │
                    ┌───────▼────────┐
                    │  RouterAgent   │  ← 1 LLM call
                    │  (GPT-4o)      │
                    └───────┬────────┘
               ┌────────────┴────────────┐
               │                         │
      ┌────────▼────────┐      ┌─────────▼────────┐
      │  NSC Agent      │      │  DSE Agent        │
      │  (Sequential)   │      │  (Sequential)     │
      └────────┬────────┘      └─────────┬─────────┘
               │                         │
    ┌──────────▼──────────┐              │
    │ Step 1: Decompose   │ ← LLM call   │
    │ Step 2: Date Tag    │ ← LLM call   │
    │ Step 3: Facet Map   │ ← LLM call + Milvus
    │ Step 4: Format      │ ← LLM call   │
    └──────────┬──────────┘              │
               │                         │
               └──────────┬──────────────┘
                           │
                    ┌──────▼───────┐
                    │  PostgreSQL  │
                    └──────────────┘

PROBLEMS: Sequential, no caching, no memory, no evals,
          no multi-tenant, no fallbacks, no improvement
```

### Target Architecture (To-Be)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PERCEPTION LAYER                             │
│   Input Normalizer → Intent Router → Permission Checker → Enricher  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                         MEMORY GATEWAY                               │
│   Short-Term (Redis) ←→ Session State ←→ Long-Term (Postgres + Vec) │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                       REASONING ENGINE                               │
│                                                                      │
│   Static System Prompt (Identity + Loop + Safety + Format)          │
│         +                                                            │
│   Dynamic Skill Loader → [Skill Registry]                           │
│         +                                                            │
│   Dynamic Knowledge Retriever → [RAG Store]                         │
│         +                                                            │
│   Dynamic Tool Registry → [Plugin Registry]                         │
│                                                                      │
│   Loop: PLAN → ACT → VERIFY → IMPROVE                               │
└────────────────────────────┬────────────────────────────────────────┘
                             │
         ┌───────────────────┼────────────────────┐
         ▼                   ▼                    ▼
┌────────────────┐  ┌────────────────┐  ┌─────────────────┐
│  Skill Runner  │  │  Tool Executor │  │ Feedback Engine │
│  (Parallel)    │  │  (w/ fallbacks)│  │ (Self-assess)   │
└───────┬────────┘  └────────┬───────┘  └────────┬────────┘
        │                    │                    │
┌───────▼────────────────────▼────────────────────▼────────┐
│                      VALIDATION LAYER                      │
│   Schema Check → Grounding Check → Business Rule Check    │
└────────────────────────────┬──────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                         EVAL GATEWAY                                 │
│   (Every response scored before returning to user)                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                       PERSISTENCE LAYER                              │
│   Redis (cache) ←→ PostgreSQL (sessions/segments) ←→ Milvus (vec)  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component-Level Upgrade Details

### 2.1 Perception Layer (New)

**Purpose:** Normalize and enrich all incoming requests before they reach the reasoning engine.

**Implementation:**

```python
# perception/input_normalizer.py
class InputNormalizer:
    """
    Transforms raw user input into structured PerceptionResult.
    Runs BEFORE any LLM call.
    """

    async def normalize(self, raw_request: RawRequest) -> PerceptionResult:
        # 1. Spell correction (no LLM needed - use pyspellchecker)
        corrected = self.spellcheck(raw_request.content)

        # 2. Intent detection (fast classifier, no GPT-4o needed)
        intent = await self.classify_intent(corrected)
        # Returns: CREATE | EDIT | DELETE | QUERY | HYPOTHESIS | OUT_OF_SCOPE

        # 3. Entity extraction (spaCy or small model)
        entities = self.extract_entities(corrected)

        # 4. Tenant context injection
        tenant_config = await self.load_tenant_config(raw_request.tenant_id)

        # 5. Permission check
        permissions = await self.check_permissions(
            raw_request.user_id,
            intent,
            tenant_config
        )

        return PerceptionResult(
            original_query=raw_request.content,
            corrected_query=corrected,
            intent=intent,
            entities=entities,
            tenant_config=tenant_config,
            permissions=permissions,
            session_id=raw_request.session_id
        )
```

**Key improvement:** Intent classification uses a fast, cheap model (not GPT-4o). Only complex reasoning tasks go to the expensive model.

---

### 2.2 Parallel Agent Execution (Replaces Sequential Pipeline)

**Before (sequential, ~15 seconds):**
```python
# All steps wait for the previous one
result_decompose = await decompose(query)      # 4s
result_dates = await tag_dates(result_decompose)  # 3s
result_facets = await map_facets(result_dates)    # 5s
result_format = await format(result_facets)       # 3s
# Total: ~15 seconds
```

**After (parallel where possible, ~8 seconds):**
```python
# Decompose first (required)
result_decompose = await decompose(query)  # 4s

# Date tagging and initial facet name shortlisting run in parallel
result_dates, facet_shortlists = await asyncio.gather(
    tag_dates(result_decompose),        # 3s
    shortlist_facets(result_decompose)  # 3s (Milvus search, no LLM)
)

# Facet mapping uses both (but now only needs LLM call, no Milvus wait)
result_facets = await map_facets(facet_shortlists, result_dates)  # 3s

# Format (quick)
result_format = await format(result_facets)  # 1s
# Total: ~8 seconds (4+3+3+1 with overlap = ~7-8s)
```

---

### 2.3 Caching Layer (Redis)

**New component:** Redis cache with intelligent TTL strategies.

```python
# cache/cache_manager.py
class CacheManager:
    def __init__(self, redis_client: Redis):
        self.redis = redis_client

    async def get_or_compute_facet_search(
        self,
        query: str,
        collection: str,
        compute_fn: Callable
    ) -> list:
        cache_key = f"milvus:{collection}:{hashlib.md5(query.encode()).hexdigest()}"

        # Try cache first
        cached = await self.redis.get(cache_key)
        if cached:
            return json.loads(cached)

        # Cache miss - compute
        result = await compute_fn(query, collection)

        # Cache with appropriate TTL
        ttl = 3600 if "facet_name" in collection else 1800
        await self.redis.setex(cache_key, ttl, json.dumps(result))

        return result

    async def get_or_compute_embedding(
        self,
        text: str,
        model: str,
        compute_fn: Callable
    ) -> list[float]:
        cache_key = f"emb:{model}:{hashlib.md5(text.encode()).hexdigest()}"
        cached = await self.redis.get(cache_key)
        if cached:
            return json.loads(cached)

        embedding = await compute_fn(text, model)
        await self.redis.setex(cache_key, 86400, json.dumps(embedding))  # 24hr TTL
        return embedding
```

**Expected impact:** 40-60% reduction in Milvus calls, 20-30% reduction in embedding computation.

---

### 2.4 Milvus Fallback (Hybrid Search)

**Before:** If Milvus fails, everything fails.

**After:** Graceful degradation with hybrid search.

```python
# search/hybrid_searcher.py
class HybridFacetSearcher:
    """
    Primary: Milvus vector search
    Fallback: Fuzzy string search against in-memory catalog
    """

    async def search_facets(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 10
    ) -> SearchResult:
        try:
            # Primary: Milvus semantic search
            results = await self.milvus_search(query, tenant_id, top_k)
            return SearchResult(results=results, source="milvus", quality="high")

        except MilvusException as e:
            logger.warning(f"Milvus failed, falling back to fuzzy search: {e}")

            # Fallback: Fuzzy string matching against loaded catalog
            catalog = await self.get_tenant_catalog(tenant_id)
            results = self.fuzzy_search(query, catalog, top_k)
            return SearchResult(results=results, source="fuzzy", quality="medium")

        except Exception as e:
            logger.error(f"All search methods failed: {e}")
            return SearchResult(results=[], source="none", quality="none")
```

---

## 3. Static Prompt Architecture

The **single most important architectural change** is separating what is static (fixed, version-controlled) from what is dynamic (loaded at runtime per request).

### What STAYS Static (The "Constitution")

```
SYSTEM PROMPT — NEVER CHANGES WITHOUT A VERSION BUMP
════════════════════════════════════════════════════

IDENTITY:
You are SegmentAI, an enterprise customer segmentation advisor.
Your goal: help users create, edit, and evaluate customer segments.

OPERATING LOOP (always follow):
1. PLAN: Identify required skills, knowledge, and tools for this request
2. ACT: Execute using loaded skills + tools; cite all retrieved facts
3. VERIFY: Check that output is grounded, valid, and complete
4. IMPROVE: If any check fails, identify the gap and retry with correction

GROUNDING RULES:
- Use only retrieved facet names and values as source of truth
- Mark any value as [RETRIEVED] if found via search, [INFERRED] if generated
- If retrieval returns nothing relevant → ask a clarifying question; do NOT invent
- Never paraphrase user sub-queries — carry them verbatim through all stages

OUTPUT FORMAT:
Always return a JSON object with this schema: {see schema below}

SAFETY:
- Reject queries that involve PII beyond allowed fields
- Reject queries that would create discriminatory segments
- Escalate to human review if confidence < 0.5

ESCALATION:
- Missing required inputs → ask exactly ONE clarifying question
- Ambiguous intent → present TWO interpretations and ask user to confirm
- Irresolvable conflict → escalate to human reviewer
════════════════════════════════════════════════════
```

### What Is DYNAMIC (Loaded Per Request)

```
[SKILL: segment_creation_v3]           ← Selected by router based on intent
[KNOWLEDGE: facet_catalog_tenant_X]    ← Retrieved from RAG based on tenant
[EXAMPLES: similar_segments_top_3]     ← Retrieved from vector store
[TOOL_REGISTRY: available_tools]       ← Registered dynamically per tenant
[TENANT_CONFIG: restrictions_tenant_X] ← Loaded from config store
[MEMORY: user_preferences_user_Y]      ← Retrieved from memory layer
```

---

## 4. Skill System Implementation

### Skill Definition Schema

Every "skill" is a versioned, testable bundle that gets injected into the agent's context:

```python
# skills/schema.py
class Skill(BaseModel):
    skill_id: str          # e.g., "segment_creation"
    version: str           # e.g., "3.2.1"
    name: str              # Human-readable
    description: str       # What this skill does (used by router)
    triggers: list[str]    # Intent patterns that activate this skill

    # The actual skill content
    instructions: str      # Step-by-step procedure for the agent
    input_schema: dict     # Expected input format
    output_schema: dict    # Expected output format
    constraints: list[str] # Explicit constraints/edge cases
    examples: list[dict]   # Few-shot examples (or "retrieve_dynamic": true)

    # Quality control
    eval_suite_id: str     # Which eval set validates this skill
    min_accuracy: float    # Must pass this threshold to be activated
    eval_gate: bool        # If True, must pass evals before live deployment

    # Metadata
    owner: str
    created_at: datetime
    updated_at: datetime
    tenant_overrides: dict  # Per-tenant customizations
```

### Example: Segment Creation Skill

```yaml
# skills/segment_creation_v3.yaml
skill_id: segment_creation
version: "3.2.1"
name: "Customer Segment Creation"
description: "Creates structured customer segment definitions from natural language queries"
triggers:
  - "create segment"
  - "build segment"
  - "define customers who"
  - "segment customers"

instructions: |
  SEGMENT CREATION PROCEDURE:

  Step 1 — DECOMPOSE
  Parse the user's query into logical sub-segments.
  - Identify: attribute groups, boolean relationships, date constraints
  - Preserve the user's EXACT phrasing for each sub-segment (do not rephrase)
  - If more than one valid decomposition exists, choose the more specific one

  Step 2 — DATE RESOLUTION
  For each sub-segment containing date references:
  - Extract date patterns (relative: "last 30 days", absolute: "Q3 FY26")
  - Resolve to exact start_date and end_date using FISCAL_YEAR_CALENDAR tool
  - If ambiguous, generate ONE clarifying question; do NOT guess

  Step 3 — FACET MAPPING
  For each sub-segment, call FACET_SEARCH tool:
  - Pass the verbatim sub-segment text, not a rephrased version
  - Accept results with confidence >= 0.6; reject below that threshold
  - For each accepted facet, mark as [RETRIEVED: facet_name]
  - For each rejected/missing facet, generate ONE clarifying question

  Step 4 — VALIDATE
  Before returning:
  - Verify all RETRIEVED facets exist in the tenant's allowed facet list
  - Verify boolean logic is valid (no circular references, valid operators)
  - Estimate segment size using SIZE_ESTIMATOR tool
  - If any validation fails, identify the issue and retry step 3

  Step 5 — FORMAT
  Return the output in the required JSON schema.
  Include: segment definition, logic expression, estimated size, confidence score

input_schema:
  user_query: str
  tenant_id: str
  user_preferences: dict
  conversation_history: list

output_schema:
  segment_definition: SegmentDefinition
  logic_expression: str
  estimated_size: int
  confidence_score: float
  clarification_questions: list[str]
  audit_trail: list[AuditEntry]

eval_suite_id: "segment_creation_v3_eval"
min_accuracy: 0.85
eval_gate: true
```

### Skill Registry

```python
# skills/registry.py
class SkillRegistry:
    def __init__(self, db: AsyncPg, redis: Redis):
        self.db = db
        self.redis = redis

    async def load_skill(
        self,
        skill_id: str,
        tenant_id: str,
        version: str = "latest"
    ) -> Skill:
        # Check cache
        cache_key = f"skill:{tenant_id}:{skill_id}:{version}"
        cached = await self.redis.get(cache_key)
        if cached:
            return Skill.model_validate_json(cached)

        # Load from DB with tenant override
        base_skill = await self.db.fetchrow(
            "SELECT * FROM skills WHERE skill_id=$1 AND version=$2",
            skill_id, version
        )

        tenant_override = await self.db.fetchrow(
            "SELECT * FROM skill_overrides WHERE skill_id=$1 AND tenant_id=$2",
            skill_id, tenant_id
        )

        skill = Skill(**base_skill)
        if tenant_override:
            skill = skill.apply_override(tenant_override)

        await self.redis.setex(cache_key, 300, skill.model_dump_json())
        return skill

    async def route_to_skill(self, intent: str, tenant_id: str) -> list[Skill]:
        """Select 1-3 relevant skills based on detected intent."""
        # Vector similarity on skill descriptions + trigger patterns
        skill_embeddings = await self.get_skill_embeddings()
        intent_embedding = await self.embed(intent)

        similarities = cosine_similarity(intent_embedding, skill_embeddings)
        top_skills = sorted(zip(skills, similarities), key=lambda x: x[1], reverse=True)[:3]

        return [skill for skill, score in top_skills if score > 0.7]
```

---

## 5. Knowledge and RAG Layer

### Knowledge Contracts

Every piece of knowledge in the system must have explicit metadata:

```python
# knowledge/schema.py
class KnowledgeChunk(BaseModel):
    chunk_id: str
    content: str

    # Metadata for retrieval quality
    domain: str           # e.g., "facet_catalog", "business_rules", "templates"
    tenant_id: str        # "global" or specific tenant
    freshness_date: date  # When this was last verified
    max_age_days: int     # After this, flag as potentially stale
    owner: str            # Who is responsible for keeping this current
    version: str

    # Anti-hallucination enforcement
    citation_required: bool  # If True, LLM must cite this source
    confidence_threshold: float  # Min retrieval score to use
```

### RAG Retrieval with Anti-Hallucination

```python
# knowledge/retriever.py
class GroundedRetriever:
    async def retrieve_and_ground(
        self,
        query: str,
        knowledge_type: str,
        tenant_id: str
    ) -> GroundedKnowledge:
        # Retrieve from Milvus
        results = await self.milvus.search(
            query=query,
            filters={"tenant_id": [tenant_id, "global"], "domain": knowledge_type},
            top_k=5
        )

        # Filter by freshness
        fresh_results = [
            r for r in results
            if (date.today() - r.freshness_date).days <= r.max_age_days
        ]

        if not fresh_results:
            return GroundedKnowledge(
                content=[],
                instruction="No relevant knowledge found. Ask clarifying question.",
                grounding_status="no_knowledge"
            )

        # Build citation map for LLM
        citations = {
            f"[RETRIEVED-{i+1}]": chunk.content
            for i, chunk in enumerate(fresh_results)
        }

        return GroundedKnowledge(
            content=fresh_results,
            citation_map=citations,
            instruction="Use only the RETRIEVED sources above. Do not use your own knowledge.",
            grounding_status="grounded"
        )
```

---

## 6. Memory System Design

### Two-Tier Memory Architecture

```
MEMORY ARCHITECTURE
═══════════════════════════════════════════════════════

TIER 1: SHORT-TERM (Redis, TTL 24 hours)
─────────────────────────────────────────
• Current conversation state (all turns)
• Ambiguity resolution history (what was asked, what user answered)
• Current segment being built (intermediate state)
• Current task state (which step we're in)

TIER 2: LONG-TERM (PostgreSQL + Milvus, no TTL)
─────────────────────────────────────────────────
• User preferences (date interpretation, terminology preferences)
• Successful segment patterns (vectorized for retrieval)
• Error patterns (what query types failed, why, how it was fixed)
• Tenant-level patterns (common segment structures for this tenant)
• User vocabulary mapping (user says "big spenders" = facet "purchase_value > 500")

═══════════════════════════════════════════════════════
```

### Memory Manager

```python
# memory/manager.py
class MemoryManager:
    def __init__(self, redis: Redis, pg: AsyncPg, milvus: MilvusClient):
        self.redis = redis
        self.pg = pg
        self.milvus = milvus

    # ── SHORT-TERM MEMORY ──────────────────────────────────

    async def get_session_state(self, session_id: str) -> SessionState:
        raw = await self.redis.get(f"session:{session_id}")
        return SessionState.model_validate_json(raw) if raw else SessionState()

    async def update_session_state(self, session_id: str, updates: dict):
        state = await self.get_session_state(session_id)
        state = state.merge(updates)
        await self.redis.setex(
            f"session:{session_id}",
            86400,  # 24 hour TTL
            state.model_dump_json()
        )

    async def get_ambiguity_history(self, session_id: str) -> list[AmbiguityRecord]:
        """What questions have been asked in this session? Prevents duplicate questions."""
        raw = await self.redis.lrange(f"ambiguity:{session_id}", 0, -1)
        return [AmbiguityRecord.model_validate_json(r) for r in raw]

    async def record_ambiguity_resolution(
        self,
        session_id: str,
        question: str,
        resolution: str
    ):
        record = AmbiguityRecord(question=question, resolution=resolution)
        await self.redis.rpush(f"ambiguity:{session_id}", record.model_dump_json())
        await self.redis.expire(f"ambiguity:{session_id}", 86400)

    # ── LONG-TERM MEMORY ──────────────────────────────────

    async def get_user_preferences(self, user_id: str) -> UserPreferences:
        """Load learned preferences: date interpretation, terminology, etc."""
        row = await self.pg.fetchrow(
            "SELECT preferences FROM user_memory WHERE user_id=$1",
            user_id
        )
        return UserPreferences.model_validate_json(row["preferences"]) if row else UserPreferences()

    async def update_user_preferences(
        self,
        user_id: str,
        preference_key: str,
        preference_value: Any,
        confidence: float
    ):
        """Update learned preference with confidence weighting."""
        current = await self.get_user_preferences(user_id)
        current.update(preference_key, preference_value, confidence)

        await self.pg.execute("""
            INSERT INTO user_memory (user_id, preferences, updated_at)
            VALUES ($1, $2, NOW())
            ON CONFLICT (user_id) DO UPDATE
            SET preferences = $2, updated_at = NOW()
        """, user_id, current.model_dump_json())

    async def find_similar_segments(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 3
    ) -> list[SegmentMemory]:
        """Find historically successful segments similar to this query."""
        embedding = await self.embed(query)

        results = await self.milvus.search(
            collection="successful_segments",
            query_vectors=[embedding],
            filter=f'tenant_id == "{tenant_id}" and quality_score > 0.7',
            output_fields=["segment_id", "query", "definition", "quality_score"],
            limit=top_k
        )

        return [SegmentMemory(**r) for r in results[0]]

    async def store_successful_segment(
        self,
        query: str,
        segment_definition: dict,
        quality_score: float,
        user_id: str,
        tenant_id: str
    ):
        """Store a successful segment creation for future reference."""
        embedding = await self.embed(query)

        await self.milvus.insert(
            collection="successful_segments",
            data=[{
                "segment_id": str(uuid4()),
                "query": query,
                "definition": json.dumps(segment_definition),
                "quality_score": quality_score,
                "user_id": user_id,
                "tenant_id": tenant_id,
                "embedding": embedding,
                "created_at": datetime.now().isoformat()
            }]
        )
```

### Centralized Ambiguity Resolver (Eliminates Duplicate Questions)

```python
# reasoning/ambiguity_resolver.py
class AmbiguityResolver:
    """
    Central authority for all clarification questions.
    Prevents duplicate questions across agents.
    """

    def __init__(self, memory: MemoryManager):
        self.memory = memory

    async def should_ask_or_infer(
        self,
        session_id: str,
        user_id: str,
        ambiguity: AmbiguityRequest
    ) -> AmbiguityDecision:
        # Check 1: Have we already asked this question in this session?
        history = await self.memory.get_ambiguity_history(session_id)
        for record in history:
            if self.is_same_question(ambiguity.question, record.question):
                return AmbiguityDecision(
                    action="use_memory",
                    resolution=record.resolution
                )

        # Check 2: Does user have a stored preference for this ambiguity type?
        prefs = await self.memory.get_user_preferences(user_id)
        if ambiguity.type in prefs:
            return AmbiguityDecision(
                action="use_preference",
                resolution=prefs[ambiguity.type]
            )

        # Check 3: Can this be resolved from context without asking?
        if ambiguity.can_infer_from_context and ambiguity.context_confidence > 0.8:
            return AmbiguityDecision(
                action="infer",
                resolution=ambiguity.inferred_value,
                confidence=ambiguity.context_confidence
            )

        # Default: Ask the user, but track it
        return AmbiguityDecision(action="ask", question=ambiguity.question)
```

---

## 7. Evaluation-First Infrastructure

### Eval Gate Architecture

```
EVERY CHANGE MUST PASS EVALS BEFORE PRODUCTION
═══════════════════════════════════════════════

Prompt change proposed
       ↓
  [EVAL GATE 1]
  Unit eval on changed skill/prompt
  Min accuracy: 85% on skill eval suite
       ↓
  [EVAL GATE 2]
  Regression eval vs. previous version
  No metric may drop > 3% vs baseline
       ↓
  [EVAL GATE 3]
  Shadow traffic test (1% of real traffic)
  Human review of random sample
       ↓
  PROMOTED TO PRODUCTION

═══════════════════════════════════════════════
```

### Continuous Eval Runner

```python
# evals/continuous_runner.py
class ContinuousEvalRunner:
    async def run_skill_eval(
        self,
        skill_id: str,
        version: str,
        tenant_id: str = "global"
    ) -> EvalResult:
        eval_suite = await self.load_eval_suite(skill_id)

        results = []
        for test_case in eval_suite.cases:
            # Run the skill with the test input
            output = await self.run_skill(skill_id, version, test_case.input)

            # Score using multiple metrics
            score = EvalScore(
                # Structural correctness
                schema_valid=self.validate_schema(output, test_case.expected_schema),
                # Semantic accuracy
                facet_accuracy=self.compare_facets(output.facets, test_case.expected_facets),
                # Hallucination detection
                grounding_ratio=self.check_grounding(output),
                # Business rule compliance
                rule_compliance=self.check_business_rules(output, tenant_id),
            )
            results.append(EvalCase(input=test_case, output=output, score=score))

        return EvalResult(
            skill_id=skill_id,
            version=version,
            cases=results,
            mean_accuracy=mean([r.score.facet_accuracy for r in results]),
            grounding_rate=mean([r.score.grounding_ratio for r in results]),
            passed=mean([r.score.facet_accuracy for r in results]) >= eval_suite.min_accuracy
        )

    async def run_regression_check(
        self,
        skill_id: str,
        new_version: str,
        baseline_version: str
    ) -> RegressionResult:
        new_result = await self.run_skill_eval(skill_id, new_version)
        baseline_result = await self.run_skill_eval(skill_id, baseline_version)

        regressions = []
        for metric in EvalMetric:
            delta = new_result.get_metric(metric) - baseline_result.get_metric(metric)
            if delta < -0.03:  # More than 3% drop = regression
                regressions.append(Regression(metric=metric, delta=delta))

        return RegressionResult(
            new_version=new_version,
            baseline_version=baseline_version,
            regressions=regressions,
            passed=len(regressions) == 0
        )
```

### Automatic Eval Dataset Growth

```python
# evals/dataset_builder.py
class EvalDatasetBuilder:
    """
    Continuously mines production queries to grow eval datasets.
    """

    async def mine_production_queries(self, since: datetime) -> list[EvalCandidate]:
        # Pull recent production queries from logs
        queries = await self.pg.fetch("""
            SELECT query, segment_output, user_feedback, quality_score
            FROM production_queries
            WHERE created_at > $1 AND user_feedback IS NOT NULL
            ORDER BY created_at DESC
        """, since)

        candidates = []
        for q in queries:
            # Cluster queries to ensure diversity
            embedding = await self.embed(q["query"])
            cluster = await self.assign_cluster(embedding)

            candidates.append(EvalCandidate(
                query=q["query"],
                expected_output=q["segment_output"],
                ground_truth=self.extract_ground_truth(q["user_feedback"]),
                cluster=cluster,
                quality_source="user_feedback"
            ))

        # Sample diverse set (1 per cluster to avoid over-representation)
        diverse_sample = self.sample_by_cluster(candidates)
        return diverse_sample
```

---

## 8. Auto-Improvement Pipeline

### Feedback Collection

```python
# feedback/collector.py
class FeedbackCollector:
    """Collects implicit and explicit feedback signals."""

    async def collect_implicit_feedback(
        self,
        session_id: str,
        query: str,
        output: SegmentOutput
    ) -> ImplicitFeedback:
        session = await self.memory.get_session_state(session_id)

        signals = ImplicitFeedback()

        # Signal 1: Did user immediately edit the segment? (indicates dissatisfaction)
        if session.edit_count_after_creation > 0:
            signals.edit_count = session.edit_count_after_creation
            signals.satisfaction = max(0.0, 1.0 - (signals.edit_count * 0.2))

        # Signal 2: Did user confirm without editing? (indicates satisfaction)
        if session.confirmed_without_edit:
            signals.satisfaction = 1.0

        # Signal 3: Did user abandon the conversation? (indicates failure)
        if session.session_abandoned:
            signals.satisfaction = 0.0
            signals.failure_type = "abandoned"

        return signals

    async def record_explicit_feedback(
        self,
        segment_id: str,
        rating: int,  # 1-5
        comment: str,
        user_id: str
    ):
        await self.pg.execute("""
            INSERT INTO segment_feedback
            (segment_id, rating, comment, user_id, created_at)
            VALUES ($1, $2, $3, $4, NOW())
        """, segment_id, rating, comment, user_id)
```

### Prompt Optimization Loop

```python
# improvement/optimizer.py
class PromptOptimizer:
    """
    DSPy-style prompt optimization based on failure analysis.
    """

    async def analyze_failures(
        self,
        skill_id: str,
        since: datetime
    ) -> FailureAnalysis:
        # Get all failed or low-quality outputs for this skill
        failures = await self.pg.fetch("""
            SELECT query, output, feedback_score, error_type
            FROM query_results
            WHERE skill_id=$1 AND created_at > $2 AND feedback_score < 0.7
            ORDER BY feedback_score ASC
        """, skill_id, since)

        # Cluster failures by type
        failure_clusters = self.cluster_failures(failures)

        # For each cluster, identify the root cause
        root_causes = []
        for cluster in failure_clusters:
            cause = await self.identify_root_cause(cluster)
            root_causes.append(cause)

        return FailureAnalysis(failure_clusters=failure_clusters, root_causes=root_causes)

    async def generate_improved_prompt(
        self,
        current_skill: Skill,
        failure_analysis: FailureAnalysis
    ) -> Skill:
        """
        Use an LLM to propose prompt improvements based on failure analysis.
        The proposed prompt is then tested against the eval suite before adoption.
        """
        improvement_prompt = f"""
        You are a prompt optimization expert.

        Current skill instructions:
        {current_skill.instructions}

        Failure analysis (these are cases where the current prompt fails):
        {failure_analysis.to_summary()}

        Propose specific improvements to the skill instructions that would
        address these failure patterns. Be concrete and specific.
        Return only the improved instructions text.
        """

        improved_instructions = await self.llm.complete(improvement_prompt)

        candidate_skill = current_skill.copy(
            update={
                "instructions": improved_instructions,
                "version": current_skill.next_version()
            }
        )

        return candidate_skill

    async def run_optimization_cycle(self, skill_id: str):
        """Full optimization cycle: analyze → propose → eval → promote."""
        current_skill = await self.registry.load_skill(skill_id, "latest")
        failure_analysis = await self.analyze_failures(skill_id, since=datetime.now() - timedelta(days=7))

        if failure_analysis.total_failures < 10:
            return  # Not enough data to optimize

        candidate = await self.generate_improved_prompt(current_skill, failure_analysis)

        # Run eval gate
        eval_result = await self.eval_runner.run_skill_eval(skill_id, candidate.version)
        regression = await self.eval_runner.run_regression_check(
            skill_id, candidate.version, current_skill.version
        )

        if eval_result.passed and regression.passed:
            await self.registry.promote_skill(candidate)
            logger.info(f"Promoted {skill_id} v{candidate.version}: accuracy improved")
        else:
            logger.info(f"Candidate {skill_id} v{candidate.version} failed evals: {eval_result}")
```

---

## 9. Multi-Tenant Architecture

### Tenant Configuration Schema

```python
# tenants/config.py
class TenantConfig(BaseModel):
    tenant_id: str
    name: str

    # Facet catalog
    facet_catalog_id: str   # Points to tenant-specific catalog in Milvus
    allowed_facets: list[str]  # Whitelist of allowed facets
    restricted_facets: list[str]  # Blacklisted facets (PII, etc.)

    # Skill customizations
    skill_overrides: dict[str, SkillOverride]  # Per-skill prompt customizations

    # Terminology mapping
    vocabulary_map: dict[str, str]  # "big spenders" → "purchase_value > 500"

    # Business rules
    segment_size_limits: dict[str, int]  # Min/max segment sizes
    date_conventions: DateConventions  # Fiscal year start, etc.

    # Output format
    output_schema: dict  # Tenant-specific segment format

    # Rate limits
    max_segments_per_day: int
    max_concurrent_sessions: int
```

### Tenant Isolation

```python
# tenants/isolator.py
class TenantDataIsolator:
    """Ensures strict data isolation between tenants."""

    async def get_facet_catalog(self, tenant_id: str) -> FacetCatalog:
        # Each tenant has their own Milvus collection prefix
        collection_name = f"facets_{tenant_id}"

        if not await self.milvus.has_collection(collection_name):
            # Fall back to global catalog
            collection_name = "facets_global"

        return await self.load_catalog(collection_name)

    async def get_segment_history(self, tenant_id: str, user_id: str) -> list[Segment]:
        # Strict tenant_id filter on all queries
        return await self.pg.fetch("""
            SELECT * FROM segments
            WHERE tenant_id = $1 AND user_id = $2
            ORDER BY created_at DESC LIMIT 100
        """, tenant_id, user_id)

    async def validate_tenant_context(
        self,
        tenant_id: str,
        user_id: str,
        request: Request
    ) -> ValidationResult:
        # Verify user belongs to this tenant
        user_tenant = await self.pg.fetchval(
            "SELECT tenant_id FROM users WHERE user_id=$1",
            user_id
        )

        if user_tenant != tenant_id:
            raise TenantViolationError(f"User {user_id} does not belong to tenant {tenant_id}")

        return ValidationResult(valid=True)
```

---

## 10. Observability and Tracing

### Enhanced Tracing (Beyond Current Phoenix Setup)

```python
# observability/tracer.py
class EnhancedTracer:
    """
    Structured tracing with full audit trail.
    Compatible with Phoenix/OTEL, also writes to PostgreSQL for querying.
    """

    @contextmanager
    async def trace_skill_execution(
        self,
        skill_id: str,
        version: str,
        session_id: str,
        user_id: str,
        tenant_id: str
    ):
        trace_id = str(uuid4())
        start = time.monotonic()

        # Start OpenTelemetry span
        with self.tracer.start_as_current_span(f"skill.{skill_id}") as span:
            span.set_attribute("skill.id", skill_id)
            span.set_attribute("skill.version", version)
            span.set_attribute("session.id", session_id)
            span.set_attribute("user.id", user_id)
            span.set_attribute("tenant.id", tenant_id)

            try:
                yield trace_id
                span.set_attribute("result", "success")
            except Exception as e:
                span.set_attribute("result", "error")
                span.set_attribute("error.type", type(e).__name__)
                span.record_exception(e)
                raise
            finally:
                duration_ms = (time.monotonic() - start) * 1000
                span.set_attribute("duration_ms", duration_ms)

                # Also write to PostgreSQL for structured querying
                await self.pg.execute("""
                    INSERT INTO execution_traces
                    (trace_id, skill_id, version, session_id, user_id, tenant_id, duration_ms, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                """, trace_id, skill_id, version, session_id, user_id, tenant_id, duration_ms)
```

### Segment Audit Trail

```python
# audit/trail.py
class SegmentAuditTrail:
    """
    Complete, queryable audit trail for every segment decision.
    Enables explainability, compliance, and debugging.
    """

    async def record_segment_creation(
        self,
        segment_id: str,
        query: str,
        skill_id: str,
        skill_version: str,
        retrieved_facets: list[RetrievedFacet],
        inferred_values: list[InferredValue],
        confidence_scores: dict,
        tenant_id: str,
        user_id: str
    ):
        await self.pg.execute("""
            INSERT INTO segment_audit
            (segment_id, query, skill_id, skill_version,
             retrieved_facets, inferred_values, confidence_scores,
             tenant_id, user_id, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
        """,
        segment_id, query, skill_id, skill_version,
        json.dumps([f.dict() for f in retrieved_facets]),
        json.dumps([v.dict() for v in inferred_values]),
        json.dumps(confidence_scores),
        tenant_id, user_id)

    async def explain_segment(self, segment_id: str) -> SegmentExplanation:
        """Generate human-readable explanation of segment creation decisions."""
        audit = await self.pg.fetchrow(
            "SELECT * FROM segment_audit WHERE segment_id=$1", segment_id
        )

        retrieved = json.loads(audit["retrieved_facets"])
        inferred = json.loads(audit["inferred_values"])

        explanation = SegmentExplanation(
            segment_id=segment_id,
            original_query=audit["query"],
            skill_used=f"{audit['skill_id']} v{audit['skill_version']}",
            retrieved_attributes=[
                f"{f['name']} = {f['value']} (confidence: {f['confidence']:.0%})"
                for f in retrieved
            ],
            inferred_attributes=[
                f"{v['name']} = {v['value']} [AI inference, not from catalog]"
                for v in inferred
            ]
        )
        return explanation
```

---

## 11. Cost Optimization Strategy

### Model Routing (Save 60% on LLM Costs)

```python
# routing/model_router.py
class ModelRouter:
    """
    Routes tasks to the cheapest model that can reliably handle them.
    """

    MODEL_TIERS = {
        "simple": "gpt-4o-mini",      # $0.15/1M tokens
        "standard": "gpt-4o",          # $2.50/1M tokens
        "complex": "claude-opus-4-6",   # $15/1M tokens
        "reasoning": "o3-mini",         # For complex planning
    }

    def select_model(self, task_type: str, complexity: float) -> str:
        if task_type == "intent_classification":
            return self.MODEL_TIERS["simple"]  # Greeting, routing

        if task_type == "date_parsing" and complexity < 0.4:
            return self.MODEL_TIERS["simple"]  # Simple dates

        if task_type == "facet_mapping" and complexity < 0.6:
            return self.MODEL_TIERS["standard"]  # Most facet mapping

        if task_type == "hypothesis_assessment":
            return self.MODEL_TIERS["reasoning"]  # Strategic analysis

        if task_type == "segment_format":
            return self.MODEL_TIERS["simple"]  # Format generation

        return self.MODEL_TIERS["standard"]  # Default
```

### Prompt Caching

```python
# Use Anthropic's prompt caching for static system prompts
# This reduces cost by ~90% for the static portion of the prompt

CACHED_SYSTEM_PROMPT = """
<antcaching>system_prompt_v3.2</antcaching>
You are SegmentAI...
[Full 12KB static prompt]
"""

# The static prompt is cached server-side — only charged once per cache lifetime
# Dynamic portions (skills, knowledge) are not cached (they change per request)
```

### Expected Cost Analysis

| Component | Current | Optimized | Savings |
|-----------|---------|-----------|---------|
| Router calls | GPT-4o ($2.50/1M) | GPT-4o-mini ($0.15/1M) | 94% |
| Date parsing | GPT-4o | GPT-4o-mini | 94% |
| Facet mapping | GPT-4o | GPT-4o (cached prompt) | ~80% |
| Format generation | GPT-4o | GPT-4o-mini | 94% |
| Milvus searches | 100% uncached | 40-60% cached | 40-60% |
| Embedding calls | 100% uncached | ~80% cached | 80% |
| **Total** | **Baseline** | | **~65% reduction** |

---

## 12. Concrete Transformation Examples

### Example 1: Eliminating Duplicate Clarification Questions

**Before:**
```
User: "Create a segment of recent high-value customers"

[Decomposer Agent]:
  "What does 'recent' mean? Last 7, 30, or 90 days?"

User: "30 days"

[Facet Mapper Agent — same session, 2 minutes later]:
  "For the time period, should I use last 30 days, Q4 FY25, or a custom range?"

User: "I just said 30 days!!"
```

**After:**
```python
# Centralized ambiguity resolver
resolver = AmbiguityResolver(memory)

# Decomposer asks "recent" question
resolution = await resolver.should_ask_or_infer(
    session_id, user_id,
    AmbiguityRequest(type="date_recency", question="What does 'recent' mean?")
)
# → AmbiguityDecision(action="ask", question="What does 'recent' mean?")

# User answers: "30 days"
await resolver.record_resolution(session_id, "date_recency", "30 days")

# Facet Mapper checks the same ambiguity
resolution = await resolver.should_ask_or_infer(
    session_id, user_id,
    AmbiguityRequest(type="date_recency", question="What time period?")
)
# → AmbiguityDecision(action="use_memory", resolution="30 days")
# The question is never asked again!
```

### Example 2: Intent-Preserved Through Pipeline

**Before:**
```
User input: "customers who buy baby products at least twice a week"

Decomposer output (rephrased): "high-frequency baby product purchasers"
  → [lost: "at least twice a week", "customers" implied vs "households"]

Date Tagger input: "high-frequency baby product purchasers"
  → Cannot extract "twice a week" — it was already lost!

Facet Mapper input: "high-frequency baby product purchasers"
  → Maps to generic "purchase_frequency = high" — WRONG
```

**After:**
```python
# Each agent receives the ORIGINAL sub-query verbatim + its own metadata
sub_segment = SubSegment(
    id="Seg-1",
    original_query="customers who buy baby products at least twice a week",  # PRESERVED
    decomposer_metadata={"is_frequency_based": True},
    date_metadata=None  # Not yet tagged
)

# Date tagger receives original query — extracts "at least twice a week"
date_result = await date_tagger(sub_segment.original_query)
sub_segment.date_metadata = {"frequency": ">=2/week", "type": "purchase_frequency"}

# Facet mapper receives BOTH original query AND date metadata
facet_result = await facet_mapper(
    original_query=sub_segment.original_query,  # "at least twice a week" still present
    date_metadata=sub_segment.date_metadata
)
# → Maps correctly to "purchase_frequency >= 2" with appropriate time window
```

### Example 3: Long-Term Memory Improving Accuracy

**Week 1:**
```
User: "segment customers who spent a lot last holiday"
System: "What does 'a lot' mean? What is your definition of 'high value'?"
User: "More than $200 in a single transaction"
→ Segment created, stored in memory: {user_id: "user_X", "high_value_threshold": 200}
```

**Week 3:**
```
User: "create a segment of high-value fall shoppers"
System: [checks long-term memory]
→ "high_value" preference found: > $200
System: "Creating segment for customers who spent more than $200 in a single transaction during fall (September-November)..."
→ No clarification question needed! Memory served correctly.
```

**Week 6:**
```
User: "target our biggest spenders this quarter"
System: [memory + context analysis]
→ "biggest spenders" → similar to "high value" → threshold: $200+
→ "this quarter" → Q1 FY26 (Feb-Apr)
→ Segment created with zero clarification questions
```

---

## Appendix: Cost-Saving Alternatives

> **Note:** These alternatives trade some quality for cost savings. The core proposal above represents the best-quality approach. Consider these only when cost constraints require trade-offs.

### A1. Replace Milvus with PGVector (Saves ~$2K-5K/month)

**Trade-off:** 15-20% accuracy reduction in facet search, higher PostgreSQL load

```sql
-- Store embeddings directly in PostgreSQL with pgvector
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE facet_embeddings (
    id BIGSERIAL PRIMARY KEY,
    facet_name TEXT,
    facet_value TEXT,
    embedding vector(1024),
    tenant_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX ON facet_embeddings USING ivfflat (embedding vector_cosine_ops);
```

**When to use:** Early-stage deployment, single-tenant, facet catalog < 100K items

### A2. Use Open-Source LLMs for Simple Tasks (Saves 40-60% on LLM costs)

**Trade-off:** Higher latency, requires GPU infrastructure, lower accuracy on complex tasks

- **Llama-3.1-8B:** For intent classification, date parsing simple cases
- **Mistral-7B:** For format generation
- **Keep GPT-4o/Claude:** Only for complex facet mapping and reasoning

**When to use:** High-volume deployments where token costs are dominant

### A3. Rule-Based Date Parser (Saves 90% of date tagging LLM costs)

**Trade-off:** Less flexible for unusual date expressions, requires maintenance

```python
# Instead of LLM for common date patterns:
DATE_PATTERNS = {
    r"last (\d+) days": lambda m: (today - timedelta(int(m.group(1))), today),
    r"last (\d+) weeks": lambda m: (today - timedelta(int(m.group(1))*7), today),
    r"Q(\d) FY(\d{2})": lambda m: fiscal_quarter(int(m.group(1)), int(m.group(2))),
    r"year to date": lambda m: (fiscal_year_start, today),
    r"last year": lambda m: (prev_year_start, prev_year_end),
}
```

**When to use:** If >80% of date queries use standard patterns

---

*Document produced as part of Enterprise Agentic Research — Research 1 (Sonnet Claude)*
*References: Prior Research Context from Prompt + Smart-Segmentation codebase analysis*
