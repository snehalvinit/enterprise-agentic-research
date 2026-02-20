# 05 — Integration Guide: How Skills, Memory, and Self-Improvement Actually Work Together

> **Research ID:** research_1_sonnet_claude
> **Document Purpose:** Stitches the architecture documents together. Shows the EXACT integration points — what the assembled prompt looks like, how memory flows through a request, and how a production failure becomes a better skill.
> **Prerequisite reading:** `03_concrete_upgrade_proposal.md` (architecture), `04_implementation_roadmap.md` (delivery plan)
> **Date:** February 2026

---

## The Core Question This Document Answers

The upgrade proposal defines three major systems:

- **Static Prompt** — the agent's permanent identity, loop, and safety rules
- **Skill System** — versioned, testable bundles of task-specific instructions
- **Memory System** — short-term session state and long-term learned preferences

But *how do they actually connect?* What does the LLM physically receive? When is memory read? When is it written? How does a user complaint eventually change a skill prompt?

This document answers those questions with concrete assembled examples, end-to-end request flow traces, and the full closed feedback loop.

---

## Table of Contents

1. [The Assembled Prompt — What the LLM Actually Sees](#1-the-assembled-prompt--what-the-llm-actually-sees)
2. [Skills → Static Prompt Integration](#2-skills--static-prompt-integration)
3. [Memory Integration — Read, Inject, Write Back](#3-memory-integration--read-inject-write-back)
4. [The Full Request Flow — One Request, All Systems](#4-the-full-request-flow--one-request-all-systems)
5. [The Self-Improvement Loop — Failure to Better Prompt](#5-the-self-improvement-loop--failure-to-better-prompt)
6. [The Three Integration Contracts (Summary)](#6-the-three-integration-contracts-summary)

---

## 1. The Assembled Prompt — What the LLM Actually Sees

The LLM never sees a monolithic prompt. It sees an **assembled context** built from four separate sources at request time. Here is what the assembled messages look like for a real segment creation request.

### 1.1 The Four Layers That Compose Every Request

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1: STATIC SYSTEM PROMPT  (never changes, cached)         │
│  Source: static_system_prompt.txt (committed to repo)           │
│  Token cost: ~0 (served from prompt cache after first request)  │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 2: ACTIVE SKILL  (changes per intent, versioned)         │
│  Source: SkillRegistry.load("segment_creation", tenant_id)      │
│  Token cost: ~800-1200 tokens (instructions + output schema)    │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 3: INJECTED CONTEXT  (changes per request)               │
│  Source: MemoryManager + RAG retriever                          │
│  Contents:                                                       │
│    • User preferences (from long-term memory)                   │
│    • Similar past segments (from Milvus episodic store)         │
│    • Resolved ambiguities from this session (short-term memory) │
│    • Tenant configuration overrides                             │
│  Token cost: ~500-1500 tokens                                   │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 4: CONVERSATION + CURRENT REQUEST                        │
│  Source: Session conversation history (Redis) + new user query  │
│  Token cost: ~200-2000 tokens (depending on conversation length)│
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Concrete Example: What the LLM Receives

For the user query: *"create a segment of high-value baby product buyers from last holiday season"*
(User B, Tenant: RetailCo, 3rd session — has prior preferences stored in memory)

```
══════════════════════════════════════════════════════════
SYSTEM MESSAGE (Layers 1 + 2 + partial Layer 3)
══════════════════════════════════════════════════════════

[LAYER 1 — STATIC PROMPT — CACHED, charged once per 5 min window]
─────────────────────────────────────────────────────────
You are SegmentAI, an enterprise customer segmentation advisor for RetailCo.

OPERATING LOOP — follow this for every request:
  PLAN   → Identify required facets, tools, and ambiguities in the request
  ACT    → Execute using ONLY retrieved facets; mark each as [RETRIEVED: name]
  VERIFY → Check every facet exists in tenant catalog; check logic is valid
  IMPROVE → If any check fails, identify the gap and retry with correction

GROUNDING RULES:
  • Never invent facet names — use ONLY what FACET_SEARCH returns
  • Mark retrieved values as [RETRIEVED: facet_name = value]
  • Mark any AI-generated value as [INFERRED: value] — these require approval
  • If FACET_SEARCH returns nothing → generate ONE clarifying question, do NOT guess
  • Carry user sub-queries VERBATIM through all steps — do not rephrase or paraphrase

SAFETY:
  • Reject requests involving protected characteristics (race, religion, health)
  • Reject requests that would de-anonymize individual customers
  • If confidence < 0.5 on any facet → generate a clarifying question

OUTPUT FORMAT:
  Always return valid JSON matching the schema in the active skill's output_schema.
  Include an audit_trail array listing every facet source.

[END LAYER 1]

─────────────────────────────────────────────────────────
[LAYER 2 — ACTIVE SKILL: segment_creation v3.2.1]
─────────────────────────────────────────────────────────
## ACTIVE SKILL: Customer Segment Creation

PROCEDURE — follow these steps in order:

STEP 1 — DECOMPOSE
  Parse the user's query into logical sub-segments.
  For each sub-segment:
    • Identify: attribute groups, boolean relationships, date constraints
    • Preserve the user's EXACT phrasing — do not rephrase
    • Output a SubSegment object per identified dimension

STEP 2 — DATE RESOLUTION
  For each sub-segment containing date references:
    • Extract date patterns (relative: "last 30 days"; seasonal: "holiday season")
    • Call FISCAL_YEAR_CALENDAR tool with the exact phrase
    • Tool returns: {start_date, end_date, resolution_confidence}
    • If resolution_confidence < 0.7 → generate ONE clarifying question

STEP 3 — FACET MAPPING
  For each sub-segment, call FACET_SEARCH tool:
    • Pass the verbatim sub-segment text (not rephrased)
    • Accept results with score >= 0.6; reject below that
    • For each accepted result, mark as [RETRIEVED: facet_name]
    • Rejected/missing → generate ONE clarifying question per gap

STEP 4 — VALIDATE
  Before returning:
    • Verify all [RETRIEVED] facets are in TENANT_ALLOWED_FACETS
    • Verify boolean logic has no circular references
    • Call SIZE_ESTIMATOR tool → include estimated_size in output

STEP 5 — FORMAT
  Return JSON matching output_schema below.

OUTPUT SCHEMA:
{
  "segment_definition": {
    "name": "string",
    "sub_segments": [{"id": "string", "original_query": "string", "facets": [...], "logic": "string"}],
    "boolean_relationship": "AND | OR | NOT"
  },
  "estimated_size": "integer",
  "confidence_score": "float 0-1",
  "clarification_questions": ["string"],
  "audit_trail": [{"facet": "string", "source": "RETRIEVED|INFERRED", "score": "float"}]
}

CONSTRAINTS:
  • maximum 5 sub-segments per segment
  • minimum segment size: 1000 customers (warn if below)
  • date range must fall within data availability: 2020-01-01 to present

[END LAYER 2]

─────────────────────────────────────────────────────────
[LAYER 3 — INJECTED CONTEXT — changes per request]
─────────────────────────────────────────────────────────
## USER CONTEXT (from long-term memory — user_id: user_B)

Learned preferences (apply unless user overrides in this session):
  • "high value" → purchase_value >= 200 USD (observed 3 times, confidence: 0.91)
  • "recent" → last 30 days (observed 2 times, confidence: 0.85)
  • "holiday season" → Nov 15 – Jan 5 (RetailCo fiscal definition, confidence: 1.0)

Vocabulary map (user-specific terminology):
  • "big spenders" → purchase_value >= 200 USD
  • "loyal customers" → purchase_frequency >= 2/month for 6+ months

## SIMILAR PAST SEGMENTS (retrieved from episodic memory — top 2 matches)

[Similar Segment 1 — quality_score: 0.92, created 2025-11-03]
  Query: "high-value customers who bought baby gear in Q4"
  Result: {purchase_value >= 200, product_category = "baby_products",
           date_range: 2025-10-01 to 2025-12-31, estimated_size: 45,200}
  User confirmed without edits. ✓

[Similar Segment 2 — quality_score: 0.87, created 2025-08-15]
  Query: "holiday shoppers who spent over $150"
  Result: {purchase_value >= 150, date_range: 2024-11-15 to 2025-01-05,
           estimated_size: 132,000}
  User confirmed after 1 edit (changed value threshold from 150 to 200). Note: user prefers $200+

## RESOLVED AMBIGUITIES THIS SESSION (from short-term memory)
  (None yet — this is the first query in this session)

## TENANT CONFIG — RetailCo
  Allowed facets: [purchase_value, product_category, purchase_frequency,
                   customer_tenure, geographic_region, device_type, ...]
  Restricted facets: [income_estimate, credit_score, health_interest]
  Date convention: fiscal_year_start = Feb 1, holiday_season = Nov15–Jan5
  Min segment size: 1000 customers

[END LAYER 3]

══════════════════════════════════════════════════════════
USER MESSAGE (Layer 4)
══════════════════════════════════════════════════════════
create a segment of high-value baby product buyers from last holiday season
══════════════════════════════════════════════════════════
```

### 1.3 What the LLM Produces (the response)

Because memory provided `"high value" → $200+` and `"holiday season" → Nov 15–Jan 5`, the LLM produces:

```json
{
  "segment_definition": {
    "name": "High-Value Baby Product Buyers — Holiday 2024",
    "sub_segments": [
      {
        "id": "Seg-1",
        "original_query": "high-value baby product buyers from last holiday season",
        "facets": [
          {"name": "purchase_value",     "operator": ">=", "value": 200,
           "source": "[RETRIEVED: purchase_value]", "score": 0.96},
          {"name": "product_category",   "operator": "=",  "value": "baby_products",
           "source": "[RETRIEVED: product_category]", "score": 0.94},
          {"name": "transaction_date",   "operator": "BETWEEN",
           "value": {"start": "2024-11-15", "end": "2025-01-05"},
           "source": "[RETRIEVED: fiscal_calendar — holiday_season]", "score": 1.0}
        ],
        "logic": "purchase_value >= 200 AND product_category = 'baby_products' AND transaction_date BETWEEN '2024-11-15' AND '2025-01-05'"
      }
    ],
    "boolean_relationship": "AND"
  },
  "estimated_size": 38400,
  "confidence_score": 0.96,
  "clarification_questions": [],
  "audit_trail": [
    {"facet": "purchase_value >= 200", "source": "RETRIEVED", "score": 0.96,
     "memory_note": "Applied user preference: high_value = $200+ (confidence 0.91)"},
    {"facet": "product_category = baby_products", "source": "RETRIEVED", "score": 0.94},
    {"facet": "holiday_season dates", "source": "RETRIEVED", "score": 1.0,
     "memory_note": "Applied tenant fiscal calendar: Nov 15 – Jan 5"}
  ]
}
```

**Zero clarification questions.** The user gets a result in one turn because memory answered two of the three potential ambiguities (`"high value"` and `"holiday season"`) before the LLM even had to ask.

---

## 2. Skills → Static Prompt Integration

### 2.1 The Assembly Mechanism

The static prompt and skill are assembled in `prompt_assembler.py` before every LLM call. The assembler is responsible for:

1. Loading the static prompt from disk/cache
2. Loading the active skill from the Skill Registry
3. Injecting memory context (Section 3)
4. Building the messages list for the LLM API call

```python
# prompt_assembler.py — the central integration point

from dataclasses import dataclass
from typing import Optional
import anthropic  # or openai — both work identically here

@dataclass
class AssembledPrompt:
    system_message: str     # Layers 1 + 2 + Layer 3 static parts
    injected_context: str   # Layer 3 dynamic parts (memory, few-shots)
    conversation_history: list[dict]  # Prior turns
    user_message: str       # Current user query

class PromptAssembler:
    def __init__(
        self,
        static_prompt_path: str,
        skill_registry: "SkillRegistry",
        memory_manager: "MemoryManager",
        rag_retriever: "GroundedRetriever",
    ):
        self.static_prompt = open(static_prompt_path).read()  # loaded once at startup
        self.skill_registry = skill_registry
        self.memory = memory_manager
        self.rag = rag_retriever

    async def assemble(
        self,
        user_query: str,
        intent: str,
        session: "SessionContainer",
        tenant_id: str,
    ) -> AssembledPrompt:

        # ── Layer 2: Load the right skill for this intent ─────────────────
        skill = await self.skill_registry.route_to_skill(intent, tenant_id)
        skill_block = self._format_skill_block(skill)

        # ── Layer 3: Load injected context ────────────────────────────────
        context_block = await self._build_context_block(
            user_query, session, tenant_id
        )

        # ── Compose system message (Layers 1 + 2 + static parts of 3) ────
        system_message = "\n\n".join([
            "[LAYER 1 — STATIC PROMPT — CACHED]\n" + self.static_prompt,
            "[LAYER 2 — ACTIVE SKILL: " + skill.skill_id + " v" + skill.version + "]\n" + skill_block,
            context_block,
        ])

        return AssembledPrompt(
            system_message=system_message,
            injected_context=context_block,
            conversation_history=session.state.nsc.conversational_history,
            user_message=user_query,
        )

    def _format_skill_block(self, skill: "Skill") -> str:
        """
        Converts a Skill object into the prompt text that appears in
        the system message. This is the 'injection point' for skills.
        """
        return f"""## ACTIVE SKILL: {skill.name}

PROCEDURE:
{skill.instructions}

OUTPUT SCHEMA:
{skill.output_schema_as_yaml()}

CONSTRAINTS:
{chr(10).join("  • " + c for c in skill.constraints)}
"""

    async def _build_context_block(
        self,
        user_query: str,
        session: "SessionContainer",
        tenant_id: str,
    ) -> str:
        """
        Builds the memory + RAG context block (Layer 3).
        This is where memory and knowledge get injected into the prompt.
        """
        user_id = session.state.user.user_id

        # Read from long-term memory
        prefs = await self.memory.get_user_preferences(user_id)
        similar = await self.memory.find_similar_segments(user_query, tenant_id, top_k=2)
        ambiguity_history = await self.memory.get_ambiguity_history(session.container_id)
        tenant_config = await self.memory.get_tenant_config(tenant_id)

        sections = ["[LAYER 3 — INJECTED CONTEXT]"]

        if prefs.has_preferences():
            sections.append(
                "## USER CONTEXT (from long-term memory)\n"
                "Learned preferences:\n"
                + "\n".join(f"  • {k} → {v.value} (confidence: {v.confidence:.2f})"
                            for k, v in prefs.items())
            )

        if similar:
            sections.append(
                "## SIMILAR PAST SEGMENTS (top matches from memory)\n"
                + "\n".join(self._format_similar_segment(s) for s in similar)
            )

        if ambiguity_history:
            sections.append(
                "## RESOLVED AMBIGUITIES THIS SESSION\n"
                + "\n".join(f"  • {r.question} → {r.resolution}" for r in ambiguity_history)
            )

        sections.append(
            f"## TENANT CONFIG — {tenant_config.name}\n"
            f"  Allowed facets: {tenant_config.allowed_facets[:10]}...\n"
            f"  Date convention: {tenant_config.date_conventions}"
        )

        return "\n\n".join(sections)
```

### 2.2 How Prompt Caching Works With This Assembly

The key insight: **Layer 1 (static prompt) and Layer 2 (skill) change rarely. Layer 3 and Layer 4 change every request.** Prompt caching is applied at the boundary between static and dynamic content.

```python
# llm_caller.py — how the assembled prompt maps to the API call

async def call_llm_with_cache(
    assembled: AssembledPrompt,
    model_id: str,
    client: anthropic.AsyncAnthropic,
) -> str:
    """
    Uses Anthropic prompt caching to avoid re-processing Layers 1+2 on every call.
    Cache saves ~60-70% of input token costs for the static portion.
    """
    messages = [
        # Prior conversation turns (no caching — changes every turn)
        *assembled.conversation_history,
        # Current user message (no caching — unique every request)
        {"role": "user", "content": assembled.user_message},
    ]

    response = await client.messages.create(
        model=model_id,
        max_tokens=4096,
        system=[
            # Layer 1: Static prompt — mark for caching
            # Stays in cache for 5 minutes; 90% cost reduction on cache hit
            {
                "type": "text",
                "text": assembled.system_message.split("[LAYER 2")[0],  # Layer 1 only
                "cache_control": {"type": "ephemeral"},  # ← cache this block
            },
            # Layer 2: Skill — mark for caching (skills change infrequently)
            {
                "type": "text",
                "text": "[LAYER 2" + assembled.system_message.split("[LAYER 2")[1].split("[LAYER 3")[0],
                "cache_control": {"type": "ephemeral"},  # ← cache this block too
            },
            # Layer 3: Memory/context — NOT cached (changes every request)
            {
                "type": "text",
                "text": "[LAYER 3" + assembled.system_message.split("[LAYER 3")[1],
                # No cache_control — this block is always freshly processed
            },
        ],
        messages=messages,
    )

    return response.content[0].text
```

**Cost profile per request:**
- Layer 1 (12KB static) — `cache_read` after first request: ~$0.00005
- Layer 2 (2KB skill) — `cache_read` after first request: ~$0.000009
- Layer 3 (1-3KB context) — always `input_tokens`: ~$0.00003–0.00009
- Layer 4 (0.2-2KB conversation) — always `input_tokens`: ~$0.000002–0.00002
- **Total per request: ~$0.00004–0.00015** (vs. ~$0.0003–0.0006 without caching)

### 2.3 Skill Selection — How the Router Picks the Right Skill

```python
# The intent → skill mapping. This is the "routing logic" that replaces
# the current hard-coded SegmentCreatedFlag routing.

class SkillRouter:
    # Hard-coded mapping for reliability (can add vector routing later)
    INTENT_TO_SKILL = {
        "create_segment":     "segment_creation",
        "edit_segment":       "segment_editing",
        "delete_segment":     "segment_management",
        "query_segment":      "segment_query",
        "hypothesis_assess":  "hypothesis_assessment",
        "out_of_scope":       "out_of_scope_handler",
    }

    async def route(self, intent: str, tenant_id: str) -> "Skill":
        skill_id = self.INTENT_TO_SKILL.get(intent, "segment_creation")

        # Load skill — tenant override applied inside load_skill()
        skill = await self.registry.load_skill(
            skill_id=skill_id,
            tenant_id=tenant_id,
            version="latest_passing",  # Always use latest eval-passing version
        )

        # Verify the skill passes its eval gate before using it
        if not skill.eval_gate_passed:
            # Fall back to the previous passing version
            skill = await self.registry.load_skill(skill_id, tenant_id, version="previous_passing")

        return skill
```

---

## 3. Memory Integration — Read, Inject, Write Back

Memory participates in every request at three distinct moments:

```
REQUEST START                    DURING REQUEST                  REQUEST END
      │                               │                               │
      ▼                               ▼                               ▼
┌─────────────┐              ┌────────────────┐              ┌───────────────────┐
│ MEMORY READ │              │ AMBIGUITY CHECK │              │ MEMORY WRITE-BACK │
│             │              │                │              │                   │
│ Load:       │              │ Before asking  │              │ Write:            │
│ • user prefs│              │ any question:  │              │ • new preferences │
│ • similar   │              │ check if       │              │ • successful seg  │
│   segments  │              │ already asked  │              │ • feedback signal │
│ • ambiguity │              │ or user pref   │              │ • ambiguity resol.│
│   history   │              │ answers it     │              │                   │
└─────────────┘              └────────────────┘              └───────────────────┘
```

### 3.1 Memory Read at Request Start

```python
# request_handler.py — the entry point for every request

async def handle_request(
    raw_request: RawRequest,
    memory: MemoryManager,
    session_registry: SessionRegistry,
    assembler: PromptAssembler,
) -> Response:

    # ── STEP 1: Load or create session ───────────────────────────────────
    session = session_registry.get(raw_request.session_id)
    if not session:
        user = UserContext(
            user_id=raw_request.user_id,
            user_type=raw_request.user_type,
            session_id=raw_request.session_id,
        )
        session = session_registry.create(user, raw_request.query, raw_request.caller)

    # ── STEP 2: MEMORY READ — happens before ANY LLM call ────────────────
    memory_snapshot = await memory.read_for_request(
        user_id=raw_request.user_id,
        session_id=raw_request.session_id,
        query=raw_request.query,
        tenant_id=raw_request.tenant_id,
    )
    # memory_snapshot contains:
    # .user_preferences  — long-term learned preferences
    # .similar_segments  — top-3 similar past segments from Milvus
    # .ambiguity_history — what was asked/resolved this session
    # .tenant_config     — tenant's facet restrictions + conventions

    # Attach snapshot to session for use during the request
    session.memory_snapshot = memory_snapshot

    # ── STEP 3: Classify intent ──────────────────────────────────────────
    intent = await intent_classifier.classify(raw_request.query)

    # ── STEP 4: Assemble prompt (memory is injected here) ────────────────
    assembled = await assembler.assemble(
        user_query=raw_request.query,
        intent=intent,
        session=session,
        tenant_id=raw_request.tenant_id,
    )

    # ── STEP 5: LLM call ─────────────────────────────────────────────────
    raw_output = await llm_caller.call(assembled, model=skill.model_tier)

    # ── STEP 6: Parse + validate output ──────────────────────────────────
    segment_output = SegmentOutput.model_validate_json(raw_output)
    validation_result = await validator.validate(segment_output, raw_request.tenant_id)

    if not validation_result.is_valid:
        # Retry with error context (max 2 retries)
        segment_output = await retry_with_error(assembled, segment_output, validation_result)

    # ── STEP 7: MEMORY WRITE-BACK — happens after successful output ───────
    await memory.write_back_after_request(
        user_id=raw_request.user_id,
        session_id=raw_request.session_id,
        query=raw_request.query,
        output=segment_output,
        tenant_id=raw_request.tenant_id,
    )

    return Response(output=segment_output, session_id=session.container_id)
```

### 3.2 The Memory Read Function in Detail

```python
# memory/manager.py — read_for_request

async def read_for_request(
    self,
    user_id: str,
    session_id: str,
    query: str,
    tenant_id: str,
) -> MemorySnapshot:
    """
    Called once per request, before any LLM call.
    Returns everything memory can contribute to improving this request.
    """

    # Parallel reads across all memory stores
    prefs, similar, ambiguity_history, tenant_config = await asyncio.gather(
        # Long-term: user's learned preferences (PostgreSQL)
        self.get_user_preferences(user_id),

        # Long-term: similar successful segments (Milvus vector search)
        self.find_similar_segments(query, tenant_id, top_k=2),

        # Short-term: what was asked/resolved in this session (Redis)
        self.get_ambiguity_history(session_id),

        # Config: tenant-specific rules and vocabulary (Redis cache → PostgreSQL)
        self.get_tenant_config(tenant_id),
    )

    return MemorySnapshot(
        user_preferences=prefs,
        similar_segments=similar,
        ambiguity_history=ambiguity_history,
        tenant_config=tenant_config,
        # For audit: which memory sources contributed to this request
        sources_used=[
            "long_term_preferences" if prefs.has_preferences() else None,
            "episodic_segments" if similar else None,
            "session_ambiguity" if ambiguity_history else None,
        ],
    )
```

### 3.3 The Memory Write-Back Function in Detail

```python
# memory/manager.py — write_back_after_request

async def write_back_after_request(
    self,
    user_id: str,
    session_id: str,
    query: str,
    output: "SegmentOutput",
    tenant_id: str,
) -> None:
    """
    Called after every successful request.
    Updates memory with what was learned from this interaction.
    """

    write_tasks = []

    # ── 1. Extract and update user preferences ────────────────────────────
    # Did the user reveal a preference in this query that memory didn't know?
    new_prefs = self._extract_preferences_from_output(query, output)
    for pref_key, pref_value, confidence in new_prefs:
        write_tasks.append(
            self.update_user_preference(user_id, pref_key, pref_value, confidence)
        )
    # Example: query contained "high value" → facet "purchase_value >= 200"
    # → stores: user_prefs["high_value_threshold"] = 200, confidence += 0.3

    # ── 2. Store ambiguity resolutions (short-term, session-scoped) ───────
    for qa in output.resolved_ambiguities:
        write_tasks.append(
            self.record_ambiguity_resolution(session_id, qa.question, qa.resolution)
        )
    # Prevents: same question being asked twice in same session

    # ── 3. Queue the segment for long-term episodic memory ────────────────
    # (Only stored after user confirms — see feedback section)
    write_tasks.append(
        self._queue_for_episodic_store(query, output, user_id, tenant_id)
    )

    # ── 4. Update conversation history (short-term) ───────────────────────
    write_tasks.append(
        self.append_to_conversation_history(
            session_id,
            user_turn=query,
            agent_turn=output.to_display_string(),
        )
    )

    # Execute all writes in parallel
    await asyncio.gather(*write_tasks)

async def confirm_segment_to_episodic_memory(
    self,
    segment_id: str,
    user_id: str,
    tenant_id: str,
    quality_signal: float,  # 1.0 if confirmed without edit, 0.6 if confirmed after edits
) -> None:
    """
    Called when user CONFIRMS a segment (clicks "Save" / "Use this segment").
    Only confirmed segments enter long-term episodic memory —
    unconfirmed attempts are discarded.
    This prevents memory from being polluted with bad examples.
    """
    queued = await self._get_queued_segment(segment_id)
    if not queued:
        return

    if quality_signal >= 0.7:  # Only store high-quality examples
        embedding = await self.embed(queued.query)
        await self.milvus.insert(
            collection="successful_segments",
            data=[{
                "segment_id": segment_id,
                "query": queued.query,
                "definition": queued.output.model_dump_json(),
                "quality_score": quality_signal,
                "user_id": user_id,
                "tenant_id": tenant_id,
                "embedding": embedding,
                "created_at": datetime.utcnow().isoformat(),
            }]
        )
        # Now this segment will appear as a "similar past segment" for future
        # queries that semantically resemble this one
```

### 3.4 The Ambiguity Resolution Flow — Eliminating Duplicate Questions

This is the most immediately impactful memory integration. Here is the exact flow that eliminates the "#1 UX complaint" (users answering the same question twice):

```python
# reasoning/ambiguity_resolver.py — called by ANY agent before generating a question

async def should_ask_or_resolve(
    self,
    session_id: str,
    user_id: str,
    ambiguity_type: str,          # e.g., "date_recency", "high_value_threshold"
    proposed_question: str,        # What the agent wants to ask
    inferred_value: Optional[str], # What the agent would guess if not asking
    inference_confidence: float,
) -> AmbiguityDecision:
    """
    Called by: decomposer, date_tagger, facet_mapper — any agent that wants to ask a question.
    Returns: either the resolved answer (from memory) or permission to ask.
    """

    # CHECK 1: Was this EXACT ambiguity resolved earlier in this session?
    session_history = await self.memory.get_ambiguity_history(session_id)
    for record in session_history:
        if self._semantically_same(proposed_question, record.question):
            return AmbiguityDecision(
                action="USE_SESSION_MEMORY",
                resolved_value=record.resolution,
                source="session_ambiguity_history",
                # The agent uses this value without asking the user
            )

    # CHECK 2: Does this user have a stored long-term preference?
    prefs = await self.memory.get_user_preferences(user_id)
    if ambiguity_type in prefs:
        pref = prefs[ambiguity_type]
        if pref.confidence >= 0.8:
            return AmbiguityDecision(
                action="USE_LONG_TERM_MEMORY",
                resolved_value=pref.value,
                source=f"user_preference (confidence: {pref.confidence:.0%})",
                # Still show the user what was assumed — transparency
                assumption_note=f"Using your saved preference: {ambiguity_type} = {pref.value}",
            )

    # CHECK 3: Is the inference confidence high enough to just infer?
    if inferred_value and inference_confidence >= 0.85:
        return AmbiguityDecision(
            action="INFER",
            resolved_value=inferred_value,
            source="context_inference",
            assumption_note=f"Inferred from context: {inferred_value} (confidence: {inference_confidence:.0%})",
        )

    # DEFAULT: Ask the user — but register the question so no other agent asks it again
    await self.memory.record_ambiguity_asked(session_id, ambiguity_type, proposed_question)
    return AmbiguityDecision(
        action="ASK",
        question=proposed_question,
    )
```

---

## 4. The Full Request Flow — One Request, All Systems

This is the complete end-to-end flow for a single user request, showing when each system is active:

```
USER QUERY: "create a segment of high-value baby product buyers from last holiday season"
                │
                ▼
┌───────────────────────────────────────────────────────────────────────┐
│  PERCEPTION LAYER (~50ms, no LLM)                                     │
│  1. InputNormalizer: spellcheck, normalize                             │
│  2. IntentClassifier (GPT-4o-mini, fast cheap call): → create_segment │
│  3. PermissionChecker: user B has create permission ✓                  │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│  MEMORY GATEWAY — READ (~30ms, no LLM, parallel reads)                │
│  Redis: get session state, get ambiguity history → (empty, new session)│
│  PostgreSQL: get user_prefs → {high_value: $200, holiday: Nov15-Jan5} │
│  Milvus: find_similar_segments → [2 similar past segments returned]   │
│  PostgreSQL: get tenant_config RetailCo → {allowed_facets, date_conv} │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│  PROMPT ASSEMBLER (~5ms, no LLM)                                      │
│  Layer 1: Load static_system_prompt.txt from disk (cached in memory)  │
│  Layer 2: SkillRegistry.load("segment_creation", "RetailCo") → v3.2.1 │
│  Layer 3: Build context block from memory snapshot                    │
│  Output: AssembledPrompt (3,200 tokens total)                         │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│  REASONING ENGINE — LLM CALL (~3-5s)                                  │
│  Model: GPT-4o (segment_creation is complex → standard tier)          │
│  Prompt caching: Layer 1+2 served from cache → 70% token cost saved   │
│  LLM executes PLAN → ACT → VERIFY → IMPROVE loop:                     │
│                                                                        │
│  PLAN:  Identifies 3 dimensions: value, category, date                │
│  ACT:   Calls FACET_SEARCH("high-value baby product buyers") → hits   │
│         Calls FISCAL_YEAR_CALENDAR("last holiday season") → dates     │
│         Memory pre-resolved "high value" → $200 (no question needed)  │
│         Memory pre-resolved "holiday season" → Nov15-Jan5 (no question)│
│  VERIFY: All facets in RetailCo allowed list ✓                        │
│          Logic valid ✓                                                 │
│          SIZE_ESTIMATOR returns 38,400 → above min 1,000 ✓           │
│  Output: JSON with 0 clarification questions                           │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│  VALIDATION LAYER (~20ms, no LLM)                                     │
│  StateValidator.validate(output, DSE_REQUIREMENTS) — all pass         │
│  SegmentValidator.validate(segment_def, tenant_id) — all pass         │
│  Grounding check: 3/3 facets are [RETRIEVED] → grounding_rate = 100% │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│  EVAL GATEWAY (~100ms, lightweight scoring)                            │
│  Phoenix tracer: captures span with all attributes                    │
│  LLM-as-judge scorer: rates output quality → 0.94                     │
│  Score logged to execution_traces table + Phoenix                     │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│  PERSISTENCE LAYER                                                     │
│  PostgreSQL: save segment_definition, audit_trail, skill_version used │
│  PostgreSQL: save segment_audit record (for explainability)           │
│  Redis: update session conversation history                           │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│  MEMORY GATEWAY — WRITE-BACK (~15ms, parallel writes)                 │
│  (Happens after response is returned to user, async)                  │
│  Redis: append to conversation history                                │
│  PostgreSQL: queue segment for episodic memory (pending confirmation) │
│  NOTE: long-term episodic memory ONLY written when user confirms      │
└───────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
              RESPONSE TO USER: Segment shown, zero questions asked
              Total time: ~4-6 seconds (vs. current 12-20 seconds)

══ USER CLICKS "USE THIS SEGMENT" ══
                │
                ▼
    memory.confirm_segment_to_episodic_memory(
        segment_id, quality_signal=1.0  # confirmed without edit
    )
    → Segment enters Milvus "successful_segments" collection
    → Will appear as a "similar past segment" for future queries
    → Memory now has one more example of what good looks like
```

---

## 5. The Self-Improvement Loop — Failure to Better Prompt

This is the closed feedback loop that makes the system get smarter over time. It has four stages:

```
STAGE 1: OBSERVE         STAGE 2: ANALYZE        STAGE 3: PROPOSE       STAGE 4: VALIDATE & PROMOTE
   │                         │                        │                        │
Production traces    →  Cluster failures    →  Generate candidate  →  Eval gate  →  New skill version
Phoenix captures         by error type           skill with LLM          passes?        promoted to
every request            Identify patterns        improved instructions     yes → deploy   production
                         Group by skill           Run LLM as optimizer     no → discard
```

### 5.1 Stage 1: Observe — Phoenix Captures Everything

Every request automatically generates a structured trace in Arize Phoenix:

```python
# This happens transparently via the EnhancedTracer context manager
# (defined in upgrade proposal §10)

# Example trace attributes captured per request:
{
    "trace_id": "tr_abc123",
    "skill_id": "segment_creation",
    "skill_version": "3.2.1",
    "tenant_id": "RetailCo",
    "user_id": "user_B",
    "session_id": "sess_xyz",
    "duration_ms": 4821,
    "llm_model": "gpt-4o",
    "input_tokens": 3240,
    "output_tokens": 412,
    "cache_read_tokens": 2900,
    "grounding_rate": 1.0,          # 3/3 facets RETRIEVED
    "clarification_questions": 0,
    "validation_passed": True,
    "llm_judge_score": 0.94,
    "memory_sources_used": ["long_term_preferences", "episodic_segments"],
    "user_feedback": None,          # pending — user hasn't rated yet
    "edit_count_after_creation": 0, # user confirmed without editing
}
```

### 5.2 Stage 2: Analyze — Failure Clustering

Run weekly (automated job):

```python
# improvement/failure_analyzer.py

async def run_weekly_failure_analysis(self) -> Dict[str, FailureReport]:
    """
    Pulls the past 7 days of traces from Phoenix/PostgreSQL.
    Groups failures by skill and failure type.
    Returns a FailureReport per skill that had >10 failures.
    """
    reports = {}

    for skill_id in await self.get_active_skills():
        # Pull low-quality traces for this skill
        failures = await self.pg.fetch("""
            SELECT
                t.trace_id, t.session_id, t.input_query,
                t.llm_judge_score, t.grounding_rate,
                t.clarification_questions, t.validation_passed,
                f.rating, f.edit_count, f.comment
            FROM execution_traces t
            LEFT JOIN segment_feedback f USING (trace_id)
            WHERE t.skill_id = $1
              AND t.created_at > NOW() - INTERVAL '7 days'
              AND (
                t.llm_judge_score < 0.7
                OR f.rating < 3
                OR f.edit_count > 2
                OR t.validation_passed = FALSE
              )
            ORDER BY t.llm_judge_score ASC
            LIMIT 100
        """, skill_id)

        if len(failures) < 10:
            continue  # Not enough failures to optimize from

        # Cluster failures by root cause
        failure_types = self._cluster_by_error_type(failures)
        # Returns: {"hallucinated_facet": [list of traces],
        #           "wrong_date_resolution": [list of traces],
        #           "missing_sub_segment": [list of traces], ...}

        reports[skill_id] = FailureReport(
            skill_id=skill_id,
            total_failures=len(failures),
            failure_clusters=failure_types,
            most_common_type=max(failure_types, key=lambda k: len(failure_types[k])),
            sample_failures={k: v[:3] for k, v in failure_types.items()},
        )

    return reports
```

### 5.3 Stage 3: Propose — LLM as Prompt Optimizer

```python
# improvement/prompt_optimizer.py

async def propose_skill_improvement(
    self,
    current_skill: Skill,
    failure_report: FailureReport,
) -> Skill:
    """
    Uses an LLM (Claude Opus for best reasoning) to propose
    specific improvements to the skill instructions based on
    the observed failure patterns.
    """

    # Build a summary of failures for the optimizer LLM
    failure_summary = self._build_failure_summary(failure_report)
    # Example failure_summary:
    # "Most common failure (37 cases): hallucinated_facet
    #   Pattern: when user says 'loyalty' or 'loyal customers', the agent
    #   maps to 'customer_loyalty_score' which does NOT exist in the catalog.
    #   The correct facet is 'purchase_frequency >= 2/month for 6+ months'.
    #   3 example failure traces: [...]"

    optimization_prompt = f"""You are a prompt optimization expert for an enterprise AI system.

Current skill instructions for "{current_skill.name}":
───────────────────────────────────────────────────
{current_skill.instructions}
───────────────────────────────────────────────────

Observed failure patterns over the past 7 days ({failure_report.total_failures} failures):
───────────────────────────────────────────────────
{failure_summary}
───────────────────────────────────────────────────

TASK: Propose specific, targeted improvements to the skill instructions that would
prevent the observed failure patterns. Be concrete — add specific examples,
clarify ambiguous steps, or add explicit edge case handling.

RULES:
- Do not change the overall structure (5-step procedure)
- Only modify instructions for the failing steps
- Add examples for the specific patterns that fail
- Return the complete improved instructions text (not a diff)
"""

    improved_instructions = await self.llm.complete(
        prompt=optimization_prompt,
        model="claude-opus-4-6",  # Use best model for optimization
        max_tokens=3000,
    )

    # Create candidate skill with new version
    candidate_version = current_skill.bump_patch_version()  # e.g., 3.2.1 → 3.2.2
    candidate_skill = Skill(
        **current_skill.dict(exclude={"instructions", "version"}),
        instructions=improved_instructions,
        version=candidate_version,
        eval_gate_passed=False,  # Must pass eval gate before going live
    )

    # Save candidate to registry (draft status — not yet active)
    await self.registry.save_skill(candidate_skill, status="draft")

    return candidate_skill
```

### 5.4 Stage 4: Validate and Promote — The Eval Gate

```python
# improvement/promotion_pipeline.py

async def run_optimization_cycle(self, skill_id: str) -> PromotionResult:
    """
    Full weekly optimization cycle for one skill.
    Analyze → Propose → Eval → Human Review → Promote (or Discard)
    """

    current_skill = await self.registry.load_skill(skill_id, version="latest_passing")
    failure_report = await self.failure_analyzer.get_report(skill_id)

    if not failure_report or failure_report.total_failures < 10:
        return PromotionResult(action="skipped", reason="insufficient_failure_data")

    # Stage 3: Propose improvement
    candidate = await self.optimizer.propose_skill_improvement(current_skill, failure_report)

    # ── EVAL GATE 1: Does the candidate meet minimum accuracy? ────────────
    eval_result = await self.eval_runner.run_skill_eval(
        skill_id=skill_id,
        version=candidate.version,
        eval_suite_id=current_skill.eval_suite_id,
    )
    if not eval_result.passed:
        await self.registry.update_skill_status(candidate, "rejected_eval_gate_1")
        return PromotionResult(
            action="rejected",
            reason=f"candidate accuracy {eval_result.mean_accuracy:.1%} < "
                   f"minimum {current_skill.min_accuracy:.1%}"
        )

    # ── EVAL GATE 2: Regression check — candidate must not be worse ───────
    regression = await self.eval_runner.run_regression_check(
        skill_id=skill_id,
        new_version=candidate.version,
        baseline_version=current_skill.version,
    )
    if not regression.passed:
        await self.registry.update_skill_status(candidate, "rejected_regression")
        return PromotionResult(
            action="rejected",
            reason=f"regression detected: {[r.metric for r in regression.regressions]}"
        )

    # ── HUMAN REVIEW (required before promotion) ──────────────────────────
    # Send to review queue — an engineer must approve before it goes live
    review_ticket = await self.review_queue.create(
        candidate_skill=candidate,
        eval_result=eval_result,
        regression_result=regression,
        failure_report=failure_report,
        instructions_diff=self._diff_instructions(current_skill, candidate),
    )
    # review_ticket goes to Slack / Jira / email with:
    # • Side-by-side diff of old vs new instructions
    # • Eval scores (current: 84.2% → candidate: 87.1%)
    # • Regression results (no regressions detected)
    # • Sample failure cases it's designed to fix
    # Engineer approves or rejects in the review UI

    # If approved (engineer clicks "Promote" in the review UI):
    if await self.review_queue.wait_for_decision(review_ticket.id, timeout_hours=48):
        await self.registry.promote_skill(candidate, status="active")
        return PromotionResult(
            action="promoted",
            new_version=candidate.version,
            accuracy_delta=eval_result.mean_accuracy - current_skill.last_eval_accuracy,
        )
    else:
        return PromotionResult(action="rejected_human_review")
```

### 5.5 The Complete Improvement Loop — End to End

```
WEEK 1, Monday:
  User query fails: "loyal customers who bought electronics this quarter"
  → facet_mapper hallucinates "customer_loyalty_score" (doesn't exist)
  → llm_judge_score: 0.42 (low)
  → Segment saved to execution_traces with score 0.42
  → User edits 3 times before accepting (edit_count=3, signal: dissatisfied)
  → Phoenix captures all of this

WEEK 1 (3 similar failures throughout the week):
  All 4 failures show same pattern: "loyalty" → hallucinated facet

WEEK 1, Sunday (automated weekly job):
  FailureAnalyzer pulls last 7 days of traces for skill "segment_creation"
  Finds: 4 failures, all "hallucinated_facet" type, all involve "loyalty"
  → Creates FailureReport: most_common_type="hallucinated_facet", sample=[4 traces]

WEEK 2, Monday 2am (automated):
  PromptOptimizer.propose_skill_improvement(segment_creation_v3.2.1, failure_report)
  LLM (Claude Opus) reads failure report, proposes:
    "Add to STEP 3 — FACET MAPPING:
    IMPORTANT: 'loyal customers', 'loyalty', 'loyal' do NOT map to any facet
    called 'customer_loyalty_score'. The correct representation is:
    purchase_frequency >= 2 AND customer_tenure >= 6 months.
    Always call FACET_SEARCH first; never invent facet names."
  → Saves as segment_creation_v3.2.2 (draft)

WEEK 2, Monday 3am (automated):
  EvalRunner.run_skill_eval("segment_creation", "3.2.2"):
    Runs against 300-case eval suite
    Old score: 84.2%, New score: 87.1% → passes min_accuracy (82%)
  RegressionCheck: no metric dropped >3% → passes
  → Sends review ticket to engineer with diff + scores

WEEK 2, Tuesday:
  Engineer reviews diff (5 minute task):
    Old: "For each sub-segment, call FACET_SEARCH tool"
    New: "For each sub-segment, call FACET_SEARCH tool
          IMPORTANT: 'loyal customers'... → purchase_frequency >= 2..."
  Looks good. Engineer clicks "Promote".

WEEK 2, Tuesday afternoon:
  segment_creation_v3.2.2 is now ACTIVE
  All new requests use v3.2.2
  The "loyalty" hallucination no longer occurs

WEEK 3:
  "loyal customers" queries now resolve correctly
  llm_judge_scores for these queries: 0.89 (up from 0.42)
  The system improved itself, with human oversight, in 8 days
```

---

## 6. The Three Integration Contracts (Summary)

These are the exact interfaces between the three systems. Get these right and everything else follows.

### Contract 1: Static Prompt ↔ Skill

```
STATIC PROMPT provides:          SKILL provides:
  • Identity ("You are...")        • PROCEDURE (step-by-step instructions)
  • Operating loop (PLAN→ACT→     • INPUT SCHEMA (what the LLM receives)
    VERIFY→IMPROVE)                • OUTPUT SCHEMA (what the LLM must return)
  • Grounding rules               • CONSTRAINTS (edge cases, limits)
  • Output format requirements    • MODEL TIER (which model handles this skill)
  • Safety rules                  • EVAL SUITE ID (how the skill is tested)

INTEGRATION POINT:
  PromptAssembler._format_skill_block(skill)
  → Injects skill content into system message under "[LAYER 2 — ACTIVE SKILL]"
  → Skill instructions are appended AFTER static prompt, BEFORE memory context

CACHING BOUNDARY:
  Layer 1 (static) → cached separately
  Layer 2 (skill) → cached separately
  Both invalidate independently when updated
```

### Contract 2: Memory ↔ Assembled Prompt

```
MEMORY provides:                    PROMPT consumes:
  • User preferences                  • [LAYER 3 — INJECTED CONTEXT] block
    (long-term, PostgreSQL)           • "Learned preferences:" section
  • Similar past segments             • "Similar past segments:" section
    (long-term, Milvus)               • (used as dynamic few-shot examples)
  • Ambiguity history                 • "Resolved ambiguities this session:" section
    (short-term, Redis)               • (prevents duplicate questions)
  • Tenant config                     • "Tenant config:" section
    (config, cached)                  • (enforces allowed facets, date conventions)

INTEGRATION POINT:
  PromptAssembler._build_context_block(query, session, tenant_id)
  → Calls MemoryManager.read_for_request() in parallel
  → Formats results into LAYER 3 text block
  → Injected as part of system message (NOT as user message)

WRITE-BACK TRIGGER:
  After successful output → MemoryManager.write_back_after_request()
  After user confirmation → MemoryManager.confirm_segment_to_episodic_memory()
  After user feedback → FeedbackCollector.record_explicit_feedback()
```

### Contract 3: Failure Signal ↔ Skill Improvement

```
IMPROVEMENT reads from:              IMPROVEMENT writes to:
  • execution_traces table           • skills table (new draft version)
    (Phoenix traces + scores)        • eval_results table
  • segment_feedback table           • review_queue (pending human approval)
    (user ratings, edit counts)      • skills table (promoted version)
  • skill registry (current version)

INTEGRATION POINT:
  PromotionPipeline.run_optimization_cycle(skill_id)
  → Runs weekly per skill
  → Uses FailureAnalyzer to identify patterns
  → Uses PromptOptimizer (LLM call) to propose improvements
  → Uses EvalRunner to gate on quality
  → Uses ReviewQueue to require human approval
  → Uses SkillRegistry.promote_skill() to make active

FEEDBACK LOOP TIMING:
  Failure occurs → captured in Phoenix (real-time)
  Weekly analyzer runs → identifies patterns (T+7 days)
  Candidate proposed → (T+7 days + 1 hour)
  Eval gate runs → (T+7 days + 2 hours)
  Human review → (T+7 days + 1-2 business days)
  Promoted to production → (T+8-10 days from failure)
```

---

## The Flywheel Diagram

When all three integrations are working, they form a compounding flywheel:

```
                        ┌─────────────────────┐
                        │    USER REQUEST      │
                        └──────────┬──────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │                             │
              MEMORY READ                  SKILL LOADED
            (user prefs +                (static + skill
           similar segments)              assembled)
                    │                             │
                    └──────────────┬──────────────┘
                                   │
                          ┌────────▼────────┐
                          │   LLM REASONS   │
                          │   (PLAN→ACT→    │
                          │   VERIFY→IMPROVE)│
                          └────────┬────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │                             │
              MEMORY WRITE                  PHOENIX TRACES
             (preferences +                (every request
           episodic store)                   scored)
                    │                             │
                    │                   ┌─────────▼─────────┐
                    │                   │  WEEKLY ANALYSIS   │
                    │                   │  (failure clusters) │
                    │                   └─────────┬─────────┘
                    │                             │
                    │                   ┌─────────▼─────────┐
                    │                   │  SKILL IMPROVED   │
                    │                   │  (LLM proposes +  │
                    │                   │  human approves)  │
                    │                   └─────────┬─────────┘
                    │                             │
                    └──────────────┬──────────────┘
                                   │
                    Better memory + better skill → better next request
                    The system improves with every confirmed interaction.
```

---

*Document produced as part of Enterprise Agentic Research — Research 1 (Sonnet Claude)*
*This document stitches together: 01_bottleneck_analysis.md, 03_concrete_upgrade_proposal.md, 04_implementation_roadmap.md*
*Answers: (1) how skills integrate with static prompts, (2) how memory is used at request time, (3) how the self-improvement loop closes*
