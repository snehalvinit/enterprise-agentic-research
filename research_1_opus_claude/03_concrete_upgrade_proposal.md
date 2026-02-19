# Concrete Upgrade Proposal: Smart-Segmentation → Enterprise Agentic Platform

> **Research ID**: research_1_opus_claude
> **Model**: Claude Opus 4.6
> **Date**: February 2026
> **Status**: Complete

---

## Executive Summary

This document presents a detailed reformation plan to transform Smart-Segmentation from a functional segmentation tool into an enterprise-grade agentic customer segmentation platform. The architecture follows a **layered, modular design** with pluggable skills, enterprise memory, eval-first quality gates, auto-improvement loops, and multi-tenant support.

The proposal is organized in three layers of detail:
1. **High-Level Architecture** — System-wide design and principles
2. **Component Deep-Dives** — Detailed design for each module
3. **Concrete Transformation Examples** — Before/after code with specific changes

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Layer 1: Perception — Input Handling & Routing](#2-layer-1-perception--input-handling--routing)
3. [Layer 2: Reasoning — Planning & Cognitive Logic](#3-layer-2-reasoning--planning--cognitive-logic)
4. [Layer 3: Memory — Context Management](#4-layer-3-memory--context-management)
5. [Layer 4: Action — Skill Execution & Tool Calls](#5-layer-4-action--skill-execution--tool-calls)
6. [Layer 5: Feedback — Self-Assessment & Improvement](#6-layer-5-feedback--self-assessment--improvement)
7. [Skill Architecture](#7-skill-architecture)
8. [Knowledge System (RAG)](#8-knowledge-system-rag)
9. [Multi-Tenant Support](#9-multi-tenant-support)
10. [Evaluation Framework](#10-evaluation-framework)
11. [Observability & Cost](#11-observability--cost-optimization)
12. [Concrete Transformations](#12-concrete-transformations)
13. [Appendix: Cost-Saving Alternatives](#appendix-cost-saving-alternatives)

---

## 1. High-Level Architecture

### 1.1 Target Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ENTERPRISE AGENT PLATFORM                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │  PERCEPTION  │→│   REASONING   │→│    ACTION     │→│    FEEDBACK     │ │
│  │             │  │              │  │              │  │                 │ │
│  │ • Input     │  │ • Plan       │  │ • Execute    │  │ • Verify        │ │
│  │ • Normalize │  │ • Select     │  │ • Skills     │  │ • Evaluate      │ │
│  │ • Route     │  │   Skills     │  │ • Tools      │  │ • Improve       │ │
│  │ • Auth      │  │ • Reason     │  │ • Format     │  │ • Learn         │ │
│  └─────────────┘  └──────────────┘  └──────────────┘  └─────────────────┘ │
│         ↕                ↕                 ↕                  ↕            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     SHARED INFRASTRUCTURE                           │   │
│  │                                                                     │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ ┌───────────┐  │   │
│  │  │  MEMORY  │ │  SKILLS  │ │KNOWLEDGE │ │ TOOLS  │ │  TENANT   │  │   │
│  │  │  Store   │ │ Registry │ │  Store   │ │Registry│ │  Config   │  │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └────────┘ └───────────┘  │   │
│  │                                                                     │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────────┐    │   │
│  │  │  EVAL    │ │  MODEL   │ │OBSERV-   │ │   PROMPT           │    │   │
│  │  │  Engine  │ │  Router  │ │ ABILITY  │ │   Management       │    │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Design Principles

1. **Static Prompt, Dynamic Everything Else**: The core system prompt stays fixed. Skills, knowledge, tools, and tenant configs load dynamically at runtime.
2. **Eval-First**: Every change (prompt, skill, tool) must pass evaluations before deployment.
3. **Skill-Based Extension**: New capabilities = new skills, not new code in the agent core.
4. **Memory-Driven Learning**: The agent improves through accumulated memories, not prompt mutation.
5. **Multi-Tenant by Design**: Tenant isolation at every layer from the start.
6. **Plan-Act-Verify-Improve**: Every task follows this loop for reliability.

### 1.3 Static System Prompt (What Stays Fixed)

```
You are an enterprise customer segmentation agent. You help users create,
analyze, and refine customer segments using available data and tools.

OPERATING RULES (always follow):
1. PLAN: Before acting, state your plan and reasoning.
2. ACT: Execute using available skills and tools. Follow skill instructions exactly.
3. VERIFY: After acting, verify your output against the user's intent.
4. IMPROVE: If verification fails, identify the issue and retry (max 3 attempts).

GROUNDING RULES:
- Use tools and retrieved knowledge as your source of truth.
- Never invent facet names, values, or operators. Only use what's in the catalog.
- Cite the evidence for your choices (which facets, why those operators).

ESCALATION RULES:
- If missing required information, ask a clarifying question.
- If confidence is below threshold, present options to the user.
- If no relevant tools/knowledge found, say "I don't have enough information."

OUTPUT RULES:
- Follow the output schema provided by the active skill.
- Return structured JSON when the skill requires it.
- Include reasoning in your response so the user understands your choices.
```

**Everything else loads dynamically**: skills, knowledge, tool definitions, tenant policies.

---

## 2. Layer 1: Perception — Input Handling & Routing

### 2.1 Current State

```python
# Current: RouterAgent with hardcoded sub-agents
root_agent = LlmAgent(
    name="RouterAgent",
    instruction=ROOT_AGENT_INSTRUCTION,  # monolithic prompt
    sub_agents=[NewSegmentCreationAgent, DirectSegmentEditorAgent],
)
```

### 2.2 Proposed Design

```python
# Proposed: Intent classification + skill-based routing
class PerceptionLayer:
    def __init__(self, tenant_config: TenantConfig):
        self.intent_classifier = IntentClassifier(model="haiku")  # cheap model
        self.input_validator = InputValidator()
        self.tenant_config = tenant_config

    async def process(self, request: UserRequest) -> ProcessedInput:
        # 1. Validate input (prevent prompt injection)
        validated = self.input_validator.validate(request.content)

        # 2. Classify intent
        intent = await self.intent_classifier.classify(
            query=validated.content,
            history=request.conversation_history,
            available_skills=self.tenant_config.enabled_skills
        )

        # 3. Return normalized input
        return ProcessedInput(
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            intent=intent,
            query=validated.content,
            history=request.conversation_history,
            skill_candidates=intent.matched_skills,
            metadata=request.metadata
        )
```

### 2.3 Intent Classification

```python
class IntentClassifier:
    """Uses a fast, cheap model for intent classification."""

    INTENTS = [
        "create_segment",      # New segment creation
        "edit_segment",        # Modify existing segment
        "analyze_segment",     # Analyze segment characteristics
        "hypothesis_test",     # Test a user hypothesis
        "explore_data",        # Explore available data/facets
        "campaign_recommend",  # Recommend campaign parameters
        "greeting",            # Social interaction
        "out_of_scope",        # Not related to segmentation
        "clarification",       # Response to agent's question
    ]

    async def classify(self, query, history, available_skills) -> Intent:
        response = await self.model.generate(
            system="Classify the user's intent. Return a JSON object.",
            user=f"Query: {query}\nHistory: {history}\nAvailable skills: {available_skills}",
            response_model=IntentResult  # Pydantic model
        )
        return response
```

### 2.4 Input Validation (Security)

```python
class InputValidator:
    """Prevents prompt injection and validates input."""

    def validate(self, content: str) -> ValidatedInput:
        # 1. Check for prompt injection patterns
        if self._detect_injection(content):
            raise InputValidationError("Potentially malicious input detected")

        # 2. Sanitize special characters
        sanitized = self._sanitize(content)

        # 3. Length check
        if len(sanitized) > MAX_QUERY_LENGTH:
            raise InputValidationError("Query too long")

        return ValidatedInput(content=sanitized, is_safe=True)
```

**Bottleneck Solved**: Monolithic router prompt → modular intent classification with security validation.

---

## 3. Layer 2: Reasoning — Planning & Cognitive Logic

### 3.1 Plan-Act-Verify-Improve Loop

This is the core control flow that wraps every skill execution:

```python
class ReasoningEngine:
    """Implements the Plan-Act-Verify-Improve loop."""

    MAX_RETRIES = 3

    async def execute(self, processed_input: ProcessedInput,
                      skill: Skill, context: AgentContext) -> AgentResponse:

        for attempt in range(self.MAX_RETRIES):
            # PLAN: Generate execution plan
            plan = await self._plan(processed_input, skill, context, attempt)

            if plan.needs_clarification:
                return AgentResponse(
                    type="clarification",
                    question=plan.clarification_question
                )

            # ACT: Execute the plan using the skill
            result = await self._act(plan, skill, context)

            # VERIFY: Check the result against the plan and intent
            verification = await self._verify(result, processed_input, plan)

            if verification.is_valid:
                # Store successful pattern in memory
                await context.memory.store_success(
                    query=processed_input.query,
                    plan=plan,
                    result=result
                )
                return AgentResponse(type="success", data=result, reasoning=plan.reasoning)

            # IMPROVE: Analyze failure and adjust for next attempt
            improvement = await self._improve(verification, plan, result)
            context.feedback_history.append(improvement)

        # Max retries exceeded — return best result with confidence warning
        return AgentResponse(
            type="low_confidence",
            data=result,
            warning="Could not fully verify this result. Please review."
        )
```

### 3.2 Planning Step

```python
async def _plan(self, input: ProcessedInput, skill: Skill,
                context: AgentContext, attempt: int) -> Plan:
    # Retrieve relevant memories
    similar_past = await context.memory.recall(
        query=input.query,
        namespace=f"tenant:{input.tenant_id}"
    )

    plan_prompt = f"""
    User Query: {input.query}
    Active Skill: {skill.name}
    Skill Instructions: {skill.instructions}
    Available Tools: {[t.name for t in skill.tools]}
    Relevant Past Experience: {similar_past}
    {"Previous Attempt Feedback: " + context.feedback_history[-1] if attempt > 0 else ""}

    Create a step-by-step plan to address this query.
    """

    return await self.model.generate(
        system=STATIC_SYSTEM_PROMPT,
        user=plan_prompt,
        response_model=Plan
    )
```

### 3.3 Verification Step

```python
async def _verify(self, result, input: ProcessedInput, plan: Plan) -> Verification:
    """Multi-level verification of agent output."""

    checks = []

    # Level 1: Schema validation (deterministic)
    schema_check = self._validate_schema(result, plan.expected_output_schema)
    checks.append(schema_check)

    # Level 2: Business rule validation (deterministic)
    rule_check = self._validate_business_rules(result, input.tenant_config)
    checks.append(rule_check)

    # Level 3: Semantic verification (LLM-based, uses cheap model)
    semantic_check = await self._semantic_verify(
        result=result,
        original_query=input.query,
        plan=plan,
        model="haiku"  # cheap model for verification
    )
    checks.append(semantic_check)

    return Verification(
        is_valid=all(c.passed for c in checks),
        checks=checks,
        feedback=[c.feedback for c in checks if not c.passed]
    )
```

**Bottleneck Solved**: No verification or self-correction → systematic Plan-Act-Verify-Improve loop.

---

## 4. Layer 3: Memory — Context Management

### 4.1 Memory Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                      MEMORY SYSTEM                              │
├──────────────┬────────────────┬─────────────┬─────────────────┤
│   WORKING    │   EPISODIC     │  SEMANTIC    │  PROCEDURAL     │
│   MEMORY     │   MEMORY       │  MEMORY      │  MEMORY         │
│              │                │             │                 │
│ Current      │ Past sessions  │ Facet       │ Segment         │
│ session      │ & outcomes     │ catalog     │ recipes         │
│ state        │                │ Business    │ Best            │
│              │ User           │ rules       │ practices       │
│ Conversation │ corrections    │ Domain      │ Prompt          │
│ history      │ & feedback     │ knowledge   │ templates       │
│              │                │             │                 │
│ Task state   │ Successful     │ Product     │ Error           │
│              │ patterns       │ taxonomy    │ recovery        │
│              │                │             │ strategies      │
├──────────────┴────────────────┴─────────────┴─────────────────┤
│ Namespaced by: tenant_id / user_id / global                    │
└────────────────────────────────────────────────────────────────┘
```

### 4.2 Typed State Management (Replacing God State)

```python
# BEFORE: Flat string constants in state.py
USER_ID = "user_id"
SUB_SEGMENT_QUERY_REPRESENTATION = 'sub_segment_representation'
# ... 66+ untyped string constants

# AFTER: Typed Pydantic state models
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime

class SessionState(BaseModel):
    """Typed session state — replaces flat string dictionary."""

    # User context
    user_id: str
    tenant_id: str
    session_id: str
    user_type: str = "ASSOCIATE"

    # Conversation
    conversation_history: List[ConversationTurn] = []

    # Current task
    current_skill: Optional[str] = None
    current_phase: str = "idle"  # idle, planning, executing, verifying

    # Segment construction (only populated during segment skills)
    segment: Optional[SegmentState] = None

    # Flags
    needs_clarification: bool = False
    clarification_context: Optional[ClarificationContext] = None

class SegmentState(BaseModel):
    """State for segment construction — typed and validated."""

    raw_query: str
    decomposition: Optional[DecompositionResult] = None
    date_metadata: Dict[str, DateMetadata] = {}
    facet_mappings: Dict[str, FacetMappingResult] = {}
    formatted_segment: Optional[FormattedSegment] = None

    # Quality tracking
    verification_status: str = "pending"  # pending, verified, failed
    confidence_score: Optional[float] = None

    class Config:
        # Validate on assignment — catch type errors immediately
        validate_assignment = True

class ConversationTurn(BaseModel):
    role: str  # "user", "assistant", "clarification"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict = {}
```

### 4.3 Long-Term Memory Store

```python
class MemoryStore:
    """Enterprise memory system with namespace isolation."""

    def __init__(self, vector_db, relational_db):
        self.vector_db = vector_db  # For semantic search
        self.relational_db = relational_db  # For structured queries

    async def store(self, memory: Memory):
        """Store a memory with namespace isolation."""
        # Embed the memory content
        embedding = await self.embedder.embed(memory.content)

        # Store in vector DB for semantic retrieval
        await self.vector_db.upsert(
            collection=f"memory_{memory.namespace}",
            id=memory.id,
            vector=embedding,
            metadata={
                "type": memory.type,  # episodic, semantic, procedural
                "tenant_id": memory.tenant_id,
                "user_id": memory.user_id,
                "created_at": memory.created_at.isoformat(),
                "expiry": memory.expiry.isoformat() if memory.expiry else None,
                "tags": memory.tags
            },
            content=memory.content
        )

    async def recall(self, query: str, namespace: str,
                     top_k: int = 5, filters: dict = None) -> List[Memory]:
        """Retrieve relevant memories via semantic search."""
        embedding = await self.embedder.embed(query)
        results = await self.vector_db.search(
            collection=f"memory_{namespace}",
            vector=embedding,
            top_k=top_k,
            filters=filters
        )
        return [Memory.from_result(r) for r in results]

    async def store_success(self, query: str, plan, result,
                           tenant_id: str, user_id: str):
        """Store a successful segmentation as a memory."""
        memory = Memory(
            type="episodic",
            namespace=f"tenant:{tenant_id}",
            tenant_id=tenant_id,
            user_id=user_id,
            content=f"Query: {query}\nPlan: {plan}\nResult: {result}",
            tags=["success", "segment_creation"],
            metadata={"facets_used": result.facet_names, "operators": result.operators}
        )
        await self.store(memory)
```

**Bottleneck Solved**: No memory (every session starts fresh) → structured memory system with namespace isolation.

---

## 5. Layer 4: Action — Skill Execution & Tool Calls

### 5.1 Skill Execution Engine

```python
class ActionLayer:
    """Executes skills using tools and knowledge."""

    def __init__(self, skill_registry, tool_registry, knowledge_store):
        self.skill_registry = skill_registry
        self.tool_registry = tool_registry
        self.knowledge_store = knowledge_store

    async def execute(self, plan: Plan, skill: Skill,
                      context: AgentContext) -> ActionResult:
        # 1. Load skill-specific tools
        tools = self.tool_registry.get_tools(
            skill_id=skill.id,
            tenant_id=context.tenant_id
        )

        # 2. Retrieve relevant knowledge
        knowledge = await self.knowledge_store.retrieve(
            query=plan.knowledge_query,
            tenant_id=context.tenant_id,
            domain=skill.knowledge_domain
        )

        # 3. Select appropriate model for this skill
        model = self.model_router.select(
            skill=skill,
            complexity=plan.estimated_complexity,
            tenant_config=context.tenant_config
        )

        # 4. Execute skill with tools, knowledge, and model
        result = await self._run_skill(
            skill=skill,
            plan=plan,
            tools=tools,
            knowledge=knowledge,
            model=model,
            context=context
        )

        return result
```

### 5.2 Model Router

```python
class ModelRouter:
    """Routes tasks to the optimal model based on complexity and cost."""

    MODEL_TIERS = {
        "simple": {"model": "claude-haiku-4-5", "cost_per_1k": 0.00025},
        "medium": {"model": "claude-sonnet-4-6", "cost_per_1k": 0.003},
        "complex": {"model": "claude-opus-4-6", "cost_per_1k": 0.015},
    }

    SKILL_DEFAULT_TIERS = {
        "intent_classification": "simple",
        "date_extraction": "simple",
        "segment_decomposition": "medium",
        "facet_mapping": "medium",
        "format_generation": "simple",
        "hypothesis_analysis": "complex",
        "verification": "simple",
    }

    def select(self, skill: Skill, complexity: str,
               tenant_config: TenantConfig) -> str:
        # Tenant can override model selection
        if tenant_config.model_override:
            return tenant_config.model_override

        # Use skill default, adjusted by complexity
        base_tier = self.SKILL_DEFAULT_TIERS.get(skill.id, "medium")

        if complexity == "high" and base_tier != "complex":
            # Upgrade tier for complex queries
            return self.MODEL_TIERS["complex"]["model"]

        return self.MODEL_TIERS[base_tier]["model"]
```

**Bottleneck Solved**: Single model for all tasks → intelligent model routing based on task complexity.

---

## 6. Layer 5: Feedback — Self-Assessment & Improvement

### 6.1 Feedback Loop Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     FEEDBACK LOOP                                 │
│                                                                   │
│  Result → Verify → [Pass?] ──Yes──→ Store Success → Return       │
│                       │                                           │
│                      No                                           │
│                       ↓                                           │
│              Analyze Failure                                      │
│                       ↓                                           │
│              Generate Feedback                                    │
│                       ↓                                           │
│              [Retries Left?] ──Yes──→ Improve Plan → Re-execute  │
│                       │                                           │
│                      No                                           │
│                       ↓                                           │
│              Return Best + Warning                                │
│                       ↓                                           │
│              Store Failure (for learning)                         │
└──────────────────────────────────────────────────────────────────┘
```

### 6.2 Auto-Improvement via Evaluation Feedback

```python
class AutoImprover:
    """Uses evaluation results to improve prompts and skills over time."""

    async def analyze_failures(self, eval_results: List[EvalResult]):
        """Analyze patterns in evaluation failures."""

        # Group failures by type
        failure_patterns = self._cluster_failures(eval_results)

        for pattern in failure_patterns:
            if pattern.frequency > IMPROVEMENT_THRESHOLD:
                # Generate improvement suggestion
                suggestion = await self._generate_improvement(pattern)

                # Create candidate skill/prompt update
                candidate = await self._create_candidate(
                    current_skill=pattern.skill,
                    suggestion=suggestion
                )

                # Run candidate through eval suite
                candidate_results = await self.eval_engine.evaluate(
                    candidate=candidate,
                    eval_suite=pattern.skill.eval_suite
                )

                if candidate_results.score > pattern.current_score:
                    # Submit for review (human-in-the-loop)
                    await self.submit_for_review(
                        candidate=candidate,
                        improvement=candidate_results.score - pattern.current_score,
                        evidence=failure_patterns
                    )
```

**Bottleneck Solved**: No improvement loop → automated failure analysis and candidate improvement pipeline.

---

## 7. Skill Architecture

### 7.1 Skill Definition

```python
class Skill(BaseModel):
    """A versioned, testable instruction bundle."""

    # Identity
    id: str                    # "segment_creation"
    version: str               # "2.3.1"
    name: str                  # "Create Customer Segment"
    description: str           # "Creates a new customer segment from natural language"

    # Triggers
    trigger_intents: List[str] # ["create_segment"]
    trigger_keywords: List[str] # ["create", "build", "new segment"]

    # Instructions (the "prompt" for this skill)
    instructions: str          # Detailed step-by-step procedure

    # Input/Output contracts
    input_schema: Dict         # JSON Schema for expected input
    output_schema: Dict        # JSON Schema for expected output

    # Resources
    tools: List[str]           # Tool IDs this skill can use
    knowledge_domains: List[str]  # Knowledge domains to query

    # Constraints
    max_retries: int = 3
    timeout_seconds: int = 60
    requires_verification: bool = True

    # Few-shot examples
    examples: List[SkillExample] = []

    # Evaluation
    eval_suite_id: str         # ID of the eval suite for this skill

    # Tenant configuration
    enabled_for_tenants: List[str] = ["*"]  # "*" = all tenants
    tenant_overrides: Dict[str, SkillOverride] = {}
```

### 7.2 Skill Registry

```python
class SkillRegistry:
    """Central registry for all skills. Skills loaded from DB, not code."""

    def __init__(self, db):
        self.db = db
        self._cache = {}

    async def get_skill(self, skill_id: str, version: str = "latest",
                        tenant_id: str = None) -> Skill:
        """Load a skill, applying tenant overrides if present."""

        cache_key = f"{skill_id}:{version}:{tenant_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Load base skill
        skill = await self.db.get_skill(skill_id, version)

        # Apply tenant overrides
        if tenant_id and tenant_id in skill.tenant_overrides:
            skill = self._apply_overrides(skill, skill.tenant_overrides[tenant_id])

        self._cache[cache_key] = skill
        return skill

    async def register_skill(self, skill: Skill):
        """Register a new skill version. Must pass eval gate."""

        # Run eval suite
        eval_results = await self.eval_engine.evaluate(
            skill=skill,
            eval_suite_id=skill.eval_suite_id
        )

        if not eval_results.passed:
            raise EvalGateError(
                f"Skill {skill.id} v{skill.version} failed eval gate: "
                f"{eval_results.failures}"
            )

        # Store in registry
        await self.db.store_skill(skill)

        # Invalidate cache
        self._cache = {k: v for k, v in self._cache.items()
                       if not k.startswith(skill.id)}

    async def list_skills(self, tenant_id: str = None,
                          intent: str = None) -> List[Skill]:
        """List available skills, filtered by tenant and/or intent."""
        skills = await self.db.list_skills()

        if tenant_id:
            skills = [s for s in skills
                      if "*" in s.enabled_for_tenants or tenant_id in s.enabled_for_tenants]

        if intent:
            skills = [s for s in skills if intent in s.trigger_intents]

        return skills
```

### 7.3 Concrete Skill Example: Segment Creation

```yaml
# skills/segment_creation/skill.yaml
id: segment_creation
version: "1.0.0"
name: "Create Customer Segment"
description: "Creates a new customer segment from a natural language description"

trigger_intents:
  - create_segment
  - new_segment

instructions: |
  You are executing the Segment Creation skill. Follow these steps exactly:

  STEP 1: DECOMPOSE
  Break the user's query into logical sub-segments.
  Each sub-segment should represent one filtering condition.
  Define the logical relationship (AND/OR/EXCLUDE) between sub-segments.
  Output: DecompositionResult with ruleSet and subSegments.

  STEP 2: EXTRACT TEMPORAL
  For each sub-segment, identify temporal references.
  Convert relative dates ("last 30 days") to absolute date ranges.
  Output: DateMetadata for each relevant sub-segment.

  STEP 3: MAP FACETS
  For each sub-segment:
  a. Search the facet catalog for matching facets (use facet_search tool)
  b. Select the best matching facet based on the sub-segment intent
  c. Determine the appropriate operator (is, in, >=, <=, between)
  d. Map values from the user's query to catalog values
  Output: FacetMappingResult for each sub-segment.

  STEP 4: FORMAT
  Combine all mappings into the final segment format.
  Validate that every sub-segment in the ruleSet has a mapping.
  Output: FormattedSegment JSON.

  STEP 5: VERIFY
  Check: Does this segment capture the user's full intent?
  Check: Are all facet names valid catalog entries?
  Check: Are all operators appropriate for their facet types?
  If any check fails, identify the issue and retry the relevant step.

input_schema:
  type: object
  properties:
    query: { type: string }
    conversation_history: { type: array }
  required: [query]

output_schema:
  type: object
  properties:
    segment: { $ref: "#/definitions/FormattedSegment" }
    reasoning: { type: string }
    confidence: { type: number, minimum: 0, maximum: 1 }

tools:
  - facet_search
  - date_parser
  - segment_validator
  - size_estimator

knowledge_domains:
  - facet_catalog
  - business_rules
  - segment_recipes

eval_suite_id: eval_segment_creation

examples:
  - input: "Customers who bought electronics online in the last 30 days"
    output:
      segment:
        ruleSet:
          INCLUDE: "(Seg-1 AND Seg-2)"
          EXCLUDE: ""
        subSegments:
          Seg-1:
            facet: "department_name"
            operator: "is"
            values: ["Electronics"]
          Seg-2:
            facet: "last_purchase_date"
            operator: ">="
            values: ["2026-01-19"]
      reasoning: "Decomposed into department filter + recency filter"
      confidence: 0.95
```

**Bottleneck Solved**: Hardcoded agent tree → declarative, versionable skill bundles with eval gates.

---

## 8. Knowledge System (RAG)

### 8.1 Knowledge Architecture

```python
class KnowledgeStore:
    """Enterprise knowledge management with multi-tenant support."""

    KNOWLEDGE_TYPES = {
        "facet_catalog": "Structured facet definitions with types and values",
        "business_rules": "Tenant-specific business rules and constraints",
        "segment_recipes": "Proven segmentation patterns and templates",
        "product_taxonomy": "Product hierarchy and category mappings",
        "campaign_context": "Active campaigns and their target segments",
    }

    async def retrieve(self, query: str, tenant_id: str,
                       domain: str = None, top_k: int = 10) -> List[KnowledgeItem]:
        """Retrieve relevant knowledge with tenant isolation."""

        # 1. Generate retrieval query (HyDE-style for better matching)
        enhanced_query = await self._enhance_query(query, domain)

        # 2. Search with tenant filter
        results = await self.vector_db.search(
            collection=f"knowledge_{domain}" if domain else "knowledge_all",
            query_vector=await self.embedder.embed(enhanced_query),
            top_k=top_k * 2,  # Over-retrieve for reranking
            filters={"tenant_id": {"$in": [tenant_id, "global"]}}
        )

        # 3. Rerank results
        reranked = await self.reranker.rerank(query, results, top_k=top_k)

        # 4. Apply freshness rules
        fresh = [r for r in reranked if not self._is_expired(r)]

        return fresh
```

### 8.2 Facet Catalog as Knowledge (Not Pickle Files)

```python
# BEFORE: Pickle file loaded into memory
facet_catalog = pd.read_pickle("facet_catalog_email_mobile_data.pkl")

# AFTER: Facet catalog in knowledge store
class FacetCatalogKnowledge:
    """Facet catalog managed as enterprise knowledge."""

    async def search_facets(self, query: str, tenant_id: str,
                           facet_key: str = "email_mobile") -> List[FacetMatch]:
        """Search for facets matching a natural language query."""

        results = await self.knowledge_store.retrieve(
            query=query,
            tenant_id=tenant_id,
            domain="facet_catalog",
            top_k=20
        )

        # Filter by facet key
        filtered = [r for r in results if r.metadata.get("facet_key") == facet_key]

        return [FacetMatch(
            facet_name=r.metadata["facet_name"],
            facet_type=r.metadata["facet_type"],
            description=r.content,
            operators=r.metadata["operators"],
            sample_values=r.metadata.get("sample_values", []),
            relevance_score=r.score
        ) for r in filtered]
```

**Bottleneck Solved**: Pickle files in memory → searchable knowledge store with tenant isolation.

---

## 9. Multi-Tenant Support

### 9.1 Tenant Configuration

```python
class TenantConfig(BaseModel):
    """Complete tenant configuration — stored as data, not code."""

    tenant_id: str
    tenant_name: str

    # Model configuration
    model_override: Optional[str] = None  # Force specific model
    model_tier: str = "medium"  # default model tier

    # Skill configuration
    enabled_skills: List[str] = ["*"]  # Which skills are available
    skill_overrides: Dict[str, Dict] = {}  # Per-skill config overrides

    # Knowledge configuration
    facet_catalog_id: str = "default"  # Which facet catalog to use
    knowledge_namespaces: List[str] = []  # Additional knowledge sources

    # Memory configuration
    memory_namespace: str  # Isolated memory namespace
    memory_retention_days: int = 365

    # Security
    allowed_facets: List[str] = ["*"]  # Facet access control
    restricted_facets: List[str] = []

    # Cost controls
    max_requests_per_hour: int = 1000
    max_cost_per_day: float = 100.0

    # Evaluation
    eval_suite_ids: List[str] = []  # Tenant-specific eval suites
```

### 9.2 Tenant-Aware Request Flow

```python
class TenantAwareAgent:
    """Agent that automatically applies tenant context."""

    async def process_request(self, request: UserRequest):
        # 1. Load tenant config
        tenant_config = await self.tenant_store.get_config(request.tenant_id)

        # 2. Check rate limits
        if not await self.rate_limiter.allow(request.tenant_id):
            raise RateLimitError("Tenant rate limit exceeded")

        # 3. Build tenant-scoped context
        context = AgentContext(
            tenant_id=request.tenant_id,
            tenant_config=tenant_config,
            memory=self.memory_store.scoped(tenant_config.memory_namespace),
            knowledge=self.knowledge_store.scoped(request.tenant_id),
            tools=self.tool_registry.scoped(request.tenant_id, tenant_config),
        )

        # 4. Process through standard pipeline
        result = await self.pipeline.process(request, context)

        # 5. Track cost
        await self.cost_tracker.record(
            tenant_id=request.tenant_id,
            tokens_used=result.token_usage,
            cost=result.estimated_cost
        )

        return result
```

**Bottleneck Solved**: No tenant awareness → full multi-tenant isolation with per-tenant configuration.

---

## 10. Evaluation Framework

### 10.1 Eval-First Development Workflow

```
Define Eval → Write Skill → Run Eval → [Pass?] → Deploy
                                  │
                                 No
                                  ↓
                          Fix Skill → Run Eval → ...
```

### 10.2 Three-Tier Evaluation Pyramid

```
┌──────────────────────────────────────┐
│       TIER 3: HUMAN REVIEW            │  ← 5% of evals
│  Expert review of edge cases          │  Cost: High, Accuracy: Highest
├──────────────────────────────────────┤
│       TIER 2: LLM-AS-JUDGE           │  ← 25% of evals
│  Semantic quality, intent match       │  Cost: Medium, Accuracy: Good
├──────────────────────────────────────┤
│       TIER 1: ASSERTIONS              │  ← 70% of evals
│  Schema valid, operators correct,     │  Cost: Zero, Accuracy: Perfect
│  facets exist, ruleSet consistent     │  (for what they check)
└──────────────────────────────────────┘
```

### 10.3 Concrete Eval Suite for Segment Creation

```python
class SegmentCreationEvalSuite:
    """Evaluation suite for the segment_creation skill."""

    # Tier 1: Assertion-based checks (fast, free, deterministic)
    TIER_1_CHECKS = [
        JsonSchemaCheck("output matches FormattedSegment schema"),
        RuleSetConsistencyCheck("every Seg-X in ruleSet exists in subSegments"),
        FacetExistenceCheck("every facet name exists in the catalog"),
        OperatorTypeCheck("operators match facet types (list→'is'/'in', int→comparison)"),
        DateFormatCheck("all dates are in YYYY-MM-DD format"),
        NoEmptyValuesCheck("no facet has empty values"),
        LogicalConsistencyCheck("AND/OR/EXCLUDE used correctly"),
    ]

    # Tier 2: LLM-as-judge checks (slower, costs tokens, semantic)
    TIER_2_CHECKS = [
        IntentCoverageCheck("does the segment capture the full user intent?"),
        FacetRelevanceCheck("are the chosen facets the most relevant for this query?"),
        CompletenessCheck("are there any missing conditions from the user's query?"),
        GroundingCheck("are all values grounded in the catalog, not hallucinated?"),
    ]

    async def evaluate(self, test_case: EvalTestCase) -> EvalResult:
        results = []

        # Run Tier 1 (always)
        for check in self.TIER_1_CHECKS:
            result = check.run(test_case.output, test_case.expected)
            results.append(result)

        # Run Tier 2 (if Tier 1 passes)
        if all(r.passed for r in results):
            for check in self.TIER_2_CHECKS:
                result = await check.run_async(
                    test_case.input, test_case.output, test_case.expected
                )
                results.append(result)

        return EvalResult(
            passed=all(r.passed for r in results),
            score=sum(r.score for r in results) / len(results),
            results=results
        )
```

### 10.4 CI/CD Integration

```yaml
# .github/workflows/eval-gate.yml
name: Eval Gate
on:
  push:
    paths:
      - "skills/**"
      - "prompts/**"
      - "knowledge/**"

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run Tier 1 Evaluations
        run: python -m eval_engine run --tier 1 --changed-only

      - name: Run Tier 2 Evaluations (if Tier 1 passes)
        if: success()
        run: python -m eval_engine run --tier 2 --changed-only

      - name: Compare to Baseline
        run: python -m eval_engine compare --baseline main --candidate ${{ github.sha }}

      - name: Gate Check
        run: python -m eval_engine gate --min-score 0.95 --no-regressions
```

**Bottleneck Solved**: Disconnected eval framework → eval gates in CI/CD with 3-tier pyramid.

---

## 11. Observability & Cost Optimization

### 11.1 Structured Logging

```python
class AgentLogger:
    """Structured logging for agent operations."""

    def log_skill_execution(self, skill_id, tenant_id, input_query,
                           output, tokens_used, latency_ms, model_used):
        self.logger.info("skill_execution", extra={
            "skill_id": skill_id,
            "tenant_id": tenant_id,
            "input_query_hash": hash(input_query),  # Privacy-safe
            "output_valid": output.is_valid,
            "tokens_input": tokens_used.input,
            "tokens_output": tokens_used.output,
            "latency_ms": latency_ms,
            "model_used": model_used,
            "cost_usd": self._calculate_cost(model_used, tokens_used),
            "confidence": output.confidence,
        })
```

### 11.2 Cost Dashboard Metrics

| Metric | Source | Alert Threshold |
|--------|--------|----------------|
| Cost per query | Token tracking | > $2.00 |
| Tokens per query | LLM response | > 20,000 |
| Cache hit rate | Cache layer | < 20% |
| Eval pass rate | Eval engine | < 90% |
| P95 latency | Tracing | > 30s |
| Error rate | Error tracking | > 5% |
| Clarification rate | Agent logs | > 40% |

### 11.3 Prompt Caching Implementation

```python
# Enable prompt caching for static prompt portions
class CachedPromptBuilder:
    """Builds prompts with caching markers for static content."""

    def build(self, skill: Skill, dynamic_context: Dict) -> List[Message]:
        return [
            # System prompt — CACHED (same across all calls)
            {"role": "system", "content": STATIC_SYSTEM_PROMPT,
             "cache_control": {"type": "ephemeral"}},

            # Skill instructions — CACHED (same for all calls to this skill)
            {"role": "user", "content": f"Active Skill: {skill.instructions}",
             "cache_control": {"type": "ephemeral"}},

            # Dynamic context — NOT cached (changes per request)
            {"role": "user", "content": f"""
                Query: {dynamic_context['query']}
                History: {dynamic_context['history']}
                Retrieved Knowledge: {dynamic_context['knowledge']}
            """}
        ]
```

**Bottleneck Solved**: No cost tracking → comprehensive observability with prompt caching.

---

## 12. Concrete Transformations

### 12.1 Before/After: State Management

**Before** (`state.py`):
```python
USER_ID = "user_id"
SUB_SEGMENT_QUERY_REPRESENTATION = 'sub_segment_representation'
# 66+ untyped string constants
```

**After** (`models/state.py`):
```python
class SessionState(BaseModel):
    user_id: str
    tenant_id: str
    segment: Optional[SegmentState] = None
    # Typed, validated, IDE-discoverable
```

### 12.2 Before/After: Agent Initialization

**Before** (`agent.py`):
```python
root_agent = LlmAgent(
    name="RouterAgent",
    model=gptmodel,  # Single model
    instruction=ROOT_AGENT_INSTRUCTION,  # Monolithic prompt
    sub_agents=[NewSegmentCreationAgent, DirectSegmentEditorAgent],  # Hardcoded
)
```

**After** (`agent.py`):
```python
class SegmentationAgent:
    def __init__(self):
        self.perception = PerceptionLayer()
        self.reasoning = ReasoningEngine()
        self.memory = MemoryStore()
        self.skill_registry = SkillRegistry()
        self.eval_engine = EvalEngine()

    async def handle(self, request: UserRequest) -> AgentResponse:
        tenant_config = await self.tenant_store.get(request.tenant_id)
        input = await self.perception.process(request)
        skill = await self.skill_registry.get_skill(input.intent.skill_id)
        return await self.reasoning.execute(input, skill, context)
```

### 12.3 Before/After: Facet Catalog

**Before** (`metadata.py`):
```python
facet_catalog = pd.read_pickle("facet_catalog_email_mobile_data.pkl")
```

**After** (`knowledge/facet_catalog.py`):
```python
results = await knowledge_store.retrieve(
    query="customer purchase department",
    tenant_id=tenant_id,
    domain="facet_catalog",
    top_k=10
)
```

### 12.4 Before/After: eval() Security Fix

**Before**:
```python
eval(os.environ.get('DEFAULT_USER_RESTRICTIONS'))
```

**After**:
```python
import json
json.loads(os.environ.get('DEFAULT_USER_RESTRICTIONS', '[""]'))
```

### 12.5 Before/After: Prompt Construction

**Before**:
```python
prompt = prompt.replace('{user_query}', user_query)
               .replace('{conversational_history}', history)
```

**After**:
```python
from jinja2 import Environment, StrictUndefined

env = Environment(undefined=StrictUndefined)  # Fail on missing vars
template = env.from_string(skill.instructions)
prompt = template.render(
    user_query=user_query,
    conversational_history=history,
    facet_context=knowledge_results
)
```

---

## Appendix: Cost-Saving Alternatives

### A.1 Where to Save (Without Losing Quality)

| Optimization | Savings Est. | Quality Impact | Recommendation |
|-------------|-------------|----------------|----------------|
| Model routing (Haiku for simple tasks) | 50-70% | None for simple tasks | **Do first** |
| Prompt caching | 40-60% | None | **Do first** |
| Semantic query cache | 15-25% | None (identical queries) | Do in Phase 2 |
| Embedding cache | 30-50% | None | Do in Phase 2 |
| Open-source models for routing | 80-90% | 2-5% on edge cases | Evaluate in Phase 3 |

### A.2 Where NOT to Save

| Component | Why Not |
|-----------|---------|
| Segment decomposition model | Quality here determines everything downstream |
| Facet mapping model | Incorrect facet selection = wrong segment |
| Verification model | This is the safety net — cheap model is fine but don't skip it |
| Evaluation LLM-as-judge | Under-investing in evals degrades quality silently |

### A.3 Monthly Cost Projection (10K queries/day)

| Scenario | Monthly Cost | Quality |
|----------|-------------|---------|
| Current (one model for all) | $9,000-$15,000 | Good |
| Optimized (model routing + caching) | $2,500-$4,000 | Same or better |
| Aggressive (+ open-source routing) | $1,500-$2,500 | Slightly lower on edge cases |

---

*Built for the Enterprise Agentic Research Initiative*
*Last Updated: February 2026*
