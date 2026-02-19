# 04 — Implementation Roadmap: Enterprise Agentic Segmentation Upgrade

> **Research ID:** research_1_sonnet_claude
> **Document Purpose:** Prioritized, step-by-step upgrade execution plan
> **Date:** February 2026
> **Total Estimated Duration:** 6 months (26 weeks)

---

## Executive Summary

This roadmap transforms Smart-Segmentation into an enterprise-grade agentic platform in **6 phases over 26 weeks**. Each phase delivers standalone value, allowing the system to improve incrementally rather than requiring a "big bang" rewrite.

The prioritization philosophy follows three principles:
1. **Stop the bleeding first** — fix critical reliability and quality failures before adding features
2. **Build the foundation before the house** — infrastructure changes (eval gates, caching, memory) before capability upgrades
3. **Most impactful per unit of effort** — rank by (impact × urgency) / complexity

---

## Prioritization Matrix

| Phase | Weeks | Focus | Impact | Risk | Parallelizable? |
|-------|-------|-------|--------|------|----------------|
| 1 | 1-4 | Stability & Quality Gates | 🔴 Critical | Low | Partially |
| 2 | 5-8 | Performance & Reliability | 🟠 High | Low | Yes |
| 3 | 9-14 | Skill Architecture & Dynamic Prompts | 🟠 High | Medium | Partially |
| 4 | 15-18 | Memory System | 🟠 High | Medium | No |
| 5 | 19-22 | Auto-Improvement & Eval-First | 🟡 Medium | Low | Yes |
| 6 | 23-26 | Enterprise Features (Multi-Tenant, Hypothesis) | 🟡 Medium | High | Partially |

---

## Phase 1: Stability & Quality Gates (Weeks 1-4)

### Hypothesis
The system currently ships prompt changes without any automated quality check, has silent segment validation failures, and crashes completely when Milvus is unavailable. Before adding any new capability, we must **stop the bleeding** — fix the issues that cause the most user-visible failures today.

This phase has the highest ROI of any phase: 4 weeks of work will prevent a class of production failures that happen every week.

### P1.1 — Production Eval Gate (Week 1-2)

**Priority:** First. Nothing else matters if we can't detect regressions.

**Why first:**
- Every prompt change without an eval gate risks degrading production quality
- The evaluation infrastructure already exists (Streamlit UI, eval_sets/) — we just need to connect it to CI/CD
- Blocking all other work is the right trade-off: better to slow delivery than to ship regressions

**Tasks:**
```
Week 1:
  □ Map all existing eval_sets/ to the skill they test
  □ Implement EvalGate class (run evals against a skill version)
  □ Define minimum accuracy thresholds per skill:
      - segment_decomposition: 88% accuracy
      - date_parsing: 92% accuracy
      - facet_mapping: 82% accuracy
      - format_generation: 95% accuracy
  □ Add eval gate to CI/CD pipeline (GitHub Actions or internal CI)

Week 2:
  □ Add regression detection (compare against last passing version)
  □ Implement eval result store in PostgreSQL
  □ Build eval dashboard (extend existing Streamlit eval UI)
  □ Document "how to add a new eval case" process
  □ Run first full eval baseline against current codebase
```

**Dependencies:** None (can start immediately)
**Owner:** ML/Prompt Engineering team
**Success metric:** Zero prompt changes reach production without passing eval gate

---

### P1.2 — Segment Validation Layer (Week 1-2, parallel with P1.1)

**Priority:** Critical. Silent failures damage user trust.

**Why now:** Users confirm a segment was "created successfully" but it silently fails when executed in CRM/email systems. This is discovered only when campaigns run — too late.

**Tasks:**
```
Week 1-2:
  □ Implement SegmentValidator class:
      - Schema validation (correct JSON structure)
      - Facet existence check (facet name exists in tenant's catalog)
      - Value validity check (value is valid for the given facet)
      - Logic validation (INCLUDE/EXCLUDE expressions are syntactically correct)
  □ Add validation step BEFORE segment is written to PostgreSQL
  □ On validation failure: return specific error message + retry loop
  □ Add validation results to segment record in DB for auditability
```

**Code change example:**
```python
# Before: (in segment_format_generator/agent.py)
await db.save_segment(segment_definition)
return {"success": True, "segment": segment_definition}

# After:
validation_result = await validator.validate(segment_definition, tenant_id)
if not validation_result.is_valid:
    error_context = validation_result.format_error_for_llm()
    # Feed error back to LLM for correction
    corrected = await self.correct_segment(segment_definition, error_context)
    validation_result = await validator.validate(corrected, tenant_id)

if validation_result.is_valid:
    await db.save_segment(corrected or segment_definition)
    return {"success": True, "segment": corrected or segment_definition}
else:
    return {"success": False, "error": validation_result.user_facing_error}
```

**Dependencies:** None
**Owner:** Backend team
**Success metric:** Zero segments saved with invalid facet references

---

### P1.3 — Milvus Fallback Search (Week 3)

**Priority:** Critical for reliability.

**Why:** Milvus is a single point of failure. One outage takes down the entire segment creation pipeline.

**Tasks:**
```
Week 3:
  □ Implement HybridFacetSearcher class (see upgrade proposal §2.4)
  □ Add FuzzyStringSearcher as fallback:
      - Uses rapidfuzz library for approximate matching
      - Searches against in-memory facet catalog
      - Returns results with "source=fuzzy" flag
  □ Add fallback trigger logic:
      - Trigger on: MilvusException, connection timeout >2s, empty results
  □ Implement circuit breaker pattern for Milvus
  □ Add monitoring: track fallback activation rate
  □ Test: manually trigger Milvus failure and verify degraded-but-functional behavior
```

**Dependencies:** None (can run parallel with P1.1, P1.2)
**Owner:** Backend team
**Success metric:** System continues to function at >80% quality when Milvus is unavailable

---

### P1.4 — Centralized Ambiguity Resolver (Week 3-4)

**Priority:** High. Duplicate clarification questions are the #1 UX complaint.

**Why now:** Users report being asked the same question twice in the same session. This is fixable with a simple shared state mechanism.

**Tasks:**
```
Week 3:
  □ Implement AmbiguityRegistry (Redis-backed, session-scoped)
  □ Modify each agent to check registry before generating clarification question:
      - segment_decomposer_agent.py
      - date_tagger_agent.py
      - facet_value_mapper_agent.py
  □ Add "question hash" to detect semantically identical questions

Week 4:
  □ Add user preference lookup (check if this user has answered this type of
    question before and use their preference)
  □ Test: run 20 complex queries and verify zero duplicate questions
  □ Add metric: "clarification_questions_per_segment" (target: ≤2 per segment)
```

**Dependencies:** None (can run parallel)
**Owner:** ML team
**Success metric:** Duplicate clarification rate drops to <5%

---

### Phase 1 Success Criteria

✅ All prompt changes blocked by eval gate
✅ Zero segments saved with validation errors
✅ System functional during Milvus outages (degraded but not down)
✅ No duplicate clarification questions per session
✅ Eval baseline established for all 4 core skills

---

## Phase 2: Performance & Reliability (Weeks 5-8)

### Hypothesis
With the critical failures fixed, we can now focus on performance. The system is currently 2x slower than it needs to be (sequential pipeline) and 5x more expensive in LLM calls than it needs to be (no caching, wrong models for simple tasks). This phase delivers the largest latency improvement per unit of work.

### P2.1 — Redis Caching Layer (Week 5-6)

**Why:** 40-60% of Milvus searches and 80% of embedding computations are redundant. Adding Redis eliminates them.

**Tasks:**
```
Week 5:
  □ Add Redis to Docker Compose and Kubernetes configs
  □ Implement CacheManager class (see upgrade proposal §2.3)
  □ Cache Milvus search results:
      - Key: hash(query + collection)
      - TTL: 1 hour for facet names, 30 min for facet values
  □ Cache embedding vectors:
      - Key: hash(text + model_name)
      - TTL: 24 hours (embeddings don't change)

Week 6:
  □ Cache segment size estimates (TTL: 6 hours)
  □ Cache tenant config lookups (TTL: 5 minutes)
  □ Add cache hit/miss metrics
  □ Load test to verify latency improvement
  □ Add Redis health check to app startup
```

**Expected impact:**
- P50 latency: 15s → 8s (with parallel execution in Phase 2.3)
- Milvus load: -50%
- Embedding computation: -80%

---

### P2.2 — Database Connection Pool Scaling (Week 5, parallel)

**Why:** Currently maxes out at 30 concurrent users. Enterprise needs 200+.

**Tasks:**
```
Week 5:
  □ Increase POOL_MAX from 3 to 20 connections per worker
  □ Add PgBouncer connection pooler (sits between app and PostgreSQL)
  □ Configure PgBouncer: transaction mode, 200 max client connections
  □ Add connection pool metrics (active, idle, waiting connections)
  □ Load test to verify 100+ concurrent users without connection errors
```

**Expected impact:** Concurrent user limit: 30 → 200+

---

### P2.3 — Parallel Agent Execution (Week 7-8)

**Why:** Date tagging and initial Milvus shortlisting can run in parallel after decomposition. This cuts latency by ~40%.

**Tasks:**
```
Week 7:
  □ Refactor NSC pipeline to use asyncio.gather() for parallel steps
  □ Identify exact dependency graph:
      Step 1 (decompose) → required sequential
      Step 2a (date tagging) + Step 2b (Milvus shortlisting) → parallel
      Step 3 (facet mapping, uses 2a+2b) → sequential after 2a,2b
      Step 4 (format) → sequential after 3

Week 8:
  □ Implement parallel execution for DSE edit operations:
      - Multiple sub-segment edits can run in parallel
  □ Add progress streaming (return partial results as each step completes)
  □ Test for race conditions and state consistency
  □ Measure latency improvement across 50 test queries
```

**Expected impact:** P50 end-to-end latency: 15s → 8s (combined with Redis caching)

---

### P2.4 — Model Routing (Week 8, parallel)

**Why:** Using GPT-4o for every task is like using a sledgehammer for every nail. Simple tasks (routing, formatting) can use GPT-4o-mini at 94% cost savings with equivalent quality.

**Tasks:**
```
Week 8:
  □ Implement ModelRouter class
  □ Configure task-to-model mapping:
      - Intent classification → GPT-4o-mini
      - Date parsing (simple patterns) → GPT-4o-mini
      - Facet mapping (complex) → GPT-4o
      - Format generation → GPT-4o-mini
      - Hypothesis assessment → Claude Opus / o3-mini
  □ Add A/B test to validate quality parity for simple tasks
  □ Monitor cost savings vs. baseline
```

**Expected impact:** 40-60% reduction in LLM API costs

---

### Phase 2 Success Criteria

✅ P50 latency < 9 seconds (down from 15s)
✅ System handles 200+ concurrent users
✅ Redis cache hit rate > 40% for Milvus searches
✅ LLM costs reduced by 40%+ vs baseline
✅ Zero connection pool exhaustion errors

---

## Phase 3: Skill Architecture & Dynamic Prompts (Weeks 9-14)

### Hypothesis
The biggest leverage for long-term maintainability comes from decoupling "what the agent knows" from "how the agent thinks". Currently, adding a new capability requires a code deployment. With a skill architecture, it requires only a skill definition + eval test.

This phase also eliminates the biggest source of quality degradation: paraphrase-based prompt mutation.

### P3.1 — Skill Registry Infrastructure (Week 9-10)

**Tasks:**
```
Week 9:
  □ Design and create PostgreSQL schema for skill registry:
      CREATE TABLE skills (
          skill_id TEXT,
          version TEXT,
          name TEXT,
          instructions TEXT,
          input_schema JSONB,
          output_schema JSONB,
          eval_suite_id TEXT,
          min_accuracy FLOAT,
          tenant_overrides JSONB,
          created_at TIMESTAMPTZ,
          PRIMARY KEY (skill_id, version)
      );

  □ Implement SkillRegistry class (load, route, version management)
  □ Implement SkillLoader (inject skill into agent context)
  □ Create skill versioning workflow (draft → eval → promote → active)

Week 10:
  □ Migrate existing prompts to skill definitions:
      - segment_creation.yaml (from segment_decomposer_prompt.txt + facet_value_operator_mapper_prompt.txt)
      - segment_editing.yaml (from direct_segment_editor_prompt.txt)
      - date_resolution.yaml (from date_extraction_prompt.txt)
      - format_generation.yaml (from master_format_generator_prompt.txt)
  □ Verify all migrated skills pass existing eval suites
  □ Deploy skill registry to staging
```

**Dependencies:** P1.1 (eval gates must exist before promoting skills)

---

### P3.2 — Static System Prompt Implementation (Week 11)

**Tasks:**
```
Week 11:
  □ Write the static system prompt "constitution" (identity + loop + safety + format)
  □ Remove per-task instructions from system prompt
      - System prompt: "Plan → Act → Verify → Improve"
      - Task instructions: loaded from skill registry
  □ Implement prompt caching (Anthropic/OpenAI prompt caching API)
      - Cache the static portion (~12KB)
      - Only charge for dynamic skill content per request
  □ Run full eval suite on new prompt structure
  □ A/B test: 10% traffic on new structure vs. 90% on old
  □ Promote to 100% if eval gate passes
```

---

### P3.3 — Anti-Paraphrase Query Preservation (Week 12)

**Tasks:**
```
Week 12:
  □ Add original_query field to SubSegment data model
  □ Enforce that original_query is never modified through the pipeline
  □ Add grounding enforcement: each LLM output must cite [RETRIEVED] vs [INFERRED]
  □ Add validator: reject any output where >30% of facets are [INFERRED]
  □ Update all prompts to use verbatim sub-query passing pattern
  □ Run eval suite: measure grounding rate before and after
      Target: >85% of facets should be [RETRIEVED]
```

---

### P3.4 — Dynamic Few-Shot Example Retrieval (Week 13-14)

**Tasks:**
```
Week 13:
  □ Create Milvus collection for successful segment examples:
      Collection: successful_segments
      Fields: query, segment_definition, facets_used, quality_score, tenant_id, embedding
  □ Seed with 200+ examples from production query history (anonymized)
  □ Implement few-shot retriever: given new query → find top-3 similar past segments

Week 14:
  □ Integrate retrieved examples into skill instruction context
  □ Replace hard-coded examples in facet_value_operator_mapper_prompt.txt
  □ Measure accuracy improvement on held-out eval set
      Hypothesis: 5-8% accuracy improvement from dynamic few-shot examples
  □ Track example quality: mark used examples with outcome for feedback loop
```

---

### Phase 3 Success Criteria

✅ All prompt changes go through skill registry (zero direct file edits)
✅ New capability = new skill file + eval tests (no code changes needed)
✅ Grounding rate > 85% ([RETRIEVED] citations on >85% of facets)
✅ Zero duplicate clarification questions across any agent
✅ Dynamic few-shot retrieval operational with >200 examples

---

## Phase 4: Memory System (Weeks 15-18)

### Hypothesis
Users currently re-specify the same preferences every session. Experienced users' patterns are not shared with new users. Long conversations lose context because of naive truncation. A memory system changes the system from "stateless session handler" to "learning agent that gets smarter over time."

### P4.1 — Short-Term Memory Redesign (Week 15-16)

**Tasks:**
```
Week 15:
  □ Implement SessionState v2 (typed, validated state, replaces 40 untyped variables)
  □ Move session state entirely to Redis:
      - Remove dependency on PostgreSQL for session state
      - Keep PostgreSQL only for final segment persistence
  □ Implement conversation summarization for long sessions:
      - When context > 20K tokens: summarize oldest turns
      - Keep summary + recent turns (not raw history)
      - Preserve: original intent, confirmed preferences, current segment state

Week 16:
  □ Implement AmbiguityRegistry (session-scoped, Redis-backed)
  □ Add "conversation snapshot" capability:
      - Users can "save" a session state and return to it later
  □ Test with 20+ turn conversations: verify no context loss
```

---

### P4.2 — Long-Term Memory System (Week 17-18)

**Tasks:**
```
Week 17:
  □ Create user_memory table in PostgreSQL:
      CREATE TABLE user_memory (
          user_id TEXT PRIMARY KEY,
          tenant_id TEXT,
          preferences JSONB,         -- Learned preferences
          vocabulary_map JSONB,      -- User's custom terminology
          segment_patterns JSONB,    -- Common patterns for this user
          updated_at TIMESTAMPTZ
      );

  □ Create successful_segments Milvus collection (if not done in Phase 3)
  □ Implement MemoryManager class (see upgrade proposal §6)
  □ Add preference learning:
      - Date interpretation preferences
      - Value threshold preferences ("high value" = X)
      - Terminology preferences

Week 18:
  □ Integrate memory into perception layer:
      - Load user preferences before every request
      - Inject into skill context
  □ Test: demonstrate that system learns user's "high value" definition
    after first interaction and applies it in subsequent sessions
  □ Add privacy controls: users can view and delete their stored preferences
```

---

### Phase 4 Success Criteria

✅ Session state is fully typed and validated (no 40-variable chaos)
✅ No context loss in sessions up to 50 turns
✅ System learns user preferences after 1st interaction, applies in 2nd
✅ "Similar segment" retrieval works: given new query, surfaces top-3 related past segments
✅ User vocabulary map operational for 10+ common terminology variants

---

## Phase 5: Auto-Improvement & Eval-First Maturity (Weeks 19-22)

### Hypothesis
Once we have memory, validation, and grounding, we have the data needed to automatically improve the system. Production feedback (user confirmations, edit counts, explicit ratings) can drive prompt optimization without human intervention, creating a flywheel effect where the system gets better with every use.

### P5.1 — Feedback Collection System (Week 19)

**Tasks:**
```
Week 19:
  □ Add explicit feedback UI:
      - Thumbs up/down on segment quality
      - Optional comment field
  □ Implement implicit feedback signals:
      - Track: edits after creation (signal: dissatisfaction)
      - Track: confirmation without edit (signal: satisfaction)
      - Track: session abandonment (signal: failure)
  □ Store all feedback in feedback table with segment_id, skill_id, version
  □ Build feedback dashboard: quality trends over time per skill
```

---

### P5.2 — Eval Dataset Auto-Growth (Week 20)

**Tasks:**
```
Week 20:
  □ Implement EvalDatasetBuilder (mines production queries with feedback)
  □ Auto-add high-quality examples to eval suites:
      - Condition: user confirmed without edit + rating >= 4
      - Sample diverse examples across query types
  □ Auto-flag problematic patterns:
      - Condition: edit_count > 3 OR user abandoned
      - Flag for human review and potential addition to eval suite
  □ Target: eval suite grows by 50+ cases per month automatically
```

---

### P5.3 — Prompt Optimization Loop (Week 21-22)

**Tasks:**
```
Week 21:
  □ Implement FailureAnalyzer (cluster recent low-quality outputs by failure type)
  □ Implement PromptOptimizer (uses LLM to propose skill improvements)
  □ Build optimization pipeline:
      analyze_failures → propose_improvement → eval_gate → promote if passing

Week 22:
  □ Schedule weekly optimization runs per skill
  □ Add safeguards:
      - Optimization only runs with >20 failure examples (enough signal)
      - Proposed changes must pass regression eval (no metric drops >3%)
      - Human review required for any proposed change (not fully automated)
  □ Track: prompt improvement rate over 4 weeks
      Hypothesis: 3-5% accuracy improvement per optimization cycle
```

---

### Phase 5 Success Criteria

✅ Feedback collected for >80% of completed segments
✅ Eval suites growing automatically (50+ new cases/month)
✅ Optimization pipeline running weekly with human oversight
✅ Measurable accuracy improvement after first 4-week optimization cycle
✅ "Time to detect regression" < 24 hours

---

## Phase 6: Enterprise Features (Weeks 23-26)

### Hypothesis
With a stable, self-improving system, we can now add enterprise capabilities: multi-tenant support, hypothesis assessment, and model exploration recommendations. These are high-value features that require the underlying architecture to be solid before they can be implemented correctly.

### P6.1 — Multi-Tenant Infrastructure (Week 23-24)

**Tasks:**
```
Week 23:
  □ Implement TenantConfig schema and storage (PostgreSQL)
  □ Implement TenantDataIsolator (strict tenant_id filtering on all queries)
  □ Add tenant context injection to perception layer
  □ Implement per-tenant facet catalog namespacing in Milvus
      - Collection naming: facets_{tenant_id}
      - Or: single collection with tenant_id field filter

Week 24:
  □ Implement skill overrides per tenant:
      - Tenant A: "high value" = $500+, Tenant B: "high value" = $100+
      - Stored as skill_overrides in TenantConfig
  □ Implement per-tenant vocabulary maps
  □ Implement per-tenant rate limits and usage tracking
  □ Test: create 3 test tenants with different configs, verify isolation
```

---

### P6.2 — Hypothesis Assessment Skill (Week 25)

**Tasks:**
```
Week 25:
  □ Design HypothesisAssessmentSkill:
      Input: segment definition + stated business goal
      Output: assessment score + reasoning + alternative suggestions

  □ Integrate historical campaign performance data (if available):
      - "Segments with these characteristics historically have X% conversion"

  □ Implement segment overlap detection:
      - "This segment overlaps 73% with Segment Y — consider merging"

  □ Implement segment quality scoring:
      - Size appropriateness (too small: <1000, too large: >80% of base)
      - Specificity score (highly specific vs. generic)
      - Business value estimate

  □ Build eval suite for hypothesis assessment:
      - Test cases: given segment + goal, verify assessment quality
```

---

### P6.3 — Model Exploration & Recommendation (Week 26)

**Tasks:**
```
Week 26:
  □ Implement ModelExplorationSkill:
      Analyzes: existing segments of a type
      Suggests: "You've created 12 'high purchase frequency' segments.
                 Consider building an ML model for this — here's why..."

  □ Implement rule-to-ML transition analysis:
      - Identify segments that would benefit from ML models vs. rule-based
      - Criteria: complexity, update frequency, accuracy requirements

  □ Implement segment consolidation recommendations:
      - Identify overlapping segments that could be merged
      - Identify gaps where new segments would add coverage

  □ Build eval suite for recommendations
  □ End-to-end system test: run 100 queries across all 6 phases of work
```

---

### Phase 6 Success Criteria

✅ 3+ tenants with isolated configs, data, and catalogs
✅ Hypothesis assessment skill operational with >80% user satisfaction rating
✅ Model recommendation skill identifies ML opportunities from segment patterns
✅ System passes full 100-query end-to-end eval with >88% accuracy

---

## Parallelization Opportunities

Many tasks across phases can run in parallel with proper team organization:

```
WEEK 1-2:
  Team A: P1.1 (Eval Gate)
  Team B: P1.2 (Segment Validation) + P1.3 (Milvus Fallback)

WEEK 3-4:
  Team A: P1.3 (Milvus Fallback finalization)
  Team B: P1.4 (Centralized Ambiguity Resolver)

WEEK 5-6:
  Team A: P2.1 (Redis Caching)
  Team B: P2.2 (DB Connection Pool) + P2.4 (Model Routing design)

WEEK 7-8:
  Team A: P2.3 (Parallel Execution)
  Team B: P2.4 (Model Routing implementation)

WEEK 9-10:
  Team A: P3.1 (Skill Registry infrastructure)
  Team B: Skill migration work (writing YAML skill definitions)

WEEK 19-20:
  Team A: P5.1 (Feedback Collection)
  Team B: P5.2 (Eval Auto-Growth)
```

---

## Dependency Graph

```
P1.1 (Eval Gates) ─────────────────────────────────────► P3.1 (Skill Registry)
P1.2 (Segment Validation) ──────────────────────────────► P4.2 (Memory: quality signals)
P1.3 (Milvus Fallback) ─────────────────────────────────► P2.1 (Redis: resilience)
P1.4 (Ambiguity Resolver) ──────────────────────────────► P4.1 (Short-term Memory)
P2.1 (Redis) ───────────────────────────────────────────► P4.1 (Session state in Redis)
P3.1 (Skill Registry) ──────────────────────────────────► P3.2 (Static Prompt)
                                                         ► P3.3 (Anti-Paraphrase)
                                                         ► P3.4 (Dynamic Few-Shot)
                                                         ► P5.3 (Prompt Optimization)
P4.1+P4.2 (Memory) ─────────────────────────────────────► P5.1 (Feedback Collection)
P5.1+P5.2 (Feedback+Evals) ─────────────────────────────► P5.3 (Prompt Optimization)
P3.x + P4.x + P5.x (All core) ─────────────────────────► P6.1 (Multi-Tenant)
                                                         ► P6.2 (Hypothesis)
                                                         ► P6.3 (Model Exploration)
```

---

## Risks and Mitigations

### Risk 1: Eval Infrastructure Adds Too Much Overhead
**Risk:** Teams feel slowed down by mandatory eval gates
**Probability:** Medium
**Impact:** High — could cause teams to bypass the gate
**Mitigation:**
- Make eval suites fast (<60 seconds per skill)
- Start with soft gates (alert but don't block) for 2 weeks, then hard gate
- Invest in eval tooling UX: one command to run all relevant evals
- Celebrate when evals catch a regression — make it a culture win

---

### Risk 2: Redis Adds Operational Complexity
**Risk:** Redis becomes another system to maintain, monitor, and fail
**Probability:** Low
**Impact:** Medium
**Mitigation:**
- Use managed Redis (AWS ElastiCache, GCP Memorystore)
- Design all caches as "best-effort" — always fall through to source on cache miss
- Add health checks; if Redis fails, system continues without caching (degraded performance, not failure)

---

### Risk 3: Skill Architecture Migration Breaks Existing Behavior
**Risk:** Migrating prompts to skill format introduces subtle behavior changes
**Probability:** Medium
**Impact:** High
**Mitigation:**
- Run old and new implementations in parallel during migration (shadow mode)
- A/B test: route 10% of traffic to new skill-based system, compare outcomes
- Full eval suite comparison before promoting

---

### Risk 4: Long-Term Memory Stores Incorrect Preferences
**Risk:** System learns wrong preferences (user gives one-time instruction, system treats as permanent preference)
**Probability:** Medium
**Impact:** Medium — annoying but not catastrophic
**Mitigation:**
- Add confidence threshold: only store preference after 2+ consistent observations
- Add preference expiry: preferences not reinforced in 60 days are downweighted
- Add user control: show learned preferences in profile, allow deletion
- Add override mechanism: "ignore my usual preference for this segment"

---

### Risk 5: Phase 6 Multi-Tenant Scope Creep
**Risk:** Multi-tenant support turns into a months-long infrastructure rewrite
**Probability:** Medium
**Impact:** High — could delay entire Phase 6
**Mitigation:**
- Limit Phase 6 scope to "logical tenant isolation" (config-based, not separate DB per tenant)
- Physical data isolation (separate DBs per tenant) is out of scope — defer to a potential Phase 7
- Start with 3 pilot tenants; generalize after pilot validation

---

## Key Metrics to Track Throughout

| Metric | Baseline (now) | Phase 1 Target | Phase 3 Target | Phase 6 Target |
|--------|---------------|----------------|----------------|----------------|
| P50 Latency | ~15s | ~12s | ~8s | ~6s |
| P99 Latency | ~45s | ~25s | ~15s | ~10s |
| Max Concurrent Users | ~30 | ~100 | ~200 | ~500 |
| Segment Accuracy | ~78% | ~85% | ~88% | ~90% |
| Grounding Rate | ~45% | ~65% | ~85% | ~90% |
| Duplicate Clarification Rate | ~25% | ~5% | ~2% | ~1% |
| LLM Cost per Segment | $X | $X | $0.45X | $0.35X |
| Mean Clarifications per Segment | ~3 | ~2 | ~1.5 | ~1 |
| Eval Suite Size | ~50 cases | ~150 cases | ~300 cases | ~600 cases |
| MTTR (Mean Time to Detect Regression) | Days | Hours | 24 hours | 1 hour |

---

## Team Structure Recommendation

For optimal execution, this roadmap assumes 3 teams:

**Team A: Infrastructure (2-3 engineers)**
- Owns: Redis, DB pool, parallel execution, Milvus fallback
- Phases: 2, part of 4

**Team B: ML/Prompts (2-3 engineers)**
- Owns: Eval gates, skill architecture, memory system, auto-improvement
- Phases: 1, 3, 5

**Team C: Product/Features (2-3 engineers)**
- Owns: UI changes, multi-tenant, hypothesis assessment, model exploration
- Phases: 1 (validation UI), 6

All teams share: eval infrastructure, observability, production monitoring

---

## Quick Wins (First 2 Weeks)

For stakeholders who need to see immediate results:

1. **Eval dashboard live** (Week 1): Show eval pass rates for all 4 skills
2. **Segment validation live** (Week 2): Prevent silent failures
3. **Milvus fallback live** (Week 3): First time Milvus is down, system stays up
4. **No duplicate questions** (Week 4): User feedback immediately positive

These four changes alone will significantly improve user trust and system reliability before any major architectural changes.

---

*Document produced as part of Enterprise Agentic Research — Research 1 (Sonnet Claude)*
*Prioritization based on: codebase analysis + bottleneck severity assessment + implementation complexity*
