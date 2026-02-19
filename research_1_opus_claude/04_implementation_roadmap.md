# Implementation Roadmap: Smart-Segmentation Enterprise Upgrade

> **Research ID**: research_1_opus_claude
> **Model**: Claude Opus 4.6
> **Date**: February 2026
> **Status**: Complete

---

## Executive Summary

This roadmap provides a step-by-step implementation plan for transforming Smart-Segmentation into an enterprise-grade agentic platform. Work is organized into **4 phases** across **16 weeks**, with the most impactful changes first. Parallelizable tracks are identified to maximize team throughput.

---

## Table of Contents

1. [Prioritization Framework](#1-prioritization-framework)
2. [Phase 1: Foundation (Weeks 1-4)](#2-phase-1-foundation-weeks-1-4)
3. [Phase 2: Intelligence (Weeks 5-8)](#3-phase-2-intelligence-weeks-5-8)
4. [Phase 3: Enterprise (Weeks 9-12)](#4-phase-3-enterprise-weeks-9-12)
5. [Phase 4: Auto-Improvement (Weeks 13-16)](#5-phase-4-auto-improvement-weeks-13-16)
6. [Parallel Track Map](#6-parallel-track-map)
7. [Dependencies & Critical Path](#7-dependencies--critical-path)
8. [Risks & Mitigation](#8-risks--mitigation)
9. [Success Metrics](#9-success-metrics)

---

## 1. Prioritization Framework

### Priority Criteria

Each task is scored on four dimensions:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| **Impact** | 40% | How much does this improve quality, reliability, or capability? |
| **Risk Reduction** | 25% | Does this fix security issues or prevent failures? |
| **Dependency** | 20% | How many other tasks depend on this? |
| **Effort** | 15% | Lower effort = higher priority (quick wins first) |

### Priority Tiers

| Tier | Criteria | Action |
|------|----------|--------|
| P0 | Security fix or blocking dependency | Do immediately |
| P1 | High impact, enables other work | Phase 1 |
| P2 | Significant improvement, can start after P1 foundations | Phase 2 |
| P3 | Enterprise features, needs P1+P2 foundation | Phase 3 |
| P4 | Optimization and auto-improvement | Phase 4 |

---

## 2. Phase 1: Foundation (Weeks 1-4)

> **Goal**: Fix critical issues, establish typed state, create skill architecture foundation, and add eval gates.

### Week 1: Critical Fixes & Typed State

#### Task 1.1: Security Fixes (P0)
**Priority Reasoning**: `eval()` usage and missing input sanitization are security vulnerabilities. Must fix before any other work.

| Item | Action | File(s) | Effort |
|------|--------|---------|--------|
| Remove `eval()` | Replace with `json.loads()` | `agent.py`, `sub_agents/*/agent.py` | 1 hour |
| Input sanitization | Add `InputValidator` class | New: `utils/input_validator.py` | 4 hours |
| Remove debug prints | Replace with structured logging | All agent files | 4 hours |
| Add rate limiting | Per-tenant/user rate limiter | `routes/agent_routes.py` | 4 hours |

**Dependency**: None — can start immediately.
**Verification**: Run existing test suite; add sanitization unit tests.

#### Task 1.2: Typed State Management (P1)
**Priority Reasoning**: This is the foundation for everything else. Typed state prevents silent bugs, enables IDE support, and is required for the skill architecture.

| Item | Action | File(s) | Effort |
|------|--------|---------|--------|
| Define `SessionState` Pydantic model | Replace 66+ string constants | New: `models/state.py` | 8 hours |
| Define `SegmentState` model | Typed segment construction state | New: `models/state.py` | 4 hours |
| Define `ConversationTurn` model | Typed conversation history | New: `models/state.py` | 2 hours |
| Migration adapter | Bridge old string keys → typed state | New: `utils/state_adapter.py` | 8 hours |
| Update `initialize_state` functions | Use typed state, remove duplication | `agent.py`, `sub_agents/*/agent.py` | 8 hours |

**Dependency**: None — can start immediately, parallel with Task 1.1.
**Verification**: All existing tests pass with typed state; type checking catches issues.

### Week 2: Prompt Architecture & Model Routing

#### Task 1.3: Prompt Modularization (P1)
**Priority Reasoning**: Separating static from dynamic prompt content enables prompt caching (immediate cost savings) and is required for the skill architecture.

| Item | Action | File(s) | Effort |
|------|--------|---------|--------|
| Create static system prompt | Extract common rules from all prompts | New: `prompts/system_prompt.txt` | 4 hours |
| Separate skill instructions | Extract per-skill instructions | `prompts/*.txt` | 8 hours |
| Replace `.replace()` with Jinja2 | Template engine with strict undefined | `utils/agent_prompt.py` | 4 hours |
| Prompt version tracking | Add version metadata to prompt files | `prompts/` + metadata | 4 hours |

**Dependency**: None — can start parallel with Tasks 1.1 and 1.2.

#### Task 1.4: Model Router (P1)
**Priority Reasoning**: Immediate cost savings (50-70%) with no quality loss on simple tasks. High ROI, low risk.

| Item | Action | File(s) | Effort |
|------|--------|---------|--------|
| Create `ModelRouter` class | Route tasks to appropriate models | New: `utils/model_router.py` | 8 hours |
| Define task-to-model mapping | Configure which tasks use which models | Configuration | 4 hours |
| Update agent creation | Each agent/tool uses routed model | `agent.py`, `sub_agents/*/agent.py` | 8 hours |
| Add prompt caching | Enable cache_control on static portions | `utils/agent_prompt.py` | 4 hours |

**Dependency**: Task 1.3 (prompt modularization enables caching).

### Week 3: Eval Gate Infrastructure

#### Task 1.5: Eval Gate in CI/CD (P1)
**Priority Reasoning**: Quality enforcement is the foundation of eval-first development. Without this, all other improvements lack a quality safety net.

| Item | Action | File(s) | Effort |
|------|--------|---------|--------|
| Define Tier 1 assertion checks | JSON schema, operator correctness, facet existence | New: `evaluations/tier1/` | 12 hours |
| Integrate evals with CI | GitHub Actions workflow for eval gate | `.github/workflows/eval-gate.yml` | 4 hours |
| Baseline eval dataset | Curate 50+ test cases from production | `evaluations/datasets/` | 8 hours |
| Eval dashboard | Basic metrics display | `evaluations/ui/` (extend existing) | 8 hours |

**Dependency**: None — can start parallel with all other Week 1-2 tasks.

#### Task 1.6: Structured Output Upgrade (P1)
**Priority Reasoning**: Replacing `StructuredInfero` with `Instructor` improves reliability and reduces retry overhead.

| Item | Action | File(s) | Effort |
|------|--------|---------|--------|
| Install Instructor library | Add to requirements | `requirements.txt` | 1 hour |
| Replace StructuredInfero calls | Use Instructor for structured output | `utils/pydantic_infero.py` + callers | 8 hours |
| Update Pydantic models | Add field-level validators | `data_models/*.py` | 4 hours |

**Dependency**: None — independent track.

### Week 4: Skill Architecture Foundation

#### Task 1.7: Skill Registry (P1)
**Priority Reasoning**: This is the core infrastructure that enables all future skill-based development. It decouples capabilities from the agent tree.

| Item | Action | File(s) | Effort |
|------|--------|---------|--------|
| Define `Skill` Pydantic model | Skill schema with all metadata | New: `models/skill.py` | 4 hours |
| Create `SkillRegistry` class | CRUD operations for skills | New: `core/skill_registry.py` | 12 hours |
| Implement skill loader | Load skill instructions into agent context | New: `core/skill_loader.py` | 8 hours |
| Create intent-to-skill router | Map intents to skills | New: `core/skill_router.py` | 8 hours |
| Migrate NSC to skill format | Convert NSC agent tree to a skill definition | `skills/segment_creation/` | 12 hours |
| Migrate DSE to skill format | Convert DSE agent tree to a skill definition | `skills/segment_editing/` | 8 hours |

**Dependency**: Tasks 1.2 (typed state) and 1.3 (prompt modularization).

### Phase 1 Deliverables

| Deliverable | Status Check |
|------------|-------------|
| Security vulnerabilities fixed | eval() removed, input validation active |
| Typed state management | All state is Pydantic models, tests pass |
| Prompt architecture modularized | Static/dynamic separated, Jinja2 templates |
| Model routing active | Different models for different tasks |
| Eval gate in CI | Every PR runs evals, regressions blocked |
| Structured output via Instructor | StructuredInfero replaced |
| Skill registry operational | NSC and DSE converted to skills |
| **Cost reduction achieved** | **40-60% estimated from model routing + caching** |

---

## 3. Phase 2: Intelligence (Weeks 5-8)

> **Goal**: Add memory system, Plan-Act-Verify-Improve loop, knowledge store, and enhanced evaluation.

### Week 5-6: Memory System

#### Task 2.1: Memory Store Implementation (P2)
**Priority Reasoning**: Memory is the foundation for auto-improvement and personalization. Without memory, the agent cannot learn from experience.

| Item | Action | Effort |
|------|--------|--------|
| Design memory schema | Working, episodic, semantic, procedural types | 4 hours |
| Implement `MemoryStore` class | CRUD + semantic search over memories | 16 hours |
| Add namespace isolation | Per-tenant, per-user memory scoping | 8 hours |
| Implement memory recall | Retrieve relevant memories for current task | 8 hours |
| Implement success/failure storage | Automatically store outcomes | 4 hours |
| Integration with agent pipeline | Memory recall in planning step | 8 hours |

**Dependency**: Task 1.2 (typed state for memory content).

### Week 5-6: Plan-Act-Verify-Improve Loop (Parallel Track)

#### Task 2.2: Reasoning Engine (P2)
**Priority Reasoning**: The PAVI loop is the highest-ROI reliability improvement. Self-verification catches errors that propagate through the pipeline.

| Item | Action | Effort |
|------|--------|--------|
| Implement `ReasoningEngine` class | Plan-Act-Verify-Improve orchestrator | 16 hours |
| Planning step | Generate and explain execution plan | 8 hours |
| Verification step | Multi-level output verification | 12 hours |
| Improvement step | Analyze failures, generate corrections | 8 hours |
| Integration with skill execution | Wrap all skill execution in PAVI loop | 8 hours |

**Dependency**: Task 1.7 (skill registry for skill-based execution).

### Week 7-8: Knowledge Store

#### Task 2.3: Knowledge System (P2)
**Priority Reasoning**: Moving from pickle files to a proper knowledge store enables tenant isolation, dynamic updates, and better retrieval.

| Item | Action | Effort |
|------|--------|--------|
| Design knowledge schema | Facet catalog, business rules, recipes | 4 hours |
| Implement `KnowledgeStore` class | Retrieval with tenant isolation | 16 hours |
| Migrate facet catalog | From pickle files to knowledge store | 12 hours |
| Implement HyDE-style query enhancement | Better facet retrieval | 8 hours |
| Add citation tracking | Track which knowledge was used | 4 hours |

**Dependency**: None for design; Task 1.7 for integration.

### Week 7-8: Enhanced Evaluation (Parallel Track)

#### Task 2.4: Tier 2 Evaluation (P2)
**Priority Reasoning**: LLM-as-judge evaluation provides semantic quality checks that assertions cannot.

| Item | Action | Effort |
|------|--------|--------|
| Define LLM-as-judge prompts | Grading rubrics for segment quality | 8 hours |
| Implement Tier 2 eval checks | Intent coverage, facet relevance, completeness | 12 hours |
| Build eval dataset from production | Capture real queries + expected outputs | 8 hours |
| Online evaluation sampling | Run evals on sampled production traces | 8 hours |

**Dependency**: Task 1.5 (eval gate infrastructure).

### Phase 2 Deliverables

| Deliverable | Status Check |
|------------|-------------|
| Memory system operational | Stores successes/failures, recalls relevant context |
| PAVI loop active | Every skill execution goes through Plan-Act-Verify-Improve |
| Knowledge store | Facet catalog migrated, tenant-isolated retrieval |
| Tier 2 evaluation | LLM-as-judge evals running in CI and on production samples |
| **Quality improvement achieved** | **Self-verification catching 60%+ of errors before user sees them** |

---

## 4. Phase 3: Enterprise (Weeks 9-12)

> **Goal**: Add multi-tenant support, new skills (hypothesis testing, campaign recommendation), and production observability.

### Week 9-10: Multi-Tenant Support

#### Task 3.1: Tenant Configuration System (P3)
**Priority Reasoning**: Multi-tenant is required for enterprise deployment. Must be built on the typed state and skill architecture from Phases 1-2.

| Item | Action | Effort |
|------|--------|--------|
| Define `TenantConfig` model | Full tenant configuration schema | 4 hours |
| Implement tenant store | CRUD for tenant configs | 8 hours |
| Tenant-scoped skill loading | Load skills based on tenant permissions | 8 hours |
| Tenant-scoped knowledge | Isolate knowledge retrieval by tenant | 8 hours |
| Tenant-scoped memory | Isolate memory by tenant namespace | 4 hours |
| Per-tenant cost tracking | Track and limit costs per tenant | 8 hours |
| Per-tenant rate limiting | Enforce request limits | 4 hours |

**Dependency**: Tasks 1.7, 2.1, 2.3 (skill registry, memory, knowledge).

### Week 9-10: Observability (Parallel Track)

#### Task 3.2: Production Observability (P3)
**Priority Reasoning**: Without observability, you can't measure improvements or detect degradation.

| Item | Action | Effort |
|------|--------|--------|
| Structured logging | Replace print statements with structured logs | 8 hours |
| Token/cost tracking per request | Track usage at agent-call level | 8 hours |
| Latency breakdown dashboard | Per-step latency tracking | 8 hours |
| Quality metrics dashboard | Eval scores, error rates, confidence | 8 hours |
| Alerting | Slack/PagerDuty alerts for quality drops | 4 hours |

**Dependency**: Task 2.4 (evaluation metrics to track).

### Week 11-12: New Skills

#### Task 3.3: Hypothesis Assessment Skill (P3)
**Priority Reasoning**: A highly requested capability that differentiates the platform from simple query-to-segment conversion.

| Item | Action | Effort |
|------|--------|--------|
| Define skill specification | Input/output schema, instructions | 4 hours |
| Implement hypothesis evaluation logic | Use data exploration tools | 16 hours |
| Write eval suite | Test cases for hypothesis assessment | 8 hours |
| Integration testing | End-to-end with skill registry | 4 hours |

**Dependency**: Tasks 1.7, 2.2, 2.3 (skill registry, reasoning engine, knowledge store).

#### Task 3.4: Segment Analysis Skill (P3)
**Priority Reasoning**: Users need to understand segments, not just create them. Analysis provides insights about overlap, size, characteristics.

| Item | Action | Effort |
|------|--------|--------|
| Define skill specification | Input/output schema, instructions | 4 hours |
| Implement analysis logic | Segment characteristics, overlap detection | 12 hours |
| Write eval suite | Test cases for analysis quality | 8 hours |

**Dependency**: Same as Task 3.3.

### Phase 3 Deliverables

| Deliverable | Status Check |
|------------|-------------|
| Multi-tenant support | Per-tenant configs, data isolation, cost tracking |
| Production observability | Dashboards, alerts, cost attribution |
| Hypothesis assessment skill | Users can test hypotheses with evidence |
| Segment analysis skill | Users can analyze segment characteristics |
| **Enterprise readiness** | **Multiple tenants can use the system with isolation** |

---

## 5. Phase 4: Auto-Improvement (Weeks 13-16)

> **Goal**: Close the feedback loop with auto-improvement, prompt optimization, and advanced capabilities.

### Week 13-14: Auto-Improvement Pipeline

#### Task 4.1: Feedback Collection & Analysis (P4)
**Priority Reasoning**: Auto-improvement requires data on what works and what doesn't. This task collects and analyzes that data.

| Item | Action | Effort |
|------|--------|--------|
| User feedback collection | Thumbs up/down, corrections | 8 hours |
| Failure pattern analysis | Cluster common failure modes | 12 hours |
| Improvement candidate generation | Suggest prompt/skill improvements | 12 hours |
| A/B testing framework | Test candidates against baseline | 16 hours |

**Dependency**: Tasks 2.1, 2.4 (memory for storing feedback, evals for measuring improvement).

#### Task 4.2: DSPy-Style Prompt Optimization (P4)
**Priority Reasoning**: Automated prompt optimization can improve all 23+ prompts simultaneously based on eval data.

| Item | Action | Effort |
|------|--------|--------|
| Define metrics for each prompt | What makes a good output for each agent? | 8 hours |
| Implement optimization pipeline | Run DSPy or similar on prompt set | 16 hours |
| Eval comparison | Compare optimized vs. manual prompts | 4 hours |
| Rollout framework | Gradually deploy optimized prompts | 8 hours |

**Dependency**: Task 4.1 (needs failure data to optimize against).

### Week 15-16: Advanced Capabilities

#### Task 4.3: Campaign Recommendation Skill (P4)
**Priority Reasoning**: Extends the platform from segmentation into actionable recommendations.

| Item | Action | Effort |
|------|--------|--------|
| Define skill specification | Campaign parameters, targeting | 4 hours |
| Implement recommendation logic | Connect segments to campaign types | 16 hours |
| Write eval suite | Test cases for recommendation quality | 8 hours |

#### Task 4.4: Model Exploration Skill (P4)
**Priority Reasoning**: Enables users to discover and leverage predictive models in their ecosystem.

| Item | Action | Effort |
|------|--------|--------|
| Define skill specification | Model assessment, recommendation | 4 hours |
| Implement model catalog integration | Query available models | 12 hours |
| Implement recommendation logic | Match segments to model opportunities | 12 hours |
| Write eval suite | Test cases | 8 hours |

### Phase 4 Deliverables

| Deliverable | Status Check |
|------------|-------------|
| Auto-improvement pipeline | Failures analyzed, improvements suggested, A/B tested |
| Prompt optimization | Automated optimization improving eval scores |
| Campaign recommendation | Users receive actionable campaign suggestions |
| Model exploration | Users can discover relevant models |
| **Self-improving system** | **Agent gets better over time through feedback loops** |

---

## 6. Parallel Track Map

```
Week:  1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16
       ├────┤────┤────┤────┤────┤────┤────┤────┤────┤────┤────┤────┤────┤────┤────┤

Track A (Core Architecture):
       [1.1 Security Fixes     ]
       [1.2 Typed State ───────]
                    [1.7 Skill Registry ───────]
                                   [2.2 Reasoning Engine ──────]
                                                       [3.1 Multi-Tenant ──────]
                                                                           [4.1 Auto-Improve ─────]

Track B (Prompts & Models):
       [1.3 Prompt Modular ────]
              [1.4 Model Router ]
              [1.6 Structured Output ]
                                   [2.3 Knowledge Store ──────]
                                                       [3.2 Observability ─────]
                                                                           [4.2 Prompt Optim ─────]

Track C (Evaluation):
       [1.5 Eval Gate CI ──────────────]
                                   [2.4 Tier 2 Eval ──────────]
                                                       [3.3 Hypothesis Skill ──]
                                                       [3.4 Analysis Skill ────]
                                                                           [4.3 Campaign Skill ───]

Track D (Memory):
                                   [2.1 Memory System ────────]
                                                                           [4.4 Model Explore ────]
```

**Key Parallelism**:
- Phase 1: 3 parallel tracks (security+state, prompts+models, eval)
- Phase 2: 2 parallel tracks (memory+reasoning, knowledge+eval)
- Phase 3: 2 parallel tracks (multi-tenant, observability+skills)
- Phase 4: 2 parallel tracks (auto-improvement, new skills)

**Team Sizing**:
- Track A: 1-2 senior engineers (architecture)
- Track B: 1 engineer (prompts/models)
- Track C: 1 engineer (evaluation + new skills)
- Track D: 1 engineer (memory + data)
- Total: **4-5 engineers**

---

## 7. Dependencies & Critical Path

### Critical Path

```
Security Fixes (1.1) → Typed State (1.2) → Skill Registry (1.7) → Reasoning Engine (2.2) → Multi-Tenant (3.1)
```

This is the longest chain and determines the minimum timeline. Each step must complete before the next begins.

### Dependency Graph

| Task | Depends On | Blocks |
|------|-----------|--------|
| 1.1 Security Fixes | Nothing | Nothing |
| 1.2 Typed State | Nothing | 1.7, 2.1, 2.2 |
| 1.3 Prompt Modular | Nothing | 1.4, 1.7 |
| 1.4 Model Router | 1.3 | Nothing |
| 1.5 Eval Gate CI | Nothing | 2.4 |
| 1.6 Structured Output | Nothing | Nothing |
| 1.7 Skill Registry | 1.2, 1.3 | 2.2, 2.3, 3.1, 3.3, 3.4 |
| 2.1 Memory System | 1.2 | 4.1 |
| 2.2 Reasoning Engine | 1.7 | 3.1, 3.3, 3.4 |
| 2.3 Knowledge Store | 1.7 | 3.1 |
| 2.4 Tier 2 Eval | 1.5 | 4.1 |
| 3.1 Multi-Tenant | 1.7, 2.1, 2.3 | Nothing |
| 3.2 Observability | 2.4 | Nothing |
| 3.3 Hypothesis Skill | 1.7, 2.2, 2.3 | Nothing |
| 3.4 Analysis Skill | 1.7, 2.2, 2.3 | Nothing |
| 4.1 Auto-Improve | 2.1, 2.4 | 4.2 |
| 4.2 Prompt Optim | 4.1 | Nothing |
| 4.3 Campaign Skill | 1.7 | Nothing |
| 4.4 Model Explore | 1.7 | Nothing |

---

## 8. Risks & Mitigation

### Risk 1: Breaking Existing Functionality During Refactor

| Aspect | Detail |
|--------|--------|
| **Probability** | High |
| **Impact** | High |
| **Mitigation** | Build migration adapters that bridge old interfaces to new ones. Run all existing eval tests after every change. Feature flag new components. Maintain backward compatibility during transition. |
| **Contingency** | Roll back to pre-refactor code via git. Adapters allow partial migration. |

### Risk 2: Model Routing Quality Degradation

| Aspect | Detail |
|--------|--------|
| **Probability** | Medium |
| **Impact** | Medium |
| **Mitigation** | Start conservative: only route clearly simple tasks to smaller models. Run A/B comparisons for each routing decision. Keep Tier 2 eval running in CI. |
| **Contingency** | Route all tasks back to the original model (zero quality loss, just cost). |

### Risk 3: Memory System Adds Latency

| Aspect | Detail |
|--------|--------|
| **Probability** | Medium |
| **Impact** | Low-Medium |
| **Mitigation** | Make memory recall async and non-blocking. Set timeout on memory retrieval (100ms). Memory is additive — agent works without it, just less personalized. |
| **Contingency** | Disable memory recall; system degrades to current behavior. |

### Risk 4: Multi-Tenant Data Leakage

| Aspect | Detail |
|--------|--------|
| **Probability** | Low |
| **Impact** | Critical |
| **Mitigation** | Tenant ID filtering at every data access layer. Integration tests that verify cross-tenant isolation. Security audit of all query paths. |
| **Contingency** | Fall back to single-tenant mode. Add additional isolation layers. |

### Risk 5: Team Capacity / Timeline Slip

| Aspect | Detail |
|--------|--------|
| **Probability** | Medium-High |
| **Impact** | Medium |
| **Mitigation** | Phases are independent — can ship Phase 1 value without Phase 2-4. Each phase delivers standalone improvements. Parallel tracks allow flexible resourcing. |
| **Contingency** | Extend timeline. Phases 3-4 can be deferred without losing Phase 1-2 value. |

### Risk 6: Google ADK Breaking Changes

| Aspect | Detail |
|--------|--------|
| **Probability** | Low-Medium |
| **Impact** | High |
| **Mitigation** | Wrap ADK-specific code behind interfaces. The skill architecture abstracts away ADK internals. Build adapter layer so ADK can be swapped. |
| **Contingency** | Migrate to LangGraph or custom orchestration. Skill definitions and eval suites are framework-agnostic. |

---

## 9. Success Metrics

### Phase 1 Success Criteria

| Metric | Target | Measurement |
|--------|--------|------------|
| Security vulnerabilities | 0 critical/high | Security scan |
| Eval gate active | All PRs gated | CI pipeline logs |
| Cost reduction | 40%+ | Token usage tracking |
| State type safety | 100% typed | Type checking passes |
| Test pass rate | 95%+ | CI eval results |

### Phase 2 Success Criteria

| Metric | Target | Measurement |
|--------|--------|------------|
| Self-verification catch rate | 60%+ of errors caught | Verification logs |
| Memory recall relevance | 70%+ relevant memories | Manual review sample |
| Knowledge retrieval accuracy | 85%+ relevant facets in top-10 | Retrieval eval |
| Eval coverage | 100+ test cases | Eval dataset size |

### Phase 3 Success Criteria

| Metric | Target | Measurement |
|--------|--------|------------|
| Multi-tenant isolation | 0 cross-tenant leaks | Security test suite |
| Observability coverage | All agents traced | Dashboard completeness |
| New skill quality | 90%+ eval pass rate | Skill eval suites |
| Tenant onboarding time | < 1 hour | Onboarding test |

### Phase 4 Success Criteria

| Metric | Target | Measurement |
|--------|--------|------------|
| Auto-improvement suggestions | 5+ per month | Improvement pipeline logs |
| Prompt optimization lift | 5%+ eval score improvement | A/B test results |
| User satisfaction | 85%+ positive feedback | User feedback collection |
| System improvement trend | Improving eval scores month-over-month | Eval trend charts |

### Overall Success (End of 16 Weeks)

| Dimension | Before | After |
|-----------|--------|-------|
| Cost per query | ~$1.00 | ~$0.35 |
| Segment quality (eval score) | ~75% | ~92% |
| Error rate | ~15% | ~5% |
| New skill deployment time | Weeks (code change) | Hours (skill definition) |
| Tenant onboarding | Custom code | Configuration |
| Self-improvement | None | Continuous |
| Memory / Personalization | None | Per-user/tenant |
| Observability | Basic traces | Full dashboards + alerts |

---

## Appendix: Quick Reference

### What to Build First (If Time Is Limited)

If you only have 4 weeks, do these in order:
1. **Security fixes** (Task 1.1) — 1 day
2. **Model routing + prompt caching** (Tasks 1.3, 1.4) — 1 week
3. **Typed state** (Task 1.2) — 1 week
4. **Eval gate in CI** (Task 1.5) — 1 week
5. **Skill registry** (Task 1.7) — 1 week

This delivers: 50% cost reduction, security hardening, quality gates, and the foundation for all future work.

### What to Build Next (If Time Extends)

If you have 8 more weeks after the first 4:
6. **Memory system** (Task 2.1) — 2 weeks
7. **PAVI loop** (Task 2.2) — 2 weeks
8. **Knowledge store** (Task 2.3) — 2 weeks
9. **Multi-tenant** (Task 3.1) — 2 weeks

This delivers: self-correcting agent, learning from experience, enterprise knowledge management, and multi-tenant support.

---

*Built for the Enterprise Agentic Research Initiative*
*Last Updated: February 2026*
