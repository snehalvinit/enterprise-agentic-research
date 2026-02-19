# Implementation Roadmap — Smart-Segmentation to Enterprise Agentic System

> **Research ID:** research_1_opus_copilot  
> **Date:** February 2026  
> **Model:** Claude Opus 4.6 via GitHub Copilot  
> **Scope:** Phased execution plan with priorities, dependencies, and risk mitigation

---

## Overview

This roadmap translates the Concrete Upgrade Proposal (Document 03) into an actionable, phased plan. Work is organized into **four phases** spanning approximately **16-20 weeks**, with each phase delivering independently deployable improvements.

### Core Principles

1. **Most impactful first** — Security fixes and testing unlock everything else
2. **Parallelizable where possible** — Multiple teams/individuals can work concurrently
3. **Each phase is independently deployable** — No "big bang" cutover
4. **Hypothesis-driven** — Each upgrade encodes a testable hypothesis
5. **Eval gates protect quality** — Nothing deploys without passing eval thresholds

---

## Phase 0: Emergency Fixes (Week 1-2)

**Goal:** Eliminate critical security vulnerabilities and establish baseline testing.

### Tasks

| # | Task | Effort | Dependencies | Parallelizable |
|---|---|---|---|---|
| 0.1 | Replace all `eval()` with `json.loads()` + Pydantic | 1 day | None | Yes |
| 0.2 | Enable SSL verification | 0.5 day | Cert setup | Yes |
| 0.3 | Restrict CORS to allowed origins | 0.5 day | None | Yes |
| 0.4 | Remove `import pytest` from production code | 0.5 day | None | Yes |
| 0.5 | Add input validation to API endpoints | 1 day | 0.4 | Yes |
| 0.6 | Create initial unit test suite (20+ tests) | 3 days | None | Yes |
| 0.7 | Set up CI pipeline with test gate | 1 day | 0.6 | No |
| 0.8 | Create 10-case golden eval set from existing eval data | 2 days | None | Yes |

### Hypotheses

- **H0.1:** Replacing `eval()` eliminates arbitrary code execution risk without changing behavior → **Verify:** all existing API responses are identical before/after
- **H0.2:** An initial test suite catches at least 3 existing bugs → **Verify:** run tests, count findings

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `eval()` removal changes behavior for edge cases | Medium | Medium | Run existing eval set before and after; diff outputs |
| SSL enablement breaks internal service calls | Medium | High | Test with staging certs first; have bypass flag for rollback |
| Test suite reveals many issues simultaneously | High | Low | Triage: fix P0 issues, catalog P1+ for later |

### Exit Criteria

- [ ] Zero `eval()` calls in production code
- [ ] SSL verification enabled
- [ ] CORS restricted
- [ ] ≥ 20 unit tests passing in CI
- [ ] Eval golden set established with baseline scores documented

---

## Phase 1: Efficiency & Observability (Week 3-6)

**Goal:** Reduce cost by 70-85%, reduce latency by 40%, and establish end-to-end observability.

### Tasks

| # | Task | Effort | Dependencies | Parallelizable |
|---|---|---|---|---|
| 1.1 | Implement model router with task-to-tier mapping | 3 days | Phase 0 | — |
| 1.2 | Set up Redis and implement LLM response caching | 2 days | None | With 1.1 |
| 1.3 | Implement embedding caching | 1 day | 1.2 | No |
| 1.4 | Implement Milvus search result caching | 1 day | 1.2 | No |
| 1.5 | Create Milvus connection pool singleton | 1 day | None | Yes |
| 1.6 | Create embedding model singleton | 0.5 day | None | Yes |
| 1.7 | Expand PostgreSQL connection pool to 20 | 0.5 day | None | Yes |
| 1.8 | Add Prometheus metrics for LLM calls, latency, cost | 2 days | None | Yes |
| 1.9 | Set up Grafana dashboards for agent metrics | 2 days | 1.8 | No |
| 1.10 | Evaluate BGE-small vs BGE-large accuracy trade-off | 2 days | Phase 0 eval set | Yes |

### Parallelization Guide

```
Week 3:  [1.1 Model Router ───────] [1.5 Milvus Pool] [1.6 Embed Singleton]
         [1.2 Redis + LLM Cache ──] [1.7 PG Pool    ] [1.8 Prometheus ─────]
Week 4:  [1.1 continued ──────────] [1.3 Embed Cache] [1.8 continued]
         [1.10 BGE-small eval ────]                    [1.4 Milvus Cache ──]
Week 5:  [1.9 Grafana ────────────] [Integration Testing ──────────────────]
Week 6:  [Performance Testing ────] [Staging Deployment ──────────────────]
```

### Hypotheses

- **H1.1:** Model tiering achieves 70%+ cost reduction without quality regression → **Verify:** run eval suite with tiered models, compare against baseline
- **H1.2:** LLM response caching achieves 15%+ hit rate in production → **Verify:** monitor cache hit rate metric for 1 week post-deploy
- **H1.3:** BGE-small achieves ≤ 2% accuracy drop vs BGE-large → **Verify:** run retrieval eval on 100 queries, measure recall@10
- **H1.4:** Connection pooling reduces p95 latency by ≥ 20% → **Verify:** before/after latency comparison

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Economy model degrades routing accuracy | Medium | High | Run eval gate on routing accuracy; keep Sonnet fallback |
| Cache invalidation issues (stale responses) | Low | Medium | Short TTLs (1hr for LLM, 24hr for embeddings); add cache version |
| BGE-small quality is insufficient | Low | Medium | Keep BGE-large as fallback; A/B test in production |
| Redis adds operational complexity | Low | Low | Use managed Redis (ElastiCache) or Walmart's internal Redis |

### Exit Criteria

- [ ] Model router active with ≥ 3 tiers
- [ ] LLM response cache operational with ≥ 10% hit rate
- [ ] Embedding + Milvus caching operational
- [ ] All connection pools properly sized
- [ ] Grafana dashboard showing: request rate, latency p50/p95, cost/request, cache hit rate
- [ ] Cost per request reduced by ≥ 50% (measured over 1 week)
- [ ] Eval scores maintained within 2% of Phase 0 baseline

---

## Phase 2: Reliability & Architecture (Week 7-12)

**Goal:** Refactor the monolithic route handler, implement PAVI loop, add circuit breakers, and integrate eval gates into CI/CD.

### Tasks

| # | Task | Effort | Dependencies | Parallelizable |
|---|---|---|---|---|
| 2.1 | Refactor 641-line route handler into modular pipeline | 5 days | Phase 0 tests | — |
| 2.2 | Extract typed state models (replace 60+ flat vars) | 5 days | 2.1 | No |
| 2.3 | Implement PAVI loop for segment creation | 3 days | 2.2 | — |
| 2.4 | Implement segment verifier (multi-check) | 3 days | 2.3 | No |
| 2.5 | Add circuit breakers for LLM + Milvus + Postgres | 2 days | 2.1 | Yes |
| 2.6 | Set up eval runner in CI with quality gates | 3 days | Phase 0 eval set | Yes |
| 2.7 | Expand eval set to 50+ cases | 5 days | 2.6 | Yes |
| 2.8 | Implement prompt template system (replace inline strings) | 2 days | None | Yes |
| 2.9 | Add structured output validation on all LLM calls | 2 days | 2.2 | Yes |
| 2.10 | Unify LLM client (remove dual LiteLLM + Infero) | 3 days | Phase 1 | Yes |

### Parallelization Guide

```
Week 7:  [2.1 Route Handler Refactor ─────────────────────────────────]
         [2.8 Prompt Templates ────] [2.6 Eval Runner CI ─────────────]
Week 8:  [2.1 continued ──────────────────────────────────────────────]
         [2.5 Circuit Breakers ────] [2.7 Expand Eval Set ────────────]
Week 9:  [2.2 Typed State Models ─────────────────────────────────────]
         [2.10 Unify LLM Client ──────────] [2.9 Structured Output ──]
Week 10: [2.2 continued ─────────────────] [2.7 continued ───────────]
Week 11: [2.3 PAVI Loop ─────────────────]
Week 12: [2.4 Segment Verifier ──────────] [Integration Testing ─────]
```

### Hypotheses

- **H2.1:** Modular pipeline reduces handler complexity from 641 lines to ≤ 100 lines per module → **Verify:** LOC analysis post-refactor
- **H2.2:** Typed state reduces state-related bugs by replacing flat strings with validated Pydantic models → **Verify:** count type errors caught by Pydantic in first month
- **H2.3:** PAVI loop improves first-attempt accuracy by ≥ 10% → **Verify:** compare single-pass vs PAVI accuracy on eval set
- **H2.4:** Eval gates in CI prevent quality regressions → **Verify:** simulate a breaking change, confirm CI blocks deployment
- **H2.5:** Prompt templates reduce prompt-related bugs → **Verify:** count prompt loading errors before/after

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Route handler refactoring breaks SSE streaming | Medium | High | Keep old endpoint as fallback; feature flag for new pipeline |
| State migration misses edge cases | High | Medium | Comprehensive state dump comparison: old vs new for 100 sessions |
| PAVI loop adds latency (multiple LLM calls) | High | Medium | Budget iteration count (max 2); use economy model for verification |
| Breaking eval gate blocks deployments too aggressively | Medium | Medium | Start with warning mode; switch to blocking after 2 weeks |

### Exit Criteria

- [ ] Route handler modularized: ≤ 100 LOC per module
- [ ] All state accessed through typed Pydantic models
- [ ] PAVI loop operational for segment creation
- [ ] Segment verifier validates: schema, facets, operators, NL alignment
- [ ] Circuit breakers on all external service calls
- [ ] Eval gate blocking CI on < 90% accuracy
- [ ] ≥ 50 eval test cases maintained
- [ ] Zero inline prompt strings (all in template files)

---

## Phase 3: Intelligence & Extensibility (Week 13-20)

**Goal:** Add memory system, skill architecture, multi-tenancy, and auto-improvement foundation.

### Tasks

| # | Task | Effort | Dependencies | Parallelizable |
|---|---|---|---|---|
| 3.1 | Design and implement skill registry | 3 days | Phase 2 | — |
| 3.2 | Migrate `create_segment` into skill format | 3 days | 3.1 | — |
| 3.3 | Migrate `edit_segment` into skill format | 2 days | 3.1 | With 3.2 |
| 3.4 | Implement three-tier memory system (core/recall/archival) | 5 days | Phase 2 | — |
| 3.5 | Implement recipe storage (learn from successful segments) | 2 days | 3.4 | — |
| 3.6 | Implement correction learning (learn from user edits) | 2 days | 3.4 | With 3.5 |
| 3.7 | Implement knowledge store (replace pickle files) | 5 days | Phase 2 | Yes |
| 3.8 | Implement tenant config service | 3 days | None | Yes |
| 3.9 | Add tenant isolation to all data paths | 3 days | 3.8 | No |
| 3.10 | DSPy integration for prompt optimization prototype | 5 days | Phase 2 eval set | Yes |
| 3.11 | Implement feedback loop (production → eval → optimize) | 3 days | 3.10 | No |
| 3.12 | Create `assess_hypothesis` skill | 3 days | 3.1, 3.4 | Yes |

### Parallelization Guide

```
Week 13: [3.1 Skill Registry ─────────] [3.8 Tenant Config ───────]
         [3.7 Knowledge Store ────────────────────────────────────]
Week 14: [3.2 Migrate create_segment ─] [3.3 Migrate edit_segment]
         [3.7 continued ─────────────────────────────────────────]
Week 15: [3.4 Memory System ─────────────────────────────────────]
         [3.9 Tenant Isolation ───────────────────────────────────]
Week 16: [3.4 continued ────────────────────] [3.10 DSPy ────────]
         [3.9 continued ─────────────────────]
Week 17: [3.5 Recipe Storage ─] [3.6 Correction Learning ───────]
         [3.10 continued ────────────────────────────────────────]
Week 18: [3.12 assess_hypothesis skill ──────────────────────────]
         [3.11 Feedback Loop ────────────────────────────────────]
Week 19-20: [Integration Testing] [Performance Testing] [Staging Deploy]
```

### Hypotheses

- **H3.1:** Skill architecture allows adding a new capability in < 1 day without modifying core agent code → **Verify:** time the `assess_hypothesis` skill implementation end-to-end
- **H3.2:** Memory system improves repeat-query accuracy by ≥ 15% → **Verify:** create eval set of queries similar to stored recipes, measure accuracy gain
- **H3.3:** DSPy prompt optimization improves decomposer accuracy by ≥ 5% within 1 optimization cycle → **Verify:** before/after eval comparison
- **H3.4:** Knowledge store (PostgreSQL + Milvus) matches pickle file retrieval quality → **Verify:** run full eval suite with knowledge store vs pickle, diff results
- **H3.5:** Multi-tenancy adds < 5ms overhead per request → **Verify:** latency comparison with/without tenant resolution

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Skill migration changes agent behavior | Medium | High | A/B test old vs new skill-based pipeline; eval gate comparison |
| Memory system generates noise (low-quality memories) | Medium | Medium | Confidence scoring; only store user-confirmed outcomes; decay function |
| DSPy optimization overfits to eval set | Medium | High | Hold-out test set (20% of eval cases); cross-validation |
| Multi-tenancy introduces data leakage | Low | Critical | Automated tenant isolation tests; security audit |
| Knowledge store migration loses data | Low | High | Run dual-path (pickle + store) for 2 weeks; compare results |

### Exit Criteria

- [ ] ≥ 3 skills registered (create, edit, assess_hypothesis)
- [ ] New skill can be added via configuration only (no core code changes)
- [ ] Memory system storing and retrieving segment recipes
- [ ] Correction learning active (storing user edits)
- [ ] Knowledge store replacing pickle files
- [ ] Multi-tenant config operational for ≥ 2 tenants
- [ ] DSPy optimization prototype showing measurable improvement
- [ ] Feedback loop running (production logs → eval → optimization suggestions)
- [ ] All eval gates passing at ≥ 90% accuracy

---

## Cross-Phase Dependency Graph

```
Phase 0                Phase 1                Phase 2                Phase 3
────────               ────────               ────────               ────────

[eval() fix]           
[SSL fix]              
[CORS fix]──────────┐  
[pytest remove]     │  
                    │  
[unit tests]────────┤──[eval gates CI]─────────────────────────────────────
                    │                         │
[golden eval set]───┤──[model router]─────────┤───[DSPy optimization]
                    │  [cache layer]          │   [feedback loop]
                    │  [connection pools]     │
                    │  [prometheus]────────[grafana dashboards]
                    │  [bge-small eval]─────[embedding decision]
                    │                         │
                    ├──[route refactor]────────┤
                    │  [prompt templates]     │───[skill registry]
                    │  [circuit breakers]     │   [skill migration]
                    │  [unify LLM client]     │   [assess_hypothesis]
                    │                         │
                    │  [typed state]────────────┤───[memory system]
                    │  [PAVI loop]             │   [recipe storage]
                    │  [verifier]              │   [correction learning]
                    │  [structured output]     │
                    │                          │───[knowledge store]
                    │  [expand eval set]       │   [tenant config]
                    │                          │   [tenant isolation]
                    └──────────────────────────┘
```

---

## Resource Requirements

### Team

| Role | Phase 0 | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|---|
| Backend Engineer | 1 | 1-2 | 2 | 2-3 |
| ML Engineer | 0 | 0.5 | 0.5 | 1 |
| DevOps / SRE | 0.5 | 0.5 | 0.5 | 0.5 |
| QA / Eval Engineer | 0.5 | 0.5 | 1 | 1 |

### Infrastructure

| Component | Phase 0 | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|---|
| Redis | — | Required | Required | Required |
| Grafana/Prometheus | — | Required | Required | Required |
| CI Pipeline (KITT) | Enhanced | Enhanced | Enhanced | Enhanced |
| Additional DB capacity | — | — | — | For memory + knowledge |

---

## Success Metrics (Cumulative)

| Metric | Phase 0 | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|---|
| **Eval Accuracy** | Baseline | ≥ Baseline | +10% | +15% |
| **Cost/Request** | $0.09 | $0.01-0.02 | $0.01-0.02 | $0.01-0.02 |
| **Latency p95** | 30-45s | 15-25s | 10-20s | 10-20s |
| **Test Count** | 20+ | 30+ | 60+ | 100+ |
| **Eval Cases** | 10 | 10 | 50+ | 100+ |
| **Pod RAM** | 32-64GB | 4-8GB | 4-8GB | 4-8GB |
| **Security Issues** | 0 (fixed) | 0 | 0 | 0 |
| **New Skill Time** | N/A | N/A | N/A | < 1 day |

---

## Quick Wins (Can Start Today)

These require no architecture changes and provide immediate value:

1. **Replace `eval()` calls** — 30 minutes, eliminates critical vulnerability
2. **Enable SSL verification** — 15 minutes with proper certs
3. **Restrict CORS** — 5 minutes configuration change
4. **Remove `import pytest` from production** — 5 minutes
5. **Expand PostgreSQL pool from 3 → 20** — 5 minutes configuration change
6. **Add request logging** — 1 hour, immediate debugging value
7. **Create first 5 unit tests** — 2 hours, establishes testing culture

---

## Conclusion

This roadmap delivers enterprise-grade improvements incrementally, with Phase 0 providing immediate security value and Phase 1 providing dramatic cost savings. The full four-phase plan transforms Smart-Segmentation from prototype to production-grade in approximately 20 weeks, with each phase independently deployable and measurable through eval gates.

The key insight is that **the biggest improvements (security, cost, reliability) are also the lowest risk and fastest to implement**. The more ambitious capabilities (memory, auto-improvement, skills) are built on the solid foundation created by the earlier phases.
