# 04 — Implementation Roadmap: Smart-Segmentation Enterprise Upgrade

> **Research ID:** research_2_opus_rag_extra_claude
> **Model:** Claude Opus 4.6
> **Date:** February 2026
> **Timeline:** 20 weeks (5 phases)
> **Approach:** Highest-impact, lowest-effort changes first; parallel tracks where possible

---

## Executive Summary

This roadmap transforms Smart-Segmentation from a 7-stage sequential pipeline into an enterprise-grade agentic system across **5 phases over 20 weeks**. The plan is ordered by **impact-to-effort ratio** — quick wins that deliver measurable quality and cost improvements come first, architectural changes follow.

**Phase Overview:**

| Phase | Weeks | Focus | Key Deliverable | Impact |
|-------|-------|-------|-----------------|--------|
| 1 | 1-3 | Quick Wins | Ground truth RAG + deterministic stages + hybrid search | +15-25% accuracy, -40% latency |
| 2 | 4-8 | Pipeline Refactor | 7 stages → 3 stages + cascade retrieval | +20% accuracy, -60% cost |
| 3 | 9-13 | Evaluation & Skills | CI/CD eval gates + skill registry + observability | Quality regression prevention |
| 4 | 14-17 | Multi-Tenant | Tenant config + isolation + onboarding flow | Enterprise-ready |
| 5 | 18-20 | Auto-Improve & Memory | DSPy optimization + memory system + feedback loops | Continuous improvement |

---

## Phase 1: Quick Wins (Weeks 1-3)

**Hypothesis:** The biggest accuracy gains come from using existing data (ground truth) and removing unnecessary LLM calls, not from architectural changes.

### Week 1: Deterministic Stage Replacement + Ground Truth RAG

#### Task 1.1: Replace Date Tagger with Rule-Based Code
- **What:** Replace `date_extraction_prompt.txt` LLM call with `dateparser` + custom patterns from `date_tagger_patterns.json`
- **Why:** 95% of date expressions are deterministic; saves 1 LLM call per segment
- **Effort:** 2 days
- **Risk:** Low — fallback to LLM for ambiguous dates
- **Metric:** Date extraction accuracy ≥ 95% on ground truth
- **Files to change:** `date_tagger_agent.py`, new `date_parser.py`

#### Task 1.2: Replace Formatter with Deterministic JSON Transform
- **What:** Replace `master_format_generator_prompt.txt` LLM call with Python code that transforms structured facet-value pairs to SegmentR JSON
- **Why:** JSON formatting is deterministic; eliminates hallucination risk in formatting
- **Effort:** 2 days
- **Risk:** Very low — pure data transformation
- **Metric:** Schema validity = 100%
- **Files to change:** `segment_format_generator/tools/segmentr_formatter.py`

#### Task 1.3: Implement Ground Truth Few-Shot RAG
- **What:** Embed ground truth segment descriptions; retrieve top-3 similar examples at runtime; inject into facet mapper prompt
- **Why:** Dynamic few-shot from real validated examples improves accuracy by 15-25%
- **Effort:** 3 days
- **Risk:** Medium — need to ensure example quality doesn't degrade
- **Metric:** Facet recall improvement of +15% on eval set
- **Files to create:** `utils/ground_truth_rag.py`; modify `facet_value_operator_mapper_prompt.txt`
- **Dependency:** Ground truth CSV must be accessible and cleaned

### Week 2: Enable Hybrid Search + Grounding

#### Task 2.1: Enable Hybrid BM25+Dense Search by Default
- **What:** Switch `search_mode` from `"standard"` to `"hybrid"` in `shortlist_generation.py`
- **Why:** Hybrid search is already implemented in `milvus.py` but not activated; provides 5-15% retrieval improvement
- **Effort:** 1 day (configuration change + testing)
- **Risk:** Low — implementation exists, just needs activation and validation
- **Metric:** Retrieval Recall@5 improvement of +5-10%
- **Files to change:** `tools/shortlist_generation.py` (search_mode parameter)

#### Task 2.2: Pass Original Query to All Stages
- **What:** Ensure every pipeline stage receives the original user query (not just paraphrased intermediate results)
- **Why:** Prevents semantic drift from stage-to-stage paraphrasing
- **Effort:** 1 day
- **Risk:** Very low — additive change
- **Files to change:** State management in `agent.py`, pass `ORIGINAL_USER_QUERY` to all tool calls

#### Task 2.3: Add Grounding Rule to Facet Prompts
- **What:** Add explicit instruction: "Select ONLY from provided candidates. If no match, return 'uncertain' — do NOT invent facets."
- **Why:** Eliminates hallucinated facet names/values
- **Effort:** 0.5 days
- **Risk:** Very low
- **Files to change:** `facet_value_operator_mapper_prompt.txt`, `facet_classifier_matcher_prompt.txt`

#### Task 2.4: Add Prompt Caching
- **What:** Enable prompt caching for system prompt + contextual information prefix
- **Why:** 80% input token cost reduction for repeated prefix content
- **Effort:** 1 day
- **Risk:** Very low — framework-level feature
- **Files to change:** LLM call wrapper in `utils/adk_llm_model.py`

### Week 3: Cross-Encoder Reranking + Security Fix

#### Task 3.1: Add Cross-Encoder Reranking
- **What:** After Milvus returns top-50 candidates, run BGE-reranker-large to select top-5
- **Why:** 95% of LLM reranking quality at 3x speed and 72% cost reduction
- **Effort:** 3 days
- **Risk:** Medium — new dependency (cross-encoder model), latency impact to measure
- **Metric:** Facet precision improvement of +10-15%
- **Files to create:** `utils/reranker.py`; modify `tools/shortlist_generation.py`

#### Task 3.2: Remove eval() Calls from Environment Variable Parsing
- **What:** Replace `eval(os.environ.get(...))` with `json.loads()` throughout
- **Why:** Security vulnerability — eval() on env vars is dangerous
- **Effort:** 1 day
- **Risk:** Very low
- **Files to change:** `tools/shortlist_generation.py`, anywhere `eval()` is used on env vars

#### Task 3.3: Use NER Results to Filter Milvus Search
- **What:** Use extracted entities from NER agent to narrow Milvus search scope via metadata filters
- **Why:** NER is already run but results aren't used to reduce search space
- **Effort:** 2 days
- **Risk:** Medium — need to validate that filtering doesn't exclude valid facets
- **Metric:** Search precision improvement, latency reduction
- **Files to change:** `tools/shortlist_generation.py`, `agent_tools/named_entity_recognition_agent.py`

**Phase 1 Exit Criteria:**
- [ ] Date extraction accuracy ≥ 95% without LLM
- [ ] Formatter produces 100% schema-valid JSON without LLM
- [ ] Ground truth few-shot RAG integrated and eval shows ≥ 15% recall improvement
- [ ] Hybrid search enabled and validated
- [ ] All `eval()` calls removed
- [ ] Facet precision improved by ≥ 10% with cross-encoder

---

## Phase 2: Pipeline Refactor (Weeks 4-8)

**Hypothesis:** Collapsing 7 stages to 3 reduces error accumulation (0.9^7 = 47.8% → 0.9^3 = 72.9%) and cuts latency/cost by 50-60%.

### Week 4-5: Cascade Retriever

#### Task 2.1: Implement Cascade Retriever
- **What:** Build `CascadeRetriever` class with 4-stage cascade: exact match → BM25 → type filter → embedding fallback
- **Why:** Structured first, embeddings as fallback — matches the structured nature of the facet catalog
- **Effort:** 5 days
- **Risk:** Medium — needs thorough testing against current retrieval quality
- **Metric:** Retrieval precision ≥ 85% on ground truth, latency ≤ 500ms
- **Files to create:** `utils/cascade_retriever.py`, `utils/bm25_index.py`, `utils/facet_trie.py`
- **Dependency:** Phase 1 cross-encoder must be working

#### Task 2.2: Build Facet Taxonomy Graph
- **What:** Lightweight graph encoding parent/child and linked facet relationships
- **Why:** Enables deterministic traversal for hierarchical facets (Super Department → Division → Brand)
- **Effort:** 3 days
- **Risk:** Low — additive capability
- **Files to create:** `utils/facet_taxonomy.py`
- **Testing:** Validate that taxonomy traversal returns correct related facets for all 46 ground truth segments

### Week 6-7: Stage Merge — Reason & Map

#### Task 2.3: Merge Decomposer + Facet Mapper into Single LLM Call
- **What:** Create unified "Reason & Map" prompt that decomposes, extracts dates (code pre-pass), and maps facets in one structured output call
- **Why:** Eliminates 2-3 LLM calls; prevents semantic drift; modern models handle this in one pass
- **Effort:** 5 days
- **Risk:** High — this is the core pipeline change; needs extensive eval
- **Metric:** End-to-end F1 ≥ 80% (matching or exceeding current)
- **Files to create:** `stages/reason_and_map.py`; new unified prompt
- **Files to modify:** NSC agent orchestration in `sub_agents/new_segment_creation/agent.py`
- **Testing:** A/B test against current pipeline on full eval set

#### Task 2.4: Implement Perceive Stage
- **What:** Replace Route Agent LLM call with code-based intent classifier + NER
- **Why:** Intent classification is a 4-way classification; doesn't need a full LLM
- **Effort:** 3 days
- **Risk:** Low — simple classification task
- **Files to create:** `stages/perceive.py`

### Week 8: Validate & Resolve + Integration

#### Task 2.5: Implement Validate & Resolve Stage
- **What:** Verification stage that checks completeness, resolves dependencies, handles ambiguity
- **Why:** Current pipeline has no explicit verification between mapping and formatting
- **Effort:** 3 days
- **Risk:** Medium — needs to handle all dependency patterns
- **Files to create:** `stages/validate_and_resolve.py`

#### Task 2.6: Integration Testing
- **What:** Run full eval suite on new 3-stage pipeline; compare against 7-stage baseline
- **Effort:** 2 days
- **Risk:** Findings may require adjustments
- **Acceptance criteria:**
  - End-to-end F1 ≥ 80%
  - Latency ≤ 8s (p95)
  - LLM cost ≤ 40% of baseline
  - No regression on any segment type

**Phase 2 Exit Criteria:**
- [ ] Cascade retriever deployed and retrieval precision ≥ 85%
- [ ] Pipeline reduced from 7 to 3 stages
- [ ] End-to-end F1 ≥ 80% on eval set
- [ ] Latency reduced by ≥ 50%
- [ ] LLM cost reduced by ≥ 50%

---

## Phase 3: Evaluation & Skills (Weeks 9-13)

**Hypothesis:** Eval-first development and skill-based architecture prevent quality regressions and enable rapid, safe iteration.

### Week 9-10: Evaluation Infrastructure

#### Task 3.1: Per-Stage Evaluation Framework
- **What:** Create eval metrics for each stage (intent accuracy, NER precision, facet recall@5, facet precision, schema validity)
- **Effort:** 4 days
- **Files to create:** `evaluations/per_stage_eval.py`, stage-specific eval sets
- **Dependency:** Phase 2 pipeline must be deployed

#### Task 3.2: CI/CD Eval Gates
- **What:** GitHub Actions workflow that runs evals on every PR touching prompts/code
- **Effort:** 2 days
- **Files to create:** `.github/workflows/eval-gate.yaml`
- **Metric:** 100% of prompt changes must pass eval before merge

#### Task 3.3: Retrieval Quality Metrics
- **What:** Implement Recall@K, MRR, NDCG for Milvus retrieval; track over time
- **Effort:** 2 days
- **Files to create:** `evaluations/retrieval_metrics.py`

### Week 11-12: Skill Registry

#### Task 3.4: Implement Skill Registry
- **What:** YAML-based skill definitions for each capability (decompose, map, validate, format)
- **Effort:** 4 days
- **Files to create:** `skills/` directory, `utils/skill_registry.py`, `utils/skill_loader.py`

#### Task 3.5: Convert Existing Prompts to Skills
- **What:** Migrate prompts from static .txt files to versioned skill YAML bundles with eval suites
- **Effort:** 3 days
- **Risk:** Medium — must maintain quality during migration

#### Task 3.6: Dynamic Skill Loading
- **What:** Skills loaded at runtime based on intent + tenant config (not hardcoded in code)
- **Effort:** 2 days

### Week 13: Observability

#### Task 3.7: Full Observability Stack
- **What:** Per-stage latency, token usage, cost tracking, retrieval traces, decision traces
- **Effort:** 3 days
- **Tools:** Langfuse or Arize Phoenix (already in dependencies)
- **Files to create:** `utils/observability.py`, dashboard config

#### Task 3.8: Cost Tracking Dashboard
- **What:** Real-time cost per segment creation, cost per stage, cost trend over time
- **Effort:** 2 days

**Phase 3 Exit Criteria:**
- [ ] Per-stage evals running on every PR
- [ ] CI/CD eval gates blocking regressions
- [ ] Retrieval quality metrics tracked and baselined
- [ ] Skills registry operational with versioned bundles
- [ ] Full observability stack deployed
- [ ] Cost tracking dashboard live

---

## Phase 4: Multi-Tenant (Weeks 14-17)

**Hypothesis:** Multi-tenancy can be achieved through configuration isolation without code changes, if the architecture is properly parameterized.

### Week 14-15: Tenant Configuration System

#### Task 4.1: Tenant Config Manifest
- **What:** YAML-based tenant configuration defining all tenant-specific settings
- **Effort:** 3 days
- **Files to create:** `tenants/` directory structure, `utils/tenant_config.py`

#### Task 4.2: Tenant Context Loader
- **What:** Runtime loading of contextual information, hints, and channel mappings from tenant config
- **Effort:** 3 days
- **Files to create:** `utils/tenant_context_loader.py`
- **Files to modify:** All prompt loading code to use tenant context

#### Task 4.3: Migrate Current Tenant to Config
- **What:** Extract all hardcoded current-tenant settings into a config manifest
- **Effort:** 2 days
- **Risk:** Must not break existing functionality

### Week 16: Tenant Data Isolation

#### Task 4.4: Per-Tenant Milvus Collections
- **What:** Create tenant-scoped Milvus collections with tenant_id prefix
- **Effort:** 2 days
- **Decision:** Start with silo pattern (separate collections per tenant)

#### Task 4.5: Per-Tenant Ground Truth Store
- **What:** Each tenant has their own ground truth CSV and few-shot RAG index
- **Effort:** 2 days

#### Task 4.6: Per-Tenant Eval Suite
- **What:** Eval runs are scoped to tenant — no cross-tenant data mixing
- **Effort:** 1 day

### Week 17: Tenant Onboarding Automation

#### Task 4.7: Auto-Generate Contextual Info
- **What:** LLM generates contextual_information files from new tenant's facet catalog
- **Effort:** 3 days
- **Risk:** Medium — generated files need human review

#### Task 4.8: Vocabulary Synonym Generator
- **What:** Cross-reference new tenant's facet names with existing tenant using embedding similarity
- **Effort:** 2 days

#### Task 4.9: Tenant Onboarding Integration Test
- **What:** End-to-end test: provide facet catalog + 50 examples → system onboards tenant → run evals
- **Effort:** 2 days
- **Acceptance criteria:** New tenant quality ≥ 80% of primary tenant baseline

**Phase 4 Exit Criteria:**
- [ ] Tenant config manifest schema defined and validated
- [ ] Current tenant migrated to config-based loading
- [ ] Per-tenant Milvus collections operational
- [ ] Per-tenant ground truth and eval suites working
- [ ] Onboarding flow: 50 examples → running system in < 4 hours
- [ ] No code changes needed for new tenant

---

## Phase 5: Auto-Improve & Memory (Weeks 18-20)

**Hypothesis:** Continuous improvement through automated optimization and memory-augmented reasoning closes the quality gap over time without manual intervention.

### Week 18: Auto-Improvement Pipeline

#### Task 5.1: DSPy Prompt Optimization Setup
- **What:** Configure DSPy GEPA optimizer for facet selection and decomposition prompts
- **Effort:** 3 days
- **Risk:** Medium — needs sufficient ground truth data (≥ 50 examples)
- **Metric:** Prompt optimization should improve eval metrics by ≥ 5%
- **Files to create:** `optimization/dspy_optimizer.py`

#### Task 5.2: Feedback Collection
- **What:** Track user modifications to generated segments as implicit feedback
- **Effort:** 2 days
- **Files to create:** `utils/feedback_collector.py`

### Week 19: Memory System

#### Task 5.3: Short-Term Memory (Redis)
- **What:** Session state, conversation progression, current segment draft stored in Redis
- **Effort:** 3 days

#### Task 5.4: Long-Term Memory (PostgreSQL)
- **What:** User preferences, successful segment recipes, learned corrections stored in PostgreSQL
- **Effort:** 3 days

#### Task 5.5: Memory Integration
- **What:** Retrieve relevant memories at inference time; inject as context alongside few-shot examples
- **Effort:** 2 days

### Week 20: Finalization

#### Task 5.6: A/B Testing Framework
- **What:** Framework for testing prompt variations, model choices, and retrieval strategies per-tenant
- **Effort:** 3 days

#### Task 5.7: Full System Integration Test
- **What:** End-to-end test of all components: cascade retrieval, 3-stage pipeline, ground truth RAG, multi-tenant, memory, observability
- **Effort:** 2 days

#### Task 5.8: Documentation and Handoff
- **What:** Architecture docs, runbook, operational guides
- **Effort:** 2 days

**Phase 5 Exit Criteria:**
- [ ] DSPy optimization producing measurable quality improvements
- [ ] User feedback captured and feeding into ground truth expansion
- [ ] Memory system operational (short-term + long-term)
- [ ] A/B testing framework ready for ongoing experimentation
- [ ] Full system integration test passing

---

## Parallel Track Summary

```
Week:  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20
       ─────────  ──────────────  ───────────────  ──────────  ────────
       Phase 1    Phase 2         Phase 3          Phase 4     Phase 5

Track A (Retrieval):
       [GT RAG]   [Cascade Retriever  ] [Metrics    ] [Tenant   ] [Memory]
       [Hybrid]   [BM25+Trie+Taxonomy ] [Eval gates ] [Milvus   ]
       [Rerank]                                       [isolation]

Track B (Pipeline):
       [Code   ]  [Stage  ][Merge  ]   [Skill      ] [Config   ] [DSPy ]
       [stages ]  [Merge  ][Test   ]   [Registry   ] [Loader   ] [A/B  ]
                                       [Convert    ]

Track C (Quality):
       [Ground-]  [A/B against]       [CI/CD eval ] [Onboard  ] [Feed-]
       [ing    ]  [baseline   ]       [Per-stage  ] [automate ] [back ]
                                       [Observ     ]
```

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Pipeline merge degrades quality | Medium | High | A/B test against baseline; keep 7-stage as fallback |
| Cross-encoder adds latency | Low | Medium | Benchmark before deploy; async reranking |
| Ground truth too small for few-shot | Medium | Medium | Expand to 200+ segments in parallel |
| DSPy optimization overfits | Low | Medium | Hold-out validation set; monitor on new queries |
| New tenant quality below threshold | Medium | High | Cross-tenant transfer + progressive improvement |
| Milvus migration breaks existing | Low | High | Blue-green deployment; rollback plan |
| Skill migration introduces bugs | Medium | Medium | Per-skill eval gates; incremental migration |

---

## Success Metrics Summary

| Metric | Current (Estimated) | Phase 1 Target | Phase 2 Target | Phase 5 Target |
|--------|-------------------|----------------|----------------|----------------|
| End-to-end F1 | ~65% | ~75% | ~82% | ~88% |
| Facet Recall@5 | ~60-70% | ~80% | ~88% | ~92% |
| Latency (p50) | ~15s | ~10s | ~6s | ~5s |
| LLM cost/segment | $0.15 | $0.09 | $0.05 | $0.04 |
| LLM calls/segment | 8-15 | 5-8 | 2-4 | 2-3 |
| Schema validity | ~95% | 100% | 100% | 100% |
| Eval coverage | Manual | Per-stage | CI/CD gated | Auto-optimized |
| Multi-tenant | No | No | No | Yes (config-based) |
| Memory | Session only | Session only | Session only | Short + long-term |

---

## Dependencies Map

```
Phase 1 (Quick Wins)
├── No dependencies — all tasks can start immediately
└── Provides: Ground truth RAG, hybrid search, cross-encoder

Phase 2 (Pipeline Refactor)
├── Depends on: Phase 1 cross-encoder, ground truth RAG
├── Can partially overlap with Phase 1 Week 3
└── Provides: 3-stage pipeline, cascade retriever

Phase 3 (Eval & Skills)
├── Depends on: Phase 2 pipeline (eval targets the new pipeline)
├── Observability can start during Phase 2
└── Provides: Eval gates, skill registry, observability

Phase 4 (Multi-Tenant)
├── Depends on: Phase 2 cascade retriever, Phase 3 skill registry
├── Tenant config can start during Phase 3
└── Provides: Multi-tenant isolation, onboarding flow

Phase 5 (Auto-Improve & Memory)
├── Depends on: Phase 3 eval gates (optimization needs quality metrics)
├── Memory can start during Phase 4
└── Provides: Auto-improvement, memory, A/B testing
```

---

*See also: [01_bottleneck_analysis.md](01_bottleneck_analysis.md) for the issues this roadmap addresses, and [03_concrete_upgrade_proposal.md](03_concrete_upgrade_proposal.md) for the technical details of each component.*
