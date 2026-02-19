# Research 2: Opus RAG Extra — Enterprise Agentic Customer Segmentation

## Research Metadata
- **Research ID**: research_2_opus_rag_extra_claude
- **Model**: Claude Opus 4.6
- **Source Code Analyzed**: Smart-Segmentation (`/Users/s0m0ohl/customer_segement/Smart-Segmentation`)
- **GitHub**: [snehalvinit/enterprise-agentic-research](https://github.com/snehalvinit/enterprise-agentic-research)
- **Status**: Done — All deliverables produced
- **Date**: February 2026
- **Focus**: RAG alternatives for structured catalogs, pipeline stage optimization, ground truth as runtime context, multi-tenant architecture

## Objective

Deep research and concrete upgrade plan to transform Smart-Segmentation into an enterprise-grade agentic customer segmentation system, with particular focus on:
1. **Whether embedding-based RAG is the right choice** for a finite, structured facet catalog of ~500 items
2. **Whether 7 LLM pipeline stages are necessary** or can be collapsed to 2-3
3. **How to use ground truth data at runtime** as dynamic few-shot context (not just offline eval)
4. **How to onboard new tenants** without code changes

## Methodology

1. **Codebase Deep Dive** — Complete analysis of Smart-Segmentation: 23+ prompt files, Milvus integration, shortlist generation cascade, state management (57 variables), and all 7 pipeline stages
2. **Ground Truth Analysis** — Analyzed 46 labeled segment definitions identifying systematic failure patterns: 54.3% Strict/non-Strict confusion, 30.4% brand exclusion errors, 41.3% naming inconsistencies
3. **Web Research** — 90+ sources across 17 topic areas covering enterprise agent architectures, RAG alternatives, pipeline collapse patterns, multi-tenant isolation, and auto-improvement
4. **Cross-Reference** — Findings mapped against prior research context (skill systems, memory, eval-first patterns)
5. **Proposal + Roadmap** — Concrete 20-week phased upgrade plan with code examples and transformation demonstrations

## How to View

Open `index.html` in your browser for an elegant reading experience with:
- Dark/light theme toggle (bottom-right button)
- Left sidebar navigation with document status indicators
- Breadcrumb navigation
- Card-based home dashboard
- Automatic markdown rendering

## Deliverables

| # | Document | File | Status | Key Output |
|---|----------|------|--------|-----------|
| ~ | Index (this file) | [INDEX.md](INDEX.md) | Done | Navigation + overview |
| 01 | Bottleneck Analysis | [01_bottleneck_analysis.md](01_bottleneck_analysis.md) | Done | 35+ bottlenecks across 8 dimensions with focus on retrieval, pipeline, ground truth |
| 02 | Research Compendium | [02_research_compendium.md](02_research_compendium.md) | Done | 90+ sources across 17 topics with enterprise implementation focus |
| 03 | Concrete Upgrade Proposal | [03_concrete_upgrade_proposal.md](03_concrete_upgrade_proposal.md) | Done | Full architecture + cascade retrieval + pipeline refactor + code examples |
| 04 | Implementation Roadmap | [04_implementation_roadmap.md](04_implementation_roadmap.md) | Done | 20-week phased plan with parallel tracks |

## Key Findings Summary

### What's Broken (Top 10)

1. **Embedding search is wrong for structured catalogs** — 500 typed, named, hierarchical facets should use deterministic lookup first, embeddings as fallback only
2. **7-stage pipeline compounds errors** — at 90% per-stage accuracy, end-to-end is only 47.8%; collapsing to 3 stages yields 72.9%
3. **Ground truth is only used offline** — 46 validated segment definitions sit unused at runtime; dynamic few-shot injection improves accuracy 15-25%
4. **Hybrid search exists but isn't enabled** — RRF ranker is implemented in milvus.py but `search_mode="standard"` is the default
5. **Contextual info is hardcoded per tenant** — 5 static files loaded at import time; new tenant requires code changes
6. **No eval gates in CI/CD** — prompt changes ship without quality checks
7. **Date tagger and formatter use LLM unnecessarily** — both are deterministic tasks that code handles better
8. **Semantic drift across stages** — each stage rephrases user intent; by stage 5, original meaning may be altered
9. **No memory or auto-improvement** — system doesn't learn from past interactions
10. **eval() used for env var parsing** — security vulnerability throughout shortlist generation

### What the Upgrade Delivers

| Capability | Before | After |
|-----------|--------|-------|
| Pipeline stages | 7 LLM calls | 3 stages (1-2 LLM + code) |
| P50 Latency | ~15-20s | ~5-8s |
| LLM cost/segment | ~$0.15 | ~$0.04 (70% reduction) |
| Facet retrieval precision | ~60-70% | ~85-90% |
| End-to-end F1 | ~65% | ~85%+ |
| Multi-tenant | None | Config-driven, zero-code onboarding |
| Memory | Session-only | Short-term (Redis) + Long-term (PostgreSQL) |
| Auto-improvement | Manual | DSPy-driven optimization cycles |
| Eval coverage | Manual, post-hoc | CI/CD gated, per-stage |

### The Three Root Causes

1. **"Embeddings where SQL belongs"** — Using probabilistic retrieval for a deterministic, structured catalog. Fix with cascade retrieval: exact match → BM25 → type filter → embedding fallback
2. **"Too many stages, too little verification"** — 7 LLM calls with no quality checks between them. Fix by collapsing to 3 stages with explicit verification
3. **"Built for one tenant, hardcoded everywhere"** — All contextual information, hints, and catalogs are tenant-specific but loaded statically. Fix with tenant config manifests and runtime context loading

## Document Descriptions

### 01 — Bottleneck Analysis
35+ issues organized across 8 dimensions:
- Facet Retrieval Bottlenecks (6 issues) — embedding misfit, hybrid disabled, BGE vs MiniLM unvalidated, cascade order fragile
- Pipeline Stage Overengineering (5 issues) — 7-stage error accumulation, deterministic stages using LLM, missing verification
- Ground Truth Gaps (4 issues) — not used at runtime, systematic failure patterns, small dataset
- Multi-Tenant Risks (5 issues) — hardcoded contextual info, tenant-coupled catalogs, no config manifest
- Architecture (4 issues) — sequential pipeline, single point of failure, state explosion
- Prompt Design (4 issues) — semantic drift, stale examples, verbose prompts, no grounding
- Evaluation (3 issues) — no CI/CD gates, no per-stage eval, no retrieval metrics
- Missing Capabilities (5 issues) — no memory, no auto-improvement, limited observability

### 02 — Research Compendium
90+ sources across 17 topic areas:
- RAG vs no-RAG decision framework (ICLR 2025, Microsoft)
- Structured catalog retrieval alternatives (cascade, NER, taxonomy, tools)
- Hybrid BM25+dense retrieval (BEIR, Pinecone, SIGIR)
- Cross-encoder reranking (Cohere, MTEB)
- Ground truth as runtime few-shot (DSPy, Anthropic)
- Self-RAG, CRAG, FLARE patterns (AAAI, ICLR, EMNLP)
- Pipeline stage collapse and error accumulation
- Type-aware tool calling (OpenAI, Anthropic)
- Multi-tenant RAG isolation (Pinecone, Weaviate, Qdrant)
- Knowledge graph augmentation (FalkorDB)
- Enterprise agent architectures (Salesforce, Google ADK, CRMArena)
- Marketing segmentation implementations (Walmart, Adobe, HubSpot)
- Auto-improvement (DSPy GEPA, Mem0)
- Observability (Braintrust, Databricks Mosaic)
- Cost optimization (semantic caching, model routing, prompt caching)

### 03 — Concrete Upgrade Proposal
Detailed reformation plan including:
- Target architecture diagram (Perceive → Reason & Map → Validate → Format)
- Cascade retrieval implementation with code examples
- Pipeline 7→3 stage refactor with before/after demonstrations
- Ground truth few-shot RAG implementation
- Static prompt + dynamic skill architecture with YAML skill bundles
- Multi-tenant architecture with onboarding flow (50 examples → production in 4 hours)
- Memory system design (Redis short-term + PostgreSQL long-term)
- Eval-first infrastructure with CI/CD gates
- Auto-improvement pipeline with DSPy
- 3 concrete transformation examples showing before/after pipeline execution
- Cost optimization strategy (combined 70% reduction)

### 04 — Implementation Roadmap
20-week phased plan across 5 phases:
- **Phase 1 (Weeks 1-3):** Quick Wins — Ground truth RAG, deterministic stages, hybrid search, cross-encoder, grounding, prompt caching
- **Phase 2 (Weeks 4-8):** Pipeline Refactor — Cascade retriever, taxonomy graph, stage merge, A/B testing
- **Phase 3 (Weeks 9-13):** Eval & Skills — Per-stage eval, CI/CD gates, skill registry, observability
- **Phase 4 (Weeks 14-17):** Multi-Tenant — Config manifest, context loader, isolation, onboarding automation
- **Phase 5 (Weeks 18-20):** Auto-Improve & Memory — DSPy optimization, memory system, feedback loops

## Reference
- Research prompt: provided inline (user message)
- Hub page: [../index.html](../index.html)
- Previous research: [research_1_sonnet_claude](../research_1_sonnet_claude/index.html)
- Reference project: ditto_delivery
- GitHub Pages: [snehalvinit.github.io/enterprise-agentic-research](https://snehalvinit.github.io/enterprise-agentic-research)
