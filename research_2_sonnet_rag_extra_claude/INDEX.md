# Enterprise Agentic Research — Index & Overview
## Research 2: Sonnet Claude — RAG Extra (Comprehensive Deep Dive)

**Research ID:** `research_2_sonnet_rag_extra_claude`
**Model:** Claude Sonnet 4.6
**Date:** February 2026
**Source Codebase:** `/Users/s0m0ohl/customer_segement/Smart-Segmentation`

---

## Mission

Deeply analyze the Smart-Segmentation codebase and produce a concrete, actionable upgrade plan to transform it into an enterprise-grade agentic customer segmentation system. The system must be reliable, scalable, auto-improving, and deliver state-of-the-art quality while remaining cost-conscious.

---

## Research Methodology

1. **Codebase deep-read:** All Python source files, all prompts, all contextual information files, evaluation framework
2. **Ground truth data analysis:** Full statistical analysis of the 46-row ground truth CSV
3. **Web research:** Live search across academic papers, industry blogs, engineering posts, framework documentation
4. **Cross-reference:** Findings grounded against real implementations (Adobe CDP, Salesforce Einstein, Walmart Global Tech)
5. **Synthesis:** Bottlenecks ranked by severity; upgrade proposals ranked by impact/effort ratio

---

## Key Discoveries

### Critical Finding 1: Hybrid Search is a Dead Code Path
The `_match_single_instance_hybrid_search` method in `milvus.py` is fully implemented with RRFRanker support but **never called** due to a validation typo (`"hybird"` instead of `"hybrid"`). Fixing this 1-character bug immediately enables hybrid retrieval.

### Critical Finding 2: Ground Truth has 46 Rows (Not 18,779)
The ground truth CSV file contains **46 evaluation rows** — statistically insufficient for reliable eval (±15% margin at 95% CI). The system cannot reliably detect quality improvements until this grows to ≥200 rows.

### Critical Finding 3: 47.8% Correction Rate in Ground Truth
22 of 46 ground truth rows have non-empty Remarks indicating manual corrections were needed. This is the empirical evidence of the system's quality ceiling under the current architecture.

### Critical Finding 4: 7+ LLM Stages with No Verify Step
The pipeline has 7+ sequential LLM calls with no verification between stages. If each stage is 90% accurate, the combined pipeline accuracy is ≈0.9^7 = 48%. Adding code-based verify steps between stages dramatically improves overall accuracy.

### Critical Finding 5: Static Tenant Configuration
Four critical contextual information files (`refinements.txt`, `catalog_view_description.txt`, `decomposer_hints.txt`, `fvom_hints.txt`) are loaded statically at agent startup and hardcoded to the current tenant (Walmart). Multi-tenant onboarding requires runtime tenant config loading.

---

## Document Status

| # | Document | Status | Key Content |
|---|---|---|---|
| INDEX | This file | ✅ Done | Overview and findings |
| 01 | Bottleneck Analysis | ✅ Done | 18 bottlenecks, severity matrix, code references |
| 02 | Research Compendium | ✅ Done | 20+ sources, RAG framework, enterprise patterns |
| 03 | Concrete Upgrade Proposal | ✅ Done | Cascade retrieval, 4-stage pipeline, tenant manifest |
| 04 | Implementation Roadmap | ✅ Done | 5 phases, parallel tracks, risk registry |

---

## Architecture Transformation Summary

```
CURRENT: 7+ LLM stages, 1 tenant, dense-only Milvus, static context files
   ↓
TARGET:  4 LLM stages, N tenants, cascade retrieval, runtime tenant config, ground truth RAG
```

**Phase 0 (Week 1-2):** Fix hybrid search typo, establish eval baseline, enable full tracing
**Phase 1 (Week 3-6):** Cascade retrieval (Exact → Alias → BM25 → Dense → LLM Rerank)
**Phase 2 (Week 5-8):** Pipeline compression (7+ stages → 4 stages with verify steps)
**Phase 3 (Week 7-10):** Ground truth as runtime RAG (few-shot context injection)
**Phase 4 (Week 8-12):** Multi-tenant architecture (tenant config manifest + runtime loader)
**Phase 5 (Week 12-16):** Auto-improvement (feedback loops, prompt optimization, DSPy)

---

## Estimated Impact

| Dimension | Current State | Target State |
|---|---|---|
| Facet recall | ~60% (estimated from 47.8% correction rate) | >80% |
| End-to-end latency | ~5-8 seconds | ~2-4 seconds |
| Per-request LLM cost | ~$0.053 (8 calls) | ~$0.024 (3.4 calls avg) |
| Tenant onboarding | Days-weeks (code changes) | <4 hours (config only) |
| Quality regression detection | Manual only | 100% automated |

---

## Files in This Research

```
research_2_sonnet_rag_extra_claude/
├── INDEX.md                          ← This file
├── 01_bottleneck_analysis.md         ← All 18 bottlenecks with code references
├── 02_research_compendium.md         ← State-of-the-art research with citations
├── 03_concrete_upgrade_proposal.md   ← Detailed upgrade plan with code examples
├── 04_implementation_roadmap.md      ← Phased roadmap with parallel tracks
└── index.html                        ← Interactive viewer (dark/light theme)
```

---

*Part of the Enterprise Agentic Research hub. View all research at [../index.html](../index.html)*
