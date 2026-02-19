# 03 — Concrete Upgrade Proposal: Smart-Segmentation → Enterprise-Grade Agentic System

> **Research ID:** research_2_opus_rag_extra_claude
> **Model:** Claude Opus 4.6
> **Date:** February 2026
> **Focus:** Facet Retrieval Overhaul, Pipeline Refactor, Ground Truth RAG, Multi-Tenant Architecture

---

## Executive Summary

This proposal transforms Smart-Segmentation from a 7-stage sequential LLM pipeline with embedding-only retrieval into a **3-stage agentic system** with cascade retrieval, dynamic few-shot context, and multi-tenant isolation. The upgrade delivers:

| Metric | Current | Proposed | Improvement |
|--------|---------|----------|------------|
| Pipeline stages | 7 LLM calls | 3 stages (1 LLM + 2 code/hybrid) | -57% stages |
| End-to-end latency | ~15-20s | ~5-8s | ~3x faster |
| LLM cost per segment | Baseline | ~30% of baseline | 70% reduction |
| Facet retrieval precision | ~60-70% (estimated) | ~85-90% | +20-30pp |
| Multi-tenant support | None | Config-driven isolation | Enterprise-ready |
| Memory | Session-only | Short-term + long-term | Cross-session learning |
| Auto-improvement | Manual prompt editing | DSPy-driven optimization | Continuous |

---

## Table of Contents

1. [Target Architecture Overview](#1-target-architecture-overview)
2. [Facet Retrieval Upgrade](#2-facet-retrieval-upgrade)
3. [Pipeline Stage Refactor](#3-pipeline-stage-refactor)
4. [Ground Truth as Runtime Few-Shot RAG](#4-ground-truth-as-runtime-few-shot-rag)
5. [Static Prompt + Dynamic Skill Architecture](#5-static-prompt--dynamic-skill-architecture)
6. [Multi-Tenant Architecture](#6-multi-tenant-architecture)
7. [Memory System](#7-memory-system)
8. [Evaluation-First Infrastructure](#8-evaluation-first-infrastructure)
9. [Auto-Improvement Pipeline](#9-auto-improvement-pipeline)
10. [Enterprise Domain Patterns](#10-enterprise-domain-patterns)
11. [Cost Optimization Strategy](#11-cost-optimization-strategy)
12. [Concrete Transformation Examples](#12-concrete-transformation-examples)

---

## 1. Target Architecture Overview

### 1.1 High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     ENTERPRISE AGENT LAYER                       │
│                                                                  │
│  ┌─────────┐   ┌───────────────────┐   ┌──────────────────────┐ │
│  │ PERCEIVE│──▶│  REASON & RESOLVE │──▶│  FORMAT & VALIDATE   │ │
│  │         │   │                   │   │                      │ │
│  │ • Route │   │ • Decompose       │   │ • JSON transform     │ │
│  │ • NER   │   │ • Map facets      │   │ • Schema validate    │ │
│  │ • Intent│   │ • Extract dates   │   │ • Business rules     │ │
│  │         │   │ • Resolve deps    │   │ • Grounding check    │ │
│  └────┬────┘   └────────┬──────────┘   └──────────┬───────────┘ │
│       │                 │                          │             │
│  ┌────▼─────────────────▼──────────────────────────▼───────────┐ │
│  │                   RETRIEVAL LAYER                            │ │
│  │                                                              │ │
│  │  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │ │
│  │  │ Structured  │  │  Embedding   │  │  Ground Truth     │  │ │
│  │  │ Lookup      │  │  Fallback    │  │  Few-Shot RAG     │  │ │
│  │  │             │  │              │  │                   │  │ │
│  │  │ • BM25/Trie │  │ • Milvus    │  │ • Similar history │  │ │
│  │  │ • Type filt │  │ • BGE/MiniLM│  │ • Validated pairs │  │ │
│  │  │ • NER match │  │ • Cross-enc │  │ • Edge case notes │  │ │
│  │  │ • Taxonomy  │  │   rerank    │  │                   │  │ │
│  │  └─────────────┘  └──────────────┘  └───────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │                   TENANT LAYER                               ││
│  │  ┌─────────┐  ┌─────────────┐  ┌──────────┐  ┌───────────┐ ││
│  │  │ Config  │  │ Contextual  │  │ Facet    │  │ Ground    │ ││
│  │  │ Manifest│  │ Info Store  │  │ Catalog  │  │ Truth DB  │ ││
│  │  └─────────┘  └─────────────┘  └──────────┘  └───────────┘ ││
│  └──────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │                   FEEDBACK LAYER                             ││
│  │  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐ ││
│  │  │ Memory  │  │ Eval     │  │ Auto-    │  │ Observ-     │ ││
│  │  │ System  │  │ Gates    │  │ Improve  │  │ ability     │ ││
│  │  └─────────┘  └──────────┘  └──────────┘  └─────────────┘ ││
│  └──────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 Core Design Principles

1. **Structured first, embeddings as fallback** — don't use probabilistic retrieval where deterministic lookup works
2. **Fewer stages, each with verification** — collapse 7 stages to 3; add explicit verification at each
3. **Ground truth as active context** — use validated examples at runtime, not just for offline eval
4. **Tenant config, not tenant code** — all tenant-specific content loads from config, never from code
5. **Static prompt contract + dynamic skills** — agent identity and rules are fixed; capabilities are pluggable
6. **Eval gates everywhere** — no change ships without passing automated quality checks

---

## 2. Facet Retrieval Upgrade

### 2.1 Decision Matrix: Retrieval Approaches

| Approach | Precision for Exact Names | Semantic Synonym Handling | Scales to 1500 Facets | Latency | Cost |
|---|---|---|---|---|---|
| Embedding search (current) | Medium (60-70%) | Good | Yes | 50-200ms | Low |
| BM25/keyword search | High (85-90%) | Poor | Yes | 10-30ms | Very low |
| Structured SQL/filter | Very high (95%+) | None | Yes | 5-15ms | Very low |
| Full-catalog in context | Very high (90%+) | Excellent | No (>200K token limit) | 2-5s | High |
| **Cascade (recommended)** | **Very high (90%+)** | **Good** | **Yes** | **100-500ms** | **Low-Medium** |
| Type-aware tools | High (85-90%) | Good | Yes | 200-1000ms | Medium |

### 2.2 Recommended: Cascade Retrieval Architecture

**Before (current):**
```
User Query → Embedding → Milvus Dense Search → Top-K Facets → LLM Classifier
```

**After (proposed):**
```
User Query
    │
    ▼
┌─────────────────────────────────┐
│  STAGE 1: NER + Type Detection  │  (Code, ~10ms)
│  • Extract entities (brands,    │
│    departments, channels)       │
│  • Detect query type (date,     │
│    propensity, persona, etc.)   │
│  • Exact name match against     │
│    facet catalog                │
└────────────┬────────────────────┘
             │
    ┌────────▼────────┐
    │ Exact match?    │
    │ (fuzzy ratio    │
    │  > 85%)         │
    └───┬─────────┬───┘
       YES       NO
        │         │
        ▼         ▼
  ┌──────────┐ ┌──────────────────────┐
  │ Direct   │ │ STAGE 2: Structured  │  (Code, ~30ms)
  │ return   │ │ + Embedding Search   │
  │          │ │ • BM25 on facet names│
  └──────────┘ │ • Type filter (date, │
               │   numeric, string)   │
               │ • Restriction filter │
               │ • If low confidence: │
               │   → embedding search │
               │   → RRF fusion       │
               └──────────┬───────────┘
                          │
                          ▼
               ┌──────────────────────┐
               │ STAGE 3: Cross-      │  (~200ms)
               │ Encoder Rerank       │
               │ • Score top-50       │
               │   candidates against │
               │   query              │
               │ • Return top-5       │
               └──────────┬───────────┘
                          │
                          ▼
               ┌──────────────────────┐
               │ STAGE 4: LLM Select  │  (~2s, only if needed)
               │ • Facet classifier   │
               │   with full metadata │
               │ • Few-shot from      │
               │   ground truth       │
               │ • Grounding enforced │
               └──────────────────────┘
```

### 2.3 Implementation: Cascade Retriever

```python
class CascadeRetriever:
    """
    Multi-stage retrieval for structured facet catalogs.
    Prioritizes deterministic lookup over probabilistic search.
    """

    def __init__(self, tenant_config: TenantConfig):
        self.catalog = load_catalog(tenant_config.facet_catalog_source)
        self.bm25_index = build_bm25_index(self.catalog)
        self.trie = build_trie(self.catalog["name"].tolist())
        self.milvus = MilvusDB(tenant_config.milvus_collection)
        self.cross_encoder = CrossEncoder("BAAI/bge-reranker-large")
        self.ground_truth_rag = GroundTruthRAG(tenant_config.ground_truth_csv)

    def retrieve(self, query: str, sub_segment: str,
                 query_type: str, restrictions: list) -> list[FacetCandidate]:

        # Stage 1: Exact / fuzzy name match
        exact_matches = self._exact_match(sub_segment, restrictions)
        if exact_matches and exact_matches[0].confidence > 0.85:
            return exact_matches[:5]

        # Stage 2: Structured + embedding search
        bm25_results = self._bm25_search(sub_segment, k=25)
        type_filtered = self._type_filter(bm25_results, query_type)

        if len(type_filtered) < 5:
            # Fallback to embedding for semantic coverage
            emb_results = self._embedding_search(sub_segment, k=25, restrictions=restrictions)
            candidates = self._rrf_fusion(type_filtered, emb_results, k=50)
        else:
            candidates = type_filtered

        # Stage 3: Cross-encoder rerank
        reranked = self._cross_encoder_rerank(sub_segment, candidates, k=5)

        return reranked

    def _exact_match(self, query: str, restrictions: list) -> list[FacetCandidate]:
        """Trie-based exact prefix match + fuzzy matching."""
        trie_matches = self.trie.search(query.lower())
        fuzzy_matches = [
            FacetCandidate(name=name, confidence=fuzz.ratio(query.lower(), name.lower()) / 100)
            for name in self.catalog["name"]
            if fuzz.ratio(query.lower(), name.lower()) > 70
        ]
        return sorted(trie_matches + fuzzy_matches, key=lambda x: x.confidence, reverse=True)

    def _bm25_search(self, query: str, k: int) -> list[FacetCandidate]:
        """BM25 keyword search on facet names + descriptions."""
        scores = self.bm25_index.get_scores(query.split())
        top_k_indices = scores.argsort()[-k:][::-1]
        return [FacetCandidate(name=self.catalog.iloc[i]["name"],
                               confidence=scores[i])
                for i in top_k_indices if scores[i] > 0]

    def _type_filter(self, candidates: list, query_type: str) -> list[FacetCandidate]:
        """Filter candidates by detected query type."""
        type_map = {
            "date": ["date", "datetime"],
            "numeric": ["numeric", "integer", "float"],
            "propensity": ["csv"],  # propensity facets are csv type
            "engagement": ["string"],
        }
        if query_type in type_map:
            allowed_types = type_map[query_type]
            return [c for c in candidates
                    if self.catalog[self.catalog["name"] == c.name]["type"].iloc[0] in allowed_types]
        return candidates

    def _cross_encoder_rerank(self, query: str, candidates: list, k: int) -> list[FacetCandidate]:
        """Cross-encoder reranking for final precision."""
        pairs = [(query, c.name + " " + self.catalog[self.catalog["name"] == c.name]["description"].iloc[0])
                 for c in candidates]
        scores = self.cross_encoder.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [c for c, s in ranked[:k]]
```

### 2.4 Facet Taxonomy Graph

```python
class FacetTaxonomy:
    """
    Lightweight graph encoding facet relationships.
    Enables deterministic traversal instead of probabilistic search.
    """

    def __init__(self, catalog: pd.DataFrame):
        self.graph = self._build_graph(catalog)

    def _build_graph(self, catalog: pd.DataFrame) -> dict:
        """Build adjacency list from facet metadata."""
        graph = defaultdict(lambda: {"children": [], "linked": [], "category": "", "type": ""})

        # Propensity hierarchy
        for _, row in catalog.iterrows():
            name = row["name"]
            graph[name]["type"] = row["type"]
            graph[name]["category"] = self._infer_category(name)

            # Parent-child relationships
            if "Division" in name and "Super Department" not in name:
                parent = name.replace("Division", "Super Department")
                if parent in graph:
                    graph[parent]["children"].append(name)
            if "Brand" in name and "Division" not in name:
                parent = name.replace("Brand", "Division")
                if parent in graph:
                    graph[parent]["children"].append(name)

        return dict(graph)

    def traverse(self, entry_point: str, depth: int = 2) -> list[str]:
        """Return all related facets within depth hops."""
        visited = set()
        queue = [(entry_point, 0)]
        result = []

        while queue:
            node, d = queue.pop(0)
            if node in visited or d > depth:
                continue
            visited.add(node)
            result.append(node)

            for child in self.graph.get(node, {}).get("children", []):
                queue.append((child, d + 1))
            for linked in self.graph.get(node, {}).get("linked", []):
                queue.append((linked, d + 1))

        return result
```

---

## 3. Pipeline Stage Refactor

### 3.1 Current vs Proposed Pipeline

**Current (7 stages, 8-15 LLM calls):**
```
Route → Decompose → Date Tag → Facet Map → Classify → Resolve → Format
  ↓         ↓          ↓          ↓          ↓         ↓         ↓
 LLM       LLM        LLM        LLM       LLM      LLM       LLM
```

**Proposed (3 stages + 1 code step, 2-4 LLM calls):**
```
┌──────────┐    ┌────────────────────────┐    ┌───────────────┐    ┌──────────┐
│ PERCEIVE │    │ REASON & MAP           │    │ VALIDATE &    │    │ FORMAT   │
│          │    │                        │    │ RESOLVE       │    │          │
│ • Intent │───▶│ • Decompose query      │───▶│ • Verify      │───▶│ • JSON   │
│   classify│   │ • Extract dates (code) │    │   completeness│    │   transform│
│ • NER    │   │ • Retrieve facets       │    │ • Resolve     │    │ • Schema │
│ • Route  │   │   (cascade retriever)  │    │   dependencies│    │   validate│
│          │   │ • Map facet-value pairs │    │ • Ambiguity   │    │ • Business│
│ Code/    │   │ • Few-shot from GT     │    │   questions   │    │   rules  │
│ Classifier│   │                        │    │               │    │          │
│ ~10ms    │   │ 1 LLM call, ~3-5s      │    │ 1-3 LLM calls │    │ Code     │
└──────────┘   └────────────────────────┘    │ ~2-4s          │    │ ~5ms     │
                                              └───────────────┘    └──────────┘
```

### 3.2 Stage 1: PERCEIVE (Code-Only, ~10ms)

Replaces: Route Agent (Stage 1)

```python
class PerceptionStage:
    """
    Deterministic intent classification and entity extraction.
    No LLM needed — rule-based + small classifier.
    """

    def __init__(self, tenant_config: TenantConfig):
        self.ner = NEREngine(tenant_config.facet_catalog)
        self.classifier = IntentClassifier()  # Fine-tuned BERT-tiny or rules

    def perceive(self, query: str, session_state: SessionState) -> PerceptionResult:
        # Classify intent
        intent = self.classifier.classify(query)
        # Returns: GREETING | UNINTELLIGIBLE | OUT_OF_SCOPE | CREATE_SEGMENT |
        #          EDIT_SEGMENT | FACET_INFO

        if intent in ("GREETING", "UNINTELLIGIBLE", "OUT_OF_SCOPE"):
            return PerceptionResult(intent=intent, response=self._canned_response(intent))

        if intent == "FACET_INFO":
            return PerceptionResult(intent=intent, response=self._facet_info(query))

        # Extract entities
        entities = self.ner.extract(query)

        # Detect query type
        query_type = self._detect_query_type(query, entities)

        # Check if editing existing segment
        if session_state.segment_exists:
            intent = "EDIT_SEGMENT"

        return PerceptionResult(
            intent=intent,
            entities=entities,
            query_type=query_type,
            original_query=query
        )
```

### 3.3 Stage 2: REASON & MAP (1 LLM Call, ~3-5s)

Replaces: Decomposer (Stage 2) + Date Tagger (Stage 3) + Facet Mapper (Stage 4)

```python
class ReasonAndMapStage:
    """
    Single LLM call that decomposes, extracts dates, and maps facets.
    Uses cascade retrieval + ground truth few-shot context.
    """

    SYSTEM_PROMPT = """
    You are a customer segmentation expert. Given a user query, you must:
    1. Decompose it into logical sub-segments
    2. Identify date/time constraints in each sub-segment
    3. Map each sub-segment to the most appropriate facets from the provided candidates

    RULES:
    - Use ONLY facets from the provided candidates list
    - Preserve the original user intent exactly — do not paraphrase
    - If a date is mentioned, use the date facet candidates provided
    - If unsure about a facet, mark it as "uncertain" — do NOT guess

    OUTPUT FORMAT (JSON):
    {
      "ruleSet": {"INCLUDE": "(Seg-1 AND Seg-2)", "EXCLUDE": ""},
      "subSegments": {
        "Seg-1": {
          "description": "original text from user query",
          "date_constraint": {"type": "relative", "value": "last 30 days"} or null,
          "facets": [
            {"name": "Propensity Super Department", "value": "APPAREL", "operator": "IN",
             "confidence": "high", "reasoning": "user mentioned 'clothing'"}
          ]
        }
      }
    }
    """

    def __init__(self, tenant_config: TenantConfig):
        self.retriever = CascadeRetriever(tenant_config)
        self.ground_truth = GroundTruthRAG(tenant_config.ground_truth_csv)
        self.date_parser = DateParser()  # Rule-based, not LLM

    async def reason_and_map(self, perception: PerceptionResult,
                              tenant_config: TenantConfig) -> ReasonResult:
        # 1. Parse dates deterministically (code, ~5ms)
        date_info = self.date_parser.extract_dates(perception.original_query)

        # 2. Retrieve candidate facets via cascade (code, ~100-500ms)
        candidates = self.retriever.retrieve(
            query=perception.original_query,
            sub_segment=perception.original_query,
            query_type=perception.query_type,
            restrictions=tenant_config.restrictions
        )

        # 3. Get ground truth few-shot examples (code, ~50ms)
        few_shot = self.ground_truth.get_few_shot_examples(
            perception.original_query, k=3
        )

        # 4. Single LLM call with all context (~3-5s)
        prompt = self._build_prompt(
            system=self.SYSTEM_PROMPT,
            query=perception.original_query,
            entities=perception.entities,
            date_info=date_info,
            candidate_facets=candidates,
            few_shot_examples=few_shot,
            tenant_hints=tenant_config.hints
        )

        result = await llm_call(prompt, structured_output=ReasonResult)
        return result
```

### 3.4 Stage 3: VALIDATE & RESOLVE (1-3 LLM Calls, ~2-4s)

Replaces: Classifier (Stage 5a) + Linked Facet (Stage 5b) + Ambiguity (Stage 5c)

```python
class ValidateAndResolveStage:
    """
    Verify completeness, resolve dependencies, handle ambiguity.
    May require user interaction for clarification.
    """

    def validate_and_resolve(self, reason_result: ReasonResult,
                              perception: PerceptionResult) -> ValidatedResult:
        # 1. Completeness check (code)
        missing = self._check_completeness(
            reason_result, perception.original_query
        )
        if missing:
            # Flag missing facets for LLM re-evaluation
            reason_result = self._fill_gaps(reason_result, missing)

        # 2. Dependency resolution (code + optional LLM)
        dependencies = self._identify_dependencies(reason_result)
        if dependencies.has_classifier_deps:
            resolved = await self._resolve_classifier(
                dependencies.classifier_deps,
                reason_result
            )
            reason_result = self._merge_resolved(reason_result, resolved)

        # 3. Ambiguity detection (code)
        ambiguities = self._detect_ambiguities(reason_result)
        if ambiguities:
            return ValidatedResult(
                needs_clarification=True,
                question=self._generate_question(ambiguities)
            )

        # 4. Business rule validation (code)
        violations = self._check_business_rules(reason_result)
        if violations:
            reason_result = self._fix_violations(reason_result, violations)

        return ValidatedResult(
            needs_clarification=False,
            facet_value_pairs=reason_result.facets,
            rule_set=reason_result.ruleSet
        )
```

### 3.5 Stage 4: FORMAT (Pure Code, ~5ms)

Replaces: Formatter (Stage 6)

```python
class FormatStage:
    """
    Deterministic JSON transformation. No LLM needed.
    Converts validated facet-value pairs to SegmentR format.
    """

    def format_segmentr(self, validated: ValidatedResult) -> dict:
        segment_definition = {
            "segmentDefinition": {
                "ruleSet": validated.rule_set,
                "subSegments": {}
            }
        }

        for seg_id, seg_data in validated.facet_value_pairs.items():
            segment_definition["segmentDefinition"]["subSegments"][seg_id] = {
                "facets": [
                    {
                        "name": facet.name,
                        "operator": facet.operator,
                        "value": facet.value,
                        "refinement": facet.refinement
                    }
                    for facet in seg_data.facets
                ]
            }

        # Schema validation
        validate_schema(segment_definition, SEGMENTR_SCHEMA)

        return segment_definition
```

### 3.6 Skill Bundle Examples

Each pipeline capability becomes a versioned skill:

**Skill: segment_decompose (v1.2)**
```yaml
name: segment_decompose
version: "1.2"
description: "Decompose a natural language segment query into logical sub-segments"
triggers:
  - intent: CREATE_SEGMENT
  - contains: ["segment", "audience", "customers who"]

input_schema:
  type: object
  properties:
    user_query: {type: string}
    conversation_history: {type: array}

output_schema:
  type: object
  properties:
    ruleSet: {type: object, properties: {INCLUDE: {type: string}, EXCLUDE: {type: string}}}
    subSegments: {type: object, additionalProperties: {type: string}}

instructions: |
  Break the user query into independent sub-segments where each represents
  a distinct customer attribute or condition. Rules:
  1. Each sub-segment must be self-contained (no cross-references)
  2. Negations (haven't, not, no) stay in INCLUDE; only explicit "exclude"/"excluding" goes in EXCLUDE
  3. Preserve exact user phrases — do not paraphrase
  4. Use Seg-N format strictly (Seg-1, Seg-2, ...)

constraints:
  - No duplicate sub-segments
  - Every Seg-N in ruleSet must exist in subSegments and vice versa
  - EXCLUDE must be empty string if no explicit exclude/excluding keyword

eval_suite: "evals/decomposer/test_cases.yaml"
```

**Skill: map_facets (v2.1)**
```yaml
name: map_facets
version: "2.1"
description: "Map sub-segment descriptions to facets from the tenant catalog"
triggers:
  - after: segment_decompose

input_schema:
  type: object
  properties:
    sub_segments: {type: object}
    candidate_facets: {type: array, items: {type: object}}
    date_constraints: {type: object}
    few_shot_examples: {type: array}

output_schema:
  type: object
  properties:
    facet_mappings: {type: object}
    confidence: {type: number}
    uncertain_mappings: {type: array}

instructions: |
  For each sub-segment, select the most appropriate facets from the
  provided candidates. You MUST:
  1. Only select from candidate_facets — never invent facet names
  2. Include reasoning for each selection
  3. Mark uncertain mappings with confidence < 0.7
  4. Consider the few-shot examples for similar segments

constraints:
  - Grounding: only use provided facets
  - Date facets: use date_constraints, don't re-interpret
  - If no good match: return "uncertain" rather than guessing

eval_suite: "evals/facet_mapper/test_cases.yaml"
```

---

## 4. Ground Truth as Runtime Few-Shot RAG

### 4.1 Architecture

```
┌─────────────────────────────────────────────────┐
│           GROUND TRUTH FEW-SHOT RAG             │
│                                                  │
│  ┌──────────────┐    ┌────────────────────────┐ │
│  │ Ground Truth │    │   Embedding Index      │ │
│  │ CSV (46+ rows│───▶│   (FAISS / Milvus)     │ │
│  │ per tenant)  │    │                        │ │
│  └──────────────┘    │   Indexed fields:      │ │
│                      │   • Segment Description│ │
│                      │   • Expected Facets    │ │
│                      │   • Segment Type tag   │ │
│                      └────────────┬───────────┘ │
│                                   │              │
│  At inference time:               │              │
│  ┌─────────┐                      │              │
│  │ User    │──── similarity ──────┘              │
│  │ Query   │     search                          │
│  └─────────┘                                     │
│       │                                          │
│       ▼                                          │
│  ┌─────────────────────────────────────┐        │
│  │ Top-3 Similar Examples:             │        │
│  │                                     │        │
│  │ 1. "spring fashion shoppers..."     │        │
│  │    → Persona | Propensity SD | ...  │        │
│  │    → Note: "Removed ACCESSORIES"    │        │
│  │                                     │        │
│  │ 2. "new parents with baby reg..."   │        │
│  │    → CRM Email | Persona | ...      │        │
│  │                                     │        │
│  │ 3. "electronics buyers who..."      │        │
│  │    → Propensity Brand Strict | ...  │        │
│  └─────────────────────────────────────┘        │
│       │                                          │
│       ▼                                          │
│  Injected into LLM prompt as:                    │
│  "Reference examples from validated segments..." │
└─────────────────────────────────────────────────┘
```

### 4.2 Implementation

```python
class GroundTruthRAG:
    """
    Dynamic few-shot retrieval from validated ground truth examples.
    Each tenant has their own ground truth store.
    """

    def __init__(self, csv_path: str, embedding_model: str = "BAAI/bge-large-en-v1.5"):
        self.df = pd.read_csv(csv_path)
        self.encoder = SentenceTransformer(embedding_model)
        self._build_index()

    def _build_index(self):
        """Embed segment descriptions for similarity search."""
        descriptions = self.df["Updated Segment Description with Add-on"].fillna("").tolist()
        self.embeddings = self.encoder.encode(descriptions)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)

    def get_few_shot_examples(self, query: str, k: int = 3) -> list[dict]:
        """Retrieve k most similar ground truth examples."""
        query_emb = self.encoder.encode([query])
        faiss.normalize_L2(query_emb)
        distances, indices = self.index.search(query_emb, k)

        examples = []
        for i, idx in enumerate(indices[0]):
            row = self.df.iloc[idx]
            examples.append({
                "description": row["Updated Segment Description with Add-on"],
                "expected_facets": row["updated expected facets"],
                "remarks": row["Remarks on values"] if pd.notna(row["Remarks on values"]) else "",
                "similarity": float(distances[0][i])
            })
        return examples

    def format_for_prompt(self, examples: list[dict]) -> str:
        """Format examples for injection into LLM prompt."""
        lines = ["Here are validated segment examples similar to your query:\n"]
        for i, ex in enumerate(examples, 1):
            lines.append(f"Example {i}:")
            lines.append(f"  Description: \"{ex['description']}\"")
            lines.append(f"  Correct Facets: {ex['expected_facets']}")
            if ex['remarks']:
                lines.append(f"  Note: {ex['remarks']}")
            lines.append("")
        return "\n".join(lines)
```

### 4.3 Impact Estimation

Based on research findings (DSPy, dynamic few-shot studies):
- **Facet recall improvement:** +15-25% (relevant examples provide better guidance than static examples)
- **Strict/non-Strict confusion reduction:** -40-60% (examples explicitly show correct Strict usage)
- **Value accuracy improvement:** +10-15% (Remarks column provides edge case guidance)
- **Implementation effort:** 2-3 days (embed CSV, add similarity search, inject into prompts)

---

## 5. Static Prompt + Dynamic Skill Architecture

### 5.1 What Stays Static (The "Constitution")

```python
STATIC_SYSTEM_PROMPT = """
You are a Customer Segmentation Agent. Your role is to convert natural language
descriptions into precise, validated customer segment definitions.

OPERATING RULES (always follow):
1. PLAN → ACT → VERIFY → IMPROVE
   - Plan: Understand the user's intent before acting
   - Act: Use provided tools and retrieved context
   - Verify: Check your output against the user's original intent
   - Improve: If something is wrong, fix it before responding

2. GROUNDING: Use ONLY information from:
   - Retrieved facet candidates (provided by tools)
   - Ground truth examples (provided as context)
   - Tenant contextual information (provided as context)
   - NEVER invent facet names, values, or operators

3. OUTPUT: Return structured JSON matching the required schema exactly

4. UNCERTAINTY: If unsure about any facet mapping:
   - Mark it as "uncertain" in your output
   - The system will ask the user for clarification
   - Do NOT guess — uncertainty is better than a wrong answer

5. ESCALATION: If the query is ambiguous or missing critical information,
   ask a clarifying question before proceeding
"""
```

### 5.2 What Loads Dynamically

```python
# Loaded per-request based on intent, tenant, and context
dynamic_context = {
    "skill": skill_registry.load(intent),              # From skill registry
    "knowledge": ground_truth_rag.get_examples(query),  # Few-shot from GT
    "facets": cascade_retriever.retrieve(query),         # Retrieved candidates
    "tenant_hints": tenant_config.contextual_info,       # Tenant-specific
    "memory": memory_store.get_relevant(user_id, query), # User preferences
}
```

### 5.3 Skill Registry

```python
class SkillRegistry:
    """
    Versioned skill bundles loaded dynamically based on intent.
    Skills are YAML files with instructions, constraints, examples, and eval suites.
    """

    def __init__(self, skills_dir: str):
        self.skills = {}
        for skill_file in Path(skills_dir).glob("*.yaml"):
            skill = yaml.safe_load(skill_file.read_text())
            self.skills[skill["name"]] = skill

    def load(self, intent: str) -> dict:
        """Load relevant skills for the given intent."""
        relevant = [s for s in self.skills.values()
                    if any(t.get("intent") == intent for t in s.get("triggers", []))]
        return relevant

    def update_skill(self, name: str, new_version: dict):
        """Update skill version — must pass eval gate first."""
        eval_suite = new_version.get("eval_suite")
        if not eval_suite:
            raise ValueError("Skill must have eval_suite defined")

        # Run eval gate
        results = run_eval(eval_suite, new_version)
        if results.pass_rate < 0.95:
            raise ValueError(f"Skill failed eval gate: {results.pass_rate:.1%} < 95%")

        self.skills[name] = new_version
```

---

## 6. Multi-Tenant Architecture

### 6.1 Tenant Configuration Manifest

```yaml
# tenants/retailer_a/config.yaml
tenant_id: "retailer_a"
display_name: "Retailer A"
status: "active"

data_sources:
  facet_catalog:
    type: "bigquery"
    table: "project.retailer_a.facet_catalog"
    cache_format: "pickle"
    cache_path: "tenants/retailer_a/cache/facets.pkl"
  ground_truth:
    csv_path: "tenants/retailer_a/ground_truth.csv"
    min_examples: 50  # Minimum for quality parity

retrieval:
  milvus:
    collection_prefix: "RETAILER_A"
    facet_name_collection: "RETAILER_A_FACET_NAME"
    facet_value_collection: "RETAILER_A_FACET_VALUE"
    partition_strategy: "silo"  # "silo" | "pool" | "bridge"
  embedding_model: "BAAI/bge-large-en-v1.5"
  search_mode: "hybrid"  # "standard" | "hybrid"
  enable_cross_encoder: true

contextual_info:
  base_dir: "tenants/retailer_a/contextual_information/"
  files:
    refinements: "contextual_information_on_refinements.txt"
    catalog_description: "catalog_view_description.txt"
    decomposer_hints: "segment_decomposer_hints.txt"
    mapper_hints: "facet_value_mapper_hints.txt"
    channel_map: "channel_date_attribute_map.json"

business_rules:
  default_facet_key: "email_mobile"
  refinement_categories: ["TOP 10", "OTHERS", "ALL"]
  scoring_tiers: ["Very-High", "High", "Medium-High", "Medium", "Medium-Low", "Low", "Very Low"]
  strict_preference: true  # Prefer Strict facets when available
  purchase_channels: ["ONLINE", "STORE"]

vocabulary:
  synonyms:  # Tenant-specific vocabulary mapping
    "Affinity Score": "Propensity"
    "Customer Profile": "Persona"
    "Email Opt-in": "Email Savings and Updates Opt-in"

model_config:
  primary_model: "gemini-2.0-flash"  # Cost-effective default
  complex_model: "claude-opus-4-6"   # For complex queries
  complexity_threshold: 0.7           # Route to complex model above this
```

### 6.2 Tenant Onboarding Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                    TENANT ONBOARDING FLOW                        │
│                                                                  │
│  Step 1: PROVIDE DATA (Human, ~2 hours)                         │
│  ├── Facet catalog (BigQuery table or CSV)                      │
│  ├── Ground truth CSV (minimum 50 labeled segment→facet pairs)  │
│  └── Business rules document (refinement categories, channels)   │
│                                                                  │
│  Step 2: AUTO-GENERATE CONFIG (System, ~30 min)                 │
│  ├── Parse facet catalog → infer categories, types, hierarchies │
│  ├── Generate tenant_config.yaml from template                  │
│  ├── LLM generates contextual_information files from catalog    │
│  │   └── Prompts: "Given this facet catalog, generate a         │
│  │       catalog_view_description.txt that explains each        │
│  │       capability category with examples"                     │
│  ├── LLM generates hints files from catalog + ground truth      │
│  └── Generate vocabulary synonym table (cross-reference with    │
│      existing tenant facet names using embedding similarity)    │
│                                                                  │
│  Step 3: INDEX DATA (System, ~15 min)                           │
│  ├── Create Milvus collection (silo) or partition (pool)        │
│  ├── Embed and index facet names + values                       │
│  ├── Embed ground truth descriptions for few-shot RAG           │
│  └── Build BM25 index, trie, and taxonomy graph                │
│                                                                  │
│  Step 4: VALIDATE (System, ~1 hour)                             │
│  ├── Run eval suite on ground truth examples                    │
│  ├── Compare quality metrics vs primary tenant baseline         │
│  ├── If quality < 80% of baseline:                             │
│  │   ├── Flag for human review                                  │
│  │   ├── Auto-adjust: add cross-tenant few-shot examples       │
│  │   └── Re-run eval                                            │
│  └── Generate tenant readiness report                           │
│                                                                  │
│  Step 5: ACTIVATE (Human approval, ~5 min)                      │
│  ├── Review readiness report                                    │
│  ├── Approve tenant activation                                  │
│  └── Tenant goes live — no code changes, no redeployment       │
└──────────────────────────────────────────────────────────────────┘
```

### 6.3 Tenant Context Loader

```python
class TenantContextLoader:
    """
    Loads all tenant-specific context at runtime based on tenant_id.
    No hardcoded file paths — everything from config.
    """

    def __init__(self, tenants_dir: str = "tenants/"):
        self.tenants = {}
        for config_file in Path(tenants_dir).glob("*/config.yaml"):
            config = yaml.safe_load(config_file.read_text())
            self.tenants[config["tenant_id"]] = TenantConfig(
                config=config,
                base_dir=config_file.parent
            )

    def get_tenant(self, tenant_id: str) -> TenantConfig:
        if tenant_id not in self.tenants:
            raise ValueError(f"Unknown tenant: {tenant_id}")
        return self.tenants[tenant_id]

    def get_contextual_info(self, tenant_id: str) -> dict:
        """Load all contextual information files for a tenant."""
        tenant = self.get_tenant(tenant_id)
        ctx = {}
        for key, filename in tenant.config["contextual_info"]["files"].items():
            filepath = tenant.base_dir / tenant.config["contextual_info"]["base_dir"] / filename
            ctx[key] = filepath.read_text()
        return ctx
```

### 6.4 Quality Parity Across Tenants

**Cold-start strategy for new tenants:**

1. **Cross-tenant few-shot transfer:** Use primary tenant's ground truth rows that share similar abstract segment types (persona, propensity, date-based) as few-shot examples, substituting tenant-specific facet names via the synonym table
2. **Universal abstract taxonomy:** Both tenants map to abstract categories at reasoning time:
   ```
   Abstract: "Propensity" → Tenant A: "Propensity Super Department" | Tenant B: "Affinity Score"
   Abstract: "Engagement" → Tenant A: "CRM Email Engagement" | Tenant B: "Email Activity Level"
   ```
3. **Progressive quality:** Start with cross-tenant examples + auto-generated hints; replace with tenant-specific data as ground truth accumulates
4. **Minimum viable dataset:** 50 labeled segment→facet pairs (based on enterprise RAG cold-start research)

---

## 7. Memory System

### 7.1 Two-Tier Memory Architecture

```
┌─────────────────────────────────────────┐
│            MEMORY SYSTEM                 │
│                                          │
│  SHORT-TERM (Redis, per session)         │
│  ├── Current conversation state          │
│  ├── Sub-segment query progression       │
│  ├── Clarification Q&A history           │
│  └── Current segment draft               │
│                                          │
│  LONG-TERM (PostgreSQL, per user+tenant) │
│  ├── User preferences:                   │
│  │   • Preferred facet types             │
│  │   • Strict vs non-Strict preference   │
│  │   • Common segment patterns           │
│  ├── Successful segment recipes:          │
│  │   • Query → facet mapping pairs       │
│  │   • Reusable segment templates        │
│  ├── Learned corrections:                │
│  │   • Past mistakes and fixes           │
│  │   • User feedback on accuracy         │
│  └── Tenant domain knowledge:            │
│      • Discovered synonyms               │
│      • Inferred business rules           │
└─────────────────────────────────────────┘
```

### 7.2 Memory Integration with Pipeline

```python
class MemoryStore:
    def get_relevant_memories(self, user_id: str, query: str, k: int = 3) -> list[Memory]:
        """Retrieve memories relevant to current query."""
        # User preferences
        prefs = self.get_user_preferences(user_id)

        # Similar past segments by this user
        past_segments = self.search_user_segments(user_id, query, k=k)

        # Known corrections for similar queries
        corrections = self.get_corrections(query, k=2)

        return MemoryContext(
            preferences=prefs,
            past_segments=past_segments,
            corrections=corrections
        )
```

---

## 8. Evaluation-First Infrastructure

### 8.1 Per-Stage Evaluation Metrics

| Stage | Metric | Target | How Measured |
|-------|--------|--------|-------------|
| Perceive | Intent accuracy | >98% | Classification F1 on labeled intents |
| Perceive | NER precision | >90% | Entity extraction vs. labeled entities |
| Reason & Map: Decompose | Sub-segment F1 | >85% | Compare sub-segments vs. ground truth |
| Reason & Map: Date | Date extraction accuracy | >95% | Parsed dates vs. expected dates |
| Reason & Map: Facets | Facet Recall@5 | >85% | Expected facets in top-5 retrieved |
| Reason & Map: Facets | Facet Precision | >80% | Correct facets / selected facets |
| Validate | Dependency resolution | >90% | Correct refinement values selected |
| Format | Schema validity | 100% | JSON schema validation pass rate |
| End-to-end | Segment F1 | >80% | Overall expected vs. produced |

### 8.2 CI/CD Eval Gates

```yaml
# .github/workflows/eval-gate.yaml
name: Evaluation Gate
on:
  pull_request:
    paths:
      - "agentic_framework/prompts/**"
      - "agentic_framework/contextual_information/**"
      - "agentic_framework/sub_agents/**"
      - "skills/**"

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - name: Run eval suite
        run: python -m evaluations.scripts.e2e_evaluation_test --eval-set production_core

      - name: Check quality gates
        run: |
          python -c "
          import json
          results = json.load(open('eval_results.json'))
          assert results['facet_recall_at_5'] >= 0.85, f'Facet recall too low: {results[\"facet_recall_at_5\"]}'
          assert results['facet_precision'] >= 0.80, f'Facet precision too low: {results[\"facet_precision\"]}'
          assert results['schema_validity'] >= 1.00, f'Schema validity: {results[\"schema_validity\"]}'
          assert results['end_to_end_f1'] >= 0.80, f'E2E F1 too low: {results[\"end_to_end_f1\"]}'
          print('All quality gates passed')
          "
```

---

## 9. Auto-Improvement Pipeline

### 9.1 DSPy-Based Prompt Optimization

```python
import dspy

class SegmentMapper(dspy.Module):
    """DSPy module for optimizing facet selection prompts."""

    def __init__(self):
        self.mapper = dspy.ChainOfThought("query, candidates -> facets")

    def forward(self, query: str, candidates: list[str]) -> str:
        return self.mapper(query=query, candidates=str(candidates))

# Optimize with ground truth
optimizer = dspy.GEPA(
    metric=facet_f1_metric,
    num_threads=4,
    max_bootstrapped_demos=5
)

optimized_mapper = optimizer.compile(
    SegmentMapper(),
    trainset=ground_truth_train,
    valset=ground_truth_val
)

# Optimized prompt is auto-generated — replaces manual prompt engineering
```

### 9.2 Feedback Loop

```
User creates segment
    ↓
System produces segment definition
    ↓
User reviews (accept / modify / reject)
    ↓
If modified or rejected:
    ├── Store as correction in long-term memory
    ├── Add to ground truth CSV (pending human validation)
    └── Weekly: run DSPy optimization with expanded dataset
    ↓
Optimized prompts must pass eval gates before deployment
    ↓
A/B test new prompts against current (per-tenant)
    ↓
Promote winning prompts
```

---

## 10. Enterprise Domain Patterns

### 10.1 Patterns from Leading Enterprise Implementations

**From Salesforce Agentforce:**
- **Supervisor-Specialist pattern:** Central orchestrator with domain-specific specialists
- **Topic-based routing:** Each conversation topic has its own agent with scoped tools and guardrails
- **Guardrails as infrastructure:** Safety checks at boundaries, not in prompts
- **Adoption:** Apply to Smart-Segmentation: Router as supervisor, NSC/DSE as specialists

**From Adobe Real-Time CDP:**
- **Hub-Edge architecture:** Centralized governance + millisecond edge evaluation
- **Multi-agent collaboration:** Audience Agent, Data Engineering Agent, Builder Agent
- **Adoption:** Separate governance (prompt rules, eval gates) from execution (segment building)

**From Walmart:**
- **Customer embedding layer:** Unified customer representations for cross-channel consistency
- **Domain-specific LLM (Wallaby):** Fine-tuned on retail vocabulary for better understanding
- **Adoption:** Consider lightweight domain adaptation (fine-tune small model on facet pairs)

**From HubSpot:**
- **Intent-based segmentation:** 82% conversion rate improvement vs. demographic segments
- **Adoption:** Prioritize behavioral facets (purchase, engagement) over demographic ones

### 10.2 CDP Integration Pattern

```
┌───────────────────────────────────┐
│        CUSTOMER DATA PLATFORM     │
│  (BigQuery / Snowflake / CDP)     │
│                                   │
│  • Customer profiles              │
│  • Event streams                  │
│  • Purchase history               │
│  • Engagement data                │
└──────────────┬────────────────────┘
               │ (Data source)
               ▼
┌───────────────────────────────────┐
│      FACET CATALOG LAYER          │
│                                   │
│  • Derived from CDP schema        │
│  • Typed, categorized, restricted │
│  • Versioned, per-tenant          │
└──────────────┬────────────────────┘
               │ (Retrieval target)
               ▼
┌───────────────────────────────────┐
│     AGENTIC SEGMENTATION AGENT    │
│                                   │
│  • Cascade retrieval              │
│  • Ground truth few-shot          │
│  • Memory-augmented reasoning     │
│  • Multi-tenant isolated          │
└──────────────┬────────────────────┘
               │ (Segment definition)
               ▼
┌───────────────────────────────────┐
│      ACTIVATION CHANNELS          │
│                                   │
│  • Marketing Cloud                │
│  • CRM campaigns                  │
│  • Email / Push / In-app          │
│  • Audience export (CSV / API)    │
└───────────────────────────────────┘
```

---

## 11. Cost Optimization Strategy

### 11.1 Combined Cost Reduction

| Optimization | Savings | Implementation |
|---|---|---|
| Pipeline collapse (7→3 stages) | 60% LLM calls | Weeks 3-6 |
| Date tagger → code | 100% of 1 stage | Week 1 |
| Formatter → code | 100% of 1 stage | Week 1 |
| Route agent → classifier | 90% of 1 stage | Week 2 |
| Prompt caching | 80% input tokens | Week 2 |
| Semantic caching (30% hit rate) | 30% overall | Week 4 |
| Model routing (simple/complex) | 50% on simple queries | Week 5 |
| Cross-encoder rerank (vs LLM rerank) | 72% reranking cost | Week 3 |
| **Combined** | **~70% total cost reduction** | **Weeks 1-6** |

### 11.2 Cost Monitoring

```python
class CostTracker:
    """Track and report per-segment creation costs."""

    def track_segment_cost(self, segment_id: str, traces: list[Trace]):
        cost = {
            "embedding_cost": sum(t.embedding_cost for t in traces),
            "llm_input_tokens": sum(t.input_tokens for t in traces),
            "llm_output_tokens": sum(t.output_tokens for t in traces),
            "llm_cost": sum(t.llm_cost for t in traces),
            "milvus_queries": sum(1 for t in traces if t.type == "milvus"),
            "total_cost": sum(t.total_cost for t in traces),
            "latency_ms": sum(t.latency_ms for t in traces),
        }
        self.store(segment_id, cost)
        return cost
```

---

## 12. Concrete Transformation Examples

### 12.1 Example 1: "Spring fashion shoppers looking to buy women's clothing"

**Current Pipeline (7 stages):**
```
1. Route Agent: "Create segment" → route to NSC              [LLM, ~2s]
2. Decomposer: {Seg-1: "spring fashion", Seg-2: "women's"}  [LLM, ~2s]
3. Date Tagger: "spring" → Q1/Q2 date range                  [LLM, ~2s]
4. Facet Mapper: Milvus search → 15 candidates → LLM select  [Milvus + LLM, ~4s]
5. Classifier: Propensity SD → refinement "APPAREL"          [LLM, ~2s]
6. Linked Facet: Division → "WOMEN'S CLOTHING"               [LLM, ~2s]
7. Formatter: JSON output                                     [LLM, ~2s]
Total: ~16s, 7 LLM calls, ~$0.15
```

**Proposed Pipeline (3 stages):**
```
1. Perceive: Intent=CREATE, NER=[fashion, women's, clothing]  [Code, ~10ms]
2. Reason & Map:
   - Date: "spring" → no date facet (seasonal, not temporal)  [Code, ~5ms]
   - Retrieve: NER → Propensity facets → [SD, Division, Brand] [Cascade, ~200ms]
   - Few-shot: 2 similar examples from ground truth            [RAG, ~50ms]
   - LLM: Decompose + map in one pass                         [LLM, ~3s]
3. Validate: Check completeness, no ambiguity                  [Code, ~10ms]
4. Format: SegmentR JSON                                       [Code, ~5ms]
Total: ~3.3s, 1 LLM call, ~$0.04
```

### 12.2 Example 2: "New parents with active baby registry and email engagement 0-90 days"

**Current Pipeline:**
```
1. Route: NSC                                                  [LLM, ~2s]
2. Decompose: {Seg-1: "new parents", Seg-2: "baby registry",
               Seg-3: "email engagement 0-90 days"}           [LLM, ~2s]
3. Date Tag: "0-90 days" → relative date                      [LLM, ~2s]
4. Facet Map: 3 searches → 20+ candidates → LLM picks         [Milvus + LLM, ~5s]
5. Classifier: Baby Registry Status → "Active"                [LLM, ~2s]
   + CRM Email Engagement → ambiguity → ask user              [LLM, ~2s]
6. (User responds)
7. Format: JSON                                                [LLM, ~2s]
Total: ~17s + user wait, 8 LLM calls, ~$0.20
```

**Proposed Pipeline:**
```
1. Perceive: Intent=CREATE, NER=[parents, baby, registry, email] [Code, ~10ms]
2. Reason & Map:
   - Date: "0-90 days" → relative date constraint              [Code, ~5ms]
   - Retrieve: NER → Persona + Lifecycle + Engagement facets   [Cascade, ~300ms]
   - Few-shot: Ground truth Example 3 (BabyPlanners) retrieved [RAG, ~50ms]
     → Shows exact expected facets for similar query!
   - LLM: Full decompose + map with GT example guidance        [LLM, ~4s]
3. Validate:
   - CRM Email Engagement needs value specification             [Code, ~10ms]
   - Ask user: "What engagement level?"                         [Pause for user]
4. Format: SegmentR JSON                                        [Code, ~5ms]
Total: ~4.4s + user wait, 1-2 LLM calls, ~$0.06
```

### 12.3 Example 3: Multi-Tenant — New Retailer Onboarding

**Current process (code changes required):**
1. Developer creates new pickle files from retailer's data _(manual, ~2 days)_
2. Developer modifies `metadata.py` to support new facet key _(code change)_
3. Developer creates new contextual_information files _(manual, ~1 day)_
4. Developer updates Milvus collection names in env vars _(config change)_
5. Developer creates new eval sets _(manual, ~1 day)_
6. Deploy new code version _(deployment)_
7. Total: ~5 days of developer time

**Proposed process (config-only):**
1. Retailer provides: facet catalog CSV + 50 labeled examples _(~2 hours)_
2. System auto-generates: config.yaml + contextual info files + synonym table _(~30 min)_
3. System indexes: Milvus collection + BM25 + ground truth RAG _(~15 min)_
4. System validates: Run eval suite, compare to baseline _(~1 hour)_
5. Human approves: Review readiness report _(~5 min)_
6. Total: ~4 hours, zero code changes

---

## Appendix A: Technology Stack Changes

| Component | Current | Proposed | Rationale |
|---|---|---|---|
| Agent framework | Google ADK | Google ADK (keep) | Working well, no need to change |
| Vector DB | Milvus (dense only) | Milvus (hybrid BM25+dense) | Enable hybrid search |
| Embedding model | BGE-large or MiniLM | BGE-large (primary) + cross-encoder (rerank) | Add reranking |
| LLM | Gemini (single model) | Gemini Flash (simple) + Claude Opus (complex) | Model routing |
| Memory | None | Redis (short) + PostgreSQL (long) | Two-tier memory |
| Caching | None | Redis semantic cache | 73% cost reduction |
| Eval | Manual e2e | Automated CI/CD gates + per-stage metrics | Eval-first |
| Prompt optimization | Manual editing | DSPy GEPA | Auto-improvement |
| Observability | Basic Phoenix | Phoenix + Langfuse + cost tracking | Full visibility |
| Config | Env vars + static files | Tenant YAML manifests | Multi-tenant ready |

---

*Next: See [04_implementation_roadmap.md](04_implementation_roadmap.md) for the phased implementation plan.*
