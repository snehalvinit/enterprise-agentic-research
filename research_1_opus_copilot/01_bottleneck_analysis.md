# Bottleneck Analysis — Smart-Segmentation

> **Research ID:** research_1_opus_copilot  
> **Date:** February 2026  
> **Model:** Claude Opus 4.6 via GitHub Copilot  
> **Source System:** Smart-Segmentation (Walmart CDI)

---

## Executive Summary

Smart-Segmentation is a well-conceived multi-agent system built on Google ADK for translating natural language into structured customer segment definitions. The core RAG pipeline (NER → embeddings → Milvus vector search → fuzzy matching → LLM reasoning) is sophisticated, and the hierarchical agent structure (Router → Sub-agents → Tools) demonstrates sound engineering instincts.

However, the system has accumulated significant technical debt and architectural gaps that prevent it from meeting enterprise-grade standards for reliability, scalability, security, and extensibility. This document catalogs every materially significant bottleneck across 10 dimensions.

---

## 1. Security Vulnerabilities

### 1.1 Critical: `eval()` on Untrusted Input

**Severity: CRITICAL**

The codebase contains multiple instances of Python's `eval()` called on environment variables, LLM output, and deserialized data. This is a direct code injection vulnerability.

**Affected files:**
- `agentic_framework/agent.py` — `eval(os.environ.get('DEFAULT_USER_RESTRICTIONS'))`
- `agentic_framework/sub_agents/new_segment_creation/agent.py` — `eval()` on env vars
- `agentic_framework/sub_agents/new_segment_creation/agent_tools/segment_decomposer_agent.py` — `eval()` on LLM output
- `agentic_framework/sub_agents/new_segment_creation/agent_tools/facet_value_mapper_agent.py` — `eval()` on LLM output
- `agentic_framework/sub_agents/new_segment_creation/tools/shortlist_generation.py` — `eval()` on env vars

**Impact:** An attacker who can influence environment variables, or a prompt injection attack that modifies LLM output, could execute arbitrary Python code on the server.

**Fix:** Replace all `eval()` with `json.loads()`, `ast.literal_eval()`, or typed Pydantic parsing.

### 1.2 SSL Verification Disabled

`agentic_framework/database/connection.py` disables SSL certificate verification:
```python
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE
```

This allows man-in-the-middle attacks on database connections.

### 1.3 Wide-Open CORS

`agentic_framework/api.py` sets `allow_origins=["*"]` — any domain can make authenticated requests to the API.

### 1.4 Unnecessary Production Dependencies

`import pytest` appears at the top level of `api.py` — a test framework imported into the production API server, expanding the attack surface.

---

## 2. Architecture Issues

### 2.1 Monolithic Route Handler

`agentic_framework/routes/agent_routes.py` contains a **641-line single function** (`run_sse`) that handles:
- Request parsing
- Session creation/retrieval
- Event streaming
- Response formatting
- Segment summary generation
- UI button generation
- Error handling at 3+ nesting levels

This violates single-responsibility principle and makes the code untestable, unreadable, and fragile.

### 2.2 Dual LLM Client Libraries

The system uses **two completely different LLM client libraries** for the same gateway:
1. **LiteLLM** (via `gptmodel` in `adk_llm_model.py`) — for ADK agent orchestration
2. **Infero** (via `StructuredInfero` in `pydantic_infero.py`) — for structured tool calls

This creates:
- Inconsistent retry logic and error handling
- Double configuration burden
- Different observability pathways
- Maintenance overhead when the gateway API changes

### 2.3 Single Model for All Tasks

`adk_llm_model.py` creates one `gptmodel` instance shared by all agents. There is no ability to:
- Use a cheaper, faster model for simple routing decisions
- Use a stronger reasoning model for complex decomposition
- Use a specialized model for NER extraction
- A/B test different models per task

**Cost impact:** Every routing decision (cheap) costs the same as every complex reasoning step (expensive).

### 2.4 No Connection Reuse

- **Milvus:** `connections.connect(...)` is called inside every search method — a new TCP connection per vector search
- **Embedding model:** `EmbeddingGenerator()` loads the entire BGE-large model from disk on every instantiation (no singleton, no cache)
- **GCP Logger:** Recreated per request instead of using a persistent instance

### 2.5 Sequential Pipeline with No Parallelism

A single segment creation request requires this sequential chain:

```
Decompose (LLM) → Date tag (LLM) → NER (LLM) → Embed → Milvus search → Fuzzy match → Facet map (LLM) → Dependency check → Format (LLM)
```

This is **4-6+ sequential LLM calls** plus vector searches, yielding 15-45 second latency per request. Many of these steps could be parallelized (e.g., NER + date tagging could run simultaneously).

### 2.6 Pickle File Dependencies

`agentic_framework/utils/metadata.py` loads the facet catalog from `.pkl` (pickle) files. Pickle files are:
- Opaque and impossible to diff/review
- Fragile across Python version changes
- Not versioned or cache-invalidated
- Loaded from disk on every `MetaData()` instantiation

---

## 3. State Management Problems

### 3.1 Explosive State Variable Count

`agentic_framework/state.py` declares **60+ flat string constants** with no grouping, namespacing, or type annotations. Examples:

```python
FACET_VALUE_MAPPER_SHORTLISTED_FACETS = "facet_value_mapper_shortlisted_facets"
FACET_VALUE_MAPPER_CURR_INDEX = "facet_value_mapper_curr_index"
FACET_VALUE_MAPPER_OPERATOR_ADDITIONAL_INFO = "facet_value_mapper_operator_additonal_info"  # typo
FACET_CLASSIFIER_DEPENDECY_CURR_INDEX = "facet_classifier_dependecy_curr_index"  # typo
```

Problems:
- Typos in variable names (`DEPENDECY`, `additonal_info`) that can never be fixed without migration
- No documentation of what each variable holds or its expected type
- No way to see at a glance what state a given sub-agent requires vs. produces
- No validation that state is populated before access

### 3.2 Massive State Initialization Duplication

`initialize_state_nsc` and `initialize_state_nfa` (in `new_segment_creation/agent.py`) are **90% identical** — both set 30+ state variables with the same defaults. When a new state variable is added, both must be updated, and they frequently drift out of sync.

### 3.3 No State Cleanup

The `clear_state` callback is defined as:
```python
def clear_state(callback_context):
    pass
```

State is **never cleaned up** between conversations or after errors, leading to stale data contaminating subsequent requests.

### 3.4 In-Memory Services for Artifacts and Memory

```python
artifact_service = InMemoryArtifactService()
memory_service = InMemoryMemoryService()
```

Artifacts and memory are **lost on every pod restart** and are **not shared across pods**. In a multi-pod deployment, this means inconsistent behavior depending on which pod handles the request.

---

## 4. Prompt Design Flaws

### 4.1 No Templating Engine

Prompts use raw `str.replace()` for variable substitution:
```python
prompt = prompt.replace("{variable}", value)
```

Problems:
- No escaping — if a value contains `{another_var}`, it could be interpreted
- No missing variable detection — silent failures if a placeholder isn't replaced
- No conditional sections or loops
- No Jinja2, no Mustache, no structured templating

### 4.2 All Prompts Loaded Eagerly

`agentic_framework/utils/agent_prompt.py` loads all 23 prompt files at module import time, regardless of whether they're needed. This:
- Increases startup time
- Wastes memory for unused prompts
- Prevents lazy loading based on tenant or request type

### 4.3 No Prompt Versioning or A/B Testing

Every request uses the same static prompt text. There is no infrastructure for:
- Prompt version history
- A/B testing prompt variants
- Per-tenant prompt customization
- Gradual prompt rollouts with eval gates

### 4.4 Prompts Are Monolithic

Several prompts exceed 200 lines and embed examples, constraints, output schemas, and edge cases in a single text file. If any component changes, the entire prompt must be re-evaluated. A skill-based architecture would decompose these into independently testable and versioned units.

---

## 5. Evaluation Gaps

### 5.1 Zero Automated Tests

The test suite contains:
- `test_api.py`: `assert 1 + 1 == 2` (placeholder)
- `segment_decomposer_test.py`: 50 concurrent decomposer runs requiring live services (load test, not unit test)

There are **zero unit tests, zero integration tests, zero mock-based tests, and zero tests in CI** beyond the placeholder.

### 5.2 Evaluation Framework Not CI-Integrated

The `agentic_framework/evaluations/` directory contains a substantial evaluation framework (2800+ line CLI with comparators, report generation, paraphrase testing). However:
- It's completely disconnected from CI/CD
- Must be run manually
- No eval gates block deployment of broken changes
- Eval results aren't tracked historically

### 5.3 No Output Validation in Production

The API route extracts JSON from LLM output using regex:
```python
re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
```

If the LLM produces malformed output, the regex silently fails and the response is empty or corrupted. There is no Pydantic validation on the final API response structure — only on intermediate tool outputs.

### 5.4 No Regression Detection

There is no mechanism to detect when a model upgrade, prompt change, or code change causes regression in output quality. Historical eval scores are not stored or compared.

---

## 6. Scalability Limits

### 6.1 Resource-Heavy Pods

Each pod requires **6-12 CPU and 32-64 GiB RAM** due to:
- BGE-large-en-v1.5 embedding model (~1.3 GB) loaded into memory via PyTorch
- PyTorch itself (~800 MB)
- Pickle files for facet catalogs
- In-memory state accumulation

At $0.10-0.20/GB-hour on cloud providers, each pod costs $3-6/hour just for memory, limiting horizontal scaling economics.

### 6.2 Connection Pool Starvation

`database/config.py` sets `max_connections: 3` for PostgreSQL. With `WEB_CONCURRENCY=4` (uvicorn workers), concurrent requests will starve the connection pool under moderate load.

### 6.3 No Caching Layer

There is no caching at any level:
- No LLM response cache (identical queries re-invoke the model)
- No embedding cache (identical text re-runs the embedding model)
- No Milvus result cache
- No metadata cache

### 6.4 Single-Process Server

Despite comments about multi-worker support, the deployment runs a single uvicorn process. There is no Gunicorn multi-worker setup, and asyncpg pool limits suggest the system wasn't designed for concurrent request handling.

---

## 7. Reliability Problems

### 7.1 No Circuit Breaker

When the LLM gateway, Milvus, or PostgreSQL is down, the system will:
1. Retry aggressively (10 retries for LLM)
2. Queue up requests
3. Exhaust memory and connections
4. Eventually crash

There is no circuit breaker to fast-fail when dependencies are unhealthy.

### 7.2 Rate Limiting Detection via String Matching

```python
if "rate limit" in str(e).lower():
    # handle rate limit
```

This is fragile — different providers use different error message formats. When the string doesn't match, rate limit errors propagate as generic failures.

### 7.3 No Idempotency

The API does not guarantee idempotent behavior. If a request times out and is retried, it may create duplicate sessions, re-process the same segment, or produce inconsistent state.

### 7.4 Triple-Nested Error Handling

`agent_routes.py` has a triple-nested try/except in the SSE event generator:
```python
try:
    async for event in ...:
        try:
            # process
        except:
            try:
                # fallback
            except:
                # give up
```

This obscures the actual error source and makes debugging extremely difficult.

---

## 8. Observability Gaps

### 8.1 Single Trace Backend

Arize Phoenix is the only observability tool. There are no:
- Prometheus metrics (endpoint exists but is disabled)
- Custom dashboards
- Alert rules
- SLO/SLI definitions
- Cost-per-request tracking
- Latency percentile tracking (p50, p95, p99)

### 8.2 Typo in Configuration

`TRACING_PHEONIX_ENDPOINT` (should be `PHOENIX`) — this typo has likely been copy-pasted across environments and documentation.

### 8.3 Stdout Suppression

`pydantic_infero.py` calls `disable.blockPrint()` and `enable.enablePrint()` to suppress stdout globally. This:
- Is not thread-safe
- Suppresses error messages from all libraries
- Makes debugging intermittent issues nearly impossible

### 8.4 No Cost Tracking

There is no mechanism to track LLM token usage, cost per request, or cost per facet-mapping operation. Without cost visibility, optimization is guesswork.

---

## 9. Multi-Tenancy Deficiencies

### 9.1 Current State

Multi-tenancy support is rudimentary:
- `facet_key_identifier` supports two identifier types
- `facet_user_restrictions` provides per-user facet access control
- `tenantId` defaults to a single hardcoded value
- CBBID suffix processing for tenant-specific facet naming

### 9.2 Missing Multi-Tenant Capabilities

| Capability | Status |
|---|---|
| Tenant-specific prompts | Not supported |
| Tenant-specific model selection | Not supported |
| Tenant-scoped sessions | Not supported (shared DB) |
| Tenant-specific knowledge bases | Not supported |
| Tenant-scoped rate limits | Not supported |
| Tenant usage tracking & billing | Not supported |
| Tenant-specific eval gates | Not supported |
| Data isolation enforcement | Partial (user restrictions only) |

### 9.3 Tenant Configuration as Environment Variables

Tenant-specific behavior is controlled via environment variables (`DEFAULT_USER_RESTRICTIONS`), which means:
- Every tenant change requires a deployment
- Cannot dynamically onboard new tenants
- Configuration is opaque and not auditable

---

## 10. Missing Enterprise Capabilities

### 10.1 No Memory System

The system has session persistence (PostgreSQL) but no structured memory:
- No long-term storage of successful segmentation patterns
- No learning from user corrections
- No cross-session knowledge accumulation
- No "recipes" — proven segment patterns that can be reused

### 10.2 No Auto-Improvement

There are no feedback loops:
- No prompt optimization based on eval results
- No automatic few-shot example curation from successful interactions
- No A/B testing infrastructure
- No DSPy-style programmatic prompt tuning

### 10.3 No Skill Architecture

Every capability is hardcoded into agent definitions and tool functions. Adding a new capability requires:
1. Creating a new Agent class
2. Modifying the router prompt
3. Writing new tool functions
4. Updating state initialization
5. Full redeployment

A skill-based architecture would enable adding capabilities via configuration files without code changes.

### 10.4 No Hypothesis Assessment

The system cannot:
- Evaluate user hypotheses about customer segments
- Suggest alternative segmentation approaches with reasoning
- Compare estimated segment sizes across different strategies
- Explain why certain facet combinations may be more effective

### 10.5 No Model Exploration

The system cannot:
- Assess existing ML models for segment enrichment
- Suggest building new propensity or lookalike models
- Log evidence-based model recommendations
- Connect to model registries for metadata

---

## Summary: Bottleneck Priority Matrix

| Priority | Category | Issue | Impact |
|---|---|---|---|
| P0 | Security | `eval()` on untrusted input | Code injection risk |
| P0 | Security | SSL verification disabled | MITM attacks |
| P0 | Testing | Zero automated tests | No safety net for changes |
| P1 | Architecture | Monolithic route handler | Untestable, fragile |
| P1 | Architecture | Sequential 15-45s pipeline | User experience degradation |
| P1 | State | 60+ untyped, duplicated state vars | Bugs, drift, maintenance cost |
| P1 | Reliability | No circuit breaker | Cascading failures |
| P1 | Eval | Eval framework not in CI | Silent regressions |
| P2 | Architecture | Dual LLM clients | Maintenance burden |
| P2 | Architecture | Single model for all tasks | Cost waste |
| P2 | Scalability | 32-64 GiB RAM per pod | Expensive scaling |
| P2 | Scalability | No caching layer | Redundant computation |
| P2 | Architecture | No connection pooling | Connection starvation |
| P2 | Observability | No metrics, alerts, cost tracking | Blind operations |
| P3 | Prompts | No versioning or A/B testing | Cannot iterate safely |
| P3 | Multi-tenancy | Hardcoded single tenant | Cannot onboard customers |
| P3 | Memory | No long-term learning | Repeats mistakes |
| P3 | Extensibility | No skill architecture | Every feature = code change |

---

## Conclusion

The Smart-Segmentation system has strong engineering DNA — the multi-agent hierarchy, the RAG pipeline, and the Pydantic validation pattern are solid foundations. However, the system was built as a prototype-turned-production-service without the enterprise hardening required for reliability, security, and scalability at scale.

The transformation required is not a rewrite, but a systematic layering of enterprise patterns on top of the existing architecture: fixing security vulnerabilities first, adding testing and eval gates, refactoring the monolithic handler, introducing caching and connection pooling, and building toward a pluggable skill-based architecture with structured memory and auto-improvement.

The subsequent documents in this research provide the research foundation, concrete proposal, and implementation roadmap to execute this transformation.
