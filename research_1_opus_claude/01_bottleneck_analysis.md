# Bottleneck Analysis: Smart-Segmentation Codebase

> **Research ID**: research_1_opus_claude
> **Model**: Claude Opus 4.6
> **Date**: February 2026
> **Status**: Complete

---

## Executive Summary

Smart-Segmentation is a functional, production-deployed agentic system for converting natural language queries into customer segment definitions. It uses Google's ADK (Agent Development Kit) with a multi-agent architecture: a RouterAgent delegates to NewSegmentCreationAgent (NSC) and DirectSegmentEditorAgent (DSE), each with specialized sub-agents for decomposition, entity recognition, date extraction, facet mapping, and format generation.

While the system works, it has **fundamental architectural bottlenecks** that prevent it from evolving into an enterprise-grade agentic platform. This document catalogs every significant issue across architecture, prompt design, evaluation, scalability, reliability, and missing capabilities.

---

## Table of Contents

1. [Architecture Issues](#1-architecture-issues)
2. [Prompt Design Flaws](#2-prompt-design-flaws)
3. [Evaluation Gaps](#3-evaluation-gaps)
4. [Scalability Limits](#4-scalability-limits)
5. [Reliability Problems](#5-reliability-problems)
6. [Missing Capabilities](#6-missing-capabilities)
7. [Security Concerns](#7-security-concerns)
8. [Cost & Performance Issues](#8-cost--performance-issues)
9. [Summary Severity Matrix](#9-summary-severity-matrix)

---

## 1. Architecture Issues

### 1.1 Monolithic Agent Coupling

**Severity: Critical**

The entire system is hardwired as a fixed agent tree:

```
RouterAgent → [NSC, DSE]
NSC → [SegmentDecomposer, DateTagger, FacetValueMapper] → [SegmentFormatGenerator]
DSE → [EditDecomposer, Addition, Deletion, Updation]
```

Every agent is instantiated at import time in `agent.py` and `sub_agents/*/agent.py`. There is no registry, no dynamic loading, and no way to add a new capability without modifying the agent tree code directly. Adding a new use case (e.g., "segment analysis," "campaign recommendation," "hypothesis testing") requires:

1. Creating new sub-agent files
2. Importing them in the parent agent
3. Modifying the parent agent's `sub_agents=[]` list
4. Updating the router prompt to know about the new agent
5. Redeploying the entire application

**Impact**: Every new feature requires a code change to the core agent wiring. This is the opposite of a pluggable skill architecture.

### 1.2 God State Object

**Severity: Critical**

The file `state.py` defines **66+ flat string constants** as state keys, all sharing a single global namespace via `CallbackContext.state`. There is no typing, no grouping, no namespacing:

```python
SUB_SEGMENT_QUERY_VERSION = "sub_segment_query_version"
SUB_SEGMENT_QUERY_RELATIONSHIP_REPRESENTATION = "sub_segment_relationship_representation"
SUB_SEGMENT_QUERY_REPRESENTATION = 'sub_segment_representation'
SHORTLIST_GENERATION = 'sub_segment_shortlist_generation'
OG_FACET_CLASSIFIER_DEPENDENCY_DICT = "og_facet_classifier_dependency_dict"
OG_FACET_CLASSIFIER_RESOLVER_RESPONSE = "og_facet_classifier_resolver_response"
# ... 60 more
```

State initialization is duplicated across `agent.py:initialize_state()`, `sub_agents/new_segment_creation/agent.py:initialize_state_nsc()`, and `initialize_state_nfa()` — all doing nearly identical work with copy-paste logic:

```python
if USER_ID not in callback_context.state:
    callback_context.state[USER_ID] = callback_context.state.get('user_id', "SEGMENT_AI_ADK")
if USER_TYPE not in callback_context.state:
    callback_context.state[USER_TYPE] = callback_context.state.get('user_type', "ASSOCIATE")
# repeated 30+ times across 3 functions
```

**Impact**: State bugs are invisible until runtime. No IDE autocompletion. Typos in state keys cause silent failures. State dependencies between agents are undocumented.

### 1.3 No Plugin / Skill Architecture

**Severity: Critical**

Capabilities are encoded as Python functions and agent classes, not as declarative, versionable skill bundles. There is no:

- Skill registry (no way to list, version, or A/B test skills)
- Skill router (intent → skill mapping is buried in prompt text)
- Skill loader (no dynamic injection of instructions)
- Skill schema (no input/output contracts)
- Skill eval gate (no per-skill evaluation before deployment)

**Impact**: Every capability change requires a code deployment. Cannot do per-tenant skill customization. Cannot A/B test prompt variations.

### 1.4 Tight Coupling to Google ADK

**Severity: High**

The entire system is built on `google.adk` — LlmAgent, SequentialAgent, ToolContext, CallbackContext, AgentTool. This creates vendor lock-in:

- Agent orchestration logic is ADK-specific
- State management uses ADK's `CallbackContext`
- Tool registration uses ADK's `AgentTool`
- Model initialization uses ADK's `LiteLlm` wrapper

If Google changes ADK APIs, deprecates it, or pricing changes, migration would require rewriting every agent file.

### 1.5 Single Model Configuration

**Severity: High**

A single model is configured globally via environment variable `AGENT_MODEL`:

```python
gptmodel = LiteLlm(
    model=os.environ.get("AGENT_MODEL"),
    ...
    num_retries=10
)
```

Every agent — from simple routing to complex facet mapping — uses the same model. There is no:

- Model routing (use a cheaper model for simple tasks)
- Model fallback (try model A, fall back to model B)
- Task-specific model selection
- Cost-aware model switching

**Impact**: Cost is maximized because every call uses the most expensive model. No fallback strategy if the primary model has issues.

### 1.6 No Multi-Tenant Support

**Severity: High**

The system has minimal tenant awareness. While `FACET_USER_RESTRICTIONS` and `FACET_KEY_IDENTIFIER` provide basic data filtering, there is no:

- Tenant-specific prompt configurations
- Tenant-specific model selection
- Tenant data isolation at the application level
- Tenant-specific skill enablement
- Per-tenant rate limiting or cost tracking
- Tenant-specific evaluation suites

The facet catalog is loaded from a single pickle file path hardcoded per key type:

```python
if facet_key == 'email_mobile':
    data_file_name = 'facet_catalog_email_mobile_data.pkl'
else:
    data_file_name = 'facet_catalog_cbb_id_data.pkl'
```

**Impact**: Cannot serve multiple business units with different configurations. Cannot customize behavior per tenant without code changes.

---

## 2. Prompt Design Flaws

### 2.1 Prompts Are Monolithic Text Blobs

**Severity: High**

There are 23 prompt files in `prompts/`, all loaded as raw text strings at module import time:

```python
with open(os.getcwd()+'/agentic_framework/prompts/route_agent_prompt.txt', 'r', encoding='utf-8') as file:
    ROOT_AGENT_INSTRUCTION = file.read()
```

Prompts are not:
- Versioned (no version tracking, no rollback)
- Templated (variables injected via simple `.replace()`)
- Tested (no prompt unit tests)
- Modular (each prompt is a self-contained wall of text)

### 2.2 String Replacement Injection Pattern

**Severity: High**

Prompt construction relies on naive string replacement:

```python
prompt = prompt.replace('{user_query}', user_query)
               .replace('{conversational_history}', history)
               .replace('{facet_metadata}', metadata)
```

This pattern has problems:
- No escaping — user input could contain `{template_vars}` causing silent corruption
- No validation that all placeholders were replaced
- No type checking on injected values
- Prompt mutation via paraphrase accumulation across turns

### 2.3 Static vs Dynamic Content Not Separated

**Severity: High**

System instructions, operational rules, output schemas, few-shot examples, and runtime context are all mixed together in single prompt files. The router prompt, for instance, contains:
- Agent identity and purpose
- Intent classification rules
- Output format requirements
- Delegation instructions
- Safety/compliance rules
- Contextual catalog descriptions

When any piece needs updating, the entire prompt must be edited and redeployed.

### 2.4 No Prompt Optimization Pipeline

**Severity: Medium**

Prompts were hand-written and hand-tuned. There is no:
- DSPy-style automated prompt optimization
- A/B testing framework for prompt variants
- Prompt performance tracking over time
- Automated regression detection when prompts change

### 2.5 No Structured Output Enforcement at Prompt Level

**Severity: Medium**

While Pydantic models exist for validation (`nsc_segmentation_response.py`, `facet_value_operator_response.py`), the prompts don't leverage model-native structured output modes (e.g., JSON mode, tool calling for structured responses). Instead, prompts ask for JSON in free text and then parse the response.

---

## 3. Evaluation Gaps

### 3.1 Evaluation Framework Exists but Is Disconnected

**Severity: High**

The `evaluations/` directory contains a substantial framework:
- CLI tool (`cli.py`) with `create-eval-set`, `run-evaluations`, `generate-report` commands
- Comparators (JSON rules, semantic similarity)
- Paraphraser for robustness testing
- UI with 3 screens (set creator, runner, analytics)

However, this framework is **disconnected from the deployment pipeline**. There is no:
- Eval gate in CI/CD (evaluations don't block deployments)
- Automated regression testing on prompt changes
- Continuous evaluation in production
- Per-skill evaluation suites
- Evaluation-driven prompt optimization loop

### 3.2 No Online Evaluation

**Severity: High**

Once deployed, there is no mechanism to:
- Track segment quality in production
- Detect model degradation over time
- Monitor hallucination rates
- Measure user satisfaction or correction rates
- Compare model versions in production (shadow mode)

### 3.3 Limited Evaluation Metrics

**Severity: Medium**

The evaluation framework focuses on:
- JSON structural comparison (are keys present?)
- Semantic similarity (embedding cosine distance)

Missing evaluation dimensions:
- Factual grounding (did the agent use real facet values?)
- Logical consistency (does the ruleSet match subSegments?)
- Operator correctness (is the operator appropriate for the facet type?)
- Edge case handling (dates, numeric ranges, empty results)
- Latency benchmarks
- Cost per query tracking

### 3.4 No Eval-First Development Workflow

**Severity: High**

The current workflow is: code → prompt → test manually → deploy. The enterprise workflow should be: define eval → write prompt → pass eval → deploy. Without eval-first development, there is no objective quality bar for changes.

---

## 4. Scalability Limits

### 4.1 Facet Catalog as Pickle Files

**Severity: High**

The facet catalog is loaded from local pickle files:

```python
self._FACET_CACHE_PATH = os.getcwd()+facet_cache_path+data_file_name
facet_catalog = self.read_dataframe_from_pickle(self._FACET_CACHE_PATH)
```

Problems:
- Pickle files are loaded into memory on every request (via `MetaData()` instantiation)
- No cache invalidation strategy
- No versioning of facet catalogs
- Cannot update facets without redeployment
- Memory footprint scales with catalog size
- No partitioning for multi-tenant scenarios

### 4.2 Sequential Agent Pipeline

**Severity: High**

The NSC pipeline is strictly sequential:

```
Decompose → NER → Date Tag → Facet Map → Format Generate
```

Each step must complete before the next begins. For complex queries with many sub-segments, the facet mapping step makes multiple sequential Milvus calls. There is no:
- Parallel processing of independent sub-segments
- Streaming of partial results
- Batched vector search operations
- Concurrent sub-agent execution

### 4.3 Database Connection Pool Limits

**Severity: Medium**

PostgreSQL connection pool is configured with minimal capacity:

```python
min_size=1, max_size=3
```

With keepalive pings every 240 seconds. Under concurrent load, this pool will be exhausted quickly, causing request queueing.

### 4.4 No Horizontal Scaling Strategy

**Severity: Medium**

While Kubernetes deployment exists (kitt.yml with min=1, max=3 replicas), the application has stateful components:
- In-memory pickle file loading
- Session state in PostgreSQL (shared across replicas — works)
- Milvus connections (shared — works)
- No request routing awareness

### 4.5 Vector Search Not Optimized for Scale

**Severity: Medium**

Milvus queries are made per sub-segment, per facet type. For a query like "customers who bought electronics online in the last 30 days and live in California," this means:
- 3+ sub-segments × multiple facet searches each
- Each search requires embedding generation + Milvus query
- No batching, no caching of similar queries

---

## 5. Reliability Problems

### 5.1 eval() Usage in Production Code

**Severity: Critical**

The `initialize_state` function uses `eval()` on an environment variable:

```python
callback_context.state[FACET_USER_RESTRICTIONS] = callback_context.state.get(
    'facet_user_restrictions', eval(os.environ.get('DEFAULT_USER_RESTRICTIONS'))
)
```

Using `eval()` on environment variables is a code injection vulnerability. If the environment variable is malformed or maliciously set, arbitrary code execution occurs.

### 5.2 No Retry Strategy Beyond LLM Calls

**Severity: High**

The LLM model has 10 retries configured:

```python
gptmodel = LiteLlm(model=..., num_retries=10)
```

But there are no retry strategies for:
- Milvus vector search failures
- PostgreSQL connection failures
- Facet catalog loading errors
- Network timeouts between services

### 5.3 Silent State Corruption

**Severity: High**

State is a flat dictionary with string keys. There is no validation that:
- Required state keys exist before agent execution
- State values match expected types
- State transitions are valid (e.g., can't run FacetMapper before Decomposer)

A typo in a state key (e.g., `FACET_CLASSIFIER_DEPENDECY_ALL_INDEX` — note the typo "DEPENDECY" — which is actually in the codebase) causes silent failures.

### 5.4 No Circuit Breaker Pattern

**Severity: Medium**

If Milvus, PostgreSQL, or the LLM provider goes down, the system will:
- Retry 10 times for LLM calls
- Fail with unhandled exceptions for other services
- Return a generic error to the user

There are no circuit breakers to prevent cascading failures or graceful degradation.

### 5.5 Ambiguity Handling Is Fragile

**Severity: Medium**

The ambiguity detection and resolution system relies on:
1. LLM setting `ambiguity_exists="1"` in response
2. System storing the clarification question in state
3. User providing clarification on next turn
4. LLM incorporating clarification

If the LLM fails to detect ambiguity, or the clarification loop gets stuck (user provides unhelpful response), there is no escape mechanism. No maximum retry count for clarification loops. No automatic fallback to "best guess" after N attempts.

### 5.6 Hardcoded File Paths

**Severity: Medium**

Throughout the codebase, paths are constructed with `os.getcwd()`:

```python
with open(os.getcwd()+'/agentic_framework/prompts/route_agent_prompt.txt', 'r')
```

This breaks if the working directory changes, and it prevents proper containerized deployment where paths should be configurable.

---

## 6. Missing Capabilities

### 6.1 No Memory System

**Severity: Critical**

The system has conversational history (short-term, within a session) but no long-term memory:

- No storage of successful segmentation patterns
- No learning from user corrections
- No recall of previous segment definitions
- No tenant-specific preference tracking
- No "recipes" for common segmentation tasks
- No memory of which facet mappings worked well

Every session starts from scratch. The agent cannot say "Last time you asked about electronics customers, you used these facets..."

### 6.2 No Auto-Improvement Loop

**Severity: Critical**

There is no mechanism for the system to improve over time:

- No feedback collection (was the segment useful? accurate?)
- No prompt optimization based on outcomes
- No A/B testing of agent behaviors
- No error analysis feeding back into prompt updates
- No self-reflection or verification steps
- No "did this segment actually perform well?" tracking

### 6.3 No Plan-Act-Verify-Improve Loop

**Severity: High**

The current flow is: Receive → Route → Execute → Return. There is no:

- **Plan step**: Agent doesn't explain its strategy before executing
- **Verify step**: No post-execution validation of the segment
- **Improve step**: No self-correction if output looks wrong
- **Explain step**: No reasoning chain for why specific facets were chosen

### 6.4 No Hypothesis Assessment

**Severity: High**

The system cannot:
- Evaluate a user's hypothesis ("I think young professionals buy more organic food")
- Suggest alternative segmentation approaches
- Compare different segmentation strategies
- Reason about segment overlap or exclusivity
- Recommend segment expansion or refinement

### 6.5 No Model Exploration and Recommendation

**Severity: High**

The system cannot:
- Assess existing ML models in the user's ecosystem
- Recommend building new predictive models
- Suggest features for model training based on segmentation patterns
- Connect segment definitions to model performance data

### 6.6 No Knowledge / RAG Integration

**Severity: High**

The system does not have enterprise knowledge retrieval:
- No access to business context (what campaigns are running?)
- No understanding of business rules (compliance, regional regulations)
- No product taxonomy awareness beyond facet catalog
- No integration with CRM or marketing platform data
- No ability to cite evidence for its recommendations

### 6.7 No Observability Beyond Basic Tracing

**Severity: Medium**

Phoenix tracing provides basic request tracing, but there is no:
- Token usage tracking per request
- Cost attribution per agent/skill
- Latency breakdown per pipeline step
- Quality metrics dashboard
- Alert system for anomalies
- A/B test result monitoring

### 6.8 No Campaign / Marketing Integration

**Severity: Medium**

The system creates segments but cannot:
- Recommend how to use the segment
- Suggest marketing campaign parameters
- Estimate segment reach or value
- Connect to campaign execution systems
- Track segment performance post-deployment

---

## 7. Security Concerns

### 7.1 eval() on Environment Variables

As noted in 5.1, `eval(os.environ.get('DEFAULT_USER_RESTRICTIONS'))` is a code injection vector.

### 7.2 No Input Sanitization

**Severity: High**

User queries are passed directly into prompts via string replacement without sanitization. Prompt injection attacks could:
- Override system instructions
- Extract facet catalog data
- Cause the agent to produce unauthorized outputs

### 7.3 Debug Print Statements in Production

**Severity: Medium**

The codebase contains extensive `print()` statements with ANSI color codes:

```python
print('\033[1;32m'+"========== FVOM (LLM SHORTLIST) =========="+'\033[0m'+'\n')
print("[INFO] - Sub Segment Facet to Value Pair from LLM : ",segments)
```

These leak internal state to stdout in production and could expose sensitive data in logs.

### 7.4 No Rate Limiting

**Severity: Medium**

No per-user, per-session, or per-tenant rate limiting exists. A single user could exhaust LLM quota for all users.

---

## 8. Cost & Performance Issues

### 8.1 All Calls Use Same Model

As noted in 1.5, using the most capable model for every call (routing, decomposition, NER, date extraction, facet mapping, format generation) is cost-inefficient. Routing and date extraction could use a much cheaper/faster model.

### 8.2 No Response Caching

**Severity: High**

Identical or similar queries are processed from scratch every time. There is no:
- Semantic query cache (similar queries → cached results)
- Facet mapping cache (same sub-segment → same facets)
- Embedding cache (same text → same vector)

### 8.3 Large Prompt Token Overhead

**Severity: Medium**

The facet catalog contextual information is injected into multiple prompts. This static content consumes tokens on every call. A RAG approach (retrieve only relevant facets) would significantly reduce per-call token usage.

### 8.4 Sequential Processing Maximizes Latency

**Severity: Medium**

The pipeline is fully sequential. Total latency = sum of all agent call latencies. For a typical query:
- Router: ~1s
- Decomposer: ~2s
- Date Tagger: ~1s
- Facet Mapper: ~3-5s (multiple vector searches + LLM call)
- Format Generator: ~2s
- Total: ~9-11s minimum

Parallel execution of independent sub-segments could reduce this significantly.

---

## 9. Summary Severity Matrix

| # | Issue | Severity | Category | Fix Effort |
|---|-------|----------|----------|------------|
| 1.1 | Monolithic agent coupling | Critical | Architecture | High |
| 1.2 | God state object | Critical | Architecture | Medium |
| 1.3 | No plugin/skill architecture | Critical | Architecture | High |
| 1.4 | Tight coupling to Google ADK | High | Architecture | High |
| 1.5 | Single model configuration | High | Architecture | Medium |
| 1.6 | No multi-tenant support | High | Architecture | High |
| 2.1 | Monolithic prompt text blobs | High | Prompt Design | Medium |
| 2.2 | String replacement injection | High | Prompt Design | Low |
| 2.3 | Static/dynamic content not separated | High | Prompt Design | Medium |
| 2.4 | No prompt optimization | Medium | Prompt Design | Medium |
| 2.5 | No structured output enforcement | Medium | Prompt Design | Low |
| 3.1 | Evaluation framework disconnected | High | Evaluation | Medium |
| 3.2 | No online evaluation | High | Evaluation | High |
| 3.3 | Limited evaluation metrics | Medium | Evaluation | Medium |
| 3.4 | No eval-first workflow | High | Evaluation | Medium |
| 4.1 | Facet catalog as pickle files | High | Scalability | Medium |
| 4.2 | Sequential agent pipeline | High | Scalability | Medium |
| 4.3 | DB connection pool limits | Medium | Scalability | Low |
| 4.4 | No horizontal scaling strategy | Medium | Scalability | Medium |
| 4.5 | Vector search not optimized | Medium | Scalability | Medium |
| 5.1 | eval() usage | Critical | Reliability | Low |
| 5.2 | No retry strategy beyond LLM | High | Reliability | Low |
| 5.3 | Silent state corruption | High | Reliability | Medium |
| 5.4 | No circuit breaker | Medium | Reliability | Low |
| 5.5 | Fragile ambiguity handling | Medium | Reliability | Medium |
| 5.6 | Hardcoded file paths | Medium | Reliability | Low |
| 6.1 | No memory system | Critical | Missing | High |
| 6.2 | No auto-improvement loop | Critical | Missing | High |
| 6.3 | No Plan-Act-Verify-Improve loop | High | Missing | Medium |
| 6.4 | No hypothesis assessment | High | Missing | High |
| 6.5 | No model exploration | High | Missing | High |
| 6.6 | No knowledge/RAG integration | High | Missing | High |
| 6.7 | No observability dashboard | Medium | Missing | Medium |
| 6.8 | No campaign integration | Medium | Missing | High |
| 7.1 | eval() injection | Critical | Security | Low |
| 7.2 | No input sanitization | High | Security | Low |
| 7.3 | Debug prints in production | Medium | Security | Low |
| 7.4 | No rate limiting | Medium | Security | Low |
| 8.1 | Same model for all calls | High | Cost | Medium |
| 8.2 | No response caching | High | Cost | Medium |
| 8.3 | Large prompt token overhead | Medium | Cost | Medium |
| 8.4 | Sequential processing | Medium | Cost | Medium |

### Severity Summary

| Severity | Count |
|----------|-------|
| Critical | 7 |
| High | 21 |
| Medium | 13 |
| **Total** | **41** |

### Top 10 Priority Fixes (by Impact)

1. **Remove eval() usage** — Security fix, 1 hour
2. **Introduce structured state management** — Replace flat dict with typed Pydantic state
3. **Add input sanitization** — Prevent prompt injection
4. **Build skill registry** — Enable pluggable capabilities
5. **Implement model routing** — Right model for the right task
6. **Add memory system** — Long-term learning and personalization
7. **Create eval gates in CI/CD** — Quality enforcement before deployment
8. **Separate static vs dynamic prompt content** — Modular prompt architecture
9. **Add response caching** — Reduce cost and latency
10. **Implement Plan-Act-Verify-Improve loop** — Agent reliability

---

*Built for the Enterprise Agentic Research Initiative*
*Last Updated: February 2026*
