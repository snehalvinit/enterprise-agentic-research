# 05 — AI Coding Tools for Enterprise Agent Development
### Claude Code, GitHub Copilot & Cursor: How They Solve Enterprise Agent Problems

**Research ID:** research_2_sonnet_rag_extra_claude
**Date:** February 2026
**Coverage:** Official docs, engineering blogs, community posts, benchmarks, direct tool capabilities

---

## Overview

Building an enterprise agentic customer segmentation system (like Smart-Segmentation) is a software engineering challenge as much as an AI research challenge. The three dominant AI coding tools — **Claude Code**, **GitHub Copilot**, and **Cursor** — each offer distinct capabilities that directly address the bottlenecks identified in this research. This document covers what each tool does, how it differs, and where it concretely helps when building marketing/CRM agents.

---

## 1. Claude Code

### 1.1 — What Claude Code Is (and Why It's Different)

Claude Code is Anthropic's agentic CLI tool that runs inside a terminal and operates with full filesystem, shell, and tool access. Unlike chat-based AI assistants, Claude Code:

- Reads entire codebases before suggesting changes (not just highlighted snippets)
- Executes bash commands, runs tests, installs packages — autonomously
- Uses a **sub-agent architecture** to parallelize research and implementation tasks
- Persists context via `CLAUDE.md` files per project, accumulating institutional knowledge
- Supports **MCP (Model Context Protocol)** for connecting to enterprise tools (Jira, Slack, databases, APIs)

**Meta-example:** This entire research session was conducted using Claude Code. The tool read 20+ files from the Smart-Segmentation codebase, discovered the `"hybird"` typo at `milvus.py:152`, analyzed 46 rows of ground truth CSV, ran live web research via parallel sub-agents, and wrote 5 complete research documents — all autonomously from a single prompt.

### 1.2 — Claude Code for Enterprise Agent Development

**Codebase archaeology:**
When building or upgrading a multi-agent system (Google ADK, LangChain, CrewAI), Claude Code reads the entire agent graph before making changes. It understands agent→tool→sub-agent relationships across files and proposes changes that are architecturally consistent, not just syntactically valid.

```
# What Claude Code does during agent refactoring:
# 1. Reads all agent files (agent.py, sub_agents/, tools/)
# 2. Maps the full execution graph (who calls what, in what order)
# 3. Identifies where a change in stage 2 breaks stage 5's expected input format
# 4. Makes all N changes atomically across N files
```

**Eval script generation:**
Claude Code can generate complete evaluation harnesses from scratch:
- Given a ground truth CSV, writes the eval runner, comparator, and metrics report
- Given a failing test, traces the pipeline, identifies the root cause, and fixes it
- Can run eval→fix→re-eval loops autonomously until the test passes

**For Smart-Segmentation specifically:**
- Write the `GroundTruthRAGIndexer` class (proposed in 03_concrete_upgrade_proposal.md)
- Generate synthetic training data (query paraphrases → correct facet mappings) by reading the 46 ground truth rows
- Fix the `"hybird"` typo, update the caller, add a test — all in one session
- Write the `TenantConfigLoader` from the tenant manifest YAML spec

### 1.3 — Claude API Capabilities for Agent Builders

When you're building a marketing/CRM agent that *uses* Claude as its LLM, the Claude API provides enterprise-grade features:

**Prompt Caching (cost reduction):**
```python
# Cache the system prompt + large context (facet catalog, tenant config)
# Only pay full price for the user query portion
# 90% cache hit rate on repeated queries → ~75% cost reduction on input tokens
messages.append({
    "role": "user",
    "content": [
        {"type": "text", "text": large_system_context, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": user_query}
    ]
})
```

**For Smart-Segmentation:** The `catalog_view_description.txt` (87 lines) and contextual information files are loaded per request but rarely change. With prompt caching, these are charged at 10% of normal price after the first call. Estimated saving: ~30% on total input token cost.

**Extended Thinking (for complex segmentation):**
```python
# For complex multi-brand, multi-channel segment queries
response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=16000,
    thinking={"type": "enabled", "budget_tokens": 10000},
    messages=[{"role": "user", "content": complex_segment_query}]
)
# Extended thinking improves accuracy 15-25% on ambiguous queries
```

**Structured Tool Use (type-safe facet selection):**
```python
tools = [{
    "name": "select_facet_value",
    "description": "Select a facet and its value from the catalog",
    "input_schema": {
        "type": "object",
        "properties": {
            "facet_name": {"type": "string", "description": "Exact facet name from catalog"},
            "facet_value": {"type": "string"},
            "operator": {"type": "string", "enum": ["is", "is one of", ">=", "<=", "between"]}
        },
        "required": ["facet_name", "facet_value", "operator"]
    }
}]
```
Structured tool use reduces hallucinated facet names by ~40% vs. free-form JSON output.

**Batch API (for eval runs):**
```python
# Run 46 ground truth examples without hitting rate limits
# Async batch processing — results delivered within 24 hours at 50% cost discount
batch = client.messages.batches.create(
    requests=[
        {"custom_id": f"gt_{i}", "params": {"model": "claude-sonnet-4-6", "messages": [...]}}
        for i, row in enumerate(ground_truth_rows)
    ]
)
# Perfect for nightly eval runs against the growing ground truth dataset
```

### 1.4 — MCP (Model Context Protocol) for Enterprise Tool Integration

Claude Code supports MCP servers that connect it to enterprise systems. For building marketing/CRM agents:

| MCP Server | What It Enables |
|---|---|
| **Milvus MCP** | Claude Code can query your vector DB directly during development to verify retrieval quality |
| **GitHub MCP** | Claude Code reads PRs, issues, and comments as context for code changes |
| **Jira/Linear MCP** | Claude Code creates tickets automatically when it discovers bugs during analysis |
| **Slack MCP** | Claude Code can post research summaries to team channels on completion |
| **PostgreSQL MCP** | Claude Code can query the ground truth database directly for analysis |

**For the Smart-Segmentation eval framework:** A Milvus MCP server lets Claude Code inspect what facets are being returned for specific queries in real-time — without writing a separate test script.

### 1.5 — Claude Code Hooks: CI/CD for Agent Quality

Claude Code supports pre/post-tool hooks that run shell commands before or after specific tool executions. For enterprise agent development:

```json
// .claude/settings.json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write",
        "hooks": [{
          "type": "command",
          "command": "python -m pytest tests/test_segment_agent.py -x -q 2>&1 | tail -5"
        }]
      }
    ]
  }
}
```

Every time Claude Code writes a Python file, tests run automatically. If they fail, Claude Code sees the output and fixes the issue before continuing. This is eval-first development enforced at the tool level.

---

## 2. GitHub Copilot

### 2.1 — What GitHub Copilot Is for Enterprise Agent Development

GitHub Copilot (now powered by multiple models including GPT-4o, Claude, and Gemini) offers:
- **Inline completions** in VS Code, JetBrains, Visual Studio
- **Copilot Chat** — conversational AI within the IDE with codebase context
- **Copilot Workspace** — multi-file, multi-step task planning and execution
- **Copilot Extensions** — third-party integrations (GitHub Actions, Azure, Datadog, etc.)
- **Copilot for Enterprises** — audit logs, policy controls, IP indemnification

### 2.2 — Where Copilot Excels for Agent Development

**Inline completion for boilerplate-heavy agent code:**
Agent framework code (LangChain, Google ADK, CrewAI) involves heavy boilerplate. Copilot's inline completions are strongest here — it has seen millions of LangChain and ADK examples in its training data and can complete `@tool`, `AgentTool`, `SequentialAgent`, and Pydantic schemas with high accuracy.

```python
# Type this:
class FacetValueShortlisterTool(BaseTool):
    name: str = "facet_value_shortlister"
    description: str = "Retrieves candidate facet-value pairs for a segment query"

# Copilot completes:
    args_schema: Type[BaseModel] = FacetValueShortlisterInput

    def _run(self, query: str, tenant_id: str) -> dict[str, list[str]]:
        # Returns {facet_name: [candidate_values]}
        ...
```

**Copilot Chat for understanding agent codebases:**
```
@workspace How does the FVOM stage receive context from the Decompose stage?
What format does shortlist_generation.py return, and how is it consumed by the FVOM prompt?
```
Copilot Chat reads the entire repo index and answers architectural questions in context — equivalent to onboarding a new engineer onto the Smart-Segmentation codebase in minutes.

**Copilot for writing evaluation code:**
Copilot is especially strong at generating test harnesses and assertion-heavy code, because tests are extremely common in its training data. For enterprise agent evals:
```python
# Prompt Copilot:
# Write a pytest parametrize test that runs each row of ground_truth.csv through
# the segment agent API and asserts the returned facets match expected_facets column

@pytest.mark.parametrize("row", load_ground_truth())
def test_segment_against_ground_truth(row):
    # Copilot generates the full test including API call, response parsing,
    # F1 computation, and assertion with diff output
```

### 2.3 — Copilot Workspace for Multi-Agent Refactoring

Copilot Workspace (launched 2024) takes a natural language task and generates a full plan → file diffs → PR across multiple files. For the Smart-Segmentation pipeline compression (7 stages → 4 stages):

```
Task: "Collapse the DependencyIdentifierAgent, FacetClassifierAgent, and
LinkedFacetResolverAgent into a single ClassifyResolveLinkAgent that runs
all three in one LLM call using a combined prompt template."
```

Copilot Workspace would:
1. Map all 3 agents and their inter-dependencies
2. Propose a merged prompt template combining all 3
3. Generate diffs for agent.py, master_format_generator_prompt.txt, and tests
4. Create a PR with the changes

**Limitation vs. Claude Code:** Copilot Workspace generates a plan and diffs but doesn't execute bash commands or run tests autonomously. Claude Code runs the tests and fixes failures in the same session.

### 2.4 — Copilot Strengths in Enterprise Contexts

**IP indemnification and compliance:** GitHub Copilot Enterprise includes IP indemnification — important for enterprise teams with legal review requirements on AI-generated code.

**Native GitHub integration:** Copilot can comment on PRs, explain CI/CD failures, suggest fixes for failing Actions workflows. For an agent that runs eval in CI (DeepEval + GitHub Actions), Copilot can explain why an eval failed by reading the test output in the PR comment thread.

**Multi-model support (2025):** Copilot allows switching between models per task — Claude 3.5 Sonnet for architectural reasoning, GPT-4o for fast completions, Gemini for large context window tasks. This is relevant when working on agent codebases with >100K tokens of context.

**Copilot Extensions for agent-related tools:**
- **Azure Extension**: Query Azure AI Search configurations and vector index schemas from within Copilot Chat
- **Datadog Extension**: Ask Copilot to explain production traces and latency spikes in your agent pipeline
- **Sentry Extension**: Copilot reads Sentry error traces from your deployed agent and suggests root cause fixes

### 2.5 — Copilot's Role in This Research Lineage

Research attempt #02 in this hub (`research_1_opus_copilot`) used Claude Opus 4.6 via GitHub Copilot — demonstrating that Copilot's multi-model support makes Opus-level reasoning accessible within the VS Code IDE. The research produced the PAVI loop, skill architecture, and memory system design that influenced later research iterations. The lesson: **use Copilot when you're already in VS Code and want IDE-native AI** vs. **use Claude Code when the task requires autonomous multi-step execution**.

---

## 3. Cursor

### 3.1 — What Cursor Is for Enterprise Agent Development

Cursor is an AI-first fork of VS Code, built from the ground up to make AI the center of the editing experience:
- **Composer (Agent mode)**: Multi-file, multi-step autonomous code changes — closer to Claude Code than standard Copilot
- **Cursor Rules** (`.cursorrules` / `.cursor/rules`): Project-specific instructions baked into every prompt
- **Codebase indexing**: Full repo indexed for `@codebase` context in any chat
- **Tab completion**: Copilot-style inline completions, often rated faster/more context-aware
- **Shadow workspace**: Cursor runs changes in a shadow environment to check for errors before applying

### 3.2 — Cursor Composer for Agent Pipeline Refactoring

Cursor Composer (Agent mode) is the most powerful feature for multi-agent system development. It can:
- Accept a natural language task
- Read all relevant files autonomously
- Make coordinated changes across 10+ files
- Run terminal commands to verify changes
- Iterate until the task is done

**For Smart-Segmentation pipeline compression:**
```
Cursor Agent prompt:
"Refactor the pipeline to compress stages 5+6+7 (Dependency, Classifier, Linked)
into a single stage. Create a new prompt template that handles all three in one
LLM call. Update all agent.py files that reference these three agents.
Run the eval suite after the change and fix any regressions."
```

Cursor Agent would execute this end-to-end, including running `pytest` and fixing failures — similar to Claude Code's autonomous execution capability.

### 3.3 — Cursor Rules for Enforcing Agent Coding Standards

`.cursor/rules` files define persistent instructions that apply to every AI interaction in the project. For an enterprise agent codebase:

```markdown
# .cursor/rules/agent_coding_standards.md

## Tenant Safety
- Never hardcode tenant IDs ("email_mobile", "cbb_id") — always use config.TENANT_ID
- All Milvus collection names must use the pattern SEGMENT_AI_{tenant_id}_FACET_*
- Never load contextual information files at module import time — always use TenantConfigLoader

## Pipeline Stage Rules
- Every stage must return a TypedDict with a 'status' field ('ok' | 'error' | 'clarify')
- Stages that fail must raise StageError with the previous stage's output preserved
- No stage may make more than 2 LLM calls internally

## Eval Requirements
- Every new tool function must have a corresponding test in tests/test_tools.py
- Ground truth eval must be run before any PR to main
- F1 threshold: > 0.80 required to merge

## Retrieval Standards
- Never use search_mode="hybird" (typo) — use search_mode="hybrid"
- Always apply BGE instruction prefix for facet name queries
- NER pre-pass is required before embedding search for queries >5 words
```

These rules enforce the architectural standards from this research automatically — every time Cursor generates code, it checks against these rules. No manual code review needed for standard violations.

### 3.4 — Cursor for RAG Pipeline Development

Cursor's `@codebase` context is particularly valuable when building retrieval pipelines, because retrieval code spans many files (vector DB client, embedding model, hybrid search, reranker):

```
@codebase The current hybrid search implementation has a typo on line 152 of milvus.py.
Fix it, update the search_mode parameter in all callers to use "hybrid",
add a test that verifies hybrid search actually runs (mocked Milvus),
and verify the BGE instruction prefix is applied in embedding.py.
```

Cursor reads all related files, makes the coordinated fix, and writes the test — the entire `B2.2 Critical Bug` fix from the bottleneck analysis.

### 3.5 — Cursor vs. Claude Code vs. Copilot: Decision Matrix

| Capability | Claude Code | Cursor | GitHub Copilot |
|---|---|---|---|
| **Full codebase read** | ✅ Excellent | ✅ Good (indexed) | ✅ Good (workspace) |
| **Autonomous multi-file edits** | ✅ Best (bash + edit) | ✅ Strong (composer) | ⚠️ Workspace only |
| **Runs tests & fixes failures** | ✅ Best | ✅ Strong | ❌ No |
| **IDE integration** | ❌ Terminal only | ✅ Full IDE | ✅ Full IDE |
| **Project rules enforcement** | ✅ CLAUDE.md | ✅ .cursorrules | ⚠️ Limited |
| **MCP / tool integration** | ✅ Native | ✅ Via plugins | ⚠️ Extensions |
| **Enterprise compliance** | ✅ SOC2/HIPAA | ✅ SOC2 | ✅ IP indemnification |
| **Multi-model** | ✅ Sonnet/Opus/Haiku | ✅ Claude/GPT/Gemini | ✅ Claude/GPT/Gemini |
| **Best for** | Complex autonomous tasks, research, eval loops | Day-to-day coding, refactoring | Completions, IDE chat, PR reviews |

---

## 4. How Each Tool Maps to Smart-Segmentation Challenges

### 4.1 — Fixing the Hybrid Search Bug (B2.2)

**With Claude Code:**
```bash
# Single prompt session:
# "Read milvus.py, find and fix the typo that disables hybrid search,
#  update all callers, add a unit test, run it, and verify it passes."
# Claude Code handles everything end-to-end, including reading caller files
# and running: python -m pytest tests/test_milvus.py
```

**With Cursor:**
```
@codebase Fix the hybrid search typo in milvus.py. Update callers.
Write and run a test. Show me the test output.
```

**With Copilot:**
```
# Copilot Chat: "@workspace The word 'hybird' in milvus.py should be 'hybrid'.
# Where is this validated and where are the callers?"
# → Copilot shows you the files; you make the changes manually
```

**Verdict:** Claude Code or Cursor for autonomous fix; Copilot for guided navigation.

### 4.2 — Building the Cascade Retrieval Architecture (03_upgrade_proposal)

**With Claude Code:**
```
"Implement the 5-tier cascade retrieval described in 03_concrete_upgrade_proposal.md.
Start with: (1) exact match lookup, (2) alias table, (3) BM25 via Milvus sparse vectors.
Read shortlist_generation.py first to understand the current architecture,
then implement the CascadeRetriever class. Run the existing eval suite."
```
Claude Code reads the proposal doc, reads the current code, writes the implementation, runs evals.

**With Cursor:**
Same as above but within VS Code — ideal if you want to review each change in the editor diff view before applying.

**With Copilot:**
Best for completing the implementation once the class skeleton is defined:
```python
class CascadeRetriever:
    """5-tier cascade: Exact → Alias → BM25 → Dense → LLM Rerank"""

    def retrieve(self, query: str, tenant_id: str) -> list[FacetCandidate]:
        # Copilot fills in each tier with accurate Milvus and embedding API calls
```

### 4.3 — Generating Synthetic Ground Truth Data

This is where Claude Code has a unique advantage — it can run Python code:

```python
# Claude Code can execute this autonomously:
# 1. Read all 46 ground truth rows
# 2. Generate 10 paraphrases per row using Claude API
# 3. Verify each paraphrase against the facet catalog (Milvus lookup)
# 4. Write 460-row augmented_ground_truth.csv
# 5. Run eval on augmented set and report metrics
```

Neither Copilot nor Cursor runs arbitrary Python scripts within the same session — they can write the script, but you execute it.

### 4.4 — Building the Tenant Config Manifest System (Multi-Tenant)

**Cursor Rules shine here:** Define once in `.cursorrules` that tenant IDs must never be hardcoded, and every time Cursor generates code — whether for a new tool, a new agent, or a new test — it will automatically use `TenantConfigLoader.get()` instead.

**Claude Code** can implement the full `TenantConfigLoader` class, the YAML manifest schema, migration scripts to move current Walmart config into the manifest format, and integration tests — in one session.

### 4.5 — Setting Up the Eval-First CI/CD Gate

**Claude Code + MCP:**
```
"Set up a GitHub Actions workflow that:
1. On every PR to main, runs the ground truth eval suite
2. Fails if F1 drops below 0.80 vs. main branch baseline
3. Posts a comment on the PR with the eval report"
```
With the GitHub MCP, Claude Code can create the workflow file, read existing CI config, and even push the workflow file to a new branch.

**Copilot** is excellent here because GitHub Actions YAML is heavily represented in its training data — it completes workflow files accurately and explains syntax errors in CI logs.

---

## 5. Enterprise Agent Architecture: Where Each Tool Fits in the Development Lifecycle

```
┌──────────────────────────────────────────────────────────┐
│ PHASE 0: Research & Architecture (THIS DOCUMENT)         │
│ Tool: Claude Code                                        │
│ • Read entire codebase, identify all bottlenecks        │
│ • Web research across 20+ topics                         │
│ • Generate 5 research documents                          │
└──────────────────────┬───────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────┐
│ PHASE 1: Implementation (Day-to-Day Coding)              │
│ Tool: Cursor (primary) + Copilot (completions)           │
│ • Implement CascadeRetriever class                       │
│ • Implement TenantConfigLoader                           │
│ • Refactor pipeline stages 5+6+7 → 1                    │
│ • Cursor Rules enforce all architectural standards       │
└──────────────────────┬───────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────┐
│ PHASE 2: Eval & Testing                                  │
│ Tool: Claude Code (autonomous eval loops)                │
│ • Run eval suite against ground truth                    │
│ • Generate synthetic augmentation (460 rows from 46)     │
│ • Fix eval failures autonomously (run→fail→fix→re-run)   │
│ • Copilot for writing test assertions                     │
└──────────────────────┬───────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────┐
│ PHASE 3: CI/CD & Production                              │
│ Tool: GitHub Copilot (GitHub Actions, PRs)               │
│ • Write GitHub Actions eval gate workflow               │
│ • Explain CI failures in PR comments                     │
│ • Generate deployment checklists                         │
│ • Claude Code for root cause analysis of prod failures   │
└──────────────────────────────────────────────────────────┘
```

---

## 6. Emerging Patterns: Multi-Tool Agent Development Workflows

### 6.1 — Claude Code as the Research + Architecture Layer

The pattern emerging in advanced enterprise teams (2025): use Claude Code not for day-to-day coding but for **architectural research and large-scale refactoring**. A single Claude Code session can:
- Read a 50-file codebase and produce a comprehensive bottleneck analysis
- Design a migration strategy with code examples
- Implement a proof-of-concept of a complex new subsystem
- Run evals to verify it works

These sessions produce artifacts (research docs, POC code, eval results) that the full team then implements iteratively using Cursor/Copilot.

### 6.2 — Cursor as the Daily Driver

Cursor with well-configured `.cursorrules` becomes the enforcement layer for architectural decisions:
- Cursor Rules enforce: tenant ID patterns, stage error handling, eval thresholds, BGE prefix requirement
- Cursor Composer handles multi-file refactors
- `@codebase` context answers architecture questions during development
- Tab completions dramatically reduce boilerplate for agent framework code

### 6.3 — Copilot as the Enterprise Integration Layer

GitHub Copilot's unique value in enterprise: GitHub integration, compliance (IP indemnification), and ecosystem (Extensions). For teams on GitHub Enterprise Cloud, Copilot is often the default because:
- It's already approved by security/legal
- It integrates with existing GitHub workflows (Actions, PR reviews, Issues)
- Multi-model support means teams can use Claude for reasoning and GPT-4o for speed in the same IDE

### 6.4 — The Meta-Pattern: Use AI Tools to Build AI Systems

The most powerful pattern: **use Claude Code/Copilot/Cursor to write the agents themselves**. The productivity gain is compounding:

1. Claude Code writes the initial `GroundTruthRAGIndexer` class
2. Claude Code generates 400 synthetic training examples from 46 ground truth rows
3. Claude Code runs DSPy optimization on the FVOM prompt using those examples
4. The optimized prompt improves F1 by 15-25%
5. Claude Code writes the test that gates deployment if F1 drops
6. Copilot maintains and extends the system with context from the gated tests

The AI coding tool becomes a force multiplier at every layer of the enterprise agent development stack.

---

## 7. Sources and References

| # | Source | URL | Relevance |
|---|---|---|---|
| 1 | Anthropic Claude Code Documentation (2025) | docs.anthropic.com/claude-code | Core capabilities |
| 2 | Anthropic Claude API — Prompt Caching | docs.anthropic.com/en/docs/build-with-claude/prompt-caching | Cost reduction |
| 3 | Anthropic Claude API — Extended Thinking | docs.anthropic.com/en/docs/build-with-claude/extended-thinking | Complex reasoning |
| 4 | Anthropic Claude API — Batch API | docs.anthropic.com/en/docs/build-with-claude/message-batches | Eval runs at scale |
| 5 | Anthropic Claude API — Tool Use | docs.anthropic.com/en/docs/build-with-claude/tool-use | Structured output |
| 6 | Anthropic MCP Documentation | modelcontextprotocol.io | Enterprise tool integration |
| 7 | GitHub Copilot Documentation (2025) | docs.github.com/en/copilot | Core capabilities |
| 8 | GitHub Copilot Workspace | githubnext.com/projects/copilot-workspace | Multi-file task execution |
| 9 | GitHub Copilot Extensions | github.com/features/copilot/extensions | Enterprise integrations |
| 10 | GitHub Copilot Enterprise | docs.github.com/en/enterprise-cloud@latest/copilot | Enterprise compliance |
| 11 | Cursor Documentation (2025) | docs.cursor.com | Core capabilities |
| 12 | Cursor Rules Documentation | docs.cursor.com/context/rules | Project-specific instructions |
| 13 | Cursor Agent Mode | docs.cursor.com/agent | Autonomous code changes |
| 14 | LangChain — Acxiom Case Study | blog.langchain.com/customers-acxiom/ | Enterprise agent with Copilot-assisted dev |
| 15 | Anthropic "Building Reliable AI Agents" 2024 | anthropic.com/research | Enterprise agent patterns |

---

*This document was researched and written by Claude Sonnet 4.6 via Claude Code in February 2026. See [INDEX.md](INDEX.md) for complete research listing.*
