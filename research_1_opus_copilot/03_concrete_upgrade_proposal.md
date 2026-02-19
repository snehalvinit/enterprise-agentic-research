# Concrete Upgrade Proposal — Smart-Segmentation to Enterprise Agentic System

> **Research ID:** research_1_opus_copilot  
> **Date:** February 2026  
> **Model:** Claude Opus 4.6 via GitHub Copilot  
> **Scope:** Complete architectural transformation of Smart-Segmentation

---

## Executive Summary

This document proposes a systematic upgrade of Smart-Segmentation from its current state — a well-designed but prototype-grade multi-agent system — into an enterprise-grade agentic customer segmentation platform. The transformation preserves the existing strengths (ADK multi-agent hierarchy, Milvus RAG pipeline, Pydantic validation) while layering enterprise patterns for reliability, extensibility, cost efficiency, and auto-improvement.

The upgrade is organized into five architectural layers:

1. **Foundation Layer** — Security fixes, testing, observability
2. **Efficiency Layer** — Cost optimization, caching, connection management
3. **Reliability Layer** — Eval gates, structured validation, circuit breakers
4. **Extensibility Layer** — Skills, knowledge management, multi-tenancy
5. **Intelligence Layer** — Memory, auto-improvement, hypothesis assessment

---

## 1. High-Level Architecture

### 1.1 Current Architecture

```
┌───────────────────────────────────────────────┐
│                  FastAPI Server                │
│  ┌──────────┐  ┌────────────────────────────┐ │
│  │ Routes   │  │  Google ADK Runner          │ │
│  │ (641 ln) │──│  ┌──────────────────────┐   │ │
│  │          │  │  │ RouterAgent (LLM)    │   │ │
│  └──────────┘  │  │  ├── NewSegmentAgent │   │ │
│                │  │  │  │  ├── Tools     │   │ │
│  ┌──────────┐  │  │  │  │  └── SubAgent  │   │ │
│  │ Metadata │  │  │  ├── EditAgent       │   │ │
│  │ (Pickle) │  │  │  └── ...             │   │ │
│  └──────────┘  │  └──────────────────────┘   │ │
│                └────────────────────────────┘ │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │ Milvus   │  │ Postgres │  │ BGE Model  │  │
│  │ (no pool)│  │ (3 conn) │  │ (1.3GB RAM)│  │
│  └──────────┘  └──────────┘  └────────────┘  │
└───────────────────────────────────────────────┘
```

### 1.2 Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     API Gateway / Load Balancer                  │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                    Agent Service (FastAPI)                        │
│                                                                  │
│  ┌─────────────┐  ┌──────────────────────────────────────────┐  │
│  │ Request     │  │  Agent Orchestrator                       │  │
│  │ Handler     │──│  ┌────────────┐  ┌─────────────────────┐ │  │
│  │ (modular)   │  │  │ Router     │  │ Skill Registry      │ │  │
│  │             │  │  │ (economy   │──│ ┌─────────────────┐ │ │  │
│  │ ┌─────────┐ │  │  │  model)    │  │ │ create_segment  │ │ │  │
│  │ │ Input   │ │  │  └────────────┘  │ │ edit_segment    │ │ │  │
│  │ │ Validat.│ │  │                  │ │ assess_hypothes │ │ │  │
│  │ └─────────┘ │  │  ┌────────────┐  │ │ explore_models  │ │ │  │
│  │ ┌─────────┐ │  │  │ PAVI Loop  │  │ │ ...             │ │ │  │
│  │ │ Output  │ │  │  │ Plan-Act-  │  │ └─────────────────┘ │ │  │
│  │ │ Validat.│ │  │  │ Verify-    │  └─────────────────────┘ │  │
│  │ └─────────┘ │  │  │ Improve    │                          │  │
│  └─────────────┘  │  └────────────┘  ┌─────────────────────┐ │  │
│                   │                  │ Tool Registry (MCP)  │ │  │
│                   │  ┌────────────┐  │ ┌─────────────────┐ │ │  │
│                   │  │ Memory     │  │ │ milvus_search   │ │ │  │
│                   │  │ Manager    │  │ │ metadata_lookup │ │ │  │
│                   │  │ ┌────────┐ │  │ │ segment_format  │ │ │  │
│                   │  │ │ Core   │ │  │ │ model_registry  │ │ │  │
│                   │  │ │ Recall │ │  │ └─────────────────┘ │ │  │
│                   │  │ │Archive │ │  └─────────────────────┘ │  │
│                   │  │ └────────┘ │                          │  │
│                   │  └────────────┘                          │  │
│                   └──────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Shared Services                        │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │   │
│  │  │ Model    │  │ Cache    │  │ Eval     │  │ Config  │ │   │
│  │  │ Router   │  │ Layer    │  │ Runner   │  │ Service │ │   │
│  │  │ (tiered) │  │ (Redis)  │  │ (CI-int.)│  │ (tenant)│ │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────▼────────┐ ┌───────▼────────┐ ┌───────▼────────┐
│   Milvus       │ │  PostgreSQL    │ │  Redis         │
│   (pooled)     │ │  (sessions +   │ │  (cache +      │
│                │ │   memory)      │ │   rate limits)  │
└────────────────┘ └────────────────┘ └────────────────┘
```

---

## 2. Foundation Layer: Security & Testing

### 2.1 Security Hardening

#### 2.1.1 Eliminate `eval()`

**Current (DANGEROUS):**
```python
# agent.py
user_restrictions = eval(os.environ.get('DEFAULT_USER_RESTRICTIONS'))
```

**Proposed (SAFE):**
```python
# config.py
import json
from pydantic import BaseModel

class UserRestrictions(BaseModel):
    restricted_facets: list[str] = []
    allowed_catalogs: list[str] = []

def load_user_restrictions() -> UserRestrictions:
    raw = os.environ.get('DEFAULT_USER_RESTRICTIONS', '{}')
    return UserRestrictions(**json.loads(raw))
```

**For LLM output parsing:**
```python
# BEFORE (vulnerable)
result = eval(llm_output)

# AFTER (safe)
result = json.loads(llm_output)
# Or better, with Pydantic:
result = SegmentOutput.model_validate_json(llm_output)
```

#### 2.1.2 Fix SSL Verification

```python
# BEFORE
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# AFTER
ssl_context.check_hostname = True
ssl_context.verify_mode = ssl.CERT_REQUIRED
ssl_context.load_verify_locations('/etc/ssl/certs/ca-certificates.crt')
```

#### 2.1.3 Restrict CORS

```python
# BEFORE
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# AFTER
ALLOWED_ORIGINS = os.environ.get("CORS_ORIGINS", "").split(",")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS)
```

#### 2.1.4 Remove `pytest` from Production

```python
# BEFORE (api.py)
import pytest  # Remove this entirely

# AFTER
# pytest only in test files, never in production code
```

---

### 2.2 Testing Infrastructure

#### 2.2.1 Test Hierarchy

```
tests/
├── unit/
│   ├── test_state_management.py         # State init, cleanup, validation
│   ├── test_prompt_loading.py           # Prompt template resolution
│   ├── test_output_validation.py        # Pydantic schema validation
│   ├── test_metadata_loading.py         # Facet catalog parsing
│   └── test_formatter.py               # Output formatting
├── integration/
│   ├── test_milvus_search.py            # Vector search accuracy
│   ├── test_llm_structured_output.py    # StructuredInfero with mocks
│   ├── test_session_persistence.py      # DB session lifecycle
│   └── test_api_endpoints.py            # Route handler tests
├── eval/
│   ├── test_segment_decomposition.py    # Decomposer accuracy
│   ├── test_facet_mapping.py            # Facet value mapper accuracy
│   ├── test_end_to_end.py              # Full pipeline accuracy
│   └── test_regression.py              # Comparison with baseline
└── conftest.py                          # Shared fixtures, mocks
```

#### 2.2.2 Example Unit Test

```python
# tests/unit/test_output_validation.py
import pytest
from pydantic import ValidationError
from models.segment import SegmentOutput, SubSegment

def test_valid_segment_output():
    output = SegmentOutput(
        sub_segments=[
            SubSegment(
                facet_name="age",
                operator="BETWEEN",
                values=["18", "35"],
                logical_operator="AND"
            )
        ],
        logical_operator="AND"
    )
    assert len(output.sub_segments) == 1
    assert output.sub_segments[0].operator == "BETWEEN"

def test_invalid_operator_rejected():
    with pytest.raises(ValidationError):
        SubSegment(
            facet_name="age",
            operator="INVALID_OP",
            values=["18"]
        )

def test_empty_sub_segments_rejected():
    with pytest.raises(ValidationError):
        SegmentOutput(sub_segments=[], logical_operator="AND")
```

#### 2.2.3 CI Integration

```yaml
# .github/workflows/ci.yml (or KITT equivalent)
name: CI Pipeline
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Unit Tests
        run: pytest tests/unit/ -v --tb=short
      - name: Integration Tests
        run: pytest tests/integration/ -v --tb=short
      
  eval-gate:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Run Eval Suite
        run: python -m evaluations.cli run --set core_regression --threshold 0.90
      - name: Check Eval Results
        run: python -m evaluations.cli check --min-facet-precision 0.95 --min-recall 0.90
```

---

## 3. Efficiency Layer: Cost & Performance

### 3.1 Model Tiering

#### 3.1.1 Model Router

```python
# utils/model_router.py
from enum import Enum
from dataclasses import dataclass

class ModelTier(Enum):
    ULTRA_ECONOMY = "ultra_economy"   # NER, date extraction, formatting
    ECONOMY = "economy"               # Routing, classification
    STANDARD = "standard"             # Reasoning, decomposition, mapping  
    PREMIUM = "premium"               # Complex ambiguity resolution

@dataclass
class ModelConfig:
    tier: ModelTier
    model_id: str
    max_tokens: int
    temperature: float

MODEL_CONFIGS = {
    ModelTier.ULTRA_ECONOMY: ModelConfig(
        tier=ModelTier.ULTRA_ECONOMY,
        model_id="gemini-2.5-flash-lite",  # $0.03/1M input tokens
        max_tokens=2048,
        temperature=0.0
    ),
    ModelTier.ECONOMY: ModelConfig(
        tier=ModelTier.ECONOMY,
        model_id="claude-haiku-4.5",  # $0.25/1M input tokens
        max_tokens=4096,
        temperature=0.0
    ),
    ModelTier.STANDARD: ModelConfig(
        tier=ModelTier.STANDARD,
        model_id="claude-sonnet-4.5",  # $3.00/1M input tokens
        max_tokens=8192,
        temperature=0.1
    ),
    ModelTier.PREMIUM: ModelConfig(
        tier=ModelTier.PREMIUM,
        model_id="claude-opus-4.6",  # $15.00/1M input tokens
        max_tokens=16384,
        temperature=0.2
    ),
}

# Task-to-tier mapping
TASK_TIER_MAP = {
    "route_request": ModelTier.ECONOMY,
    "decompose_segment": ModelTier.STANDARD,
    "extract_named_entities": ModelTier.ULTRA_ECONOMY,
    "tag_dates": ModelTier.ULTRA_ECONOMY,
    "map_facet_values": ModelTier.STANDARD,
    "resolve_ambiguity": ModelTier.STANDARD,
    "format_segment": ModelTier.ECONOMY,
    "generate_summary": ModelTier.ECONOMY,
    "assess_hypothesis": ModelTier.PREMIUM,
}
```

#### 3.1.2 Cost Impact Analysis

| Task | Current Cost (per call) | Proposed Cost (per call) | Monthly Savings (10K requests) |
|---|---|---|---|
| Routing | $0.015 | $0.0025 | $125 |
| Decomposition | $0.015 | $0.003 | $120 |
| NER | $0.015 | $0.0003 | $147 |
| Date Tagging | $0.015 | $0.0003 | $147 |
| Facet Mapping | $0.015 | $0.003 | $120 |
| Formatting | $0.015 | $0.0025 | $125 |
| **Total per request** | **$0.090** | **$0.011** | **$784/month** |

**Projected annual savings: ~$9,400** (at 10K requests/month)

---

### 3.2 Caching Layer

#### 3.2.1 Redis Cache Architecture

```python
# cache/cache_manager.py
import hashlib
import json
import redis.asyncio as redis
from typing import Optional

class CacheManager:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    async def get_llm_cache(self, prompt_hash: str, model: str) -> Optional[str]:
        """Cache LLM responses by prompt content hash."""
        key = f"llm:{model}:{prompt_hash}"
        return await self.redis.get(key)
    
    async def set_llm_cache(self, prompt_hash: str, model: str, response: str, ttl: int = 3600):
        key = f"llm:{model}:{prompt_hash}"
        await self.redis.setex(key, ttl, response)
    
    async def get_embedding_cache(self, text_hash: str) -> Optional[list[float]]:
        """Cache embedding vectors by text hash."""
        key = f"emb:{text_hash}"
        cached = await self.redis.get(key)
        return json.loads(cached) if cached else None
    
    async def set_embedding_cache(self, text_hash: str, vector: list[float], ttl: int = 86400):
        key = f"emb:{text_hash}"
        await self.redis.setex(key, ttl, json.dumps(vector))
    
    async def get_milvus_cache(self, query_hash: str, collection: str) -> Optional[list]:
        """Cache Milvus search results."""
        key = f"mil:{collection}:{query_hash}"
        cached = await self.redis.get(key)
        return json.loads(cached) if cached else None
    
    @staticmethod
    def hash_content(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:16]
```

#### 3.2.2 Estimated Cache Hit Rates

| Cache Layer | Expected Hit Rate | Token Savings |
|---|---|---|
| LLM Response Cache | 15-30% | 15-30% of LLM costs |
| Embedding Cache | 40-60% | 40-60% of embedding compute |
| Milvus Result Cache | 20-40% | 20-40% of Milvus query load |
| Prompt Assembly Cache | 99% | Negligible (template compilation) |

---

### 3.3 Connection Management

#### 3.3.1 Milvus Connection Pool

```python
# utils/milvus_pool.py
from pymilvus import connections, MilvusClient
from contextlib import contextmanager
import threading

class MilvusPool:
    """Singleton Milvus connection pool."""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, uri: str, max_connections: int = 10):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_pool(uri, max_connections)
            return cls._instance
    
    def _init_pool(self, uri: str, max_connections: int):
        self.uri = uri
        self.client = MilvusClient(uri=uri)
        connections.connect(alias="default", uri=uri)
        self._initialized = True
    
    def search(self, collection: str, vectors: list, top_k: int = 10, **kwargs):
        """Reuse existing connection for search."""
        return self.client.search(
            collection_name=collection,
            data=vectors,
            limit=top_k,
            **kwargs
        )
```

#### 3.3.2 Embedding Singleton

```python
# utils/embedding_singleton.py
from sentence_transformers import SentenceTransformer
import threading

class EmbeddingService:
    """Singleton embedding model — loaded once, used everywhere."""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, model_name: str = "BAAI/bge-small-en-v1.5"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.model = SentenceTransformer(model_name)
                cls._instance.model_name = model_name
            return cls._instance
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()
```

#### 3.3.3 Database Pool Expansion

```python
# BEFORE
max_connections: int = 3  # Too low

# AFTER
max_connections: int = 20  # Support concurrent requests
min_connections: int = 5   # Keep warm connections
```

---

### 3.4 Memory Reduction

| Component | Current | Proposed | Savings |
|---|---|---|---|
| Embedding model | BGE-large (1.3GB) | BGE-small (80MB) or API-based | 1.2GB |
| PyTorch | ~800MB | ONNX Runtime (~100MB) | 700MB |
| Pickle metadata | ~200MB estimated | JSON + lazy loading | ~150MB |
| **Pod RAM** | **32-64 GB** | **4-8 GB** | **87%** |

Moving the embedding model to a separate microservice (or using ONNX Runtime) is the single biggest RAM savings.

---

## 4. Reliability Layer: Validation & Resilience

### 4.1 Plan-Act-Verify-Improve (PAVI) Loop

#### 4.1.1 Integration into Segment Creation

```python
# agents/pavi_orchestrator.py
from pydantic import BaseModel
from typing import Literal

class PAVIResult(BaseModel):
    status: Literal["success", "needs_improvement", "needs_clarification", "failed"]
    segment_json: dict | None = None
    issues: list[str] = []
    improvements_applied: list[str] = []
    clarification_needed: str | None = None

async def segment_creation_pavi(
    user_query: str,
    facet_catalog: dict,
    max_iterations: int = 3
) -> PAVIResult:
    """Execute segment creation with Plan-Act-Verify-Improve loop."""
    
    for iteration in range(max_iterations):
        # ── PLAN ──
        plan = await plan_segment(user_query, facet_catalog)
        # Output: sub-segments, required facets, expected operators
        
        # ── ACT ──
        segment_json = await execute_plan(plan, facet_catalog)
        # Output: structured segment JSON
        
        # ── VERIFY ──
        verification = await verify_segment(segment_json, facet_catalog, user_query)
        # Checks: schema validity, facets exist, operators valid, 
        #         logical consistency, NL alignment
        
        if verification.all_passed:
            return PAVIResult(
                status="success",
                segment_json=segment_json,
                improvements_applied=[f"Iteration {iteration + 1}: passed verification"]
            )
        
        if verification.needs_user_clarification:
            return PAVIResult(
                status="needs_clarification",
                clarification_needed=verification.clarification_prompt
            )
        
        # ── IMPROVE ──
        user_query = await improve_with_feedback(
            user_query, segment_json, verification.issues
        )
    
    return PAVIResult(status="failed", issues=["Max iterations reached"])
```

#### 4.1.2 Verification Checks

```python
# agents/verifier.py
class SegmentVerification(BaseModel):
    schema_valid: bool
    all_facets_exist: bool
    operators_valid: bool
    values_in_range: bool
    logical_consistency: bool
    nl_alignment_score: float  # 0-1, does output match input intent?
    all_passed: bool
    needs_user_clarification: bool = False
    clarification_prompt: str | None = None
    issues: list[str] = []

async def verify_segment(
    segment_json: dict,
    facet_catalog: dict,
    original_query: str
) -> SegmentVerification:
    """Multi-check verification of generated segment."""
    
    issues = []
    
    # Check 1: Schema validity
    try:
        SegmentOutput.model_validate(segment_json)
        schema_valid = True
    except ValidationError as e:
        schema_valid = False
        issues.append(f"Schema invalid: {e}")
    
    # Check 2: All facets exist in catalog
    all_facets_exist = True
    for sub in segment_json.get("sub_segments", []):
        if sub["facet_name"] not in facet_catalog:
            all_facets_exist = False
            issues.append(f"Unknown facet: {sub['facet_name']}")
    
    # Check 3: Operators are valid for facet type
    operators_valid = True
    for sub in segment_json.get("sub_segments", []):
        facet_type = facet_catalog.get(sub["facet_name"], {}).get("type")
        if not is_operator_valid(sub["operator"], facet_type):
            operators_valid = False
            issues.append(f"Invalid operator {sub['operator']} for {facet_type} facet {sub['facet_name']}")
    
    # Check 4: NL alignment (does the output match the intent?)
    nl_alignment_score = await check_nl_alignment(original_query, segment_json)
    
    all_passed = all([schema_valid, all_facets_exist, operators_valid]) and nl_alignment_score > 0.85
    
    return SegmentVerification(
        schema_valid=schema_valid,
        all_facets_exist=all_facets_exist,
        operators_valid=operators_valid,
        values_in_range=True,  # Implement per facet type
        logical_consistency=True,  # Implement contradiction detection
        nl_alignment_score=nl_alignment_score,
        all_passed=all_passed,
        issues=issues
    )
```

---

### 4.2 Circuit Breaker Pattern

```python
# utils/circuit_breaker.py
import time
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, reject immediately
    HALF_OPEN = "half_open" # Testing if service recovered

class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 2
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
    
    async def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker open. Retry after {self.recovery_timeout}s"
                )
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

# Usage
llm_circuit = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
milvus_circuit = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
```

---

### 4.3 Route Handler Refactoring

#### 4.3.1 Current: 641-Line Monolith

The current `run_sse` function handles everything in a single massive async generator.

#### 4.3.2 Proposed: Modular Pipeline

```python
# routes/agent_routes.py (refactored)
from fastapi import APIRouter, Depends
from routes.handlers import (
    validate_request, 
    resolve_session,
    run_agent_pipeline,
    format_response,
    stream_response
)

router = APIRouter(prefix="/agent")

@router.post("/run_sse2")
async def run_sse(request: AgentRequest):
    # Step 1: Validate
    validated = await validate_request(request)
    
    # Step 2: Session
    session = await resolve_session(validated)
    
    # Step 3: Execute
    result = await run_agent_pipeline(session, validated)
    
    # Step 4: Format
    response = await format_response(result, session)
    
    # Step 5: Stream
    return stream_response(response)
```

Each step is independently testable with clear inputs and outputs.

---

### 4.4 Eval Gates in CI/CD

```python
# evaluations/eval_gate.py
class EvalGate:
    """Deployment gate based on eval results."""
    
    MINIMUM_THRESHOLDS = {
        "facet_precision": 0.95,
        "facet_recall": 0.90,
        "operator_accuracy": 0.95,
        "value_accuracy": 0.90,
        "schema_validity": 1.00,
        "hallucination_rate": 0.01,  # Max 1%
        "latency_p95_seconds": 30.0,
    }
    
    def check(self, eval_results: dict) -> tuple[bool, list[str]]:
        failures = []
        for metric, threshold in self.MINIMUM_THRESHOLDS.items():
            actual = eval_results.get(metric, 0)
            if metric == "hallucination_rate":
                if actual > threshold:
                    failures.append(f"{metric}: {actual:.3f} > {threshold}")
            elif metric == "latency_p95_seconds":
                if actual > threshold:
                    failures.append(f"{metric}: {actual:.1f}s > {threshold}s")
            else:
                if actual < threshold:
                    failures.append(f"{metric}: {actual:.3f} < {threshold}")
        
        return len(failures) == 0, failures
```

---

## 5. Extensibility Layer: Skills, Knowledge & Multi-Tenancy

### 5.1 Skill Architecture

#### 5.1.1 Skill Definition

```
skills/
├── create_segment/
│   ├── SKILL.md               # Trigger + main instructions
│   ├── schema.py              # Input/output Pydantic models
│   ├── pipeline.py            # Execution steps
│   ├── evals/
│   │   ├── test_cases.yaml    # Eval test cases
│   │   └── run_evals.py       # Eval runner
│   └── versions/
│       ├── v1.0.md            # Versioned instruction text
│       └── v1.1.md
├── edit_segment/
│   ├── SKILL.md
│   ├── schema.py
│   ├── pipeline.py
│   └── evals/
├── assess_hypothesis/
│   ├── SKILL.md
│   ├── schema.py
│   └── pipeline.py
└── explore_models/
    ├── SKILL.md
    ├── schema.py
    └── pipeline.py
```

#### 5.1.2 Skill Registry

```python
# skills/registry.py
from dataclasses import dataclass
from typing import Callable

@dataclass
class Skill:
    id: str
    version: str
    description: str
    trigger: str  # "when to use" description
    instruction_file: str
    input_schema: type  # Pydantic model
    output_schema: type  # Pydantic model
    pipeline: Callable
    eval_threshold: float = 0.90

class SkillRegistry:
    def __init__(self):
        self._skills: dict[str, Skill] = {}
    
    def register(self, skill: Skill):
        self._skills[skill.id] = skill
    
    def get(self, skill_id: str) -> Skill:
        return self._skills[skill_id]
    
    def list_for_routing(self) -> list[dict]:
        """Return skill descriptions for the router LLM."""
        return [
            {"id": s.id, "description": s.description, "trigger": s.trigger}
            for s in self._skills.values()
        ]
    
    def load_instructions(self, skill_id: str) -> str:
        """Load skill instructions into agent context."""
        skill = self._skills[skill_id]
        with open(skill.instruction_file) as f:
            return f.read()
```

#### 5.1.3 Adding a New Skill (No Code Change Required)

To add "hypothesis assessment" capability:

1. Create `skills/assess_hypothesis/SKILL.md`:
```markdown
# Skill: Assess Customer Segment Hypothesis

## Trigger
Use when the user asks to evaluate, assess, or test a hypothesis about a 
customer segment, or asks "what if" questions about segment composition.

## Instructions
1. Parse the hypothesis from the user query
2. Identify relevant facets and their expected behavior
3. Query historical segment performance data
4. Compare hypothesis against known patterns in memory
5. Provide assessment with confidence score and reasoning
6. Suggest alternative hypotheses if the original is weak

## Output Format
Return a HypothesisAssessment object with:
- hypothesis_text: The parsed hypothesis
- confidence_score: 0-100
- supporting_evidence: List of evidence items
- counter_evidence: List of counter-evidence items
- alternative_hypotheses: List of suggested alternatives
- recommendation: ACCEPT, REJECT, or INVESTIGATE_FURTHER
```

2. Create `skills/assess_hypothesis/schema.py`:
```python
from pydantic import BaseModel
from typing import Literal

class HypothesisAssessment(BaseModel):
    hypothesis_text: str
    confidence_score: int  # 0-100
    supporting_evidence: list[str]
    counter_evidence: list[str]
    alternative_hypotheses: list[str]
    recommendation: Literal["ACCEPT", "REJECT", "INVESTIGATE_FURTHER"]
```

3. Register in config — **no agent.py or router prompt modification needed**.

---

### 5.2 Knowledge Management

#### 5.2.1 Knowledge Store Architecture

```python
# knowledge/store.py
from pydantic import BaseModel
from datetime import datetime

class KnowledgeEntry(BaseModel):
    id: str
    domain: str  # "facet_catalog", "business_rules", "segment_patterns"
    tenant_id: str
    content: str
    metadata: dict
    version: int
    created_at: datetime
    expires_at: datetime | None = None
    
class KnowledgeStore:
    """Unified knowledge store with versioning and tenant isolation."""
    
    async def retrieve(
        self,
        query: str,
        domain: str,
        tenant_id: str,
        top_k: int = 10,
        freshness_days: int | None = None
    ) -> list[KnowledgeEntry]:
        """Retrieve knowledge with tenant isolation and freshness filter."""
        results = await self.vector_search(query, domain, top_k)
        
        # Filter by tenant
        results = [r for r in results if r.tenant_id == tenant_id or r.tenant_id == "global"]
        
        # Filter by freshness
        if freshness_days:
            cutoff = datetime.now() - timedelta(days=freshness_days)
            results = [r for r in results if r.created_at > cutoff]
        
        return results
```

#### 5.2.2 Migrating from Pickle to Knowledge Store

**Current:** Facet catalog in `.pkl` files, loaded from disk each time.

**Proposed migration:**
1. Export pickle data to JSON/Parquet format
2. Load into PostgreSQL with facet metadata table
3. Index in Milvus with per-tenant partitions
4. Serve via `KnowledgeStore` with caching

```python
# knowledge/facet_catalog.py
class FacetCatalog:
    def __init__(self, knowledge_store: KnowledgeStore, tenant_id: str):
        self.store = knowledge_store
        self.tenant_id = tenant_id
        self._cache: dict | None = None
    
    async def get_facets(self) -> dict:
        if self._cache is None:
            entries = await self.store.retrieve(
                query="*",
                domain="facet_catalog",
                tenant_id=self.tenant_id
            )
            self._cache = {e.metadata["facet_name"]: e for e in entries}
        return self._cache
    
    async def search_facets(self, query: str, top_k: int = 10) -> list:
        return await self.store.retrieve(
            query=query,
            domain="facet_catalog",
            tenant_id=self.tenant_id,
            top_k=top_k
        )
```

---

### 5.3 Multi-Tenant Configuration

#### 5.3.1 Tenant Config Model

```python
# config/tenant.py
from pydantic import BaseModel
from typing import Optional

class TenantModelConfig(BaseModel):
    router_model: str = "claude-haiku-4.5"
    reasoning_model: str = "claude-sonnet-4.5"
    premium_model: str = "claude-opus-4.6"
    embedding_model: str = "bge-small-en-v1.5"

class TenantConfig(BaseModel):
    tenant_id: str
    display_name: str
    model_config_: TenantModelConfig = TenantModelConfig()
    facet_catalog_version: str = "latest"
    knowledge_base_id: str = "default"
    prompt_overrides: dict[str, str] = {}  # skill_id -> prompt version
    feature_flags: dict[str, bool] = {}
    rate_limits: dict[str, int] = {"requests_per_minute": 60}
    user_restrictions: list[str] = []
    custom_skills: list[str] = []

class TenantConfigService:
    """Load tenant configs from database, not environment variables."""
    
    async def get_config(self, tenant_id: str) -> TenantConfig:
        # Query from PostgreSQL config table
        row = await self.db.fetchrow(
            "SELECT config FROM tenant_configs WHERE tenant_id = $1",
            tenant_id
        )
        if not row:
            return TenantConfig(tenant_id=tenant_id)  # Defaults
        return TenantConfig(**json.loads(row["config"]))
```

---

## 6. Intelligence Layer: Memory & Auto-Improvement

### 6.1 Three-Tier Memory System

#### 6.1.1 Memory Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Memory Manager                        │
│                                                          │
│  ┌─────────────────┐  ┌──────────────────────────────┐  │
│  │  CORE MEMORY    │  │  RECALL MEMORY               │  │
│  │  (always in     │  │  (conversation history)       │  │
│  │   context)      │  │                              │  │
│  │                 │  │  Stored: PostgreSQL           │  │
│  │  - Current      │  │  Indexed: Full-text search    │  │
│  │    segment def  │  │                              │  │
│  │  - User prefs   │  │  Per-user conversation logs   │  │
│  │  - Active task  │  │  Searchable by content/date   │  │
│  │    state        │  │                              │  │
│  └─────────────────┘  └──────────────────────────────┘  │
│                                                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │  ARCHIVAL MEMORY (long-term learned patterns)      │  │
│  │                                                    │  │
│  │  Stored: Milvus + PostgreSQL                       │  │
│  │  Indexed: Vector similarity                        │  │
│  │                                                    │  │
│  │  - Successful segment recipes                      │  │
│  │  - Common facet combinations                       │  │
│  │  - User correction patterns                        │  │
│  │  - Facet usage statistics                          │  │
│  │  - Learned tenant preferences                      │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

#### 6.1.2 Memory Manager Implementation

```python
# memory/manager.py
from pydantic import BaseModel
from datetime import datetime

class MemoryEntry(BaseModel):
    id: str
    type: str  # "recipe", "correction", "preference", "pattern"
    content: str
    metadata: dict
    user_id: str | None = None
    tenant_id: str
    confidence: float = 1.0
    access_count: int = 0
    created_at: datetime
    last_accessed: datetime

class MemoryManager:
    def __init__(self, db, milvus, embedding_service):
        self.db = db
        self.milvus = milvus
        self.embedder = embedding_service
    
    async def store_recipe(
        self,
        user_query: str,
        segment_json: dict,
        user_confirmed: bool,
        tenant_id: str
    ):
        """Store a successful segment creation as a reusable recipe."""
        if user_confirmed:
            embedding = self.embedder.embed([user_query])[0]
            entry = MemoryEntry(
                id=generate_id(),
                type="recipe",
                content=json.dumps({
                    "query": user_query,
                    "segment": segment_json
                }),
                metadata={
                    "facets_used": extract_facet_names(segment_json),
                    "complexity": count_sub_segments(segment_json)
                },
                tenant_id=tenant_id,
                created_at=datetime.now(),
                last_accessed=datetime.now()
            )
            await self._persist(entry, embedding)
    
    async def recall_similar_recipes(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 3
    ) -> list[MemoryEntry]:
        """Find similar past segments for reference."""
        embedding = self.embedder.embed([query])[0]
        results = self.milvus.search(
            collection="segment_recipes",
            vectors=[embedding],
            top_k=top_k,
            filter=f'tenant_id == "{tenant_id}"'
        )
        return [self._to_entry(r) for r in results]
    
    async def store_correction(
        self,
        original_query: str,
        original_segment: dict,
        corrected_segment: dict,
        tenant_id: str
    ):
        """Learn from user corrections to avoid repeating mistakes."""
        entry = MemoryEntry(
            id=generate_id(),
            type="correction",
            content=json.dumps({
                "query": original_query,
                "original": original_segment,
                "corrected": corrected_segment,
                "diff": compute_segment_diff(original_segment, corrected_segment)
            }),
            tenant_id=tenant_id,
            created_at=datetime.now(),
            last_accessed=datetime.now()
        )
        embedding = self.embedder.embed([original_query])[0]
        await self._persist(entry, embedding)
```

---

### 6.2 Auto-Improvement Pipeline

#### 6.2.1 DSPy Integration for Prompt Optimization

```python
# auto_improve/optimizer.py
import dspy

class SegmentDecomposerModule(dspy.Module):
    """DSPy module wrapping the segment decomposer."""
    
    def __init__(self):
        self.decompose = dspy.ChainOfThought(
            "user_query, facet_catalog_summary -> sub_segments, logical_operators"
        )
    
    def forward(self, user_query: str, facet_catalog_summary: str):
        return self.decompose(
            user_query=user_query,
            facet_catalog_summary=facet_catalog_summary
        )

def optimize_decomposer(train_data: list[dict]):
    """Optimize the segment decomposer prompt using eval data."""
    
    # Define metric
    def facet_accuracy(example, prediction, trace=None):
        expected = set(example.expected_facets)
        predicted = set(extract_facets(prediction.sub_segments))
        return len(expected & predicted) / max(len(expected), 1)
    
    # Create optimizer
    optimizer = dspy.MIPROv2(
        metric=facet_accuracy,
        auto="medium",
        num_threads=4
    )
    
    # Optimize
    module = SegmentDecomposerModule()
    optimized = optimizer.compile(module, trainset=train_data)
    
    return optimized
```

#### 6.2.2 Feedback Loop Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Production  │────>│  Eval Runner │────>│  Optimizer   │
│  Requests    │     │  (weekly)    │     │  (DSPy/OPRO) │
│              │     │              │     │              │
│  Log: query, │     │  Metrics:    │     │  Output:     │
│  output,     │     │  - accuracy  │     │  - improved  │
│  user action │     │  - latency   │     │    prompt    │
│  (confirm/   │     │  - cost      │     │  - eval      │
│   edit/      │     │              │     │    results   │
│   reject)    │     └──────┬───────┘     └──────┬───────┘
└──────────────┘            │                     │
                            ▼                     ▼
                    ┌──────────────┐     ┌──────────────┐
                    │  Eval Gate   │────>│  Deploy New  │
                    │  (threshold  │     │  Prompt      │
                    │   check)     │     │  Version     │
                    └──────────────┘     └──────────────┘
```

---

### 6.3 Observability Upgrade

#### 6.3.1 Prometheus Metrics

```python
# observability/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
REQUESTS_TOTAL = Counter(
    "agent_requests_total",
    "Total agent requests",
    ["tenant_id", "skill", "status"]
)

REQUEST_LATENCY = Histogram(
    "agent_request_latency_seconds",
    "Request latency in seconds",
    ["tenant_id", "skill"],
    buckets=[1, 5, 10, 15, 20, 30, 45, 60, 90, 120]
)

# LLM metrics
LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Total LLM tokens consumed",
    ["model", "task", "direction"]  # direction: input/output
)

LLM_COST_TOTAL = Counter(
    "llm_cost_usd_total",
    "Total LLM cost in USD",
    ["model", "task"]
)

LLM_CALL_LATENCY = Histogram(
    "llm_call_latency_seconds",
    "Individual LLM call latency",
    ["model", "task"],
    buckets=[0.5, 1, 2, 3, 5, 8, 13, 21, 34]
)

# Cache metrics
CACHE_HITS = Counter("cache_hits_total", "Cache hits", ["cache_layer"])
CACHE_MISSES = Counter("cache_misses_total", "Cache misses", ["cache_layer"])

# Quality metrics (from evals)
EVAL_SCORE = Gauge(
    "eval_score",
    "Latest eval score",
    ["metric_name", "skill"]
)
```

---

## 7. State Management Redesign

### 7.1 Typed State with Namespacing

```python
# state/models.py
from pydantic import BaseModel
from typing import Optional

class DecomposerState(BaseModel):
    """State for the segment decomposer stage."""
    sub_segments: list[dict] = []
    logical_operators: list[str] = []
    requires_clarification: bool = False
    clarification_question: Optional[str] = None

class FacetMapperState(BaseModel):
    """State for the facet-value mapper stage."""
    shortlisted_facets: list[dict] = []
    mapped_values: list[dict] = []
    current_index: int = 0
    additional_info: dict = {}

class ConversationState(BaseModel):
    """Top-level conversation state."""
    conversation_id: str
    user_id: str
    tenant_id: str
    current_skill: Optional[str] = None
    decomposer: DecomposerState = DecomposerState()
    facet_mapper: FacetMapperState = FacetMapperState()
    history: list[dict] = []
    segment_json: Optional[dict] = None
    
    def reset_pipeline(self):
        """Clean reset between pipeline runs."""
        self.decomposer = DecomposerState()
        self.facet_mapper = FacetMapperState()
        self.segment_json = None
```

This replaces the 60+ flat string constants with typed, grouped, documented state models that can be validated at runtime.

---

## Appendix A: Cost-Optimized vs State-of-the-Art Alternatives

| Component | State-of-the-Art | Cost-Optimized Alternative | Quality Impact |
|---|---|---|---|
| **Reasoning model** | Claude Opus 4.6 ($15/1M) | Claude Sonnet 4.5 ($3/1M) | Minimal for segment creation |
| **Routing model** | Claude Sonnet 4.5 ($3/1M) | Claude Haiku 4.5 ($0.25/1M) | None — routing is simple classification |
| **NER model** | Claude Sonnet 4.5 ($3/1M) | Gemini Flash Lite ($0.03/1M) | Minimal — NER is well-constrained |
| **Embedding model** | BGE-large (1.3GB, high accuracy) | BGE-small (80MB, ~2% less accurate) | Negligible in production |
| **Vector DB** | Dedicated Milvus cluster | Milvus Lite in-process | Acceptable for < 1M vectors |
| **Memory store** | Dedicated Redis cluster | PostgreSQL JSONB columns | Acceptable for < 100K entries |
| **Observability** | Datadog / New Relic ($$$) | Phoenix + Prometheus + Grafana (free) | Equivalent for agent workloads |

**Recommendation:** Use cost-optimized alternatives for all components except the reasoning model (facet mapper, decomposer). These tasks require nuanced understanding of natural language to structured mapping and benefit from a stronger model.

---

## Appendix B: Migration Path from Current to Proposed

| Current Component | Proposed Replacement | Migration Risk |
|---|---|---|
| `eval()` calls | `json.loads()` + Pydantic | Low — direct substitution |
| SSL bypass | Proper SSL verification | Low — cert configuration |
| Pickle metadata | PostgreSQL + API | Medium — data migration |
| Single model | Model router | Medium — testing per task |
| No tests | Test suite | Low — additive |
| 641-line route handler | Modular pipeline | Medium — refactoring |
| 60+ state vars | Typed Pydantic state | High — touches everywhere |
| Manual evals | CI-integrated eval gates | Medium — pipeline setup |
| In-memory artifacts | PostgreSQL / Redis | Low — configuration change |
| No caching | Redis cache layer | Low — additive |
| No memory system | Three-tier memory | Medium — new subsystem |
| Hardcoded skills | Skill registry | High — architectural shift |
| Single tenant | Multi-tenant config | High — pervasive change |

---

## Conclusion

The proposed upgrade transforms Smart-Segmentation from a prototype-grade system into an enterprise-grade agentic platform through five incremental layers. Each layer builds on the previous one, and each provides independent value.

The most impactful changes — security fixes, testing, model tiering, and caching — are also the lowest risk and highest ROI. The more ambitious changes — skill architecture, memory system, auto-improvement — build on the stable foundation created by the first layers.

See the Implementation Roadmap (Document 04) for the prioritized execution plan.
