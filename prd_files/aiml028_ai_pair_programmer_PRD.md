# Product Requirements Document (PRD) / # `aiml028_ai_pair_programmer`

Project ID: aiml028  
Category: AI/ML – Code Intelligence, Developer Productivity  
Status: Draft for review  
Version: 1.0  
Last Updated: 2025-11-25

## 1. Overview
### 1.1 Executive Summary
aiml028_ai_pair_programmer is an AI-assisted pair programming platform that provides repo-aware code completions, in-editor chat, automated refactoring, code review, test generation, and multi-step fix agents. It uses retrieval-augmented generation (RAG) optimized for code with dual indexing (file and symbol level), code-aware chunking via AST, hybrid search+rereanking, and grammar-guided decoding to output actionable patches and JSON-structured commands. It integrates with IDEs (VS Code, JetBrains), version control, CI, and cloud LLM providers. The goal is to increase developer velocity and code quality while reducing defects, context-switching, and cognitive load.

### 1.2 Document Purpose
This PRD defines scope, requirements, architecture, data model, APIs, UX, security, performance, testing, deployment, and success metrics for aiml028_ai_pair_programmer to align engineering, product, and stakeholders.

### 1.3 Product Vision
Build a trustworthy, repo-aware AI pair programmer that understands the entire codebase, reasons across files and symbols, and produces grounded, auditable, and safe edits that developers accept with confidence.

## 2. Problem Statement
### 2.1 Current Challenges
- Generic code assistants hallucinate APIs, ignore project conventions, and lack repo context.
- Developers waste time navigating large codebases, reading docs, and resolving dependency chains.
- Refactors and bug fixes are multi-step and error-prone across files and tests.
- Tooling fragmentation (linters, tests, search) leads to context switching.
- Inconsistent code quality and slow onboarding for new contributors.

### 2.2 Impact Analysis
- Productivity loss: 20–40% time spent on code navigation, boilerplate, and debugging.
- Quality issues: Higher defect rates from incomplete context and missed edge cases.
- Onboarding drag: Ramp-up delays for new engineers.
- Cost: Increased CI re-runs and review cycles.

### 2.3 Opportunity
- Provide context-grounded assistance with code-specific RAG and AST-aware retrieval.
- Automate multi-step fixes using tool-augmented agents.
- Deliver structured, auditable outputs (diffs, tests, commit messages).
- Improve velocity, quality, and developer satisfaction with measurable ROI.

## 3. Goals and Objectives
### 3.1 Primary Goals
- Repo-aware completions and chat grounded in current code, tests, and docs.
- Automated edits as diffs with rationale and generated/updated tests.
- Tool-driven agents to run linters, tests, and static analyzers for validation.
- Robust retrieval and decoding to minimize hallucinations and enforce constraints.

### 3.2 Business Objectives
- Increase code change acceptance rate by >30%.
- Reduce PR cycle time by 25%.
- Improve pass@k for coding tasks to >70% on SWE-bench-lite and HumanEval+ variants.
- Achieve <500 ms p95 latency for inline completions, 99.5%+ service uptime.
- Expand to 10+ languages and frameworks.

### 3.3 Success Metrics
- Edit acceptance rate ≥ 50% within 90 days of deployment.
- Repair rate (auto-fix that passes tests) ≥ 35%.
- pass@3 on benchmark tasks ≥ 65%.
- Inline completion p95 latency ≤ 500 ms; chat p95 ≤ 2 s.
- User NPS ≥ 45; daily active devs retention D30 ≥ 60%.

## 4. Target Users/Audience
### 4.1 Primary Users
- Software engineers (backend, frontend, full-stack)
- Data/ML engineers and data scientists writing production code
- QA/SDET engineers

### 4.2 Secondary Users
- Engineering managers and tech leads
- DevOps/SRE
- Security engineers (code scanning, policy enforcement)

### 4.3 User Personas
- Persona 1: Maya Singh – Senior Backend Engineer
  - Background: 8 years in Python/Go microservices, leads service maturity efforts.
  - Pain points: Large repos, cross-service dependencies, flaky tests, boilerplate for APIs and database migrations.
  - Goals: Faster refactors, reliable suggestions grounded in repo conventions, reduce defects.
- Persona 2: Lucas Chen – Frontend Engineer
  - Background: 5 years React/TypeScript; owns design system.
  - Pain points: Prop drilling, state management complexity, inconsistent patterns, accessibility compliance.
  - Goals: Context-aware component completions, refactors with codemods, auto-generated tests and stories.
- Persona 3: Ana Rodriguez – Data Scientist
  - Background: Python, notebooks, ML pipelines (Airflow/Prefect).
  - Pain points: Translating notebooks to production code, DAG debugging, docstring/tests backlog.
  - Goals: Generate clean modules from notebooks, test scaffolds, pipeline fix suggestions.
- Persona 4: Omar Farouk – Engineering Manager
  - Background: Leads platform team across languages.
  - Pain points: Onboarding new hires, code quality consistency, review bottlenecks.
  - Goals: Policy-driven guardrails, analytics on assistant effectiveness, reduced cycle time.

## 5. User Stories
- US-001: As a backend engineer, I want inline completions that use my project’s APIs so that I don’t waste time searching docs.
  - Acceptance: Completion references existing symbols; no broken imports; median latency ≤ 250 ms.
- US-002: As a developer, I want to ask chat “why is this test failing?” and receive a diagnosis referencing stack traces and related files.
  - Acceptance: Response cites retrieved files/snippets; provides patch diff or steps; includes confidence score.
- US-003: As a frontend engineer, I want a refactor to migrate legacy components to hooks with tests updated automatically.
  - Acceptance: Generated diff compiles; tests updated/added; lint passes; rationale provided.
- US-004: As a data scientist, I want to convert a notebook cell block into a production-ready module with docstrings and unit tests.
  - Acceptance: Produces .py module, docstrings, pytest; passes black/flake8/mypy gates.
- US-005: As an SDET, I want suggested tests for new functions to increase coverage.
  - Acceptance: Suggested tests increase coverage by ≥ 10% for target module; runnable locally.
- US-006: As a tech lead, I want PR review summaries and risk highlights.
  - Acceptance: Summary cites files; flags risky patterns; provides configurable policy checks.
- US-007: As a developer, I want multi-step fix agents to run linters and tests and iterate until green.
  - Acceptance: Agent executes plan with max steps/k budget; reports each step; final status green/red with logs.
- US-008: As an admin, I want RBAC and SSO to control org/repo access.
  - Acceptance: Users inherit correct permissions; audit logs recorded.

## 6. Functional Requirements
### 6.1 Core Features
- FR-001 Repo indexing: ingest files, build ASTs, symbol graphs; incremental updates on change.
- FR-002 Dual retrieval: file- and symbol-level embeddings with hybrid lexical+semantic search.
- FR-003 Reranking: cross-encoder or late-interaction reranker; MMR for diversity; recency bias.
- FR-004 Prompt building: window packing, dedup, provenance tags, policy filters.
- FR-005 Inline completions: IDE plugin with low-latency streaming and fallback suggestions.
- FR-006 Chat: grounding in open buffers, retrieved context; citations with line ranges.
- FR-007 Structured outputs: diffs/patches, JSON edits, commands, commit messages via schema/grammar-constrained decoding.
- FR-008 Code review: summarize PRs, flag issues, suggest patches, license/policy checks.
- FR-009 Test generation: propose unit/integration tests; update snapshots; coverage goals.
- FR-010 Tool use: function-calling agents to run tests, linters, static analyzers, dependency search, code search.
- FR-011 Multi-language support: Python, TypeScript/JS, Go, Java, C#, Rust, Ruby; extensible parsers.
- FR-012 Analytics: capture acceptance rate, latency, token cost; per-team dashboards.
- FR-013 Safety/guardrails: PII redaction, license detection, secret scanning, policy enforcement.
- FR-014 IDE integrations: VS Code, JetBrains; web dashboard for repo and agent runs.

### 6.2 Advanced Features
- FR-015 Plan-execute agent with retry and backoff capped by budgets.
- FR-016 Multi-turn memory: session vectors + persistent repo knowledge; recency decay.
- FR-017 Context orchestration: active file bias, neighborhood expansion on call graph.
- FR-018 Branch-aware suggestions: understand feature branches and pending PRs.
- FR-019 Knowledge grounding: link to internal docs and ADRs via RAG.
- FR-020 Offline cache: local embedding cache for speed and privacy-mode operation.

## 7. Non-Functional Requirements
### 7.1 Performance
- Inline completion p50 ≤ 200 ms, p95 ≤ 500 ms.
- Chat p50 ≤ 800 ms, p95 ≤ 2 s.
- RPS baseline: 50 requests/sec per org bursting to 200 with autoscale.

### 7.2 Reliability
- Uptime 99.5% monthly; 99.9% target with multi-region.
- Zero data loss for persisted changes; at-least-once delivery for events.

### 7.3 Usability
- IDE interactions within 2 clicks/keystrokes.
- Clear citations and diffs; undo at all times.

### 7.4 Maintainability
- 80% unit test coverage backend; 70% plugin.
- Modular services with clear contracts and versioned APIs.

## 8. Technical Requirements
### 8.1 Technical Stack
- Backend: Python 3.11+, FastAPI 0.115+, Uvicorn, Pydantic v2.
- Workers: Python 3.11, Celery 5.x or Arq; Redis 7.2+.
- Frontend: React 18+, TypeScript 5+, Vite 5+, TailwindCSS 3+.
- IDE: VS Code extension API (1.90+), JetBrains plugin (2023.3+).
- Databases: Postgres 15+ with pgvector 0.5+, Redis 7+ for caching.
- Search: OpenSearch 2.11+ or Elasticsearch 8+ (BM25).
- Vector DB (optional alternative): Qdrant 1.8+ or Weaviate 1.24+.
- Message bus: Kafka 3.6+ or NATS 2.10+.
- Orchestration: Docker, Kubernetes 1.29+, Helm.
- Observability: OpenTelemetry, Prometheus, Grafana, Loki.
- Auth: OAuth2/OIDC, JWT, SAML SSO.
- Cloud: AWS (EKS, S3, RDS, MSK) or GCP/Azure equivalents.

### 8.2 AI/ML Components
- LLMs: 
  - Hosted: OpenAI GPT-4.1/4o, Anthropic Claude 3.5 Sonnet, Google Gemini 1.5 Pro.
  - Self-hosted: Llama 3.1 Code 70B, CodeLlama, DeepSeek-Coder variants via vLLM 0.5+.
- Embeddings:
  - text-embedding-3-large (OpenAI), Jina Code Embeddings v2, Cohere Embed-english-light-v3, or Voyage-code-2.
  - Per-language metadata for embedding vectors.
- Rerankers: Cohere Rerank v3 or cross-encoder ms-marco-MiniLM-L-6-v2 fine-tuned on code.
- AST Parsers: tree-sitter for multi-language; language-specific parsers (lib2to3/LibCST for Python, TypeScript Compiler API).
- Decoding: constrained decoding via JSON schema/railroad grammars, stop sequences, logprobs.
- Planning/Tools: function-calling with typed tool schemas; retry policies; safety filters.
- Evaluation suite: HumanEval+, MBPP, SWE-bench-lite; custom internal code tasks.
- Metrics: pass@k, retrieval precision/recall, MRR, edit acceptance, repair rate, latency, token cost.

## 9. System Architecture
### 9.1 High-Level Architecture
```
+---------------------+        +--------------------+
| IDE Plugins (VS/JB) |<------>|  API Gateway       |
| - Inline, Chat      |  Web   |  FastAPI + Auth    |
+----------+----------+        +-----+--------------+
           |                          |
           |                          v
           |                  +-------+--------+
           |                  | Orchestrator   |
           |                  | (Requests Bus) |
           |                  +-------+--------+
           |                          |
           v                          v
+----------+----------+      +--------+---------+        +----------------+
| Prompt Builder      |----->| LLM Router       |<------>| Safety/Policy  |
| Context Orchestration|     | (Hosted/Self)    |        | Filters        |
+----------+----------+      +--------+---------+        +----------------+
           ^                          |
           |                          v
+----------+----------+      +--------+---------+        +----------------+
| Retrieval Service   |<---->| Tool Exec Workers|<------>| CI/Code Tools  |
| - BM25 + Vector     |      | (tests, linters) |        | (pytest, eslint|
| - Reranker          |      +------------------+        |  mypy, etc.)   |
+----------+----------+                                    +---------------+
           |
           v
+----------+-----------+    +------------------+    +--------------------+
| Postgres (meta, logs)|    | Vector Store     |    | Object Store (S3)  |
+----------------------+    +------------------+    +--------------------+
           |
           v
+----------------------+
| Telemetry & Analytics|
| (Prom, Grafana, OTel)|
+----------------------+
```

### 9.2 Component Details
- IDE Plugins: Provide completions, chat, diff apply/undo, telemetry opt-in, local cache.
- API Gateway: Authentication, rate limiting, request validation, AB testing flags.
- Retrieval Service: Maintains dual indices, hybrid search, AST-aware chunking, call graph.
- Prompt Builder: Packs relevant context with dedup and provenance tags.
- LLM Router: Routes to providers/models based on policy, cost/latency, and task.
- Tool Exec Workers: Isolated sandboxes to run linters, tests, static analyzers.
- Safety/Policy: PII redaction, license/provenance tags, secret scanning.
- Storage: Postgres for metadata, Vector store for embeddings, Object store for logs/artifacts.
- Telemetry: Metrics and traces with user privacy controls.

### 9.3 Data Flow
1) IDE sends completion/chat request with active file and cursor;  
2) Gateway authenticates and enriches with org/repo;  
3) Retrieval fetches nearest symbols/files and reranks;  
4) Prompt Builder assembles input with citations and constraints;  
5) LLM Router selects model and decodes with grammar;  
6) Optional tools run (lint/test) via agent;  
7) Response returns diffs/commands/tests and rationale;  
8) Telemetry logs outcomes, acceptance, latency, cost.

## 10. Data Model
### 10.1 Entity Relationships
- Org 1—N User
- Org 1—N Repo
- Repo 1—N File
- File 1—N Symbol (function, class)
- File/Symbol 1—N Chunk (embedded)
- Repo 1—N IndexVersion
- User 1—N Session 1—N Message
- Session 1—N Patch (Diff)
- Patch 1—N ToolRun (lint/test)
- Repo 1—N RetrievalLog
- Org 1—N Policy
- Org 1—N APIKey
- Org 1—N BillingRecord
- EvaluationRun N—N TaskCase

### 10.2 Database Schema (selected tables)
- org(id, name, tier, created_at)
- user(id, org_id, email, auth_provider, role, created_at)
- repo(id, org_id, name, url, default_branch, index_status, last_indexed_at)
- file(id, repo_id, path, language, hash, size, updated_at)
- symbol(id, file_id, name, kind, start_line, end_line, signature, references_json)
- chunk(id, repo_id, file_id, symbol_id, start_line, end_line, text, embedding vector, hash)
- index_version(id, repo_id, version, created_at, stats_json)
- session(id, user_id, repo_id, type, created_at)
- message(id, session_id, role, content, citations_json, created_at)
- patch(id, session_id, base_commit, diff_text, rationale, status, created_at)
- tool_run(id, patch_id, type, command, logs_uri, status, duration_ms, created_at)
- retrieval_log(id, session_id, query, results_json, lat_ms, created_at)
- policy(id, org_id, name, rules_json, enabled)
- api_key(id, org_id, name, hashed_key, scopes, created_at, last_used_at)
- billing_record(id, org_id, period, usage_tokens, cost_usd)
- evaluation_run(id, org_id, suite, pass_at_k, metrics_json, created_at)

### 10.3 Data Flow Diagrams
```
[IDE] -> [Gateway] -> [Retrieval] -> [Prompt Builder] -> [LLM Router]
                                     |                         |
                                     v                         v
                                  [Storage]               [Tool Workers]
                                     ^                         |
                                     +-----------logs----------+
```

### 10.4 Input Data & Dataset Requirements
- Source: Git repositories, local or cloud-hosted; docs and ADRs.
- Metadata: Language tags, framework, dependency manifests.
- Evaluation datasets: HumanEval+, MBPP, SWE-bench-lite; internal annotated tasks.
- Constraints: PII redaction before storage; configurable file ignore patterns; license/provenance tagging.
- Size: Support repos up to 10M LOC; incremental indexing on change.

## 11. API Specifications
### 11.1 REST Endpoints
- POST /v1/auth/token — exchange code for JWT
- POST /v1/repos — register repo
- POST /v1/repos/{id}/index — trigger indexing
- GET /v1/repos/{id}/status — index status
- POST /v1/complete — inline completion
- POST /v1/chat — chat with context
- POST /v1/diff — request patch generation
- POST /v1/tools/run — run tool (lint/test)
- GET /v1/sessions/{id} — retrieve session
- POST /v1/eval/run — run evaluation suite
- GET /v1/analytics/metrics — org metrics
- POST /v1/keys — create API key; DELETE /v1/keys/{id}

### 11.2 Request/Response Examples
- Completion
Request:
```
POST /v1/complete
Authorization: Bearer <JWT>
{
  "repo_id": "r_123",
  "file_path": "src/service/user.py",
  "language": "python",
  "cursor": {"line": 120, "column": 8},
  "buffer": "<current file content>",
  "editor_state": {"open_files": ["src/service/user.py"], "branch": "feature/x"},
  "preferences": {"temperature": 0.2, "top_p": 0.9}
}
```
Response:
```
{
  "id": "c_789",
  "choices": [
    {"text": "def get_user_by_id(self, user_id: str) -> User:\n    ...", "confidence": 0.74}
  ],
  "latency_ms": 180,
  "citations": [
    {"path": "src/models/user.py", "lines": [10, 45]}
  ]
}
```

- Diff generation
Request:
```
POST /v1/diff
{
  "repo_id": "r_123",
  "instructions": "Fix bug in pagination when page_size=0; add tests.",
  "scope": {"paths": ["src/service/pager.py", "tests/test_pager.py"]},
  "budgets": {"max_tokens": 4000, "max_tool_runs": 3}
}
```
Response:
```
{
  "patch_id": "p_456",
  "diff": "diff --git a/src/service/pager.py b/src/service/pager.py\n--- a/src/service/pager.py\n+++ b/src/service/pager.py\n@@ ...",
  "rationale": "Handled division by zero and added default page size.",
  "tests_added": ["tests/test_pager.py::test_default_page_size"],
  "tool_runs": [{"type": "pytest", "status": "passed", "duration_ms": 9321}],
  "status": "ready"
}
```

### 11.3 Authentication
- OAuth2/OIDC for web; SAML SSO for enterprises; Personal Access Tokens for CI.
- JWT (RS256) for API; scopes: read:repo, write:repo, run:tools, view:metrics.
- Rate limiting per org and per user; HMAC verification for webhooks.

## 12. UI/UX Requirements
### 12.1 User Interface
- IDE Plugin:
  - Inline ghost text completions; accept/next toggle; streaming.
  - Chat side panel with citations, apply patch button, and test run status.
  - Diff viewer with color-coded changes; partial apply; undo.
  - Settings: provider selection, privacy mode, policy view.
- Web Dashboard:
  - Repos list/status; indexing health; search.
  - Sessions, patches, tool run logs.
  - Evaluation and analytics dashboards.
  - Admin: RBAC, API keys, policies.

### 12.2 User Experience
- Minimal friction: default-on after install; clear opt-in for telemetry.
- Consistent keyboard shortcuts; contextual suggestions adapt to active file.
- Explanations and rationales accompany diffs.

### 12.3 Accessibility
- WCAG 2.1 AA compliance; color contrast; keyboard navigation; screen reader labels.
- Configurable font sizes and high-contrast themes.

## 13. Security Requirements
### 13.1 Authentication
- SSO (SAML/OIDC), MFA enforcement, short-lived JWTs, refresh tokens with rotation.

### 13.2 Authorization
- RBAC on org/repo/file; least privilege; server-side checks on every request.
- Signed audit logs for sensitive actions.

### 13.3 Data Protection
- Encryption at rest (AES-256) and in transit (TLS 1.2+).
- Secret scanning; prevent suggestions containing detected secrets.
- PII redaction; configurable data residency; privacy mode to keep code on-prem/self-host.

### 13.4 Compliance
- SOC 2 Type II and ISO 27001 alignment roadmap.
- GDPR/CCPA support: data export/delete; DPA; subprocessor transparency.

## 14. Performance Requirements
### 14.1 Response Times
- Inline completion: p50 200 ms, p95 500 ms, p99 800 ms.
- Chat: p50 800 ms, p95 2 s, p99 4 s.
- Indexing: 1M LOC initial index ≤ 30 min; incremental updates < 3 s/file.

### 14.2 Throughput
- 200 RPS sustained per region; autoscale to 1,000 RPS with 5-min warmup.

### 14.3 Resource Usage
- LLM router CPU < 1 vCPU per 50 RPS; GPU-backed self-host LLM at 70% utilization target.
- Vector search p95 < 50 ms for top-50 candidates.

## 15. Scalability Requirements
### 15.1 Horizontal Scaling
- Stateless API; scale deployments via HPA on CPU/RPS.
- Vector indices sharded by repo_id; replication factor 3.

### 15.2 Vertical Scaling
- GPU pools for self-hosted LLMs (A100/H100 class) with vLLM tensor parallelism; autoscale node groups.

### 15.3 Load Handling
- Circuit breakers, fallback models, and degrade to lexical-only retrieval under extreme load.
- Caching: embedding cache; retrieval result cache with TTL 10–60 min.

## 16. Testing Strategy
### 16.1 Unit Testing
- Backend: pytest; coverage ≥ 80%; mocks for providers and storage.
- Retrieval: golden tests for precision/recall on curated datasets.

### 16.2 Integration Testing
- End-to-end flows: index->retrieve->prompt->LLM->tools->patch.
- IDE plugin integration with emulator; snapshot tests for UI.

### 16.3 Performance Testing
- Load tests with Locust/K6; latency SLO validation; vector store benchmarks.

### 16.4 Security Testing
- SAST, DAST, dependency scanning, container image scans.
- Pen tests and threat modeling; authz fuzzing; policy bypass tests.

## 17. Deployment Strategy
### 17.1 Deployment Pipeline
- GitHub Actions: lint, unit tests, build, security scans, container build/push.
- Canary deployments via Argo Rollouts; automated health checks.

### 17.2 Environments
- Dev: feature branches; ephemeral preview environments.
- Staging: nightly sync; synthetic load and eval runs.
- Prod: multi-region; blue/green with canary.

### 17.3 Rollout Plan
- Phase 1: Internal teams; gather telemetry; iterate.
- Phase 2: Design partners; enable SSO and enterprise policies.
- Phase 3: General availability with billing.

### 17.4 Rollback Procedures
- Immediate traffic shift to previous stable; feature flag disablement.
- DB migrations reversible; backup restore runbooks.

## 18. Monitoring & Observability
### 18.1 Metrics
- Request latency (p50/p95/p99) by endpoint.
- pass@k, repair rate, edit acceptance rate.
- Retrieval precision/recall, MRR.
- Token usage and cost per request.
- Index freshness, failure rates.

### 18.2 Logging
- Structured JSON logs with request IDs; PII-scrubbed.
- Tool run logs stored in object store with retention policies.

### 18.3 Alerting
- On-call alerts for SLO breaches, error spikes, index lag, cost anomalies.

### 18.4 Dashboards
- Grafana: latency, throughput, error budgets.
- Product analytics: adoption, acceptance, retention.

## 19. Risk Assessment
### 19.1 Technical Risks
- Hallucinations or unsafe edits.
- LLM provider outages or rate limits.
- Cost overruns due to token usage.
- Retrieval drift on evolving repos.
- Tooling environment inconsistencies.

### 19.2 Business Risks
- Low adoption if trust not established.
- Compliance concerns for code data.
- Vendor lock-in perceptions.

### 19.3 Mitigation Strategies
- Strict grounding with citations; grammar-constrained outputs.
- Multi-provider router; local fallback models.
- Budget caps, caching, prompt compression.
- Continuous evaluation; incremental indexing; recency bias.
- Standardized containers for tools; reproducible environments.

## 20. Timeline & Milestones
### 20.1 Phase Breakdown
- Phase 0 (2 weeks): Discovery, requirements, design spikes.
- Phase 1 (6 weeks): Retrieval/indexing, embeddings, reranker MVP.
- Phase 2 (6 weeks): Prompt builder, LLM router, inline completions.
- Phase 3 (6 weeks): Chat with citations, diff generation, tool integration.
- Phase 4 (4 weeks): IDE plugins polish, dashboards, analytics.
- Phase 5 (4 weeks): Security hardening, SSO, compliance prep, load tests.

Total: 28 weeks to GA.

### 20.2 Key Milestones
- M1 (Week 8): Indexing + retrieval precision@10 ≥ 0.65 on internal set.
- M2 (Week 14): Inline completion p95 ≤ 600 ms; basic chat with citations.
- M3 (Week 20): Diff+tests agent producing green builds on 25% of curated bugs.
- M4 (Week 24): IDE plugins beta with 5 design partners.
- GA (Week 28): Uptime 99.5%, acceptance rate ≥ 40%, pass@3 ≥ 60%.

Estimated Costs (first 6 months):
- Team: 8 FTE (3 backend, 2 ML, 1 frontend, 1 plugin, 1 SRE) ≈ $1.6M all-in.
- Infra: $20–60k/month (LLM provider + GPUs optional), $10k/month storage/DB/observability.
- Misc/compliance: $50k.

## 21. Success Metrics & KPIs
### 21.1 Measurable Targets
- Edit acceptance rate ≥ 50% @ 6 months.
- Repair rate ≥ 35% with agent loop ≤ 3 tool runs median.
- pass@3 ≥ 65% on SWE-bench-lite; HumanEval+ ≥ 90% accuracy.
- Inline completion p95 ≤ 500 ms; chat p95 ≤ 2 s.
- Retrieval precision@10 ≥ 0.75; MRR ≥ 0.6.
- Uptime ≥ 99.5%; error rate < 0.5%.
- DAUs ≥ 200 per mid-size org; NPS ≥ 45.

## 22. Appendices & Glossary
### 22.1 Technical Background
- Retrieval-augmented generation for code leverages hybrid search over code embeddings and lexical signals to gather precise, syntactically coherent context. AST-aware chunking improves symbol integrity and reduces context fragmentation. Grammar-constrained decoding ensures model outputs adhere to expected formats (diffs, JSON), reducing hallucinations and enabling direct application. Tool-augmented agents execute and validate hypotheses via tests and linters, closing the loop on code correctness and improving trust.

### 22.2 References
- SWE-bench: https://www.swe-bench.com/
- HumanEval: https://github.com/openai/human-eval
- MBPP: https://github.com/google-research/google-research/tree/master/mbpp
- tree-sitter: https://tree-sitter.github.io/tree-sitter/
- vLLM: https://github.com/vllm-project/vllm
- OpenTelemetry: https://opentelemetry.io/

### 22.3 Glossary
- RAG: Retrieval-Augmented Generation; using external context retrieved at inference time.
- AST: Abstract Syntax Tree representing code structure.
- pass@k: Probability that at least one of k samples solves a task.
- MMR: Maximal Marginal Relevance; balances relevance and diversity in retrieval.
- MRR: Mean Reciprocal Rank; metric for ranking quality.
- Constrained decoding: Forcing outputs to follow a schema/grammar.
- Function calling: LLM capability to invoke tools via structured arguments.
- Prompt compression: Reducing prompt length via summarization, deduplication, or packing.
- p95 latency: 95th percentile response time.
- RBAC: Role-Based Access Control.

Repository Structure
- /
  - README.md
  - configs/
    - default.yaml
    - providers.yaml
  - src/
    - api/
      - main.py
      - routers/
        - auth.py
        - repos.py
        - complete.py
        - chat.py
        - diff.py
        - tools.py
        - eval.py
    - retrieval/
      - indexer.py
      - chunking/
        - ast_chunker.py
      - search.py
      - rerank.py
      - schema.py
    - llm/
      - router.py
      - prompts.py
      - decoding.py
      - schemas.py
    - agents/
      - planner.py
      - tools/
        - pytest_runner.py
        - linter_runner.py
        - code_search.py
    - safety/
      - pii.py
      - license.py
      - secrets.py
    - storage/
      - db.py
      - vector.py
      - objects.py
    - telemetry/
      - metrics.py
      - logging.py
    - utils/
      - diff.py
      - vcs.py
  - notebooks/
    - evaluation.ipynb
    - retrieval_tuning.ipynb
  - tests/
    - api/
    - retrieval/
    - llm/
    - agents/
  - data/
    - eval/
    - samples/
  - plugins/
    - vscode/
    - jetbrains/

Sample Config (configs/default.yaml)
```
app:
  environment: "prod"
  region: "us-east-1"
providers:
  llm:
    strategy: "latency_cost"
    defaults:
      model: "gpt-4.1"
      temperature: 0.2
      max_tokens: 1024
  embeddings:
    model: "text-embedding-3-large"
retrieval:
  top_k: 40
  rerank_k: 12
  mrr_bias: 0.2
  recency_half_life_hours: 72
policies:
  pii_redaction: true
  allow_external_snippets: false
```

API Code Snippet (FastAPI)
```
from fastapi import FastAPI, Depends
from pydantic import BaseModel

app = FastAPI()

class CompletionReq(BaseModel):
    repo_id: str
    file_path: str
    language: str
    cursor: dict
    buffer: str
    editor_state: dict | None = None
    preferences: dict | None = None

@app.post("/v1/complete")
async def complete(req: CompletionReq, user=Depends(require_auth)):
    ctx = await build_context(req)
    prompt = build_prompt(ctx, req)
    result = await llm_router.complete(prompt, schema=None, temperature=req.preferences.get("temperature", 0.2))
    return {"id": gen_id(), "choices": result.choices, "latency_ms": result.latency_ms, "citations": ctx.citations}
```

Editor Patch Command Example (JSON schema output)
```
{
  "type": "patch",
  "patches": [
    {
      "path": "src/service/pager.py",
      "op": "replace_range",
      "start": {"line": 42, "column": 0},
      "end": {"line": 58, "column": 0},
      "content": "def paginate(items, page, page_size=20):\n    if page_size <= 0: page_size = 20\n    ..."
    }
  ],
  "commit_message": "Fix pagination default and add guard for page_size<=0",
  "tests": ["tests/test_pager.py::test_default_page_size"]
}
```

Service-level Objectives
- Accuracy: HumanEval+ ≥ 90% tasks correct; SWE-bench-lite pass@3 ≥ 65%.
- Latency: inline p95 ≤ 500 ms; chat p95 ≤ 2 s.
- Availability: ≥ 99.5% monthly uptime.

End of PRD.