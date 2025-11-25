# Product Requirements Document (PRD) / # `aiml030_interactive_data_analysis_assistant`

Project ID: AIML-030
Category: AI/ML – Interactive Analytics Assistant
Status: Draft for Review
Version: 1.0
Last Updated: 2025-11-25

## 1. Overview
### 1.1 Executive Summary
aiml030_interactive_data_analysis_assistant is a conversational analytics agent that enables users to explore, analyze, and visualize data using natural language. It orchestrates an LLM-driven planning loop to clarify goals, generate SQL/DataFrame code, run computations in a secure sandbox, produce charts and narratives, and iterate interactively. It connects to files, databases, and object storage, automatically discovers schemas, and grounds its reasoning using retrieval-augmented generation (RAG) over data dictionaries, metric definitions, and prior analyses to reduce errors and hallucinations. The assistant provides robust guardrails, observability, and enterprise-grade security.

### 1.2 Document Purpose
This PRD defines objectives, scope, requirements, architecture, data model, APIs, UI/UX, security, performance, testing, deployment, risks, milestones, and success metrics for building and launching the product.

### 1.3 Product Vision
Deliver a trustworthy, low-latency, self-healing conversational interface that democratizes data analysis for technical and non-technical users by combining LLM orchestration, code execution, data connectors, visualization, and RAG grounding into a cohesive, secure, and delightful experience.

## 2. Problem Statement
### 2.1 Current Challenges
- Analysts spend excessive time translating stakeholder questions into SQL/DataFrame code and visualization.
- Knowledge of schemas, metrics, and business logic is siloed; high onboarding cost.
- Manual iteration cycles between question, code, run, debug, and chart are slow and error-prone.
- Existing BI tools require rigid dashboards; ad-hoc analysis is cumbersome.
- LLMs without grounding hallucinate columns/metrics and generate incorrect code.

### 2.2 Impact Analysis
- Slow insights delay decisions; high opportunity cost.
- Inconsistent definitions lead to conflicting metrics.
- Engineering bottlenecks for ad-hoc requests.
- Repeated work due to lack of reusable analysis snippets and memory.

### 2.3 Opportunity
- A conversational, grounded, tool-using agent can accelerate analysis, reduce errors, capture institutional knowledge, and empower broader audiences.
- RAG over data dictionaries/metrics/playbooks improves correctness.
- Secure sandboxing and guardrails enable safe automation of code/SQL execution.

## 3. Goals and Objectives
### 3.1 Primary Goals
- Provide natural-language to analysis (SQL/DataFrames/charts) with high accuracy and transparency.
- Ensure safe, reproducible execution via sandboxing, validation, and audit trails.
- Support multi-source data connectivity and schema discovery.
- Deliver concise narrative summaries with citations and confidence indicators.

### 3.2 Business Objectives
- Reduce time-to-insight by 60% for ad-hoc queries.
- Decrease analyst workload for routine questions by 50%.
- Improve data tool adoption and satisfaction (CSAT > 4.3/5).
- Enable new revenue via premium enterprise features (SSO, RBAC, audit).

### 3.3 Success Metrics
- NL2SQL exact match accuracy ≥ 90% on curated benchmark.
- Task completion rate ≥ 85% on representative tasks.
- Median end-to-chart latency ≤ 3s; cached responses ≤ 500ms.
- Uptime ≥ 99.5%.
- Data error/hallucination incidents ≤ 1 per 1,000 queries.

## 4. Target Users/Audience
### 4.1 Primary Users
- Data analysts and analytics engineers
- Product managers and business analysts
- Data scientists

### 4.2 Secondary Users
- Operations, marketing, finance stakeholders
- Customer success managers
- Engineering managers needing quick metrics

### 4.3 User Personas
- Persona 1: Maya Chen – Senior Data Analyst
  - Background: 6+ years SQL, Python (pandas), builds dashboards and ad-hoc analyses.
  - Pain Points: Context switching across tools, repetitive boilerplate, tribal knowledge of metric definitions, manual chart formatting.
  - Goals: Rapid prototyping, reusable snippets, trustworthy NL2SQL, transparent provenance and auditability.
- Persona 2: Alex Rodriguez – Product Manager
  - Background: SQL novice, uses spreadsheets and dashboards; needs fast answers for roadmap decisions.
  - Pain Points: Long wait times for analyst support, confusing schemas, inconsistent metric definitions.
  - Goals: Ask questions in natural language, get clear visualizations and plain-language summaries with caveats.
- Persona 3: Priya Nair – Data Scientist
  - Background: Python/R, ML modeling; uses notebooks; needs exploratory data analysis before modeling.
  - Pain Points: Time spent on data wrangling and plotting; documenting EDA steps; context on data quality.
  - Goals: Conversational EDA, reproducible code blocks, export to notebooks, profiling insights and anomalies.
- Persona 4: Jordan Lee – BI Platform Owner
  - Background: Manages data platform, security, and governance.
  - Pain Points: Data access control, auditing usage, cost monitoring, PII handling.
  - Goals: Enterprise SSO/RBAC, guardrails, observability, spend controls.

## 5. User Stories
- US-001: As a PM, I want to ask “What were weekly active users last quarter by region?” so that I can make prioritization decisions.
  - Acceptance: Assistant clarifies metric definition if ambiguous; generates SQL/DataFrame; returns correct grouped results with chart and narrative; cites metric definition and tables used.
- US-002: As an analyst, I want to upload a CSV and join it with a database table so that I can compare cohorts.
  - Acceptance: Upload succeeds, schema inferred, data profile shown; join preview on sample, full run on confirm; result chart + downloadable CSV.
- US-003: As a data scientist, I want to detect missing values and outliers automatically so that I can assess data quality.
  - Acceptance: Profiling report with nulls, type detection, distributions, outlier flags; downloadable report and code cell.
- US-004: As a platform owner, I want RBAC and audit logs so that I meet compliance requirements.
  - Acceptance: Roles and permissions enforced; all queries, tool runs, and data access logged with timestamp, user, dataset; exportable audit.
- US-005: As an analyst, I want the assistant to self-correct errors so that I don’t debug stack traces manually.
  - Acceptance: On error, assistant retries with fix (≤2 retries), shows diagnosis and final working result or clear fallback.
- US-006: As a PM, I want consistent color palettes and labels so that charts are presentation-ready.
  - Acceptance: Visuals adhere to theme, sensible defaults, accessible colors, proper labels/units/formatting.
- US-007: As a user, I want to save and share sessions so that my team can reproduce results.
  - Acceptance: Sessions persist; shared via link with proper permissions; re-runs deterministic given same data snapshot.

## 6. Functional Requirements
### 6.1 Core Features
- FR-001: Conversational interface with LLM-driven planning, tool selection, and iterative clarification.
- FR-002: Natural language to SQL with schema-aware prompting, guarded execution, and dry-run validation.
- FR-003: Natural language to DataFrame operations (pandas/polars): joins, filters, aggregates, windows.
- FR-004: Data connectors for CSV/Parquet/Excel, SQLite/Postgres/BigQuery, S3/GCS/Azure Blob.
- FR-005: Schema discovery, profiling (nulls, types, stats), metric/dimension catalog build.
- FR-006: Visualization auto-selection and spec generation (Vega-Lite/Altair/Plotly), with consistent theming.
- FR-007: Narrative generation with uncertainty and caveats; citation of sources and definitions.
- FR-008: Secure code execution sandbox with resource limits, allowlisted packages, and output capture.
- FR-009: RAG grounding over data dictionaries, ERDs, metric definitions, prior notebooks/code and playbooks.
- FR-010: Guardrails: SQL linting, schema constraints, sampling, reasonableness checks.
- FR-011: Error handling and self-repair using exception parsing and retry strategies.
- FR-012: Memory and session state: short-term conversational context and long-term preferences.
- FR-013: Export: code cells, notebooks, CSV/Parquet results, chart images/JSON specs.
- FR-014: Audit, provenance, and query cost/latency telemetry.

### 6.2 Advanced Features
- AFR-001: Semantic cache for prompts/results; parameterized template reuse.
- AFR-002: Hybrid retrieval: dense embeddings + BM25 with re-ranking; metadata filtering.
- AFR-003: Semantic code retrieval for prior successful SQL/DataFrame and visualization recipes.
- AFR-004: Interactive chart editing and “what-if” parameterization.
- AFR-005: Scheduled insights with drift detection and anomaly alerts.
- AFR-006: Multi-tenant enterprise mode with SSO, fine-grained RBAC, and data masking.
- AFR-007: Budget-aware execution planning (latency/cost budgets) and streaming partial outputs.

## 7. Non-Functional Requirements
### 7.1 Performance
- P95 chat-to-first-token latency ≤ 700ms (LLM streaming).
- P50 end-to-chart for simple queries ≤ 3s; cached ≤ 500ms.
- Query timeouts default 30s; cancelable.

### 7.2 Reliability
- Uptime ≥ 99.5%.
- Durable session storage; no data loss on restarts.
- Idempotent API for retries; exactly-once execution for tool runs via job IDs.

### 7.3 Usability
- Learnable in <10 minutes with onboarding tips.
- Accessibility AA compliance.
- Consistent UI patterns, undo for destructive actions.

### 7.4 Maintainability
- Modular services with clear contracts.
- >85% unit test coverage on core libraries; linting, type checks (mypy).
- Backward-compatible API versioning for 12 months.

## 8. Technical Requirements
### 8.1 Technical Stack
- Backend: Python 3.11+, FastAPI 0.115+, Uvicorn 0.30+, SQLAlchemy 2.0+.
- Frontend: React 18+, TypeScript 5+, Next.js 14+, Chakra UI or MUI, Vega-Lite 5+/Plotly 5+.
- Data: PostgreSQL 15+ (primary), Redis 7+ (cache/queues), object storage (S3/GCS), pgvector 0.5+ or Milvus 2.3+ (vector index).
- Sandbox: Docker 26+ with gVisor or Firecracker; optional Pyodide/Wasm for browser-only mode.
- Orchestration: Celery 5+ or Arq; Kubernetes 1.29+ for deployment.
- Observability: OpenTelemetry 1.25+, Prometheus 2.49+, Grafana 10+, Loki or ELK, Sentry.
- Auth: OAuth2/OIDC (Auth0, Okta, Azure AD), JWT (RS256).
- CI/CD: GitHub Actions, Docker Buildx, Helm 3+.

### 8.2 AI/ML Components
- LLM: Hosted API or self-hosted (OpenAI GPT-4o/GPT-4.1, Anthropic Claude 3.5, or Llama 3.1 70B via vLLM).
- Embeddings: text-embedding-3-large or bge-large; fallback open-source (gte-large).
- Re-ranking: cross-encoder (e.g., ms-marco-MiniLM-L-6-v2) via sentence-transformers.
- Vector DB: FAISS for local dev; pgvector/Milvus production.
- Evaluation: NL2SQL benchmarks (Spider-like, custom corpora), retrieval precision/recall, human preference for charts.

## 9. System Architecture
### 9.1 High-Level Architecture (ASCII diagram)
+-----------------------+        +-------------------+
|      Web Client       | <----> |  API Gateway      |
| React/Next + Websockets|       |  FastAPI + Auth   |
+----------+------------+        +---------+---------+
           |                               |
           |                               v
           |                    +----------+-----------+
           |                    |  Orchestrator        |
           |                    |  (LLM Agent, Planner)|
           |                    +----+--------+--------+
           |                         |        |
           |                  tool calls       | RAG
           |                         |        v
           |                    +----+--------+-----------+
           |                    |   Tools & Executors     |
           |                    | - SQL Runner            |
           |                    | - Py Sandbox            |
           |                    | - Viz Generator         |
           |                    +----+--------+-----------+
           |                         |        |
           v                         v        v
+----------+-----------+   +---------+--+  +--+------------------+
|  Postgres (metadata) |   |  Redis     |  | Vector Index        |
| users/sessions/data  |   | cache/queue|  | (pgvector/Milvus)   |
+----------+-----------+   +------------+  +---------------------+
           |                         |
           v                         v
+----------+-----------+   +-------------------+
| Data Sources         |   | Object Storage    |
| Files/DB (connectors)|   | Results, Artifacts|
+----------------------+   +-------------------+

### 9.2 Component Details
- API Gateway: REST + WebSocket for streaming, auth, rate limiting.
- Orchestrator: Manages conversation state, tool selection, planning, retries, and memory; applies cost/latency budgets and semantic cache.
- Tools & Executors:
  - SQL Runner: Schema-aware SQL generation with dry-run, sampling, and constraints.
  - Python Sandbox: Executes DataFrame code (pandas/polars), generates plots (Altair/Plotly), enforces time/memory/CPU limits.
  - Viz Generator: Produces declarative chart specs with theme and accessibility checks.
- RAG Layer: Hybrid retrieval over data dictionaries, metric definitions, ERDs, code snippets, prior notebooks, and analysis playbooks; re-ranking and metadata filtering.
- Data Connectors: CSV/Excel/Parquet, SQLite/Postgres/BigQuery, S3/GCS/Azure.
- Observability: Telemetry for tool calls, costs, latencies; logging and tracing.
- Storage: Postgres for metadata; Redis for caching and queues; vector index for embeddings.

### 9.3 Data Flow
1) User asks question via chat.
2) Orchestrator retrieves relevant schema/metrics/playbooks via RAG.
3) Plan decomposed into steps (clarify, draft SQL/DataFrame, validate, execute).
4) SQL/DataFrame generated; dry-run on sample; guardrails applied.
5) Sandbox executes code; results returned; visualization spec generated.
6) Narrative generated with citations; memory updated; telemetry logged.
7) UI streams intermediate status and final results.

## 10. Data Model
### 10.1 Entity Relationships
- User 1—* Session
- Session 1—* Message
- User *—* DataSource (via Permission)
- DataSource 1—* Table 1—* Column
- MetricDefinition *—* Table/Column (via mapping)
- AnalysisTask 1—* ToolRun
- Session *—* Artifact (ChartSpec, Dataset Export)
- VectorIndexEntry linked to Doc (dictionary, notebook, code snippet)
- AuditLog linked to User, DataSource, ToolRun

### 10.2 Database Schema (selected tables)
- users(id PK, org_id, email UNIQUE, name, role, prefs JSONB, created_at)
- sessions(id PK, user_id FK, title, context JSONB, created_at, updated_at)
- messages(id PK, session_id FK, role ENUM, content TEXT, artifacts JSONB, created_at)
- datasources(id PK, org_id, type ENUM, name, config JSONB, created_at)
- permissions(id PK, user_id, datasource_id, role ENUM, created_at)
- tables(id PK, datasource_id, name, schema JSONB, profile JSONB, created_at)
- columns(id PK, table_id, name, dtype, description, pii_tag, stats JSONB)
- metric_definitions(id PK, name, description, sql_expr, grain, owner, version, tags, created_at)
- vector_docs(id PK, org_id, doc_type, title, content TEXT, metadata JSONB, created_at)
- vector_index(id PK, doc_id FK, embedding VECTOR(1536), metadata JSONB)
- tool_runs(id PK, session_id, tool_name, status, input JSONB, output JSONB, error TEXT, duration_ms, cost_cents, created_at)
- chart_specs(id PK, session_id, spec JSONB, theme, created_at)
- audit_logs(id PK, user_id, action, target_type, target_id, details JSONB, created_at)
- cache_entries(key PK, value BYTEA, ttl, created_at)

### 10.3 Data Flow Diagrams (ASCII)
User -> Orchestrator -> RAG -> Plan -> SQL/Py -> Sandbox -> Results -> Viz -> Narrative -> UI
[User] -> [Orchestrator] -> [Retriever] -> [Planner]
[Planner] -> [SQL Gen] -> [Dry-run] -> [Executor] -> [Results]
[Planner] -> [Py Gen] -> [Sandbox] -> [Artifacts]
[Results] -> [Viz] -> [Narrative] -> [UI]
Telemetry: [All steps] -> [Metrics/Logs/Traces]

### 10.4 Input Data & Dataset Requirements
- File formats: CSV, Parquet, Excel (XLSX).
- DBs: SQLite, Postgres, BigQuery; read-only credentials recommended.
- Object storage: S3/GCS/Azure; signed URLs or IAM roles.
- Schema discovery: infer types, datetime parsing, unit detection; sample ≥ 10k rows where feasible.
- PII tagging: optional detection via regex/ML; support masking policies.

## 11. API Specifications
### 11.1 REST Endpoints
- POST /v1/auth/login – OIDC code exchange (if not using hosted).
- GET /v1/me – current user profile.
- POST /v1/sessions – create a session {title, context}.
- GET /v1/sessions/{id} – get session details and messages.
- POST /v1/sessions/{id}/message – send message {content}; streams response via WS.
- POST /v1/datasources – register data source {type, config}.
- GET /v1/datasources – list data sources.
- POST /v1/datasources/{id}/profile – trigger schema discovery.
- POST /v1/query/sql – execute generated SQL (guarded).
- POST /v1/query/python – run DataFrame code in sandbox.
- GET /v1/charts/{id} – fetch chart spec.
- GET /v1/metrics/definitions – list metric definitions.
- POST /v1/exports – export results (CSV/Parquet/Notebook).
- GET /v1/audit – audit logs (admin).
- GET /v1/health – liveness/readiness.

WebSocket:
- /v1/stream/{session_id} – stream tokens, statuses, and events.

### 11.2 Request/Response Examples
- Create session:
  Request:
  POST /v1/sessions
  { "title": "WAU by region Q3", "context": {"timezone":"UTC"} }

  Response:
  201
  { "id":"sess_123", "title":"WAU by region Q3", "created_at":"..." }

- Send message:
  POST /v1/sessions/sess_123/message
  { "content": "Show weekly active users by region last quarter" }

  Response (initial):
  202
  { "status":"streaming", "stream_url":"/v1/stream/sess_123" }

- SQL query:
  POST /v1/query/sql
  { "datasource_id":"ds_42", "sql":"SELECT region, COUNT(DISTINCT user_id) ...", "sample": true }

  Response:
  200
  { "rows":[{"region":"EMEA","wau":12345}], "columns":[{"name":"region","type":"TEXT"},{"name":"wau","type":"INT"}], "sampled": true }

- Chart fetch:
  GET /v1/charts/ch_789
  Response:
  200
  { "spec": { "$schema":"https://vega.github.io/schema/vega-lite/v5.json", "mark":"bar", ... }, "theme":"default" }

### 11.3 Authentication
- OAuth2/OIDC with hosted IdP; JWT (RS256) access tokens; optional API keys for service-to-service.
- Scopes: read:session, write:session, run:query, admin:audit, manage:datasource.
- CSRF protection for cookie-based sessions; HTTPS enforced.

## 12. UI/UX Requirements
### 12.1 User Interface
- Chat workspace with:
  - Message composer with slash-commands (/upload, /profile, /explain).
  - Streaming assistant responses with step badges (retrieving, planning, executing, visualizing).
  - Result blocks: data grid preview, charts, narratives, citations.
  - Side panel: data catalog (tables, columns, metrics), recent snippets, session memory.
  - Upload area for files; connector wizard.
- Theme: light/dark; accessible color palettes; consistent spacing and typography.

### 12.2 User Experience
- Onboarding tips and sample questions; tooltips for metric definitions.
- Clarification questions when ambiguity detected.
- Inline edit of chart spec and code cell with “apply” and “revert”.
- Share and export in one click.
- Undo/redo, copy SQL/Python.

### 12.3 Accessibility
- WCAG 2.1 AA: keyboard navigation, ARIA roles, focus management.
- High-contrast theme; colorblind-safe palettes.
- Alt text for charts; descriptive labels.

## 13. Security Requirements
### 13.1 Authentication
- OIDC with MFA support; short-lived tokens; refresh token rotation.
- Service accounts for scheduled jobs; least-privilege scopes.

### 13.2 Authorization
- RBAC: Admin, Editor, Viewer; per-datasource permissions.
- Row/column-level security via source policies when available.
- Data masking for PII fields; policy-based redaction.

### 13.3 Data Protection
- Encryption in transit (TLS 1.2+); at rest (KMS-managed keys).
- Secret management via Vault or cloud KMS; no secrets in code.
- Sandbox isolation (namespaces, seccomp, no outbound network by default).

### 13.4 Compliance
- Logging and audit trails retained ≥ 12 months.
- Data retention and deletion per org policies; GDPR/CCPA support.
- DLP hooks for uploads; consent and purpose limitation.

## 14. Performance Requirements
### 14.1 Response Times
- Token streaming start P50 ≤ 700ms; P95 ≤ 1.2s.
- Simple cached analysis P50 ≤ 500ms.
- Heavy queries P95 ≤ 10s with progress updates.

### 14.2 Throughput
- Support 200 concurrent active sessions per node; scale linearly with nodes.
- 50 QPS API sustained with auto-scaling.

### 14.3 Resource Usage
- Sandbox: default 2 vCPU, 2GB RAM, 60s hard timeout; configurable.
- LLM cost budget per session with alerts; semantic cache hit rate target ≥ 40%.

## 15. Scalability Requirements
### 15.1 Horizontal Scaling
- Stateless API pods; HPA based on CPU/RPS.
- Separate worker pool for sandbox execution.

### 15.2 Vertical Scaling
- Larger nodes for vector search and embeddings; GPU optional for self-hosted LLM.

### 15.3 Load Handling
- Backpressure via queues; rate limiting per org/user.
- Priority lanes for enterprise tier; graceful degradation to cached answers.

## 16. Testing Strategy
### 16.1 Unit Testing
- Python: pytest, hypothesis for code-gen edge cases; coverage ≥ 85%.
- Frontend: Jest/RTL; snapshot tests for components.

### 16.2 Integration Testing
- NL2SQL end-to-end on synthetic and real schemas.
- Sandbox execution, data connectors (using ephemeral containers).
- RAG retrieval correctness with golden sets.

### 16.3 Performance Testing
- Locust/Gatling for API load; k6 for WebSocket streaming.
- Benchmark NL2SQL latency and accuracy; vector retrieval p50/p95.

### 16.4 Security Testing
- SAST (Bandit), DAST (OWASP ZAP), dependency scanning.
- Pen tests on sandbox isolation and auth flows.
- Fuzzing on SQL generator and parser.

## 17. Deployment Strategy
### 17.1 Deployment Pipeline
- GitHub Actions: lint, type-check, tests, build images, push to registry.
- Staging deploy via Helm; smoke tests; manual approval for prod.
- Blue/green or canary with 10%-50%-100% ramp.

### 17.2 Environments
- Dev (shared), Staging (prod-like with masked data), Prod (HA).
- Feature preview envs via ephemeral namespaces.

### 17.3 Rollout Plan
- Internal alpha (2 weeks) → Design partner beta (6 weeks) → GA.
- Feature flags for advanced features (enterprise RBAC, scheduling).

### 17.4 Rollback Procedures
- Helm chart version pinning; one-click rollback.
- DB migrations backward-compatible; shadow writes before cutover.
- Preserve previous container images for 30 days.

## 18. Monitoring & Observability
### 18.1 Metrics
- API: RPS, latency (p50/p95), error rates.
- LLM: token usage, cost per session, cache hit rate.
- RAG: retrieval precision/recall on sampled queries.
- SQL: query duration, timeouts, row counts.
- Sandbox: CPU/mem, run time, failure rate.

### 18.2 Logging
- Structured JSON logs with correlation IDs; PII redaction.
- Separate channels for access, app, and audit logs.

### 18.3 Alerting
- SLO breaches (latency, error rate).
- Cost anomalies (token or compute spikes).
- Data source failures and connector errors.

### 18.4 Dashboards
- Grafana: API health, LLM usage, RAG quality, query performance.
- Sentry issues dashboard for client and server.

## 19. Risk Assessment
### 19.1 Technical Risks
- LLM hallucinations leading to incorrect SQL/code.
- Sandbox escape/abuse risks.
- Data source schema drift breaking queries.
- Vendor lock-in for LLMs/embeddings.

### 19.2 Business Risks
- User trust erosion from incorrect answers.
- High inference costs exceeding budgets.
- Adoption blocked by security/compliance concerns.

### 19.3 Mitigation Strategies
- Strong RAG grounding, schema-constrained generation, dry-runs, and reasonableness checks.
- Isolation layers (gVisor/Firecracker), no outbound network, allowlist packages.
- Schema change detection and auto-adaptation; notify users.
- Pluggable LLM providers; on-prem option; semantic caching.

## 20. Timeline & Milestones
### 20.1 Phase breakdown
- Phase 0 (2 weeks): Discovery, requirements, data schemas inventory, eval plan.
- Phase 1 (4 weeks): Core backend (sessions, auth, connectors), basic NL2SQL, sandbox MVP.
- Phase 2 (4 weeks): Visualization, narrative, RAG v1, guardrails, telemetry.
- Phase 3 (6 weeks): Advanced retrieval, semantic cache, RBAC/audit, UI polish.
- Phase 4 (6 weeks): Beta hardening, performance, security, docs, enterprise features.
- Total: ~22 weeks to GA.

Estimated team: 1 PM, 1 Designer, 3 Backend, 2 Frontend, 1 MLE, 1 DevOps, 0.5 SecEng.

Estimated cost (6 months):
- Personnel: ~$1.6M (fully loaded, region-dependent).
- Infra (staging+prod): ~$10k–$30k/month depending on LLM usage.
- Pen test and compliance: ~$50k.

### 20.2 Key Milestones
- M1 (Week 4): Sessions, file uploads, basic chat, CSV profiling.
- M2 (Week 8): NL2SQL + SQL execution with guardrails; basic charts.
- M3 (Week 12): RAG v1 over dictionaries/metrics; narratives with citations.
- M4 (Week 18): RBAC, audit, semantic cache; UI v2 with chart editor.
- GA (Week 22): Performance SLOs met; docs, onboarding, support.

## 21. Success Metrics & KPIs
### 21.1 Measurable targets
- NL2SQL exact match ≥ 90%; F1 ≥ 93%.
- Retrieval precision@5 ≥ 0.85; recall@20 ≥ 0.9.
- Task success rate (human-judged) ≥ 85%.
- P50 end-to-chart ≤ 3s; cached ≤ 500ms; uptime ≥ 99.5%.
- User CSAT ≥ 4.3/5; NPS ≥ 35.
- Analyst time saved ≥ 50% for routine tasks (survey and time logs).
- Chart quality preference win-rate ≥ 70% vs. baseline.
- Cost per answered question ≤ $0.05 (cached) / ≤ $0.50 (fresh).

## 22. Appendices & Glossary
### 22.1 Technical Background
- Orchestrated LLM agents with tool use: function-calling APIs allow the model to plan and call tools for SQL/Python execution and visualization.
- RAG: Hybrid retrieval (dense + sparse) with re-ranking to ground generation in authoritative docs and schema metadata, reducing hallucinations.
- Query decomposition: Self-ask and step-back strategies to clarify metric definitions and disambiguate columns; session-aware retrieval to respect active datasets.
- Guardrails: Schema-constrained SQL generation, linting for dangerous operations, dry-run on samples, and cross-checks (totals, constraints).
- Observability: Token/cost tracking, vector retrieval metrics, and offline evals for NL2SQL accuracy and chart quality.

### 22.2 References
- Yu et al., Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL.
- Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP.
- Wei et al., Chain-of-Thought Prompting Elicits Reasoning (applied via concise reasoning variants).
- Sentence-Transformers models for re-ranking.
- Vega-Lite documentation for declarative visualization.
- Pandas, Polars, Altair, Plotly documentation.

### 22.3 Glossary
- LLM: Large Language Model used for planning and generation.
- RAG: Retrieval-Augmented Generation to ground responses in documents/data.
- NL2SQL: Natural language to SQL translation.
- DataFrame: In-memory tabular data structure (pandas/polars).
- Guardrails: Constraints and validations applied to generated code/queries.
- Semantic cache: Cache keyed by meaning rather than exact text.
- Vector database: Index for embeddings to support semantic retrieval.
- Re-ranking: Secondary scoring to improve retrieval quality.
- RBAC: Role-Based Access Control.
- PII: Personally Identifiable Information.

Repository Structure
- /
  - README.md
  - pyproject.toml
  - package.json
  - src/
    - api/
      - main.py
      - routers/
        - auth.py
        - sessions.py
        - datasources.py
        - query.py
        - charts.py
        - audit.py
      - middleware/
      - models/
      - schemas/
    - orchestrator/
      - agent.py
      - planner.py
      - tools/
        - sql_runner.py
        - python_sandbox.py
        - viz_generator.py
        - retriever.py
        - guardrails.py
      - memory/
        - short_term.py
        - long_term.py
        - semantic_cache.py
    - rag/
      - indexer.py
      - retrievers/
      - rerankers/
    - connectors/
      - files.py
      - postgres.py
      - sqlite.py
      - bigquery.py
      - s3.py
      - gcs.py
    - observability/
      - telemetry.py
      - tracing.py
      - logging.py
    - security/
      - auth.py
      - rbac.py
      - masking.py
    - workers/
      - worker.py
  - frontend/
    - app/
      - pages/
      - components/
      - hooks/
      - styles/
  - tests/
    - unit/
    - integration/
    - e2e/
  - configs/
    - config.yaml
    - logging.yaml
    - helm/
  - notebooks/
    - evaluation/
    - examples/
  - data/
    - samples/
    - dictionaries/
  - scripts/
    - seed_data.py
    - migrate.py
  - docker/
    - Dockerfile.api
    - Dockerfile.sandbox
    - docker-compose.yml

Code Snippets

- FastAPI router (sessions)
from fastapi import APIRouter, Depends
from .schemas import SessionCreate, SessionOut
from .auth import get_current_user
from .db import create_session

router = APIRouter(prefix="/v1/sessions", tags=["sessions"])

@router.post("", response_model=SessionOut, status_code=201)
async def create_session_ep(body: SessionCreate, user=Depends(get_current_user)):
    session = await create_session(user.id, body)
    return session

- Python client usage
import requests
r = requests.post("https://api.example.com/v1/sessions", json={"title":"Exploration"})
sid = r.json()["id"]
msg = requests.post(f"https://api.example.com/v1/sessions/{sid}/message",
                    json={"content":"Show revenue by month this year"}).json()
print(msg)

- Config sample (config.yaml)
server:
  host: 0.0.0.0
  port: 8080
llm:
  provider: openai
  model: gpt-4o
  temperature: 0.2
  max_tokens: 1500
retrieval:
  index: pgvector
  top_k: 8
  reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
sandbox:
  cpu: "2"
  memory: "2Gi"
  timeout_sec: 60
security:
  allowlist_packages:
    - pandas==2.2.2
    - polars==1.6.0
    - numpy==2.1.1
    - altair==5.3.0
    - plotly==5.24.0

Evaluation Plan
- NL2SQL: curated benchmark across org schemas; exact match, exec accuracy, F1.
- Retrieval: precision/recall with annotated relevant docs (metrics/table docs).
- Task completion: human evaluation of 100 tasks per persona.
- Chart quality: pairwise preference vs. baseline templates.
- Latency and cost: continuous monitoring with targets as above.

Additional Design Details
- Semantic caching and deduplication: parametric templates; normalize “last month” relative dates.
- Metadata filtering: by table, column type, metric tag, PII tag, freshness windows.
- Citation and provenance: include sources (tables/docs/snippets) and confidence scores.
- Error handling: capture stack traces, retry with fixes, fall back to sample size or simplified query, present informative error narratives.
- Memory: per-session short-term context; per-user long-term preferences (e.g., default currency, time zone), and successful query/code snippets for reuse.

Performance Optimizations
- Hybrid indexes for retrieval; MMR diversity selection; cross-encoder re-ranking.
- Semantic cache for frequent intents; streaming partial outputs for responsiveness.
- Pushdown predicates to source DB; vectorized DataFrame ops; data sampling and progressive results.

This PRD specifies the end-to-end requirements, architecture, and delivery plan for aiml030_interactive_data_analysis_assistant to achieve high accuracy, low latency, strong security, and a delightful user experience.