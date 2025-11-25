# Product Requirements Document (PRD)
# `Aiml015_Multi_Agent_Collaborative_Coding_System`

Project ID: Aiml015_Multi_Agent_Collaborative_Coding_System
Category: General AI/ML – Multi-Agent Systems for Software Engineering
Status: Draft for Review
Version: v1.0
Last Updated: 2025-11-25

## 1. Overview
### 1.1 Executive Summary
Aiml015_Multi_Agent_Collaborative_Coding_System is a multi-agent AI platform that plans, writes, tests, reviews, and integrates code changes across repositories with human-in-the-loop checkpoints. It uses graph-based orchestration, code-aware retrieval-augmented generation (RAG), tool-augmented reasoning (linters, unit tests, vulnerability scanning), and sandboxed execution to deliver high-quality, auditable patches. The system reduces cycle time for software tasks (bug fixes, refactors, documentation, test authoring) and integrates with existing developer workflows (Git, CI/CD, issue trackers).

### 1.2 Document Purpose
This PRD defines product vision, scope, requirements, architecture, data model, APIs, UI/UX, security, performance, scalability, testing, deployment, monitoring, risks, timeline, and KPIs for the initial GA release.

### 1.3 Product Vision
Enable engineering teams to safely scale software delivery through autonomous yet governed AI agents collaborating with developers. The system becomes a reliable teammate: planning tasks, proposing diffs, self-testing, surfacing risks, and learning from feedback, while maintaining transparency, traceability, and compliance.

## 2. Problem Statement
### 2.1 Current Challenges
- Developers spend significant time on repetitive tasks: boilerplate, refactors, test updates, dependency bumps, and documentation.
- Single-agent code assistants lack robust grounding, often hallucinate, and struggle with multi-file, multi-step tasks.
- Tool use is ad hoc; minimal observability into agent decisions, costs, or performance.
- Code context exceeds LLM context windows; naive retrieval misses symbol-level dependencies.
- Limited governance: hard to enforce policy (licenses, security), gates, and rollbacks.

### 2.2 Impact Analysis
- Slow velocity: prolonged PR cycles and high context-switch overhead.
- Quality risks: insufficient tests and undetected regressions.
- Cost: duplicated effort and high manual toil in triage and code review.
- Developer experience: cognitive overload and low trust in AI outputs.

### 2.3 Opportunity
A multi-agent system with clear division of labor, code-aware RAG, and tool-augmented reasoning can autonomously deliver high-quality patches with fewer failures, reduced cycle time, and improved developer trust.

## 3. Goals and Objectives
### 3.1 Primary Goals
- Automate end-to-end coding tasks with multi-agent collaboration.
- Ground generations in repository context via semantic, AST-aware retrieval.
- Provide robust self-testing and reviewer-critic loops before human approval.
- Offer full transparency: plans, diffs, tests, costs, traces, and risks.

### 3.2 Business Objectives
- Reduce mean time-to-merge by 40% within 3 months of adoption.
- Decrease manual toil by 50% for targeted task categories (e.g., dependency updates).
- Improve test coverage deltas by +5% on automated changes.
- Achieve 99.5% service uptime and predictable cost-per-patch.

### 3.3 Success Metrics
- pass@1 on internal coding benchmark: ≥60% (≥80% pass@5)
- Automated PR acceptance rate without edits: ≥35% by GA, ≥50% by +2 quarters
- Median patch delivery time for small tasks (<50 LOC): ≤10 minutes
- Retrieval latency p95: ≤500 ms; task orchestration latency p95: ≤3 s for planning
- Cost per successful small patch: ≤$0.80 (API LLM) or ≤$0.20 (self-hosted LLM)

## 4. Target Users/Audience
### 4.1 Primary Users
- Software Engineers (Backend, Frontend, Full-stack)
- QA/Test Engineers
- DevOps/Platform Engineers

### 4.2 Secondary Users
- Engineering Managers and Tech Leads
- Security Engineers
- Documentation Writers/Technical Writers

### 4.3 User Personas
- Persona 1: Alex Kim, Senior Backend Engineer
  - Background: 8 years in microservices (Python/Go), owns several APIs.
  - Pain Points: Context switching across services; repetitive bug triage; keeping tests current.
  - Goals: Faster PRs, reliable patches, high test coverage, minimal babysitting of bots.
- Persona 2: Priya Desai, QA Automation Engineer
  - Background: 6 years in test frameworks (JavaScript/Python), CI pipelines, flaky test triage.
  - Pain Points: Time spent writing regression tests; debugging flaky tests; maintaining test data.
  - Goals: Automated test generation, reproducible runs in sandboxes, clear failure triage.
- Persona 3: Marco Alvarez, Platform Engineer
  - Background: 10 years in cloud infra/K8s, security policies, SRE.
  - Pain Points: Enforcing policy (licenses, versions), dependency risks, limited auditability.
  - Goals: Policy-aware automation, SBOMs, reliable rollbacks, robust observability.
- Persona 4: Sara Williams, Engineering Manager
  - Background: Leads 12 engineers, responsible for delivery and quality metrics.
  - Pain Points: Bottlenecks in code review; unclear ROI of AI tools; lack of visibility.
  - Goals: Dashboards for throughput, quality KPIs, cost controls, opt-in policies.

## 5. User Stories
- US-001: As a developer, I want to submit a Jira/GitHub issue URL so the system plans and proposes a fix PR.
  - Acceptance: System creates a plan, proposes a diff, runs tests, and opens a PR with passing tests or annotated failures.
- US-002: As a developer, I want to inspect the agent’s plan and retrieved context before execution.
  - Acceptance: UI shows plan steps and context snippets with sources.
- US-003: As a reviewer, I want to see diffs with explanations and confidence scores.
  - Acceptance: PR contains inline rationale, risk flags, and a confidence score 0–1.
- US-004: As a QA engineer, I want auto-generated unit tests for new code paths.
  - Acceptance: System adds tests with coverage delta ≥ +2% where applicable.
- US-005: As a platform engineer, I want policy checks (licenses/security) to gate merges.
  - Acceptance: Changes violating policy are blocked with clear remediation.
- US-006: As a developer, I want sandboxed execution for builds/tests.
  - Acceptance: Jobs run in isolated containers; logs captured and viewable.
- US-007: As a developer, I want to approve or request changes at checkpoints.
  - Acceptance: UI shows approval gates; system halts until approval.
- US-008: As a security engineer, I want vulnerability scanning results on PRs.
  - Acceptance: PR shows scan report with severity and remediation suggestions.
- US-009: As a manager, I want dashboards for throughput, quality, and cost.
  - Acceptance: Grafana panels show pass@k, coverage delta, MTTR, cost per patch.
- US-010: As a developer, I want rollback of a merged change.
  - Acceptance: One-click revert branch/PR with context of prior change.
- US-011: As a developer, I want concurrent agent edits without conflicts.
  - Acceptance: System uses branches and auto-merge or CRDT-style edits; conflicts surfaced.
- US-012: As a developer, I want to configure LLM provider and model.
  - Acceptance: Admin UI/Config supports multiple providers/models with quotas.
- US-013: As a developer, I want to chat with agents about rationale/logs.
  - Acceptance: Threaded chat with links to traces, tools, and artifacts.
- US-014: As a data scientist, I want to export decision traces for analysis.
  - Acceptance: JSONL exports via API with PII-safe redaction.
- US-015: As a dev, I want to run the system on a private repo in air-gapped mode.
  - Acceptance: Offline mode with local models, vector DB, and policy bundles.

## 6. Functional Requirements
### 6.1 Core Features
- FR-001: Task intake from UI/API integrating with Git platforms and issue trackers.
- FR-002: Planner/Decomposer agent creates a DAG of subtasks with success criteria.
- FR-003: Coder/Implementer agent generates patch-oriented diffs (not full files).
- FR-004: Tester agent runs builds/unit tests/integration tests in sandbox.
- FR-005: Reviewer/Critic agent analyzes diffs and test results; proposes fixes.
- FR-006: Integrator/PR Bot creates branches, opens PRs, handles merges/rollbacks.
- FR-007: Doc/Comment Writer updates READMEs, in-code comments, and PR descriptions.
- FR-008: Repo/Context Curator builds and queries code-aware indices (AST, symbols).
- FR-009: Guardrails: timeouts, retries, circuit breakers; per-agent budgets.
- FR-010: Human-in-the-loop checkpoints with approvals and policy enforcement.
- FR-011: Observability: traces of tool calls, logs, metrics, and costs.
- FR-012: Security scanning and license checks integrated into agent flow.

### 6.2 Advanced Features
- FR-013: Experience memory: retrieve prior successful patches and prompts.
- FR-014: Flaky test detection and quarantine suggestions.
- FR-015: Multi-repo changes with dependency graph awareness.
- FR-016: Uncertainty estimation and confidence scoring for diffs.
- FR-017: Adaptive context packing with token-budget optimization.
- FR-018: Self-hosted LLM support with adapters/LoRA for style alignment.
- FR-019: Streaming SSE for real-time plan updates and tool logs.
- FR-020: Natural language chat to refine task scope and constraints.

## 7. Non-Functional Requirements
### 7.1 Performance
- Retrieval p95 ≤ 500 ms; plan synthesis p95 ≤ 3 s; diff generation p95 ≤ 10 s for small tasks.
- Test sandbox spin-up ≤ 5 s p95 (warm pool).
### 7.2 Reliability
- Uptime ≥ 99.5%; durable event storage; at-least-once processing with idempotency.
### 7.3 Usability
- Onboarding ≤ 30 minutes; frictionless PR review UX; keyboard shortcuts.
### 7.4 Maintainability
- Modular services, typed interfaces, comprehensive tests (≥85% coverage on core orchestrator).

## 8. Technical Requirements
### 8.1 Technical Stack
- Backend: Python 3.11+, FastAPI 0.115+, LangGraph/LangChain 0.2+ (graph orchestration)
- Frontend: React 18+, Next.js 14+, TypeScript 5.6+, TailwindCSS 3+
- Message Bus: Redis Streams 7+ or Kafka 3+
- Vector DB: PostgreSQL 16 + pgvector 0.7+ or Weaviate 1.24+ or Pinecone (serverless)
- Metadata DB: PostgreSQL 16+
- Artifact Store: S3-compatible (AWS S3 or MinIO)
- Containerization: Docker 24+, Kubernetes 1.29+
- CI/CD: GitHub Actions / GitLab CI
- Observability: OpenTelemetry 1.27+, Prometheus 2.53+, Grafana 11+, Loki 2.9+ or ELK
- Auth: OAuth2/OIDC (Okta/GitHub/GitLab/Google), JWT
- Static Analysis/Tools: ruff/mypy/flake8, eslint/tsc, bandit, trivy/grype, license-checker
- Test Runners: pytest, jest, go test (pluggable)
- Security: HashiCorp Vault or cloud KMS

### 8.2 AI/ML Components
- LLM Providers: OpenAI, Anthropic, Google, Azure OpenAI, self-hosted Llama 3.1+/Code Llama variants via vLLM 0.5+.
- Embeddings: text-embedding models (e.g., OpenAI text-embedding-3-large, bge-large, e5-large), code-aware embeddings (e.g., jina-code).
- RAG: Hybrid search (BM25 + dense), AST-aware chunking, symbol index, dependency graph expansion.
- Prompting: System prompts per agent role; function calling for tool use; temperature/top-p tuned per agent.
- Fine-Tuning: Optional adapters/LoRA for organizational code style and doc tone.

## 9. System Architecture
### 9.1 High-Level Architecture (ASCII)
+-------------------+         +--------------------+          +-------------------+
|   Web UI (Next)   |  HTTPS  |    API Gateway     |  gRPC/   |  Auth/OIDC        |
|  Task/PR Console  +-------->+  FastAPI Backend   +----------+  Provider         |
|  Live Streams     |         |                    |          |                   |
+-------------------+         +----------+---------+          +-------------------+
                                      |
                                      v
                           +----------+-----------+
                           |  Orchestrator (DAG) |
                           |  LangGraph State    |
                           +----+-----------+----+
                                |           |
                     events/msgs|           |tool calls
                              +-v-+       +-v-+
                              |Bus|       |Tooling Svc (linters, tests,
                              |   |       |builders, scanners, formatters)|
                              +---+       +-------------------------------+
                                |
                                v
     +-------------------+   +--------------------+   +--------------------+
     | Vector DB (RAG)   |   | Metadata DB        |   | Artifact Store     |
     | Code/Symbol Index |   | Tasks, Plans,      |   | Logs, Reports,     |
     | Embeddings        |   | Diffs, Traces      |   | SBOMs, Artifacts   |
     +---------+---------+   +---------+----------+   +---------+----------+
               |                       |                        |
               v                       v                        v
     +---------+---------+   +---------+----------+   +---------+----------+
     | Repo Connector    |   | Sandbox Runner     |   | Observability Stack|
     | Git Providers     |   | Containers, FS ISO |   | OTel, Prom, Grafana|
     +-------------------+   +--------------------+   +--------------------+

### 9.2 Component Details
- Orchestrator: Graph/state-machine coordinating agents with guardrails, timeouts, retries, budgets, and human gates.
- Agents:
  - Planner/Decomposer: builds DAG, defines success metrics.
  - Coder/Implementer: generates diffs, uses formatters and static analysis.
  - Tester: runs test suites; triages failures.
  - Reviewer/Critic: inspects diffs/tests; proposes corrections; assigns confidence.
  - Doc Writer: updates docs/comments/PR messaging.
  - Integrator: branches, PRs, merges, rollbacks; resolves conflicts.
  - Repo/Context Curator: maintains indices; handles RAG queries.
- Tooling Service: Wraps linters, type checkers, formatters, build/test runners, vulnerability/license scanners, diff generators.
- Message Bus: Event-driven handoffs; blackboard pattern with status signals.
- Persistence: Postgres for metadata; Vector DB for embeddings; Artifact store for logs/reports.
- Sandbox Runner: Kubernetes job pods or Firecracker-based isolation; resource quotas; network/FS isolation; log capture.
- Observability: OpenTelemetry traces, Prometheus metrics, structured logs, dashboards.

### 9.3 Data Flow
1) Task created via API/UI -> API Gateway -> Orchestrator.
2) Planner queries Vector DB via Repo Curator -> collects context -> plan saved.
3) Coder requests context slices -> generates diffs -> runs formatters/linters -> posts patch to Integrator.
4) Tester spins sandbox -> builds/tests -> captures logs -> updates status.
5) Reviewer examines diffs + test results -> suggests edits -> loop until green or budget/time exhausted.
6) Integrator opens PR/branch -> triggers policy scanning -> human checkpoint -> merge or request changes.
7) Observability captures traces, metrics, and costs across steps.

## 10. Data Model
### 10.1 Entity Relationships
- User (1..n) -> Task (1..n)
- Task (1..n) -> Plan (1)
- Task (1..n) -> Patch (n)
- Patch (1..n) -> TestRun (n)
- Task (1..n) -> Message/Trace (n)
- Repo (1) -> Index (n versions)
- Patch (1) -> PR (0..1)
- Policy (n) applied to Repo (n)
- ToolCall (n) linked to AgentRun (n)

### 10.2 Database Schema (key fields)
- users: id, email, name, role, org_id, created_at
- repos: id, provider, repo_slug, default_branch, install_id, created_at
- tasks: id, repo_id, title, description, issue_url, priority, status, creator_id, created_at, updated_at
- plans: id, task_id, dag_json, context_refs[], success_criteria, created_at
- patches: id, task_id, branch, diff_summary, risk_score, confidence, status, created_at
- testruns: id, patch_id, sandbox_id, outcome, coverage_delta, logs_uri, duration_ms, created_at
- agent_runs: id, task_id, agent_type, input_refs, output_refs, cost_usd, tokens_in, tokens_out, created_at
- tool_calls: id, agent_run_id, tool_name, params_json, result_json, duration_ms, success, created_at
- pr_records: id, patch_id, pr_number, url, status, merged_at, created_at
- indices: id, repo_id, commit_sha, index_type, version, stats_json, created_at
- embeddings: id, index_id, uri, symbol, lang, vector, metadata_json
- policies: id, org_id, name, rules_json, enforcement_level, created_at
- approvals: id, patch_id, approver_id, decision, comment, created_at
- traces: id, task_id, span_id, parent_span_id, kind, payload_json, created_at
- budgets: id, task_id, max_cost_usd, max_tokens, time_limit_s, spent_cost_usd, spent_tokens

### 10.3 Data Flow Diagrams (ASCII)
Task Intake -> Plan -> Patch -> Test -> Review -> Integrate

[User/API] -> (Create Task)
   -> [Planner] -> (Plan stored)
   -> [Coder] -> (Patch created)
   -> [Tester] -> (TestRun stored)
   -> [Reviewer] -> (Patch updated)
   -> [Integrator] -> (PR record)
   -> [Approver] -> (Approval stored)

### 10.4 Input Data & Dataset Requirements
- Source code repositories: languages (Python, JS/TS, Go, Java), tests, configs.
- Embedding indices: function/class AST chunks, symbol tables, dependency graphs, documentation.
- Optional fine-tuning corpora: internal style guides, past PRs and reviews, successful patches, test cases.
- Policy datasets: license allow/deny lists, security CWE/CVEs, coding standards.

## 11. API Specifications
### 11.1 REST Endpoints
- POST /v1/tasks
- GET /v1/tasks/{task_id}
- GET /v1/tasks/{task_id}/events (SSE)
- POST /v1/repos/index
- POST /v1/patches/{patch_id}/approve
- POST /v1/patches/{patch_id}/rollback
- GET /v1/patches/{patch_id}
- POST /v1/config/models
- GET /v1/metrics
- GET /v1/traces/{task_id}
- POST /v1/chat/{task_id}

### 11.2 Request/Response Examples
POST /v1/tasks
Request:
{
  "repo": "github.com/acme/api-service",
  "base_branch": "main",
  "issue_url": "https://github.com/acme/api-service/issues/123",
  "title": "Fix 500 on /users when email missing",
  "description": "Requests without email should return 400 with error JSON.",
  "budgets": {"max_cost_usd": 2.0, "time_limit_s": 1800},
  "checkpoints": ["pre-integrate"]
}
Response:
{
  "task_id": "tsk_01HXP...89",
  "status": "queued",
  "url": "https://app.acme.ai/tasks/tsk_01HXP...89"
}

GET /v1/tasks/{task_id}
Response:
{
  "task_id": "tsk_01HXP...89",
  "status": "review_required",
  "plan": {...},
  "patch": {"branch": "ai/fix-issue-123", "risk_score": 0.22, "confidence": 0.71},
  "tests": {"outcome": "passed", "coverage_delta": 0.03},
  "pr": {"url": "https://github.com/acme/api-service/pull/456"}
}

POST /v1/repos/index
Request:
{
  "repo": "github.com/acme/api-service",
  "commit_sha": "abc123",
  "modes": ["ast", "symbols", "docs"],
  "embeddings_model": "bge-large"
}
Response: {"status": "ok", "indexed_files": 842, "symbols": 10543}

POST /v1/patches/{patch_id}/approve
Request:
{"decision": "approve", "comment": "Looks good."}
Response:
{"status": "merged", "pr_url": "https://github.com/acme/api-service/pull/456"}

### 11.3 Authentication
- OAuth2/OIDC login for UI; API uses Bearer JWT with scopes:
  - tasks:read/write, patches:approve, repos:index, metrics:read, traces:read, config:write
- PAT (personal access tokens) for CI integrations.
- Fine-grained RBAC per org/repo.

## 12. UI/UX Requirements
### 12.1 User Interface
- Task Console: create tasks, view status, budgets, and timelines.
- Plan Viewer: DAG graph, step details, retrieved context snippets with sources.
- Diff Viewer: side-by-side diffs, inline explanations, risk/confidence badges.
- Test/Logs: structured test summaries, raw logs, flaky indicators.
- PR Panel: policies, approvals, merge/rollback controls.
- Settings: model/provider selection, budgets, policies, repo connections.
- Dark mode; responsive design; keyboard shortcuts.

### 12.2 User Experience
- Progressive disclosure: high-level summary first, drill-down to traces/logs.
- Real-time updates via SSE; optimistic UI for approvals.
- Clear affordances for human gates and rollback.
- Copyable commands and links to Git/CI.

### 12.3 Accessibility
- WCAG 2.1 AA: proper contrast, ARIA roles, keyboard navigation, screen-reader labels.
- Avoid color-only status signaling; provide text labels.

## 13. Security Requirements
### 13.1 Authentication
- OIDC/OAuth2 with MFA support; short-lived JWTs; refresh tokens with rotation.
### 13.2 Authorization
- RBAC by org/repo/task; least privilege; approval actions audited.
### 13.3 Data Protection
- TLS 1.2+ in transit; AES-256 at rest; secrets in Vault/KMS.
- PII redaction in logs/traces; scoped data access for agents.
### 13.4 Compliance
- SOC 2-ready controls, audit logs retention (≥ 365 days).
- SBOM generation for containers; dependency scanning in CI.

## 14. Performance Requirements
### 14.1 Response Times
- API p95: < 300 ms for metadata; < 500 ms for retrieval queries; streaming for long tasks.
### 14.2 Throughput
- Support 200 concurrent tasks per org; 2,000 across cluster (baseline).
### 14.3 Resource Usage
- Sandbox CPU: default 2 vCPU, 4 GB RAM; burstable pools.
- Vector queries: cache hot set; target QPS 500 with p95 < 500 ms.

## 15. Scalability Requirements
### 15.1 Horizontal Scaling
- Scale orchestrator workers, tooling services, and sandbox runners independently.
- Partition message streams by task_id; sharded vector indices.
### 15.2 Vertical Scaling
- Increase pod resources for heavy builds/tests; node pools with GPU for local LLMs.
### 15.3 Load Handling
- Autoscaling by queue depth and p95 latencies; backpressure with admission control.
- Priority queues (urgent vs. batch).

## 16. Testing Strategy
### 16.1 Unit Testing
- ≥85% coverage for orchestrator, agents interfaces, RAG components, API.
- Deterministic tests via seed and mock tool adapters.
### 16.2 Integration Testing
- End-to-end workflows: issue -> plan -> diff -> tests -> PR.
- Sandbox integration across languages (Python, JS/TS, Go, Java).
- Policy enforcement scenarios and rollback tests.
### 16.3 Performance Testing
- Load tests for 2,000 concurrent tasks; vector DB QPS benchmarking.
- Sandbox pool warm/cold start measurements.
### 16.4 Security Testing
- SAST/DAST; dependency scanning; secrets scanning.
- Pen tests for authz bypass; rate-limiting and DoS resilience.

## 17. Deployment Strategy
### 17.1 Deployment Pipeline
- GitOps with trunk-based development; CI: lint/test/build; CD: canary with progressive rollout.
- SBOM generation; image signing (cosign); policy checks (OPA/Gatekeeper).
### 17.2 Environments
- Dev (shared), Staging (prod-like), Prod (HA); optional On-Prem profile.
### 17.3 Rollout Plan
- Wave 1: internal repos; Wave 2: pilot customers; Wave 3: GA.
- Feature flags for advanced features.
### 17.4 Rollback Procedures
- Blue/green switches; immediate rollback to prior image tag.
- PR rollback automation and index version pinning.

## 18. Monitoring & Observability
### 18.1 Metrics
- pass@k, patch acceptance, test pass rate, coverage delta.
- Latencies per stage; sandbox success rate; vector QPS/latencies.
- Cost per task; tokens per agent; tool success rate; flakiness rate.
### 18.2 Logging
- Structured JSON logs; tool invocation logs with redaction.
- Retain 30 days hot, 365 days cold.
### 18.3 Alerting
- SLO breaches (latency, error rate); cost anomalies; queue backlog.
### 18.4 Dashboards
- Operator: health, throughput, latencies, errors.
- Manager: velocity, acceptance, cost, quality KPIs.
- Security: policy violations, vulnerabilities, dependency risks.

## 19. Risk Assessment
### 19.1 Technical Risks
- Hallucinations or unsafe code generation.
- Stale indices leading to irrelevant context.
- Non-deterministic tests/flaky results.
- Toolchain incompatibilities across languages.
- Rate limits/cost spikes from LLM providers.
### 19.2 Business Risks
- Low developer trust/adoption.
- Vendor lock-in for LLM/vector DB.
- Data residency/compliance requirements.
### 19.3 Mitigation Strategies
- Strong grounding via code-aware RAG; uncertainty surfacing; conservative policies.
- Versioned indices with diff-based updates; staleness detection.
- Flakiness detection/quarantine; retrials with isolation.
- Pluggable tool adapters; provider abstraction; local LLM fallback.
- Budget controls, caching, and batching to manage cost.

## 20. Timeline & Milestones
### 20.1 Phase breakdown
- Phase 0 (2 weeks): Requirements finalization, design sign-off.
- Phase 1 (4 weeks): Core orchestrator, task intake, basic planner/coder, Git integration.
- Phase 2 (4 weeks): RAG (AST/symbol index), tester agent with sandbox, doc writer.
- Phase 3 (4 weeks): Reviewer/critic loops, policy scanning, approvals.
- Phase 4 (4 weeks): Observability, dashboards, cost tracking, SSE streaming.
- Phase 5 (4 weeks): Hardening, performance, security, on-prem profile.
- Phase 6 (2 weeks): Beta rollout, feedback integration.
- Phase 7 (2 weeks): GA release.
Total: ~22 weeks (~5.5 months)

Estimated team: 1 PM, 1 Designer, 5 Backend/Platform, 2 Frontend, 2 ML/Agents, 1 DevRel.
Rough cost (cloud + LLM during dev/beta): $8–12k/month.

### 20.2 Key Milestones
- M1: Task->Plan->Diff E2E demo (Week 4)
- M2: Tests passing in sandbox with logs (Week 8)
- M3: Reviewer loop and PR creation (Week 12)
- M4: Policy gates and human approvals (Week 16)
- M5: Observability dashboards (Week 20)
- M6: Beta (Week 22), GA (Week 26 with buffer)

## 21. Success Metrics & KPIs
### 21.1 Measurable targets
- Service uptime: ≥99.5%
- Retrieval p95 latency: ≤500 ms
- Median patch time (small tasks): ≤10 min
- pass@1 ≥60%, pass@5 ≥80% on internal eval
- Automated PR acceptance without edits: ≥35% (GA), ≥50% (+2Q)
- Coverage delta on automated patches: ≥+3%
- Cost per successful small patch: ≤$0.80 (API) / ≤$0.20 (self-hosted)

## 22. Appendices & Glossary
### 22.1 Technical Background
- Multi-agent coordination via DAG/state machine (LangGraph-style), enabling explicit handoffs and retries with budgets and timeouts.
- Code-aware RAG:
  - AST/function/class chunking; symbol-level indices.
  - Hybrid search (BM25 + dense embeddings); MMR for diversity.
  - Neighborhood expansion along import/dependency graphs.
  - Context packing heuristics with deduplication and priority scoring.
- Tool-augmented reasoning:
  - Linters, type checkers, test runners, formatters, vulnerability/license scanners.
  - Diff generators and conflict resolution (branches, auto-merge, or CRDT-like edits).
- Memory architecture:
  - Short-term buffers for current task.
  - Long-term vector memory for successful patches, prompts, and reflections; versioned embeddings per commit.
  - Episodic task logs for retrieval (“what fixed this before?”).
- Governance and observability:
  - Policy constraints, cost/time budgets, telemetry on tool calls, pass@k, coverage deltas, flaky test detection.

### 22.2 References
- OpenTelemetry, Prometheus, Grafana documentation
- LangGraph/LangChain for orchestration patterns
- RAG best practices (hybrid search, MMR, AST-aware chunking)
- vLLM for efficient inference
- pgvector and vector search strategies
- Secure coding guides (OWASP), SBOM practices

### 22.3 Glossary
- Agent: Role-specific LLM process with tools (e.g., Planner, Coder, Tester).
- RAG: Retrieval-Augmented Generation; grounding model outputs with retrieved context.
- AST: Abstract Syntax Tree; structure representing source code syntax.
- pass@k: Probability that at least one of k generated solutions passes tests.
- Confidence score: Model-estimated likelihood of correctness for a patch.
- Sandbox: Isolated environment for running builds/tests with resource controls.
- Vector database: Store of embeddings enabling similarity search.
- Embedding: Numeric representation of text/code semantics.
- RBAC: Role-Based Access Control.
- SBOM: Software Bill of Materials.
- SSE: Server-Sent Events for real-time streaming updates.

Repository Structure
- notebooks/
  - experiments_rag.ipynb
  - eval_passk.ipynb
- src/
  - api/
    - main.py
    - routes/
      - tasks.py
      - patches.py
      - repos.py
      - metrics.py
      - chat.py
  - orchestrator/
    - graph.py
    - states.py
    - agents/
      - planner.py
      - coder.py
      - tester.py
      - reviewer.py
      - integrator.py
      - doc_writer.py
      - curator.py
    - tools/
      - linters.py
      - test_runner.py
      - formatter.py
      - vuln_scanner.py
      - diff_utils.py
  - rag/
    - indexer.py
    - retriever.py
    - ast_chunker.py
    - symbol_index.py
  - services/
    - sandbox.py
    - git_client.py
    - policy.py
    - budget.py
    - auth.py
    - telemetry.py
  - ui/ (Next.js app)
- tests/
  - unit/
  - integration/
  - perf/
  - security/
- configs/
  - config.yaml
  - models.yaml
  - policies/
    - default_policy.yaml
- data/
  - sample_repos/
  - indices/
- scripts/
  - deploy_k8s.sh
  - index_repo.py

Code Snippets

FastAPI example (task creation):
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from auth import get_current_user
from orchestrator.graph import submit_task

app = FastAPI()

class TaskReq(BaseModel):
    repo: str
    base_branch: str = "main"
    issue_url: str | None = None
    title: str
    description: str
    budgets: dict | None = None
    checkpoints: list[str] | None = None

@app.post("/v1/tasks")
async def create_task(req: TaskReq, user=Depends(get_current_user)):
    task_id = await submit_task(user_id=user.id, **req.model_dump())
    return {"task_id": task_id, "status": "queued"}

Example orchestrator step (LangGraph-style pseudocode):
from langgraph.graph import StateGraph

def planner(state): ...
def coder(state): ...
def tester(state): ...
def reviewer(state): ...
def integrator(state): ...

g = StateGraph()
g.add_node("planner", planner)
g.add_node("coder", coder)
g.add_node("tester", tester)
g.add_node("reviewer", reviewer)
g.add_node("integrator", integrator)

g.add_edge("planner", "coder")
g.add_edge("coder", "tester")
g.add_conditional_edges("tester", lambda s: "reviewer" if not s.tests_passed else "integrator", ["reviewer", "integrator"])
g.add_edge("reviewer", "coder")

app = g.compile()

Config sample (configs/config.yaml):
orchestrator:
  timeouts:
    planner_s: 20
    coder_s: 60
    tester_s: 600
  retries:
    coder: 2
    tester: 1
budgets:
  default:
    max_cost_usd: 2.0
    max_tokens: 200000
llm:
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.2
rag:
  embedding_model: "bge-large"
  top_k: 12
  hybrid: true
sandbox:
  runtime: "k8s"
  cpu: "2"
  memory: "4Gi"
security:
  policy_file: "configs/policies/default_policy.yaml"

Example retrieval call:
results = retriever.search(
    query="validate email presence in user creation",
    filters={"lang": ["python"], "path_prefix": "services/users"},
    top_k=12,
    mmr=True
)

Performance and SLO Targets
- Accuracy: pass@1 ≥60%, pass@5 ≥80%
- Latency: retrieval p95 ≤500ms; planning p95 ≤3s; small diff generation p95 ≤10s
- Availability: ≥99.5%
- Cost: ≤$0.80 per successful small patch (API), ≤$0.20 (self-hosted)

Scalability Notes
- Horizontal scaling for agents and sandbox pools; sharded vector indices; Redis Streams/Kafka partitions.
- Caching embeddings and search results; warm sandbox pools.

End of PRD.