# Product Requirements Document (PRD)
# `Aiml016_Autonomous_Research_Assistant`

Project ID: Aiml016_Autonomous_Research_Assistant
Category: AI/ML, NLP, Agentic Systems, RAG
Status: Draft for Review
Version: 1.0
Last Updated: 2025-11-25

## 1. Overview
### 1.1 Executive Summary
Aiml016_Autonomous_Research_Assistant is an agentic, Retrieval-Augmented Generation (RAG) research assistant that autonomously plans, searches, reads, analyzes, and synthesizes findings from web and document sources. It provides traceable, citation-grounded outputs with evidence maps, contradiction checks, and iterative refinement. The system uses a planner-executor loop with tool calling for web search, scraping, parsing, reranking, and exporting structured deliverables (Markdown/JSON). It targets researchers, analysts, product managers, and data scientists who need high-quality, credible research quickly.

### 1.2 Document Purpose
This PRD defines product scope, requirements, system architecture, data models, APIs, UI/UX, security, performance, deployment, testing, and success metrics to guide design, implementation, and validation.

### 1.3 Product Vision
Empower knowledge workers to produce trustworthy, well-cited research in minutes, not days, by combining state-of-the-art LLMs, hybrid retrieval, evidence management, and robust guardrails in a user-friendly, collaborative platform.

## 2. Problem Statement
### 2.1 Current Challenges
- Manual research is time-consuming: finding, reading, and synthesizing sources.
- Information overload and credibility concerns; difficult to track provenance.
- LLM outputs may hallucinate or miss key evidence without proper grounding.
- Fragmented tools: search, notes, and citation managers aren’t integrated.
- Limited support for iterative research and hypothesis-driven exploration.

### 2.2 Impact Analysis
- Delayed decision-making and opportunity costs.
- Inconsistent quality and missed insights due to incomplete coverage.
- Increased risk of incorrect claims without verifiable citations.

### 2.3 Opportunity
Deliver an end-to-end, autonomous agent that systematically plans research, retrieves relevant, credible sources, maps claims to citations, highlights contradictions, and synthesizes structured reports—improving speed, quality, and trust.

## 3. Goals and Objectives
### 3.1 Primary Goals
- Plan → Search → Ingest → Retrieve → Analyze → Synthesize → Cite → Review → Iterate agentic loop with tool calling.
- High-faithfulness, citation-grounded outputs with span-level provenance.
- Hybrid retrieval (dense + sparse) with reranking and freshness controls.
- Exportable outputs (Markdown, JSON) and project knowledge base.

### 3.2 Business Objectives
- Reduce research cycle time by 70%+.
- Drive product adoption via superior accuracy, trust, and usability.
- Enable team collaboration and governance for enterprise customers.

### 3.3 Success Metrics
- Faithfulness/grounding score (RAGAS/TruLens) ≥ 0.8.
- Answer citation coverage ≥ 95% of claims grounded to spans.
- Retrieval Recall@10 ≥ 0.85; nDCG@10 ≥ 0.75.
- End-to-end latency: P95 ≤ 6s; retrieval step ≤ 500ms; rerank ≤ 300ms.
- Uptime ≥ 99.5%; user NPS ≥ 45.
- Time-to-first-draft reduction ≥ 70%.

## 4. Target Users/Audience
### 4.1 Primary Users
- Research analysts, data scientists, product managers, academic researchers, technical writers.

### 4.2 Secondary Users
- Legal/compliance reviewers, knowledge managers, executives consuming reports.

### 4.3 User Personas
1) Name: Maya Chen
- Role: Research Analyst at a consulting firm
- Background: MS in Data Science; synthesizes market/tech landscapes weekly
- Pain points: Hunting credible sources; manual dedup; time pressure
- Goals: Fast, trustworthy briefs with citations; easy export to client decks
- Tools: Chrome, Notion, Slack, Python notebooks

2) Name: Alex Romero
- Role: Product Manager in SaaS
- Background: Computer Engineering; prioritizes features using market/user insights
- Pain points: Drowning in articles; wants concise, compare/contrast output
- Goals: Structured recommendations, risks, and open questions with sources
- Tools: Google Docs, Jira, Confluence

3) Name: Dr. Fatima Khan
- Role: Academic PI
- Background: NLP researcher; writes grant proposals and literature reviews
- Pain points: Coverage and recency; needs span-precise citations and contradictions
- Goals: Comprehensive lit review drafts with evidence graphs and export to LaTeX/Markdown
- Tools: Zotero, Overleaf, Python, ArXiv

4) Name: Jordan Smith
- Role: Compliance Reviewer
- Background: Legal analytics; checks claims and licenses
- Pain points: Verifying quotes and data lineage
- Goals: Fast provenance audit, flagged risks, and line-level citation mapping
- Tools: MS Office, internal policy knowledge base

## 5. User Stories
US-001: As a research analyst, I want the system to autonomously create a research plan from my query so that I see subtopics and hypotheses to explore.
- Acceptance: Given a query, the Plan view shows at least 5 subtopics, hypotheses, and prioritized tasks with tool calls.

US-002: As a user, I want the assistant to search and scrape credible sources so that I only review relevant materials.
- Acceptance: At least 20 candidate sources retrieved; top-10 precision improved by reranker; sources include credibility scores.

US-003: As a user, I want span-level citations for claims so that I can verify statements.
- Acceptance: Every non-trivial claim includes passage offsets with link anchors or PDF page/section mapping.

US-004: As a user, I want contradiction detection across sources so that conflicts are highlighted.
- Acceptance: Contradictory claims are flagged with NLI-based labels (entails/neutral/contradicts) and linked to sources.

US-005: As a user, I want deduplication and clustering so that redundant articles are merged.
- Acceptance: Near-duplicates are clustered; merged artifacts show canonical source and alternates.

US-006: As a user, I want export to Markdown and JSON so that I can integrate with my workflow.
- Acceptance: Exports include findings, citations, gaps, risks; validate JSON schema.

US-007: As a user, I want iterative refinement so that I can ask follow-ups and update outputs.
- Acceptance: Follow-up prompts update plan and synthesis; versioned artifacts maintained.

US-008: As a user, I want a project knowledge base with hybrid retrieval so that prior context is reused.
- Acceptance: New queries retrieve from both project KB and the web; RRF fusion applied.

US-009: As an admin, I want role-based access so that workspace data is protected.
- Acceptance: RBAC enforced for Owner/Admin/Editor/Viewer; audit logs recorded.

US-010: As a compliance reviewer, I want a provenance report so that I can audit data lineage and licenses.
- Acceptance: Export lists each source URL, license metadata, access timestamp, and extraction method.

US-011: As a user, I want latency below 6s so that the tool feels responsive.
- Acceptance: P95 end-to-end ≤ 6s on typical workloads (query + 10 sources cached).

US-012: As a data scientist, I want observable traces so that I can debug retrieval and synthesis.
- Acceptance: Traces store top-k snapshots, vector distances, reranker scores, and prompts with redactions.

US-013: As a PM, I want a compare/contrast section so that tradeoffs are clear.
- Acceptance: Generated section includes at least 3 contrasted alternatives with evidence and confidence.

US-014: As a user, I want safety filtering so that harmful or disallowed content is not surfaced.
- Acceptance: Toxic/unsafe content is blocked or annotated per policy.

US-015: As a researcher, I want a note-taking sidebar so that I can curate highlights and quotes.
- Acceptance: Highlights linked to source spans; notes export alongside report.

## 6. Functional Requirements
### 6.1 Core Features
FR-001: Agentic planning (ReAct/Plan-and-Execute) with tool calling.
FR-002: Web search API integration (e.g., Google Custom Search, SerpAPI, Bing Web).
FR-003: Webpage/PDF scraping with boilerplate removal and rate-limited crawling.
FR-004: Table/figure extraction from PDFs.
FR-005: Citation parsing and span-level mapping with offsets and page numbers.
FR-006: Hybrid retrieval (dense + BM25), reciprocal rank fusion, metadata filtering.
FR-007: Vector indexing with chunking strategies (semantic split, sliding windows).
FR-008: Cross-encoder reranking; MMR for diversity.
FR-009: Deduplication via MinHash/SimHash and clustering.
FR-010: Evidence management, provenance tracking, and credibility scoring.
FR-011: Multi-document summarization with outline-guided synthesis.
FR-012: Contradiction/entailment detection using NLI models.
FR-013: Uncertainty annotations and confidence scores.
FR-014: Session-aware memory (episodic + semantic vectorstore).
FR-015: Safety guardrails: toxicity filters, hallucination checks, duplication control.
FR-016: Exports: Markdown, JSON (schema-defined), optional HTML.
FR-017: Project knowledge base with upsert/update and reindex scheduling.
FR-018: UI Plan view, Evidence map, Synthesis editor, Citations panel.
FR-019: Note-taking and highlight linking.
FR-020: Evaluation dashboard (RAGAS/TruLens metrics) and regression tests.
FR-021: Authentication (OAuth2/API keys) and RBAC.
FR-022: Audit logging and run traces.
FR-023: Caching, batching, and rate limiting.
FR-024: Configurable prompt templates and few-shot exemplars.
FR-025: Source filters: domain/recency/language/license.
FR-026: Source credibility scoring: domain authority, citation count, recency.
FR-027: Fact-check tool leveraging targeted retrieval and NLI.
FR-028: Export to references manager formats (BibTeX/CSL JSON).
FR-029: API-first design for headless use.
FR-030: SDKs: Python/JS minimal clients.

### 6.2 Advanced Features
AF-001: Multi-agent mode: planner, researcher, critic, editor.
AF-002: Query expansion via HyDE and doc2query.
AF-003: Adaptive context window packing and hierarchical summarization.
AF-004: Time-decay weighting and freshness-aware re-ranking.
AF-005: Learning-to-rank personalization per project.
AF-006: Interactive knowledge graph of claims and sources.
AF-007: Offline batch ingestion pipelines for folders/cloud drives.
AF-008: Cost-aware routing among LLM providers; caching policy optimization.

## 7. Non-Functional Requirements
### 7.1 Performance
- P95 end-to-end latency ≤ 6s (cached sources), ≤ 12s (fresh crawl up to 5 pages).
- Retrieval step ≤ 500ms P95; rerank ≤ 300ms; synthesis streaming TTFB ≤ 1s.
### 7.2 Reliability
- Uptime ≥ 99.5%; error budget ≤ 3.6h/month.
- Durable storage with daily backups; RPO ≤ 4h; RTO ≤ 2h.
### 7.3 Usability
- Onboarding time ≤ 10 minutes; task completion success ≥ 90% in usability tests.
### 7.4 Maintainability
- 80%+ unit test coverage; modular services; semantic versioning; IaC with reproducible environments.

## 8. Technical Requirements
### 8.1 Technical Stack
- Backend: Python 3.11+, FastAPI 0.110+, Uvicorn/Gunicorn.
- Frontend: React 18+, TypeScript 5+, Vite or Next.js 14 (app router).
- Orchestration/Workers: Celery 5.3+ or Prefect 2+, Redis 7+ or RabbitMQ 3.12+.
- Storage: PostgreSQL 15+ (metadata), S3-compatible object store (minio/AWS S3).
- Vector DB: Qdrant 1.8+ or Milvus 2.3+; FAISS for local dev.
- Sparse Search: Elasticsearch 8.11+ or OpenSearch 2.11+ (BM25/SPLADE optional).
- Caching: Redis 7+.
- Message/Events: Kafka 3+ (optional) or Redis streams.
- Infra: Docker, Kubernetes 1.28+, Helm, Terraform.
- Observability: OpenTelemetry, Prometheus, Grafana, Loki.
- Auth: OAuth2/OIDC (Auth0/Okta), JWT, API keys.
- CI/CD: GitHub Actions, Dependabot.
- Testing: PyTest, Playwright, k6/Locust.

### 8.2 AI/ML Components
- LLMs: OpenAI GPT-4 Turbo or Anthropic Claude 3 family; on-prem alt: Llama 3 70B via vLLM.
- Embeddings: bge-large-en-v1.5 or E5-large-v2 (cosine similarity with normalization).
- Reranker: bge-reranker-v2 or Cohere Rerank.
- NLI: roberta-large-mnli for entailment/contradiction.
- Summarization: Long T5/Flan-T5-XL for cascading summarization (optional).
- Safety: Moderation endpoints + local toxicity classifiers (e.g., Detoxify).
- Evaluation: RAGAS, TruLens/G-Eval style prompt-based judging.

## 9. System Architecture
### 9.1 High-Level Architecture (ASCII)
[User Browser]
    |
    v
[React UI] <---> [Websocket Stream]
    |
    v
[API Gateway (FastAPI)] --- [Auth Service (OIDC/JWT)]
    |                               |
    |                               v
    v                       [RBAC/Policy Engine]
[Orchestrator Service] <---- events/jobs ----> [Worker Pool (Celery)]
    |                         |        |        \
    |                         |        |         \
    v                         v        v          v
[Tooling Layer]         [Retrieval] [Reranker] [Fact-Check/NLI]
  |   |   |                 |            |             |
  |   |   |                 v            v             v
[Search APIs][Scrapers][PDF Parser]  [Qdrant]    [NLI/Checks]
                            |            \
                            v             \
                       [Elasticsearch]    [Object Store]
                            |
                            v
                        [PostgreSQL]
                            |
                            v
                    [Eval/Tracing/Logs] --> [Prometheus/Grafana/Loki]
                            |
                            v
                        [Exporter (MD/JSON/BibTeX)]

### 9.2 Component Details
- Orchestrator: Implements agentic loop (planner-executor, Reflexion), tool routing, retries.
- Tooling: Search connectors, crawlers, parsers, table/figure extractors, citation parsers.
- Retrieval: Hybrid search (dense Qdrant + BM25 Elasticsearch), fusion, filters, freshness.
- Evidence Manager: Tracks span IDs, offsets, citations, contradictions, credibility scores.
- Synthesis Engine: Outline-guided, multi-doc summarization; structured JSON outputs.
- Memory: Episodic (session transcripts in Postgres) and semantic (vector DB) with time-decay.
- Guardrails: Safety filters, hallucination checks, dedup controls.
- Observability: Traces for each step; metrics; logs; evaluation artifacts.

### 9.3 Data Flow
1) User submits query.
2) Planner creates sub-tasks and tool calls.
3) Search APIs return candidates; scraping/parsing extracts text, tables, metadata.
4) Indexing: Chunking + embeddings → vector upserts; sparse index updates.
5) Retrieval: Hybrid search + rerank + MMR; results returned with metadata.
6) Evidence mapping: Span offsets, citation extraction, contradiction checks.
7) Synthesis: Structured report with citations, uncertainties, compare/contrast.
8) Export: Markdown/JSON; optional BibTeX/CSL JSON.
9) Evaluation: RAG metrics computed; traces/logs stored.
10) Memory updated for future queries.

## 10. Data Model
### 10.1 Entity Relationships
- User (1..*) Workspace
- Workspace (1..*) Project
- Project (1..*) Run
- Run (1..*) Artifact (plans, notes, reports)
- Source (URL/PDF) (1..*) DocumentChunk
- DocumentChunk (1..*) Embedding
- Artifact (1..*) Claim
- Claim (1..*) Citation (to DocumentChunk spans)
- Claim (0..*) Contradiction (linking opposing claims)
- ToolInvocation linked to Run
- Feedback linked to Artifact/Claim
- EvalMetrics linked to Run/Artifact

### 10.2 Database Schema (PostgreSQL, simplified)
- users(id, email, name, auth_provider, created_at)
- workspaces(id, name, owner_id, created_at)
- workspace_members(id, workspace_id, user_id, role)
- projects(id, workspace_id, name, description, created_at, updated_at)
- runs(id, project_id, status, started_at, finished_at, cost_usd, provider_stats jsonb)
- sources(id, project_id, url, title, author, published_at, license, credibility_score, hash, created_at)
- documents(id, source_id, mime_type, storage_uri, pages jsonb, created_at)
- chunks(id, document_id, chunk_index, text, start_offset, end_offset, page_num, metadata jsonb)
- embeddings(id, chunk_id, model_name, dim, vector vector/array, created_at)
- artifacts(id, run_id, type, title, content_md, content_json jsonb, version, created_at)
- claims(id, artifact_id, text, confidence, section, created_at)
- citations(id, claim_id, chunk_id, span_start, span_end, quote, page_num, url_anchor)
- contradictions(id, claim_a_id, claim_b_id, nli_label, score, created_at)
- tool_invocations(id, run_id, tool_name, input jsonb, output jsonb, started_at, finished_at, success)
- feedback(id, artifact_id, user_id, rating, comment, created_at)
- eval_metrics(id, run_id, metric_name, metric_value, details jsonb, created_at)
- audit_logs(id, user_id, action, target, metadata jsonb, created_at)
- memory_sessions(id, project_id, summary, last_accessed_at)
- api_keys(id, user_id, key_hash, scopes, created_at, revoked_at)

Vector data stored in Qdrant; sparse index in Elasticsearch.

### 10.3 Data Flow Diagrams
- Ingestion: Source → Parser → Chunks → Embeddings → Vector Upsert → Sparse Upsert.
- Retrieval: Query → Embed + BM25 → Fusion → Rerank → MMR → Top-k → Evidence Map.
- Synthesis: Top-k → Outline → Section generation → Claims → Citations → Export.

### 10.4 Input Data & Dataset Requirements
- Inputs: URLs, PDFs (scientific articles, blogs, reports), user notes.
- Dataset for evaluation: Curated query set with gold citations; synthetic HyDE queries; negative controls for contradiction tests.
- Embedding updates: Re-embed schedule when model drifts or thresholded performance drops.
- Language: English initial; multilingual optional via models supporting it.

## 11. API Specifications
### 11.1 REST Endpoints
- POST /v1/auth/token (OAuth2 exchange)
- POST /v1/projects
- GET /v1/projects/:id
- POST /v1/projects/:id/runs (start agent run)
- GET /v1/runs/:id/status
- POST /v1/query (one-shot answer w/ citations)
- POST /v1/ingest (URL/PDF upload)
- GET /v1/sources/:id
- GET /v1/projects/:id/artifacts
- GET /v1/artifacts/:id
- POST /v1/export (format=md|json|bibtex)
- POST /v1/eval (trigger evaluation on last run)
- GET /v1/trace/:run_id
- POST /v1/feedback

### 11.2 Request/Response Examples
Request: POST /v1/query
{
  "project_id": "prj_123",
  "query": "Summarize recent advances in retrieval-augmented generation with citations.",
  "constraints": {"recency_days": 365, "domains_allow": ["arxiv.org","aclweb.org"]},
  "output": {"format": "json", "sections": ["findings","citations","gaps","risks"]}
}

Response:
{
  "run_id": "run_987",
  "findings": [
    {"claim": "Hybrid retrieval improves precision@10.",
     "confidence": 0.86,
     "citations": [
       {"source_id":"src_22","url":"https://arxiv.org/abs/xxxx",
        "page_num": 3, "span": {"start": 1520, "end": 1690}, "quote": "…"},
       {"source_id":"src_25","url":"https://aclweb.org/…","page_num": 5, "span":{"start": 320, "end": 470}}
     ],
     "contradictions": []
    }
  ],
  "gaps": ["Limited multilingual retrieval benchmarks."],
  "risks": ["Model drift affecting embedding relevance."],
  "metrics": {"retrieval_recall@10": 0.88, "nDCG@10": 0.77, "faithfulness": 0.82}
}

Request: POST /v1/ingest
{
  "project_id":"prj_123",
  "sources":[
    {"type":"url","value":"https://example.com/post"},
    {"type":"pdf","value":"s3://bucket/path/paper.pdf"}
  ],
  "metadata":{"license":"CC-BY-4.0"}
}

Response:
{"ingested":[{"source_id":"src_31","status":"ok"},{"source_id":"src_32","status":"ok"}]}

### 11.3 Authentication
- OAuth2/OIDC with JWT; API key for service-to-service.
- Scopes: read:project, write:project, run:agent, admin:workspace.
- Rate limits: per key, burst + sustained, configurable.

## 12. UI/UX Requirements
### 12.1 User Interface
- Research Canvas: plan tree, tasks, and progress indicators.
- Evidence Panel: top-k documents, spans, highlights, credibility.
- Synthesis Editor: outline view, sections, inline citations with hover previews.
- Contradictions View: grouped conflicts with NLI labels and scores.
- Notes Sidebar: user highlights and sticky notes linked to spans.
- Export Modal: format selection, schema validation.
- Settings: model/provider selection, filters, prompt templates.

### 12.2 User Experience
- Streaming responses with partial results and live citations.
- One-click “Drill deeper” on subtopics; “Why this source?” explainability.
- Keyboard shortcuts; undo/redo; version history and diffs.
- Collaboration: share links, comment mode (phase 2).

### 12.3 Accessibility
- WCAG 2.1 AA: color contrast, keyboard navigation, ARIA labels.
- Screen reader support for citations and highlights.

## 13. Security Requirements
### 13.1 Authentication
- OIDC with rotating refresh tokens; short-lived JWT access tokens.
### 13.2 Authorization
- RBAC at workspace/project; attribute-based checks for exports and ingestion.
### 13.3 Data Protection
- TLS 1.2+ in transit; AES-256 at rest; KMS-managed keys.
- Secret management via Vault/SM; periodic key rotation.
- PII redaction in logs/traces; prompt redaction policy.
### 13.4 Compliance
- SOC 2 readiness: logging, access reviews, change management.
- GDPR: data subject requests, data residency (EU region option).
- Content usage: respect robots.txt, site terms, and license metadata.

## 14. Performance Requirements
### 14.1 Response Times
- Retrieval P95 ≤ 500ms; rerank ≤ 300ms; first token ≤ 1s; end-to-end P95 ≤ 6s (cached).
### 14.2 Throughput
- Support 50 RPS steady with autoscaling; burst to 200 RPS; queue backpressure with graceful degradation.
### 14.3 Resource Usage
- Median CPU utilization per worker ≤ 65%; memory headroom ≥ 25%.
- Cost per query target ≤ $0.03 (provider-dependent), with caching.

## 15. Scalability Requirements
### 15.1 Horizontal Scaling
- Stateless API and workers; HPA on CPU/RPS; partitioned vector indexes.
### 15.2 Vertical Scaling
- Larger memory for rerank/long-context; SSD-backed storage for Elasticsearch.
### 15.3 Load Handling
- Rate limiting, circuit breakers; fallbacks to cached results; degrade features (e.g., reduce k) under load.

## 16. Testing Strategy
### 16.1 Unit Testing
- 80%+ coverage for parsers, chunking, embedding, retrieval fusion, citation mapping.
### 16.2 Integration Testing
- End-to-end agent runs on sandbox queries; deterministic responses via fixtures; mock search APIs.
### 16.3 Performance Testing
- k6/Locust for RPS and latency SLOs; shadow traffic on staging.
### 16.4 Security Testing
- SAST/DAST; dependency scanning; secrets detection; authz fuzzing; red team scenarios for prompt injection.

## 17. Deployment Strategy
### 17.1 Deployment Pipeline
- GitHub Actions: lint/test → build Docker → push → Helm deploy to staging → smoke tests → canary to prod.
### 17.2 Environments
- Dev (local Docker Compose), Staging (mirrors prod), Prod (multi-AZ).
### 17.3 Rollout Plan
- Canary 10% → 50% → 100% with automated metrics checks; feature flags for risky features.
### 17.4 Rollback Procedures
- Helm rollback to prior release; DB migrations with down scripts; circuit breakers for external providers.

## 18. Monitoring & Observability
### 18.1 Metrics
- Latency P50/P95/P99, error rates, throughput.
- Retrieval metrics: Recall@k, nDCG, MRR; reranker uplift.
- Faithfulness and citation coverage; hallucination rate.
- Cost per run; cache hit rate; token usage by provider.
### 18.2 Logging
- Structured JSON logs with correlation IDs; redacted prompts/responses.
### 18.3 Alerting
- On-call alerts for SLO violation, error spikes, provider failures.
### 18.4 Dashboards
- Grafana: API latency, worker queue depth, vector search times, rerank scores, evaluation KPIs.

## 19. Risk Assessment
### 19.1 Technical Risks
- Hallucinations and mis-citations despite guardrails.
- Provider outages or pricing changes.
- Web scraping blocked; robots.txt or TOS constraints.
- Index drift causing relevance degradation.
### 19.2 Business Risks
- Adoption friction if trust/UX not compelling.
- Data retention and compliance concerns for enterprise.
- Cost overruns without caching/routing optimization.
### 19.3 Mitigation Strategies
- Multi-provider fallback; caching and offline ingestion.
- RAG evaluation and regression gates before releases.
- Legal review for scraping; respect site policies; allow allowlist-only mode.
- Embedding drift monitors; scheduled re-embeds; A/B test retrieval variants.

## 20. Timeline & Milestones
### 20.1 Phase Breakdown (Approx. 24 weeks)
- Phase 0 (2w): Requirements, architecture, designs, infra scaffolding.
- Phase 1 (6w): Core ingestion, parsing, chunking, embeddings, vector + BM25.
- Phase 2 (6w): Agentic loop, tool calling, hybrid retrieval, reranking, evidence mapping.
- Phase 3 (4w): Synthesis engine, citations, contradiction detection, exports.
- Phase 4 (3w): UI polish, notes, plan view, observability, guardrails.
- Phase 5 (3w): Evaluation suite (RAGAS/TruLens), performance tuning, beta.
### 20.2 Key Milestones
- M1 (Week 2): Dev environment + CI/CD operational.
- M2 (Week 8): Ingest → Retrieve → Basic answer with citations.
- M3 (Week 14): Full agent loop with hybrid retrieval + rerank + MMR.
- M4 (Week 18): Structured synthesis and exports; contradiction checks.
- M5 (Week 21): UI/UX complete; observability dashboards live.
- M6 (Week 24): Beta release with SLOs met; security review passed.

Estimated Team: 6–8 (PM, EM, 3–4 backend/ML, 1–2 frontend, DevOps).
Estimated Cloud/Provider Monthly Cost at beta scale: $12k–$25k (LLM usage variable).

## 21. Success Metrics & KPIs
### 21.1 Measurable Targets
- Uptime ≥ 99.5%.
- End-to-end P95 latency ≤ 6s (cached scenarios).
- Retrieval Recall@10 ≥ 0.85; nDCG@10 ≥ 0.75.
- Faithfulness score ≥ 0.8; hallucination rate ≤ 5%.
- Citation coverage ≥ 95%; span precision ≥ 90%.
- User productivity: time-to-first-draft reduced by ≥ 70%.
- Adoption: 200+ weekly active users within 3 months of GA.
- NPS ≥ 45; CSAT ≥ 4.3/5.

## 22. Appendices & Glossary
### 22.1 Technical Background
- Agentic loop implements planner-executor with Reflexion: the LLM critiques intermediate outputs and revises steps.
- Retrieval uses hybrid dense (Qdrant HNSW) + BM25 from Elasticsearch; reciprocal rank fusion improves robustness.
- Reranking via cross-encoders lifts precision at small k; MMR ensures diversity.
- Evidence grounding uses chunk-level IDs and span offsets for exact provenance.
- Contradiction detection leverages NLI models; claims are linked to supportive or opposing evidence.
- Guardrails include safety classifiers, prompt hardening, and fact-check sub-queries.
- Observability captures query traces, top-k snapshots, and vector distances.

### 22.2 References
- ReAct: Reasoning and Acting in Language Models.
- Plan-and-Execute; Reflexion: Language Agents with Verbal Reinforcement Learning.
- HyDE: Hypothetical Document Embeddings for retrieval.
- RAGAS: Framework for evaluating RAG systems.
- MMR: Maximal Marginal Relevance for diversification.
- Qdrant, FAISS, Elasticsearch BM25 documentation.
- TruLens, G-Eval methods for LLM evaluation.

### 22.3 Glossary
- Agent: An LLM-driven process that plans and executes tasks using tools.
- RAG: Retrieval-Augmented Generation; using external knowledge to ground LLM outputs.
- Dense Retrieval: Embedding-based similarity search in a vector database.
- Sparse Retrieval: Keyword-based scoring like BM25.
- HNSW: Graph-based ANN index enabling fast nearest neighbor searches.
- Reranker: Model that reorders retrieved candidates for higher precision.
- MMR: Algorithm balancing relevance and diversity in selections.
- NLI: Natural Language Inference for entailment/contradiction detection.
- Provenance: Lineage linking claims to exact source spans.
- Faithfulness: Degree to which generated text is supported by sources.
- Hallucination: Unsupported or fabricated content from a model.
- Credibility Score: Heuristic combining domain authority, citations, and recency.

Repository Structure
- /README.md
- /LICENSE
- /configs/
  - app.yaml
  - providers.yaml
  - retrieval.yaml
- /src/
  - api/
    - main.py
    - routers/
      - auth.py
      - projects.py
      - runs.py
      - query.py
      - ingest.py
      - export.py
      - eval.py
  - orchestrator/
    - agent.py
    - planner.py
    - tools.py
    - memory.py
  - retrieval/
    - hybrid.py
    - embeddings.py
    - rerank.py
    - indexers/
      - qdrant_index.py
      - elastic_index.py
  - parsers/
    - html_parser.py
    - pdf_parser.py
    - table_extractor.py
    - citation_parser.py
  - evidence/
    - citations.py
    - contradictions.py
    - credibility.py
  - synthesis/
    - outline.py
    - summarizer.py
    - exporter.py
  - guardrails/
    - safety.py
    - faithfulness.py
  - eval/
    - ragas_runner.py
    - trulens_runner.py
  - utils/
    - logger.py
    - config.py
    - cache.py
- /tests/
  - unit/
  - integration/
  - e2e/
  - perf/
- /notebooks/
  - retrieval_eval.ipynb
  - chunking_ablation.ipynb
  - reranker_uplift.ipynb
- /data/
  - samples/
  - fixtures/
- /scripts/
  - migrate_db.py
  - load_demo_data.py
- /infra/
  - helm/
  - terraform/

Sample Config (configs/retrieval.yaml)
retrieval:
  dense:
    provider: qdrant
    collection: "aiml016_chunks"
    embedding_model: "bge-large-en-v1.5"
    similarity: "cosine"
    top_k: 50
  sparse:
    provider: elasticsearch
    index: "aiml016_docs"
    top_k: 100
  fusion:
    method: "rrf"
    k: 10
  rerank:
    model: "bge-reranker-v2"
    top_k: 15
  mmr:
    lambda: 0.7
    k: 10
  filters:
    recency_days: 365
    languages: ["en"]

Sample API (FastAPI) Snippet (src/api/routers/query.py)
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from ..deps import get_orchestrator

router = APIRouter(prefix="/v1")

class QueryReq(BaseModel):
    project_id: str
    query: str
    constraints: dict | None = None
    output: dict | None = None

@router.post("/query")
async def query(req: QueryReq, orch=Depends(get_orchestrator)):
    run_id, result = await orch.run_query(
        project_id=req.project_id,
        query=req.query,
        constraints=req.constraints or {},
        output=req.output or {}
    )
    return {"run_id": run_id, **result}

Sample Export JSON Schema (key sections)
{
  "type": "object",
  "properties": {
    "findings": {
      "type": "array",
      "items": {
        "type":"object",
        "properties":{
          "claim":{"type":"string"},
          "confidence":{"type":"number"},
          "citations":{"type":"array"},
          "contradictions":{"type":"array"}
        },
        "required":["claim","citations"]
      }
    },
    "gaps":{"type":"array","items":{"type":"string"}},
    "risks":{"type":"array","items":{"type":"string"}}
  },
  "required":["findings"]
}

Additional Performance Targets
- Accuracy: citation span precision ≥ 90%.
- Latency: retrieval ≤ 500ms; full synthesis streaming with TTFB ≤ 1s.
- Availability: 99.5% uptime SLO.

End of PRD.