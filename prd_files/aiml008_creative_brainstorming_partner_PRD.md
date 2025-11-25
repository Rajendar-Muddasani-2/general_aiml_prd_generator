# Product Requirements Document (PRD)
# `Aiml008_Creative_Brainstorming_Partner`

Project ID: aiml008  
Category: AI/ML – Generative AI, RAG, UX Productivity  
Status: Draft for Review  
Version: v1.0  
Last Updated: 2025-11-25

## 1. Overview
### 1.1 Executive Summary
Aiml008_Creative_Brainstorming_Partner is an LLM-powered ideation assistant that helps individuals and teams generate, curate, and refine creative ideas. It combines structured prompting (SCAMPER, Six Thinking Hats, TRIZ, Jobs-To-Be-Done), deliberate reasoning (Tree-of-Thought/Graph-of-Thought), and Retrieval-Augmented Generation (RAG) with diversity controls (MMR, clustering) to produce relevant, novel, and on-brief ideas. The product supports divergent and convergent modes, personalized memory, safety guardrails, and structured outputs for downstream ranking and execution.

### 1.2 Document Purpose
This PRD outlines the problem space, goals, target users, detailed requirements, system architecture, data models, APIs, UI/UX, security, performance/scalability, testing, deployment, monitoring, risks, timeline, KPIs, and appendices necessary to build and launch the product.

### 1.3 Product Vision
Be the go-to creative partner for knowledge workers—turning vague prompts into actionable idea portfolios—by combining state-of-the-art generative AI with rigorous curation, safety, and personalization. Deliver fast, diverse, and feasible ideas, integrated into existing workflows and tools.

## 2. Problem Statement
### 2.1 Current Challenges
- Unstructured ideation leads to repetitive, low-diversity ideas.
- Human sessions are time-consuming and often lack broad inspiration sources.
- AI tools can hallucinate, ignore constraints, or fail to adapt to user preferences.
- Teams struggle with moving from ideation to prioritization and next steps.
- Lack of robust safety, IP sensitivity, and instruction adherence.

### 2.2 Impact Analysis
- Inefficient brainstorming delays product discovery and campaign launches.
- Missed opportunities due to narrow exploration.
- Inconsistent quality harms stakeholder trust in AI tools.
- Fragmented tools impede collaboration and traceability.

### 2.3 Opportunity
- Use LLMs with RAG, deliberate reasoning, and diversity control to generate high-quality, novel ideas quickly.
- Personalization via memory and preferences.
- Structured outputs make ideas actionable and measurable.
- Multi-tenant SaaS with APIs for embedding into products.

## 3. Goals and Objectives
### 3.1 Primary Goals
- Generate diverse, relevant, and safe ideas from user briefs.
- Support structured ideation frameworks (SCAMPER, Six Hats, TRIZ, JTBD).
- Provide curation: scoring, clustering, pairwise preference modeling, deduplication.
- Enable convergent decision-making: ranking, shortlisting, next steps.

### 3.2 Business Objectives
- Drive adoption among product, marketing, and design teams.
- Offer flexible pricing (per-seat and API usage).
- Achieve 99.5% uptime in first 6 months.
- Integrate with key tools (Slack, Notion, Jira, Figma) to reduce churn.

### 3.3 Success Metrics
- ≥90% instruction adherence score (rubric-based) across sessions.
- ≥4.5/5 user satisfaction (CSAT) after sessions.
- ≥30% reduction in time-to-first-good-idea vs baseline.
- ≥25% more unique idea directions per session (semantic diversity).
- 99.5% service uptime; p95 retrieval latency <500ms; p95 time-to-first-token <500ms.

## 4. Target Users/Audience
### 4.1 Primary Users
- Product managers, UX designers, marketers, content strategists, innovation teams, startup founders.

### 4.2 Secondary Users
- Sales enablement, customer success, educators, research analysts, consultants, students.

### 4.3 User Personas
1) Name: Priya Nair  
   Role: Senior Product Manager at B2B SaaS  
   Background: 8 years PM; runs discovery sprints; familiar with JTBD and OKRs.  
   Pain Points: Stakeholders expect novel angles fast; team repeats old solutions; long synthesis cycles.  
   Goals: Quickly explore solution spaces, align with constraints, generate prioritized roadmap ideas.  
   Tools: Jira, Confluence, Slack, Miro.  
   Success: Diverse options, evidence-backed ideas, clear next steps.

2) Name: Alex Romero  
   Role: Creative Director at Digital Agency  
   Background: Leads multi-channel campaigns; consumer insights driven.  
   Pain Points: Campaign fatigue; too many meetings; difficulty balancing novelty and brand safety.  
   Goals: High-variance ideation with strict brand guardrails; fast curation for client decks.  
   Tools: Figma, Notion, Google Workspace, Slack.  
   Success: Safe, on-brand, presentation-ready concepts with rationale and risks.

3) Name: Mei Chen  
   Role: Startup Founder (Consumer App)  
   Background: Technical; wears many hats; data-driven experiments.  
   Pain Points: Limited time; needs rapid hypothesis generation; must avoid reinventing the wheel.  
   Goals: Generate experiments, growth loops, and feature ideas; rank by impact/effort.  
   Tools: Linear, Notion, Amplitude.  
   Success: Actionable experiments with metrics, quick export to backlog.

4) Name: Daniel López  
   Role: Instructional Designer  
   Pain Points: Course ideation stalls; needs structured prompts and examples.  
   Goals: Lesson and assessment concept generation with learner alignment and accessibility.

## 5. User Stories
- US-001: As a PM, I want divergent brainstorming using SCAMPER so that I get varied solution angles.  
  Acceptance: Given a problem brief, system generates ≥7 distinct SCAMPER-derived ideas with titles, rationales, and target users.

- US-002: As a marketer, I want Six Thinking Hats mode so that I can see optimistic, critical, customer, and creative perspectives.  
  Acceptance: At least 5 role-perspective variants per idea, with color-coded tags.

- US-003: As a founder, I want convergence support so that I can shortlist top 5 ideas.  
  Acceptance: System provides scoring (novelty, feasibility, alignment, impact) and allows sorting and selection with export.

- US-004: As a user, I want the system to consider my past preferences so that it avoids repetition.  
  Acceptance: Similar ideas across sessions are deduplicated; preference profile influences ranking.

- US-005: As a team lead, I want safe outputs so that we avoid policy and IP risks.  
  Acceptance: All outputs pass moderation; flagged content is withheld with explanations.

- US-006: As a researcher, I want inspiration mining from adjacent domains so that I can cross-pollinate ideas.  
  Acceptance: Retrieves at least one exemplar per cluster with citations/links.

- US-007: As a user, I want to upload an image for inspiration so that ideas are grounded in visual cues.  
  Acceptance: System extracts tags/alt text and uses them in RAG; ideas reference visual themes.

- US-008: As an analyst, I want JSON-structured ideas so that I can pipe them to analytics.  
  Acceptance: Ideas conform to schema; validation passes 100%.

- US-009: As an admin, I want API keys and usage analytics so that I can manage costs.  
  Acceptance: Create/rotate/revoke keys; view per-key usage and cost dashboard.

## 6. Functional Requirements
### 6.1 Core Features
- FR-001: Divergent brainstorming modes (SCAMPER, Six Hats, TRIZ, JTBD).
- FR-002: Convergent curation (scoring rubric: novelty, feasibility, alignment, impact; pairwise preference modeling; ranking).
- FR-003: RAG pipeline using user briefs, constraints, trend data, and exemplars.
- FR-004: Diversity controls (MMR, semantic clustering, self-consistency sampling, semantic deduplication).
- FR-005: Memory (short-term session and long-term user preference profiles).
- FR-006: Safety and guardrails (content moderation, IP sensitivity heuristics, bias checks, instruction adherence).
- FR-007: Structured outputs in JSON schema; export CSV/JSON/Markdown.
- FR-008: Multimodal hooks: ingest images/links; generate alt text and concept tags.
- FR-009: Collaboration: comments, reactions, shared workspaces, role prompts.
- FR-010: Streaming responses with partial updates.
- FR-011: Templates library (prompt scaffolds, few-shot exemplars).
- FR-012: Inspiration mining from vector store; analogical reasoning mapping.

### 6.2 Advanced Features
- FR-013: Graph-of-Thought exploration with branching and merge; visualize idea trees.
- FR-014: Cross-encoder re-ranking for top-K ideas/inspirations.
- FR-015: Session-aware retrieval weighting recency and user embeddings.
- FR-016: Automatic critique-then-rewrite loop for top N ideas.
- FR-017: A/B testing of prompts and sampling parameters.
- FR-018: Slack/Notion/Jira/Figma integrations for push/pull of briefs and exports.
- FR-019: Team analytics: idea diversity index, selection rate, time-to-decision.
- FR-020: Fine-tunable scoring model (preference modeling from feedback).

## 7. Non-Functional Requirements
### 7.1 Performance
- p95 retrieval latency <500ms; p95 time-to-first-token <500ms; p95 end-to-end under 2.5s for short briefs.
### 7.2 Reliability
- 99.5% uptime; graceful degradation with model/provider failover; idempotent APIs.
### 7.3 Usability
- Onboarding within 5 minutes; accessibility compliant; intuitive mode switching.
### 7.4 Maintainability
- Modular services; typed APIs; CI/CD; comprehensive tests; observability with traces.

## 8. Technical Requirements
### 8.1 Technical Stack
- Backend: Python 3.11+, FastAPI 0.115+, Uvicorn, Pydantic v2.
- Frontend: React 18+, TypeScript 5+, Vite, TailwindCSS.
- Orchestration: Kubernetes 1.29+, Helm, KEDA for autoscaling.
- Data: PostgreSQL 15+ (with pgvector 0.5+), Redis 7+, MinIO/S3 for assets.
- Vector DB: Milvus 2.3+ or Pinecone; local dev: Chroma/FAISS (HNSW).
- Retrieval: SentenceTransformers/all-MiniLM-L12-v2 for fast; bge-large-en-v1.5 for quality.
- Re-ranker: cross-encoder/ms-marco-MiniLM-L-6-v2 or Cohere Rerank.
- LLM Providers: OpenAI, Anthropic, Azure OpenAI; optional self-host Llama 3.1 70B with vLLM 0.5+.
- Queue/Workers: Celery + Redis or Kafka 3+.
- Monitoring: Prometheus, Grafana, OpenTelemetry, Jaeger.
- Auth: OAuth2/OIDC, JWT, Keycloak or Auth0.
- Cloud: AWS (EKS, RDS, S3, CloudFront) or GCP/Azure equivalents.

### 8.2 AI/ML Components
- Prompt Orchestrator with system/role prompts and templates.
- RAG pipeline: multi-query expansion, HyDE, ANN retrieval, MMR blending, cross-encoder re-rank.
- Deliberate reasoning: Tree-of-Thought/Graph-of-Thought; self-consistency sampling.
- Diversity & dedup: k-means/HDBSCAN clustering; semantic deduplication thresholding.
- Scoring: lightweight rubric LLM; learned preference model trained on pairwise feedback.
- Safety: content moderation API; toxicity/hate checks; prompt-injection defense; IP sensitivity heuristics.
- Personalization: user embeddings; context weighting; recurrence penalties.
- Multimodal tagging: CLIP/Vit-based embeddings for images; Alt-text generation.

## 9. System Architecture
### 9.1 High-Level Architecture (ASCII diagram)
                     +-----------------+
User (Web/SDK) --->  |  API Gateway    |  <---> Auth/OIDC
          SSE/HTTP   +--------+--------+
                             |
                             v
                     +------------------+
                     |  Brainstorm Svc  |---> Redis Cache
                     |  (FastAPI)       |
                     +----+----+---+----+
                          |    |   |
           +--------------+    |   +----------------+
           |                   |                    |
           v                   v                    v
+------------------+   +---------------+   +--------------------+
| RAG Orchestrator |   | LLM Router    |   | Scoring/Rank Svc   |
| (retrieval, MMR) |   | (providers)   |   | (rubric, pairwise) |
+---+----------+---+   +-------+-------+   +---------+----------+
    |          |               |                         |
    v          v               v                         v
+-------+  +--------+   +-------------+          +--------------+
|Vector |  |Doc Store|  | Providers   |          | Analytics/   |
| DB    |  |(S3/MinIO)| | (OpenAI/..) |          | Telemetry    |
+---+---+  +----+---+   +------+------+          +------+-------+
    |           |               |                        |
    v           v               v                        v
+-------+  +-----------+  +-----------+         +----------------+
|Postgres|  | Asset Svc |  | Worker(s) |         | Monitoring     |
+-------+  +-----------+  +-----------+         +----------------+

### 9.2 Component Details
- API Gateway: Routing, rate limiting, JWT validation.
- Brainstorm Service: Session handling, templates, pipelines, streaming.
- RAG Orchestrator: Query expansion, ANN retrieval, MMR, re-ranking.
- LLM Router: Provider abstraction, retries, temperature/top-p control, cost tracking.
- Scoring/Rank Service: Rubric scoring via LLM and learned model for pairwise preferences.
- Vector DB: Stores embeddings for inspirations, templates, session memory.
- Postgres: Metadata for users, sessions, ideas, feedback, configs.
- Asset Service: Uploads images/links; generates tags/alt text.
- Workers: Long-running tasks (Graph-of-Thought expansion, clustering).
- Analytics/Telemetry: Usage metrics, costs, quality signals.
- Monitoring: Traces/logs/alerts.

### 9.3 Data Flow
1) User submits brief (+ optional image/links).  
2) RAG Orchestrator expands query (multi-query + HyDE), retrieves inspirations via ANN + MMR, re-ranks.  
3) LLM Router generates ideas using structured prompts, role perspectives, and deliberate reasoning.  
4) Diversity module clusters and deduplicates ideas; self-consistency picks robust candidates.  
5) Scoring/Rank service scores rubric; optional pairwise comparisons refine top list.  
6) Safety filters applied; flagged outputs revised or elided.  
7) Results streamed; user interacts (feedback, edits).  
8) Feedback updates preference model and memory; data stored for analytics.

## 10. Data Model
### 10.1 Entity Relationships
- User (1..n) Workspace (1..n) Session.
- Session (1..n) Idea.
- Idea (1..n) IdeaScore, (1..n) Feedback.
- InspirationItem (n..n) Session via SessionInspiration.
- PromptTemplate (1..n) Session.
- Attachment (image/link) (1..n) Session.
- PreferenceProfile (1..1) User.
- APIKey (1..n) Workspace.
- ModerationEvent (1..n) Session/Idea.

### 10.2 Database Schema (PostgreSQL)
- users(id PK, email unique, name, org_id, created_at, prefs_jsonb)
- workspaces(id PK, name, owner_user_id FK, plan_tier, created_at)
- workspace_members(user_id FK, workspace_id FK, role, joined_at)
- sessions(id PK, workspace_id FK, user_id FK, title, brief_text, mode enum[divergent,convergent], status, created_at)
- ideas(id PK, session_id FK, title, rationale, target_user, risks, constraints_satisfied jsonb, next_steps jsonb, source enum[llm,user], cluster_id, parent_idea_id, created_at)
- idea_scores(id PK, idea_id FK, novelty int, feasibility int, alignment int, impact int, total float, model_version, created_at)
- feedback(id PK, idea_id FK, user_id FK, type enum[like,dislike,comment,compare], details jsonb, created_at)
- inspiration_items(id PK, title, url, source_type, content_text, embedding vector, created_at)
- session_inspirations(session_id FK, inspiration_id FK, rank, score)
- prompt_templates(id PK, name, framework enum[SCAMPER,SIXHATS,TRIZ,JTBD], text, examples jsonb, created_at)
- attachments(id PK, session_id FK, type enum[image,link], url, tags jsonb, alt_text, embedding vector, created_at)
- preference_profiles(user_id PK, vector vector, weights jsonb, updated_at)
- moderation_events(id PK, session_id FK, idea_id FK, type, reason, action, created_at)
- api_keys(id PK, workspace_id FK, name, key_hash, scopes jsonb, created_at, last_used_at)
- provider_configs(id PK, workspace_id FK, provider, model, api_key_ref, settings jsonb)

### 10.3 Data Flow Diagrams (ASCII)
User -> Session -> RAG Retrieve -> LLM Generate -> Diversity Filter -> Score/Rank -> Safety -> Stream -> Feedback -> Preference Update

[User]
  | create brief
  v
[Session] --> [RAG Retrieve] --> [LLM Generate]
                                   |
                                   v
                          [Cluster & Dedup] --> [Score]
                                   |                |
                                   v                v
                                 [Safety] <---- [Feedback]
                                   |
                                   v
                                [Results]

### 10.4 Input Data & Dataset Requirements
- User briefs, constraints, goals, style guides.
- Inspiration corpora: public web pages (licensed), internal docs (user-provided), case studies, trend reports.
- Embeddings for texts and images; metadata for provenance and citations.
- Feedback logs (pairwise choices, likes/dislikes) for preference model training.
- Safety datasets for moderation tuning (if applicable).
- All data use must respect user consent, access controls, and content licenses.

## 11. API Specifications
### 11.1 REST Endpoints
- POST /v1/auth/token — OAuth2 token exchange.
- POST /v1/sessions — Create session.
- GET /v1/sessions/{id} — Get session.
- POST /v1/sessions/{id}/attachments — Upload/link asset.
- POST /v1/sessions/{id}/brainstorm — Start brainstorming (stream or non-stream).
- GET /v1/sessions/{id}/ideas — List ideas.
- POST /v1/ideas/{id}/score — Rescore idea.
- POST /v1/ideas/{id}/feedback — Submit feedback (like/dislike/comment/compare).
- POST /v1/ideas/rerank — Rerank a set of ideas with preference model.
- GET /v1/sessions/{id}/clusters — Get clusters and representatives.
- GET /v1/templates — List prompt templates.
- POST /v1/inspiration/search — Search vector store for inspirations.
- POST /v1/admin/apikeys — Create API key (admin).
- GET /v1/admin/usage — Usage and cost metrics.
- GET /v1/stream/sessions/{id}/events — SSE endpoint for streaming updates.

### 11.2 Request/Response Examples
- Create Session
Request:
POST /v1/sessions
{
  "workspace_id": "w_123",
  "title": "Onboarding revamp",
  "brief_text": "Improve first-week activation for our mobile app",
  "mode": "divergent",
  "frameworks": ["SCAMPER","SIXHATS"],
  "constraints": ["Comply with brand tone", "Mobile-first"]
}
Response:
{
  "id": "s_789",
  "status": "created",
  "created_at": "2025-11-25T10:00:00Z"
}

- Start Brainstorm (stream=false)
POST /v1/sessions/s_789/brainstorm
{
  "temperature": 0.9,
  "top_p": 0.95,
  "num_ideas": 20,
  "use_graph_of_thought": true,
  "retrieve_inspirations": true,
  "safety_level": "standard"
}
Response:
{
  "ideas": [
    {
      "id": "i_001",
      "title": "Onboarding Quest with Social Proof",
      "rationale": "Leverages intrinsic motivation...",
      "target_user": "New mobile users",
      "constraints_satisfied": ["Brand tone", "Mobile-first"],
      "risks": ["Over-gamification"],
      "next_steps": ["A/B test quest vs control", "Design badges"],
      "scores": {"novelty":4,"feasibility":4,"alignment":5,"impact":4,"total":4.25}
    }
  ],
  "clusters": [{"cluster_id": 0, "rep_idea_id":"i_001", "size":7}],
  "inspirations": [{"title":"Case study: Habit loops","url":"https://..."}]
}

- SSE Stream
GET /v1/stream/sessions/s_789/events
Event: idea.partial
data: {"id":"i_tmp","title":"...", "partial":true}

- Rerank
POST /v1/ideas/rerank
{
  "idea_ids": ["i_001","i_002","i_003"],
  "objective": "maximize_impact_alignment",
  "user_id": "u_123"
}
Response:
{"ranking": [{"idea_id":"i_003","score":0.82},{"idea_id":"i_001","score":0.78},{"idea_id":"i_002","score":0.64}]}

### 11.3 Authentication
- OAuth2/OIDC with JWT bearer tokens for users.
- API keys for service-to-service; scopes per workspace.
- Token rotation; least privilege; IP allowlists (optional).

## 12. UI/UX Requirements
### 12.1 User Interface
- Left: Session list and templates; Center: Brief editor and results; Right: Parameters (temperature, top-p, frameworks).
- Toggle for Divergent vs Convergent modes.
- Role prompt chips: Optimist, Critic, Customer, Futurist.
- Clustering map view with representative ideas.
- Idea cards with scores, risks, constraints, next steps; inline edit.
- Safety badges; rationale tooltips; citations panel.
- Export buttons (JSON/CSV/Markdown); shareable link; integrations panel.

### 12.2 User Experience
- Onboarding wizard with example templates.
- Streaming results with skeleton loaders; first token <500ms p95.
- Keyboard shortcuts (generate: Cmd/Ctrl+Enter; next: J/K).
- Undo/redo; draft history; autosave.
- Feedback interactions: like/dislike, pairwise compare, comment threads.

### 12.3 Accessibility
- WCAG 2.1 AA compliance: color contrast, focus states, ARIA roles.
- Keyboard navigability; screen reader labels; captions for multimedia.

## 13. Security Requirements
### 13.1 Authentication
- OIDC/OAuth2; MFA optional; session timeout configurable; device management.

### 13.2 Authorization
- RBAC: owner, admin, editor, viewer; row-level workspace isolation.
- Signed URLs for asset access; strict bucket policies.

### 13.3 Data Protection
- TLS 1.3 in transit; AES-256 at rest; secrets in KMS.
- PII minimization; data retention policies; deletion on request.
- Content provenance tracking; prompt and output logs protected.

### 13.4 Compliance
- GDPR/CCPA readiness; SOC 2-aligned controls; DPA and data residency options.
- Audit logs for sensitive operations; privacy by design.

## 14. Performance Requirements
### 14.1 Response Times
- p95: retrieval <500ms, time-to-first-token <500ms, full generation <2.5s for short prompts; larger tasks stream progressively.

### 14.2 Throughput
- Handle 100 RPS baseline; burst to 300 RPS with autoscaling; backpressure via queues.

### 14.3 Resource Usage
- GPU utilization >50% for self-hosted LLM; CPU utilization <70% p95; memory per worker capped with OOM guards.
- Cache hit rate >60% for embeddings and inspirations.

## 15. Scalability Requirements
### 15.1 Horizontal Scaling
- Stateless API pods with HPA; worker pool autoscaling via KEDA on queue depth.

### 15.2 Vertical Scaling
- LLM inference nodes scaled up with GPU memory tiers; vector DB shards with segment memory optimized.

### 15.3 Load Handling
- Rate limiting per API key; circuit breakers and provider failover; request hedging for tail latency.

## 16. Testing Strategy
### 16.1 Unit Testing
- 80%+ coverage for core modules: prompts, scoring, retrieval, API handlers.
- Snapshot tests for prompt templates.

### 16.2 Integration Testing
- End-to-end RAG pipeline with mocked providers; contract tests for vector DB and re-ranker.

### 16.3 Performance Testing
- k6 load tests for 100–300 RPS; latency SLO checks; chaos tests for provider outages.

### 16.4 Security Testing
- Static analysis (Bandit), dependency scanning, secret scanning.
- DAST; red-team prompt injection tests; moderation evasion tests.

## 17. Deployment Strategy
### 17.1 Deployment Pipeline
- GitHub Actions CI: lint, type-check, tests, build images.
- CD via ArgoCD/Helm; automated migrations; feature flags.

### 17.2 Environments
- Dev (shared), Staging (pre-prod with masked data), Prod (multi-AZ).

### 17.3 Rollout Plan
- Canary (10% -> 50% -> 100%) with metrics gates; blue-green fallback.

### 17.4 Rollback Procedures
- Helm rollback with pinned image; DB rollback via transactional migrations and backups; feature flag disables.

## 18. Monitoring & Observability
### 18.1 Metrics
- Latency (p50/p95/p99), TTFB, throughput, error rates by endpoint.
- Idea quality: instruction adherence %, diversity index, safety pass rate.
- Cost per 1000 tokens; cache hit rate; provider health.

### 18.2 Logging
- Structured JSON logs; request/response metadata (redacted); sampling for high volume.

### 18.3 Alerting
- On-call alerts for SLO breaches; anomaly detection on cost spikes; provider outage notifications.

### 18.4 Dashboards
- Ops: latency/throughput/errors; Quality: adherence/diversity/safety; Business: MAU, sessions, idea selections, conversion.

## 19. Risk Assessment
### 19.1 Technical Risks
- Provider outages; high variance in LLM outputs; RAG drift with stale corpora; prompt injection; cost overruns.

### 19.2 Business Risks
- Low adoption if not integrated into workflows; perceived lack of originality; privacy concerns.

### 19.3 Mitigation Strategies
- Multi-provider routing; caching; periodic corpus refresh and evaluations; guardrails and policy tuning; cost budgets and alerts; strong integrations.

## 20. Timeline & Milestones
### 20.1 Phase breakdown
- Phase 0 (Week 1): Discovery, PRD sign-off, architecture, infra scaffolding.
- Phase 1 (Weeks 2–5): Core pipeline (RAG, generation, diversity, scoring), DB schemas, basic UI.
- Phase 2 (Weeks 6–8): Safety guardrails, memory, clustering UI, streaming, templates.
- Phase 3 (Weeks 9–11): Advanced features (Graph-of-Thought, re-ranker, preference model), integrations (Slack/Notion).
- Phase 4 (Weeks 12–13): Performance hardening, monitoring, analytics, multi-tenant, billing prep.
- Phase 5 (Weeks 14–16): Beta, A/B testing, polish, docs, launch.

### 20.2 Key Milestones
- M1 (Week 3): First ideas end-to-end with RAG; p95 <3s.
- M2 (Week 6): Safety pass rate >95%; structured outputs; clustering in UI.
- M3 (Week 9): Preference model v1; pairwise comparisons live.
- M4 (Week 12): p95 retrieval <500ms; TTFB <500ms; 99.5% uptime target feasible.
- M5 (Week 16): Public beta with integrations; CSAT ≥4.3.

Estimated Costs (monthly, post-beta):  
- Cloud (compute, storage, network): $12k–$18k  
- Managed vector DB or self-host: $3k–$8k  
- LLM usage (beta scale): $8k–$15k  
- Monitoring/logging: $1k–$3k  
- Total: $24k–$44k

## 21. Success Metrics & KPIs
### 21.1 Measurable targets
- Instruction adherence accuracy ≥90% (rubric-based evals).  
- Safety pass rate ≥97%.  
- Idea diversity index ≥0.65 (intra-session cosine dispersion).  
- Deduplication rate ≥30% reduction in near-duplicates (threshold cosine >0.9).  
- User CSAT ≥4.5/5.  
- Time-to-first-token p95 <500ms; end-to-end p95 <2.5s.  
- Uptime ≥99.5%.  
- Weekly active teams: 200+ by Month 3 post-launch.  
- Conversion to paid: ≥8% of active workspaces.

## 22. Appendices & Glossary
### 22.1 Technical Background
- Structured prompting frameworks: SCAMPER (Substitute, Combine, Adapt, Modify, Put to another use, Eliminate, Reverse), Six Thinking Hats (role perspectives), TRIZ (inventive principles), JTBD (jobs, pains, gains).
- Deliberate reasoning: Tree-of-Thought, Graph-of-Thought, self-consistency sampling.
- RAG: multi-query retrieval, HyDE for hypothetical documents, MMR for diversity, re-ranking with cross-encoders, session-aware weighting.
- Diversity and clustering: k-means/HDBSCAN, reciprocal rank fusion, topic maps.
- Evaluation: rubric scoring (novelty, feasibility, alignment, impact), pairwise preference learning.
- Guardrails: content filtering, IP sensitivity heuristics, bias checks, instruction adherence policies.

### 22.2 References
- Wei et al., Chain-of-Thought; Yao et al., Tree-of-Thought.
- Lewis et al., RAG: Retrieval-Augmented Generation.
- SentenceTransformers documentation.
- OpenAI/Anthropic/Azure OpenAI API docs.
- Milvus/Pinecone/FAISS docs.
- MMR and Reciprocal Rank Fusion literature.

### 22.3 Glossary
- ANN: Approximate Nearest Neighbor search for embeddings.
- CLIP: Contrastive Language-Image Pretraining for image/text embeddings.
- Cross-encoder: Model scoring pairs for precise re-ranking.
- Embedding: Numeric vector representation of text/image for similarity.
- Graph-of-Thought: Multi-branch reasoning structure.
- HyDE: Hypothetical Document Embeddings for enriched retrieval.
- JTBD: Jobs-To-Be-Done framework.
- LLM: Large Language Model.
- MMR: Maximal Marginal Relevance for balancing relevance and novelty.
- RAG: Retrieval-Augmented Generation.
- SCAMPER: Ideation framework for structured creativity.
- Self-consistency: Sampling strategy to select robust answers.
- Vector DB: Database optimized for vector similarity.

Repository Structure
- /README.md
- /docs/
- /configs/
  - app.yaml
  - providers.example.yaml
- /src/
  - backend/
    - main.py
    - routers/
      - sessions.py
      - ideas.py
      - admin.py
    - services/
      - rag.py
      - llm_router.py
      - scoring.py
      - safety.py
      - clustering.py
      - preferences.py
    - models/
      - schemas.py
      - db.py
    - workers/
      - tasks.py
  - frontend/
    - src/
      - App.tsx
      - components/
      - pages/
      - store/
- /tests/
  - unit/
  - integration/
  - e2e/
- /notebooks/
  - preference_model.ipynb
  - retrieval_eval.ipynb
- /data/
  - inspirations/
- /infra/
  - helm/
  - terraform/

Config Sample (configs/providers.example.yaml)
providers:
  - name: openai
    model: gpt-4o
    api_key: ${OPENAI_API_KEY}
    settings:
      temperature: 0.8
      top_p: 0.95
  - name: anthropic
    model: claude-3-5-sonnet
    api_key: ${ANTHROPIC_API_KEY}
    settings:
      temperature: 0.7

API Example (curl)
curl -X POST https://api.example.com/v1/sessions \
 -H "Authorization: Bearer $TOKEN" \
 -H "Content-Type: application/json" \
 -d '{
  "workspace_id":"w_123",
  "title":"Q3 Growth Ideas",
  "brief_text":"Increase activation and retention",
  "mode":"divergent",
  "frameworks":["SCAMPER","JTBD"]
}'

Python SDK Snippet
import requests, sseclient, json

BASE="https://api.example.com"
token="..."; headers={"Authorization":f"Bearer {token}"}

s = requests.post(f"{BASE}/v1/sessions", json={"workspace_id":"w_123","title":"Campaign","brief_text":"Launch idea","mode":"divergent"}, headers=headers).json()
sid = s["id"]

requests.post(f"{BASE}/v1/sessions/{sid}/brainstorm", json={"temperature":0.9,"num_ideas":15}, headers=headers)

stream = sseclient.SSEClient(f"{BASE}/v1/stream/sessions/{sid}/events", headers=headers)
for event in stream.events():
    if event.event == "idea":
        print(json.loads(event.data)["title"])

JSON Idea Schema (for validation)
{
  "type": "object",
  "required": ["title","rationale","target_user","next_steps","constraints_satisfied","risks"],
  "properties": {
    "title": {"type":"string"},
    "rationale": {"type":"string"},
    "target_user": {"type":"string"},
    "constraints_satisfied": {"type":"array","items":{"type":"string"}},
    "risks": {"type":"array","items":{"type":"string"}},
    "next_steps": {"type":"array","items":{"type":"string"}}
  }
}

Specific Metrics Targets
- Instruction adherence accuracy ≥90%.
- Safety pass rate ≥97%.
- Retrieval latency p95 <500ms; TTFB p95 <500ms.
- End-to-end idea generation p95 <2.5s (short briefs).
- Uptime ≥99.5%.
- Deduplication reduces redundancy ≥30%.
- Diversity index ≥0.65.

End of PRD.