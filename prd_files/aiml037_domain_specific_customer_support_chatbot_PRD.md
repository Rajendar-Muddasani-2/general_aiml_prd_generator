# Product Requirements Document (PRD)
# `Aiml037_Domain_Specific_Customer_Support_Chatbot`

Project ID: Aiml037_Domain_Specific_Customer_Support_Chatbot
Category: General AI/ML – NLP, RAG, Conversational AI
Status: Draft for Review
Version: v1.0
Last Updated: 2025-11-25

## 1. Overview
### 1.1 Executive Summary
Aiml037_Domain_Specific_Customer_Support_Chatbot is a domain-specific, retrieval-augmented conversational assistant for customer support. It ingests enterprise knowledge (FAQs, manuals, policy docs, past tickets) and provides grounded, multilingual answers with citations, performs account-aware actions (e.g., order lookup, entitlement checks), and escalates to human agents when confidence is low. The system combines hybrid retrieval (BM25 + dense vectors), cross-encoder reranking, policy-compliant generation, and robust observability to deliver high accuracy and low latency at scale.

### 1.2 Document Purpose
This PRD defines the product vision, requirements, technical architecture, data model, APIs, UI/UX, security, performance, testing, deployment, monitoring, risks, and timelines needed to build and launch the chatbot across web, mobile, and agent tooling.

### 1.3 Product Vision
Deliver a trustworthy, fast, and extensible support assistant that:
- Understands domain-specific terminology and context
- Grounds answers in verified sources with citations and confidence
- Works across languages and channels
- Connects to enterprise systems for personalized, authenticated support
- Improves continuously via feedback, analytics, and active learning

## 2. Problem Statement
### 2.1 Current Challenges
- High volume of repetitive inquiries increases agent workload and response times.
- Knowledge is fragmented across disparate systems (Zendesk, Salesforce, Confluence, SharePoint).
- Existing chatbots are brittle and frequently hallucinate or lack citations and personalization.
- Difficult to ensure compliance with policies and handle PII safely.
- Limited analytics to identify content gaps and continuously improve answers.

### 2.2 Impact Analysis
- Slow time-to-resolution reduces CSAT and increases churn.
- Rising operating costs due to manual escalations.
- Inconsistent support quality across regions and languages.
- Risk exposure from ungrounded advice and mishandling of sensitive data.

### 2.3 Opportunity
- Deflect 35–50% of tier-1 tickets via accurate self-serve answers.
- Increase CSAT by 15–20% with faster, consistent responses.
- Reduce average handle time (AHT) for agents by 25–35% with summarization and suggested replies.
- Build a centralized, continuously improving knowledge layer.

## 3. Goals and Objectives
### 3.1 Primary Goals
- Provide grounded, cited, and action-enabled support answers with <2s TTFB (P95) and <5s full response (P95).
- Achieve >90% exact/semantic match on gold FAQ evaluation; >95% groundedness (no unsupported claims).
- Integrate with key enterprise systems (CRM, ticketing, identity, order systems) with robust safety/guardrails.

### 3.2 Business Objectives
- Increase self-service resolution rate (deflection) to ≥35% within 6 months.
- Improve CSAT by ≥15% and reduce AHT by ≥25%.
- Reduce cost per contact by ≥30%.
- Achieve 99.5% monthly uptime.

### 3.3 Success Metrics
- Answer quality: exact match, semantic similarity (BERTScore), groundedness score, citation coverage.
- Operational: latency, throughput, uptime, autoscaling effectiveness, semantic cache hit rate.
- Business: deflection rate, CSAT, AHT, conversion on assisted flows, escalation precision/recall.

## 4. Target Users/Audience
### 4.1 Primary Users
- End customers seeking support via web/mobile chat.
- Support agents using AI assistance in their console.

### 4.2 Secondary Users
- Support managers/operations analyzing performance and content gaps.
- Knowledge managers curating and publishing content.
- Developers/IT managing integrations and deployments.
- Compliance and security teams auditing logs and controls.

### 4.3 User Personas
1) Carla Nguyen – End Customer
- Background: Small business owner; moderate technical proficiency; operates across US and Canada.
- Pain points: Long wait times; inconsistent answers; product SKUs and versions are confusing; needs multilingual support (EN/FR).
- Goals: Quick, accurate answers; self-serve returns and order status; confidence that info is current.

2) Alex Rivera – Tier-1 Support Agent
- Background: 2 years experience; handles 50–70 tickets/day in CRM; works evening shifts.
- Pain points: Repetitive questions; context switching across tools; pressure to reduce AHT; uncertainty with complex version-specific policies.
- Goals: Reliable suggested replies with citations; automatic ticket summaries; clear escalation criteria.

3) Priya Sharma – Support Operations Manager
- Background: Manages 40 agents; responsible for SLAs, CSAT, QA; collaborates with Knowledge team.
- Pain points: Limited visibility into knowledge gaps; hard to measure deflection; onboarding takes long; compliance concerns.
- Goals: Dashboards for performance and content health; feedback loop to update articles; policy-compliant automations.

4) Daniel Lee – Integration Engineer
- Background: Owns API integrations; maintains identity and data pipelines.
- Pain points: Fragile connectors; schema drift; rate limits; security reviews.
- Goals: Stable, documented APIs/SDKs; observability; configuration-driven connectors; safe rollouts.

## 5. User Stories
US-001
- As a customer, I want to ask free-form questions and get answers with citations so that I can trust the information.
- Acceptance: Response includes at least one relevant citation with source title/URL and confidence score; groundedness >95%.

US-002
- As a customer, I want the chatbot to understand my product version and region so that I receive tailored instructions.
- Acceptance: Metadata filters (version, region, language) applied; retrieved contexts reflect correct filters.

US-003
- As a customer, I want to check my order status after signing in so that I can get personalized updates.
- Acceptance: Authenticated session; tool call to order API; masked PII; response latency <3s P95.

US-004
- As an agent, I want summarized conversation context and suggested replies so that I can respond faster.
- Acceptance: Summary <150 tokens; top-3 suggested replies with references; accuracy >85% on internal eval.

US-005
- As a manager, I want dashboards for deflection rate, CSAT, and content gaps so that I can improve operations.
- Acceptance: Weekly and monthly aggregates; drill-down by intent, language, product; export to CSV.

US-006
- As a knowledge manager, I want to sync content from Confluence and Zendesk with de-duplication and PII redaction so that the corpus remains clean and safe.
- Acceptance: Nightly sync; duplicate detection >95% precision; PII redaction recall >98%.

US-007
- As a compliance officer, I want audit logs of all tool calls and redaction events so that audits are streamlined.
- Acceptance: Immutable logs with user/session IDs, timestamps, action, resource, decision; retention 13 months.

US-008
- As a developer, I want a clear REST API for chat, feedback, and knowledge sync so that I can integrate easily.
- Acceptance: OpenAPI spec; example clients; rate limits; sandbox environment.

US-009
- As a customer, I want the assistant to abstain and escalate when uncertain so that I avoid incorrect guidance.
- Acceptance: Low confidence triggers clarification or escalation; escalation notes include context and top citations.

US-010
- As a global user, I want multilingual support so that I can chat in my preferred language.
- Acceptance: Language auto-detected; answer in same language; BLEU/COMET score >0.6 on test set; right-to-left support.

## 6. Functional Requirements
### 6.1 Core Features
FR-001 RAG Pipeline: Hybrid retrieval (BM25 + dense) with cross-encoder reranking and MMR for diversity.
FR-002 Knowledge Ingestion: Connectors for Zendesk, Salesforce, Confluence, SharePoint; dedup; PII redaction; semantic chunking; parent-child indexing.
FR-003 Domain Adaptation: Ontologies, controlled vocab, synonym maps, metadata filters (SKU, version, entitlement, region, language).
FR-004 Query Understanding: Intent detection, query rewriting/expansion, spelling correction, multilingual normalization.
FR-005 Grounded Answering: Structured answers with citations and confidence; JSON schema enforcement; abstain on low confidence.
FR-006 Conversation Orchestration: Memory with short-term summary; tool/function calling for account/order/ticket; safe-action policies.
FR-007 Authentication & Personalization: OAuth2/OIDC SSO; user profile and entitlements; region/language preferences.
FR-008 Feedback Loop: Thumbs up/down, reason codes; embed resolved tickets and successful answers to enrich corpus.
FR-009 Observability & Analytics: Retrieval recall@k, NDCG, groundedness; query analytics; A/B testing; semantic cache hit rate.
FR-010 Escalation: Seamless handoff to agents with context packet (chat transcript, intents, top-k passages, confidence).

### 6.2 Advanced Features
- Multilingual embeddings and generation with language-aware reranking.
- Image-in-text troubleshooting (optional): OCR of screenshots; captioning to extract error codes.
- Voice input (optional): ASR for voice-to-text; TTS for readout.
- Proactive guidance: Dynamic forms/flows for troubleshooting trees; multi-hop retrieval for step-by-step procedures.
- Agent co-pilot: Suggested macros, auto-summarization of long tickets, policy checks before sending.
- Semantic caching: Approximate match cache for repeated questions; streaming responses.
- Time-aware retrieval: Prefer fresher content; decay older items unless authoritative.

## 7. Non-Functional Requirements
### 7.1 Performance
- TTFB <2s P95; full response <5s P95; tool calls <1.5s P95.
- Retrieval+rerank <500ms P95 with ANN.
- Streaming enabled for partial tokens within 500ms.

### 7.2 Reliability
- 99.5% uptime monthly; zero data loss RPO; 15-min RTO for stateless services.
- Idempotent ingestion; exactly-once semantics for document versioning.

### 7.3 Usability
- Simple, accessible chat UI; clear citations; language persistence; dark mode.
- Agent console plugin integrates into existing CRM with minimal training.

### 7.4 Maintainability
- Modular microservices with clear contracts; infra as code; configuration-driven connectors; automated tests and linting.

## 8. Technical Requirements
### 8.1 Technical Stack
- Backend: Python 3.11+, FastAPI 0.115+, Uvicorn, Gunicorn
- Frontend: React 18+, TypeScript 5+, Next.js 14+, Chakra UI or MUI
- Vector DB: Qdrant 1.7+ or Milvus 2.4+
- Full-text: OpenSearch 2.13+ or Elasticsearch 8.13+
- Database: PostgreSQL 15+ (SQLAlchemy 2.x)
- Cache/Queue: Redis 7+, Kafka 3.6+
- ML: PyTorch 2.3+, Hugging Face Transformers 4.44+, Sentence-Transformers 2.7+, ONNX Runtime 1.18+
- Orchestration: LangChain 0.2+ or LlamaIndex 0.10+; Pydantic 2+
- Deployment: Docker 24+, Kubernetes 1.30+, Helm 3.15+
- Cloud: AWS (EKS, S3, SQS, Lambda, CloudWatch) or GCP/Azure equivalents
- Auth: OAuth2/OIDC (Auth0/Okta/Azure AD), JWT
- Monitoring: OpenTelemetry 1.27+, Prometheus, Grafana, Sentry
- Feature flags: OpenFeature 1.4+

### 8.2 AI/ML Components
- Embeddings: Multilingual model (e.g., sentence-transformers/all-MiniLM-L12-v2-multilingual or e5-multilingual-large) with 768–1024 dims; domain-tuned via continued pretraining if needed.
- Reranker: Cross-encoder (e.g., cross-encoder/ms-marco-MiniLM-L-6-v2 or monoT5) fine-tuned on domain data.
- LLM: Hosted or self-hosted foundation model (e.g., GPT-4.1/GPT-4o-mini, Llama 3.1 70B instruct, Mistral Large) with tool calling and JSON mode; instruction-tuned prompts with policy layers.
- Intent Classification: Lightweight transformer (DistilBERT) or zero-shot classifier; confidence calibration (temperature scaling).
- NER/PII Redaction: spaCy 3.x or Presidio; custom patterns for SKUs, emails, phone, addresses.
- OCR (optional): Tesseract 5+ or cloud OCR; image captioning with BLIP-2 or Florence-2 for error extraction.
- Evaluation: Ragas, RetrievaLLM, custom harness for recall@k, NDCG, groundedness, semantic similarity.

## 9. System Architecture
### 9.1 High-Level Architecture (ASCII diagram)
+-------------------+        +--------------------+        +-------------------+
|  Web/Mobile Chat  |<------>|  API Gateway       |<-----> |  Auth (OIDC)      |
|  React Widget     |        |  (FastAPI)         |        |  SSO/JWT          |
+---------+---------+        +----------+---------+        +---------+---------+
          |                             |                           |
          v                             v                           |
+---------+---------+        +----------+---------+        +---------v---------+
| Conversation Svc  |<------>| Orchestrator/Agent |<-----> |  Tooling/Functions|
| Memory, Sessions  |        | RAG, Tool Calling  |        |  (Order, Ticket,  |
+---------+---------+        +----------+---------+        |   Entitlement)    |
          |                             |                  +---------+---------+
          v                             v                            |
+---------+---------+        +----------+---------+                  |
| Retrieval Service |<------>|  Re-ranker         |                  |
| Hybrid: BM25+Vec  |        | Cross-Encoder/MMR  |                  |
+---------+---------+        +----------+---------+                  |
          |                             |                            |
          v                             v                            v
+---------+---------+        +----------+---------+        +---------+---------+
| Vector DB (Qdrant)|        | Search (OpenSearch)|        |  Postgres (Meta)  |
+---------+---------+        +----------+---------+        +---------+---------+
          |                             |                            |
          +--------------+--------------+----------------------------+
                         |
                         v
                +--------+--------+
                |  Object Store   |
                |  (Docs, Chunks) |
                +--------+--------+
                         |
                         v
                +--------+--------+
                | Ingestion/ETL   |
                | Connectors, PII |
                +-----------------+

### 9.2 Component Details
- Frontend Chat Widget: Embeddable React component; streaming; file/screenshot upload; localization.
- API Gateway: Rate limiting, authentication, routing to services; OpenAPI docs.
- Orchestrator/Agent: Implements retrieval, reranking, generation, tool calling, guardrails, confidence/abstention.
- Retrieval Service: Hybrid retrieval with reciprocal rank fusion; session-aware query expansion.
- Re-ranker: Cross-encoder scoring top-50, output top-5 with MMR.
- Conversation Service: Session store, semantic buffer, summarization for long threads.
- Tools/Functions: Order status, ticket creation, user profile/entitlement, knowledge article fetch; governed by policies.
- Ingestion/ETL: Source connectors; dedup; chunking; metadata enrichment; multilingual processing; re-embedding scheduler.
- Data Stores: Vector DB for embeddings; OpenSearch for BM25 and fielded filters; Postgres for metadata, sessions, configs; Object store for raw docs; Redis for semantic cache; Kafka for events.
- Observability: Metrics, logs, traces; analytics pipeline; dashboards.

### 9.3 Data Flow
1) Ingestion: Connectors pull content → PII redaction → chunking (heading-aware, sliding window) → embeddings computed → store chunks/metadata → index in vector DB and search engine.
2) Query: User message → language detect → intent classify → query rewrite/expansion → hybrid retrieval → rerank → assemble context (parent-child) → LLM generate with citations → confidence compute → respond/stream.
3) Tool Use: If intent requires action, call function (order/ticket) → redact → incorporate results → respond.
4) Feedback: User rates answer → log → update analytics → eligible data added to training buffers/eval sets after review.

## 10. Data Model
### 10.1 Entity Relationships
- User 1—N Session
- Session 1—N Message
- Document 1—N Chunk
- Chunk 1—1 Embedding
- Document N—M Source (if multi-source lineage)
- Session 1—N ToolCall
- User 1—N Feedback
- Product 1—N Entitlement
- Intent N—N Message
- Ticket 1—1 Session (escalation)

### 10.2 Database Schema (PostgreSQL)
- users(id, ext_id, email_hash, locale, region, created_at)
- sessions(id, user_id, channel, language, started_at, last_active_at, status)
- messages(id, session_id, role ENUM(user,assistant,system), text, lang, tokens, created_at)
- tool_calls(id, session_id, tool_name, request_json, response_json, status, latency_ms, created_at)
- documents(id, source, title, uri, version, language, region, created_at, updated_at, is_active)
- chunks(id, document_id, chunk_idx, text, start_offset, end_offset, parent_span, metadata JSONB)
- embeddings(id, chunk_id, model_name, vector VECTOR(768/1024), created_at)
- intents(id, name, confidence_threshold, created_at)
- message_intents(message_id, intent_id, confidence)
- feedback(id, session_id, message_id, rating INT, reason ENUM(accuracy,helpfulness,tone,other), comment, created_at)
- tickets(id, session_id, external_id, status, priority, created_at, updated_at)
- products(id, sku, name, version, attributes JSONB)
- entitlements(id, user_id, product_id, start_date, end_date, tier)
- configs(id, key, value JSONB, updated_at)
- audits(id, actor, action, resource, decision, redactions JSONB, timestamp)

Indexes: GIN on metadata, trigram on titles, btree on timestamps, composite on (source, version, is_active).

### 10.3 Data Flow Diagrams (ASCII)
[Ingestion]
Sources -> ETL -> Redaction -> Chunking -> Embed -> Index
        +--------> Postgres meta + Object store

[Query]
User -> API -> Orchestrator -> Hybrid Retrieval -> Re-ranker -> Context Builder -> LLM -> Response (+Citations) -> Feedback

[Tooling]
Orchestrator -> Policy Check -> Function Call -> Mask PII -> Merge -> Respond

### 10.4 Input Data & Dataset Requirements
- Sources: Zendesk articles, Salesforce knowledge, Confluence spaces, SharePoint libraries, PDFs/HTML/Markdown, historical tickets (sanitized).
- Formats: HTML, Markdown, PDF, DOCX, CSV; API-based content via connectors.
- Chunking: 300–800 tokens with heading-aware splits; sliding window 15–20% overlap; parent span 1–2k tokens.
- Metadata: title, author, created/updated, language, region, SKU/product, version, entitlement tier, tags.
- Evaluation Sets: 1–2k gold Q/A pairs per domain; multilingual coverage; policy-sensitive queries; adversarial hallucination probes.

## 11. API Specifications
### 11.1 REST Endpoints
- POST /v1/chat
  - Body: { session_id?, message, locale?, context?, channel?, streaming? }
  - Returns: answer, citations[], confidence, actions[], usage, streaming_url?
- POST /v1/sessions
  - Body: { user_token?, locale?, channel }
  - Returns: { session_id }
- GET /v1/sessions/{id}
- POST /v1/feedback
  - Body: { session_id, message_id, rating, reason?, comment? }
- POST /v1/ingest/run
  - Body: { sources:[], full_refresh?:bool }
- POST /v1/ingest/document
  - Body: { uri, source, language?, region?, metadata? }
- GET /v1/search
  - Query: q, k?, language?, filters?
  - Returns: passages with scores and metadata
- POST /v1/tools/order_status
  - Body: { order_id, last4_email? }
- POST /v1/tools/create_ticket
  - Body: { session_id, severity, summary, details }
- GET /v1/analytics/metrics
  - Query: from, to, dimension?
- GET /v1/config
- POST /v1/config

### 11.2 Request/Response Examples
Request:
POST /v1/chat
{
  "session_id": "sess_123",
  "message": "How do I reset my Pro 2.1 device in Germany?",
  "locale": "en-DE",
  "streaming": true
}

Response:
{
  "answer": "Here are the steps for Pro 2.1 in Germany: ...",
  "citations": [
    {"title": "Reset Guide Pro 2.1 (EU)", "uri": "https://kb/reset-pro-21-eu", "span": "Steps 1-3", "score": 0.84}
  ],
  "confidence": 0.78,
  "actions": [],
  "usage": {"prompt_tokens": 842, "completion_tokens": 126, "latency_ms": 1780}
}

Search example:
GET /v1/search?q=refund%20policy&k=5&language=en&filters=region:EU;version:latest

### 11.3 Authentication
- OAuth2/OIDC with JWT bearer tokens.
- Scopes: chat:read, chat:write, tools:execute, ingest:write, analytics:read, config:write.
- CSRF protection for web; PKCE for public clients.
- mTLS between internal services.

Code snippet (FastAPI):
from fastapi import FastAPI, Depends
from pydantic import BaseModel

class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str
    locale: str | None = "en"
    streaming: bool | None = False

app = FastAPI()

@app.post("/v1/chat")
async def chat(req: ChatRequest, user=Depends(authn)):
    resp = await orchestrator.handle(req, user)
    return resp

## 12. UI/UX Requirements
### 12.1 User Interface
- Embeddable chat widget: header brand, input box with language autodetect, attach file/screenshot, quick-reply chips, toggle citations.
- Agent assist panel: conversation summary, suggested replies, top sources, escalation button.
- Admin console: ingestion status, metrics, feedback queue, content gap insights.

### 12.2 User Experience
- Streaming responses with visible citations and copy-able links.
- Clarifying questions when confidence low; explicit escalate option.
- Personalization indicators (e.g., “Answer tailored to Pro 2.1, EU”).
- Undo/redo after agent suggested reply insertion.
- Offline/limited network handling with retries.

### 12.3 Accessibility
- WCAG 2.2 AA compliance; ARIA labels; keyboard navigation.
- High-contrast mode; adjustable font sizes; screen reader tested.
- Right-to-left layout for relevant locales.

## 13. Security Requirements
### 13.1 Authentication
- OIDC SSO; MFA optional; session expiry configurable; JWT rotation and short TTLs.

### 13.2 Authorization
- Role-based access control (RBAC): roles for customer, agent, manager, admin, service.
- Attribute-based controls (region, tenant, entitlement tier).
- Fine-grained scopes for tool execution.

### 13.3 Data Protection
- TLS 1.3 in transit; AES-256 at rest.
- PII masking/redaction pre-index; tokenization of sensitive fields.
- Secrets management via AWS KMS/HashiCorp Vault; key rotation 90 days.
- Data retention policies by tenant; right-to-erasure workflows.

### 13.4 Compliance
- SOC 2 Type II, ISO 27001 alignment.
- GDPR/CCPA support: consent logging, data subject access requests.
- Audit logs for all admin actions and tool calls with immutable storage.

## 14. Performance Requirements
### 14.1 Response Times
- TTFB <2s P95; full response <5s P95.
- Retrieval+rerank <500ms P95; cache hits <150ms.

### 14.2 Throughput
- 200 RPS sustained; burst 500 RPS for 5 minutes with autoscaling.
- Background ingestion up to 1M chunks/day.

### 14.3 Resource Usage
- GPU-backed generation pools for heavy models; CPU-only path for small models/fallback.
- Memory per orchestrator pod <4GB P95; vector queries <50ms median on warmed index.

## 15. Scalability Requirements
### 15.1 Horizontal Scaling
- Kubernetes HPA based on CPU, GPU utilization, and tokens/sec.
- Separate autoscaling for retrieval, reranker, and generation tiers.

### 15.2 Vertical Scaling
- Scale vector DB nodes with additional RAM/NVMe; shard by tenant or product family.
- Model serving with tensor/kv cache optimization and quantization for smaller instances.

### 15.3 Load Handling
- Semantic caching; request coalescing; circuit breakers; rate limiting per tenant.
- Priority queues for authenticated users and SLA-bound channels.

## 16. Testing Strategy
### 16.1 Unit Testing
- 85%+ coverage for orchestration, parsing, filters, policies.
- Redaction and metadata handling tests with edge cases.

### 16.2 Integration Testing
- Connector end-to-end sync with sandbox tenants.
- Hybrid retrieval correctness against known corpora.
- Tool calls with mocked backends and negative tests.

### 16.3 Performance Testing
- Load tests (k6/Locust): latency/throughput under RPS ramps.
- Soak tests 24–72 hours; memory leak detection.
- A/B latency tests for rerankers and caching policies.

### 16.4 Security Testing
- Static analysis (Bandit), dependency scanning (Snyk).
- DAST; pen-tests; fuzzing for JSON schema and tool inputs.
- Privacy tests for PII leakage and redaction failures.

## 17. Deployment Strategy
### 17.1 Deployment Pipeline
- GitHub Actions/CircleCI: lint, unit tests, build images, integration tests, security scans, deploy to staging, manual approval to prod.
- Infrastructure as Code (Terraform + Helm).

### 17.2 Environments
- Dev (shared), Staging (tenant-like), Production (multi-tenant).
- Feature flags for gradual enablement; canary subsets.

### 17.3 Rollout Plan
- Phase 1: Internal users (10% traffic).
- Phase 2: Beta tenants (25–50% traffic).
- Phase 3: GA (100%); A/B test rerankers/models.

### 17.4 Rollback Procedures
- Blue/green deployments; instant traffic switch.
- DB migrations backward-compatible; feature flags to disable new features.
- Data snapshots before schema changes.

## 18. Monitoring & Observability
### 18.1 Metrics
- Latency: TTFB, full response, retrieval, rerank, generation, tool call.
- Quality: groundedness, citation coverage, exact/semantic match, recall@k, NDCG, cache hit rate.
- Business: deflection, CSAT proxy (feedback), AHT impact, escalation rate.
- Ingestion: docs processed, dedup rate, redaction recall.

### 18.2 Logging
- Structured JSON logs with trace IDs; PII-scrubbed.
- Separate audit logs for admin/tool actions.

### 18.3 Alerting
- SLO breaches (latency, error rate, uptime).
- Index freshness lag; connector failures; embedding drift.
- Security anomalies: unusual tool usage patterns.

### 18.4 Dashboards
- Grafana: service performance; cost per 1k tokens; GPU/CPU utilization.
- Product analytics: intents, satisfaction, content gaps.
- Ingestion health: per-source success/latency.

## 19. Risk Assessment
### 19.1 Technical Risks
- Hallucinations despite RAG.
- Embedding/model drift reducing retrieval quality.
- Latency spikes during traffic bursts.
- Connector rate limits and schema changes.

### 19.2 Business Risks
- User mistrust from incorrect answers.
- Content ownership/licensing issues.
- Cost overruns from model usage.
- Regulatory penalties for PII mishandling.

### 19.3 Mitigation Strategies
- Strict groundedness checks; abstention thresholds; clarify questions.
- Periodic re-embedding; drift monitoring; synthetic probes.
- Semantic caching; autoscaling; rate limiting; circuit breakers.
- Legal/content review workflows; PII redaction; DLP; privacy training.

## 20. Timeline & Milestones
### 20.1 Phase breakdown
- Phase 0 (2 weeks): Discovery, requirements, data audit, security review.
- Phase 1 (3 weeks): Ingestion MVP (Zendesk/Confluence), PII redaction, chunking/indexing.
- Phase 2 (6 weeks): RAG baseline, hybrid retrieval, reranking, chat API/UI, citations, feedback.
- Phase 3 (4 weeks): Tooling (order status, tickets), auth/entitlements, agent assist panel.
- Phase 4 (4 weeks): Multilingual, analytics dashboards, A/B framework, HPA scaling, SLOs.
- Phase 5 (2 weeks): Hardening, perf/security tests, documentation, training.
Total: ~21 weeks to GA.

Estimated Team: 6–8 FTE (2 BE, 1–2 ML, 1 FE, 1 Data/ETL, 1 DevOps, 0.5–1 PM/Design)
Estimated Infra Cost at GA: $12k–$30k/month (varies with traffic, model choice, GPUs)

### 20.2 Key Milestones
- M1 (Week 3): Ingestion pipeline live; first index built.
- M2 (Week 9): RAG chat MVP with citations; internal beta.
- M3 (Week 13): Tooling + auth integrated; pilot with selected tenants.
- M4 (Week 17): Multilingual + dashboards; performance SLOs met.
- M5 (Week 21): GA release; 99.5% uptime target; support playbooks ready.

## 21. Success Metrics & KPIs
### 21.1 Measurable targets
- Quality: >90% exact/semantic match on FAQ gold set; >95% groundedness; citation coverage >98%.
- Performance: TTFB <2s P95; full response <5s P95; retrieval+rerank <500ms P95.
- Reliability: 99.5% uptime; error rate <0.5% P95.
- Business: ≥35% deflection in 6 months; +15% CSAT; -25% AHT; ≥20% semantic cache hit rate.
- Safety: PII leakage rate 0; redaction recall ≥98%; policy violations <0.1%.

## 22. Appendices & Glossary
### 22.1 Technical Background
- Retrieval-Augmented Generation: Combining hybrid retrieval (BM25 + dense ANN) with LLM generation and reranking improves precision and groundedness over pure generation.
- Parent-child and hierarchical indexing: Retrieve granular chunks with parent spans to preserve context and reduce hallucinations.
- Confidence and abstention: Calibrated thresholds ensure the assistant escalates or asks clarifying questions when uncertain.
- Domain adaptation: Ontologies, synonyms, and metadata filters focus retrieval/generation on relevant subsets of content.

### 22.2 References
- Karpukhin et al., Dense Passage Retrieval
- MS MARCO reranking datasets
- Sentence-Transformers documentation
- Ragas: Evaluation for RAG
- OpenTelemetry specification
- OWASP ASVS and Top 10

### 22.3 Glossary
- RAG: Retrieval-Augmented Generation, combines information retrieval with text generation.
- BM25: A scoring function for ranking documents based on term frequency and document length.
- ANN: Approximate Nearest Neighbor search for fast vector similarity.
- Cross-encoder: Model that jointly encodes query and document to produce a relevance score.
- MMR: Maximal Marginal Relevance, balances relevance and diversity in selections.
- Groundedness: Degree to which an answer is supported by retrieved sources.
- Entitlement: User’s access rights to products/services or content tiers.
- PII: Personally identifiable information requiring special handling.
- TTFB: Time to first byte, measures responsiveness of the first streamed token.
- AHT: Average handle time for human agents.

Repository Structure
- root/
  - README.md
  - configs/
    - app.yaml
    - retrieval.yaml
    - redaction.yaml
  - src/
    - api/
      - main.py
      - routes/
        - chat.py
        - tools.py
        - ingest.py
        - analytics.py
    - core/
      - orchestrator.py
      - retriever.py
      - reranker.py
      - generator.py
      - memory.py
      - guardrails.py
    - ml/
      - embeddings.py
      - intent.py
      - ner_redaction.py
      - eval/
        - harness.py
        - metrics.py
    - connectors/
      - zendesk.py
      - salesforce.py
      - confluence.py
      - sharepoint.py
    - db/
      - models.py
      - repos.py
    - tools/
      - order_status.py
      - ticketing.py
      - user_profile.py
    - utils/
      - logging.py
      - caching.py
      - auth.py
  - tests/
    - unit/
    - integration/
    - performance/
    - security/
  - notebooks/
    - retrieval_evals.ipynb
    - reranker_tuning.ipynb
    - redaction_quality.ipynb
  - data/
    - raw/
    - processed/
    - eval/
  - scripts/
    - ingest_run.py
    - reembed.py
    - build_index.py
  - infra/
    - helm/
    - terraform/

Configuration Samples
configs/retrieval.yaml
retrieval:
  hybrid: true
  dense:
    model: "intfloat/e5-multilingual-large"
    top_k: 50
  sparse:
    engine: "opensearch"
    bm25:
      k1: 1.2
      b: 0.75
    top_k: 50
  fusion: "rrf"
  reranker:
    model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: 5
    mmr_lambda: 0.3
  filters:
    metadata: ["language", "region", "sku", "version", "entitlement"]
  parent_child:
    parent_span_tokens: 1500
    child_chunk_tokens: 500
  cache:
    semantic_cache: true
    threshold: 0.92

Example Retrieval Pipeline (pseudo-Python)
def answer(query, session):
    lang = detect_language(query)
    intent, conf = intent_model.predict(query)
    q_norm = normalize_query(query, lang)
    q_expanded = expand_query(q_norm, domain_ontology)
    sparse = bm25.search(q_expanded, k=50, filters=session.filters)
    dense = vectors.search(embed(q_expanded), k=50, filters=session.filters)
    fused = rrf_fuse(sparse, dense)
    reranked = cross_encoder.score(query, fused)[:5]
    context = build_parent_child_context(reranked)
    result = llm.generate(prompt=compose_prompt(query, context, policy), tools=tools_for(intent))
    if result.confidence < CONF_THRESHOLD:
        return ask_clarifying_or_escalate(query, context)
    return add_citations_and_log(result)

API Client Snippet (TypeScript)
const res = await fetch("/v1/chat", {
  method: "POST",
  headers: { "Authorization": `Bearer ${token}`, "Content-Type": "application/json" },
  body: JSON.stringify({ session_id, message, locale: "en-US", streaming: true })
})

Latencies and SLOs
- Retrieval+Rerank: <500ms P95
- TTFB: <2s P95
- Full Response: <5s P95
- Uptime: ≥99.5%

Cost Considerations (monthly, typical mid-scale)
- Compute (K8s, autoscaled): $6k–$12k
- Managed Vector/Search/DB: $2k–$6k
- Model serving/API usage: $3k–$10k
- Monitoring/Logs: $1k–$2k

End of PRD.