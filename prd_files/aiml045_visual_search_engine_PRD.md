# Product Requirements Document (PRD) / # `aiml045_visual_search_engine`

Project ID: aiml045  
Category: AI/ML - Computer Vision, Information Retrieval, Multimodal  
Status: Draft for Review  
Version: 1.0  
Last Updated: 2025-11-25

## 1. Overview
### 1.1 Executive Summary
aiml045_visual_search_engine is a cloud-native, multimodal visual search solution that enables users to search, discover, and organize images using either images or text queries. It leverages state-of-the-art multimodal embeddings (e.g., CLIP/OpenCLIP, SigLIP) and approximate nearest neighbor (ANN) indexing (e.g., FAISS HNSW/IVF-PQ, optional Qdrant/Milvus) for scalable, low-latency retrieval. The product targets e-commerce, media libraries, digital asset management, and knowledge platforms, providing object-level search, hybrid sparse+dense retrieval, safety filtering, and re-ranking for high-quality results.

### 1.2 Document Purpose
This PRD specifies requirements, architecture, algorithms, data model, APIs, UI/UX, security, performance, scalability, testing, deployment, monitoring, risk, and timelines necessary to build and operate the visual search engine from MVP through GA.

### 1.3 Product Vision
Deliver a best-in-class visual search experience that feels instantaneous, accurate, and intuitive across modalities—text-to-image and image-to-image—while being flexible, secure, and cost-efficient. The system will continually learn from user interactions and support enterprise-grade multi-tenancy, observability, and data governance.

## 2. Problem Statement
### 2.1 Current Challenges
- Traditional keyword search fails on visual similarity and aesthetic queries.
- Existing solutions lack multimodal understanding (text+image).
- Scaling to millions/billions of images with low latency is hard.
- Lack of object-level search limits fine-grained discovery in complex scenes.
- Freshness vs. index rebuilds is poorly managed; near-real-time ingestion is missing.
- Safety/content moderation and access controls are inconsistent.

### 2.2 Impact Analysis
- Lost conversions in e-commerce due to poor discovery.
- Increased manual tagging costs.
- Inefficient media management; duplicated assets persist.
- User frustration and churn due to slow or irrelevant results.
- Compliance and brand risks from unsafe or restricted content surfacing.

### 2.3 Opportunity
- Provide a plug-and-play visual search platform with API-first design.
- Boost engagement and revenue via better retrieval quality and speed.
- Reduce manual cataloging by automated captions and metadata enrichment.
- Differentiate with object-level search, hybrid retrieval, and adaptive re-ranking.
- Offer enterprise features: multi-tenant isolation, RBAC, observability, and SLAs.

## 3. Goals and Objectives
### 3.1 Primary Goals
- Enable fast, accurate image retrieval for text and image queries.
- Support object-level search within images.
- Provide production-grade APIs with multi-tenant metadata filtering.
- Achieve <500 ms P95 search latency at 1M–100M images scale.

### 3.2 Business Objectives
- Increase customer engagement (CTR on search results +20%).
- Reduce operational tagging costs by 30% via auto-captioning and deduplication.
- Support 3+ enterprise tenants by GA with 99.5% uptime SLA.
- Achieve unit-economics optimized cost <$1 per 10k searches at scale.

### 3.3 Success Metrics
- Offline: Recall@10 ≥ 0.90 on curated benchmark, mAP@10 ≥ 0.60, NDCG@10 ≥ 0.75.
- Online: Success@k (user clicks/saves) ≥ 0.45 at k=10; CTR uplift ≥ 15% vs baseline.
- Latency: P50 ≤ 250 ms, P95 ≤ 500 ms, P99 ≤ 900 ms (1M items, 1 shard).
- Uptime: ≥ 99.5% monthly.
- Safety: ≥ 98% ROC-AUC on safety classifier; <0.5% unsafe false negatives.

## 4. Target Users/Audience
### 4.1 Primary Users
- E-commerce product teams seeking similar item search and style discovery.
- Digital asset managers and content librarians.
- Developers integrating search via APIs into apps and websites.

### 4.2 Secondary Users
- Data scientists optimizing embeddings and re-rankers.
- Marketing teams curating brand-consistent visuals.
- Customer support teams handling content governance.

### 4.3 User Personas
1) Alex Chen, Product Manager (E-commerce)
- Background: 7 years in marketplace product; drives discovery and conversion.
- Pain points: Keyword search misses visually similar items; high bounce; manual curation time.
- Goals: Improve “find similar” conversion by 20%; robust filters; reliable SLAs.

2) Priya Iyer, ML Engineer (Platform)
- Background: MS in ML; maintains retrieval services; familiar with PyTorch and FAISS.
- Pain points: Maintaining indices and drift; slow rebuilds; lacking observability; model A/B friction.
- Goals: Seamless pipelines, clear metrics, quick rollback, re-ranker experimentation.

3) Maria Gomez, Digital Asset Manager (Media)
- Background: Oversees 5M+ images across brands; ensures tagging and compliance.
- Pain points: Duplicates, poor metadata, unsafe content leaks, limited object-level search.
- Goals: Deduplication, auto-captioning, safety filters, region-based tagging/search.

4) Dan Park, Frontend Developer
- Background: React developer integrating visual search widget.
- Pain points: Complex APIs, CORS/auth issues, inconsistent results under load.
- Goals: Simple SDK, predictable response structure, fast CDN delivery.

## 5. User Stories
US-001: As a user, I want to search by image (drag-and-drop) so that I can find visually similar items.  
Acceptance: Given an uploaded image, when I query, then the system returns ≥10 similar images within 500 ms P95 with Recall@10 ≥ 0.9 on curated set.

US-002: As a user, I want to search by text so that I can find images matching my description.  
Acceptance: Given a text query, results appear within 500 ms P95 with NDCG@10 ≥ 0.75 and safety filters applied.

US-003: As an admin, I want to ingest new images in near-real-time so that they are searchable within minutes.  
Acceptance: New assets appear in search within ≤ 5 minutes, with delta index merge in ≤ 30 minutes.

US-004: As a user, I want object-level search so that I can find similar objects inside scenes.  
Acceptance: Given a region selection or object name, return top-10 similar objects with mAP@10 ≥ 0.55.

US-005: As a developer, I want hybrid retrieval (metadata filters + dense search) so that I can constrain results.  
Acceptance: Provide metadata filter syntax; queries respect tenant/category/brand filters 100% of the time.

US-006: As a user, I want diversity in results so that I don’t see many near-duplicates.  
Acceptance: MMR/diversity ensures max 2 near-duplicates in top-10; dedup by pHash threshold.

US-007: As a security officer, I need content moderation so that unsafe content is filtered.  
Acceptance: NSFW classifier enabled by policy; unsafe content excluded from results with <0.5% FN rate.

US-008: As an ML engineer, I want A/B testing of embedding models so that I can optimize retrieval.  
Acceptance: Traffic split toggled via config; evaluation dashboard shows recall/latency deltas.

US-009: As a tenant admin, I want RBAC so that only authorized users can manage indices.  
Acceptance: Roles (viewer, editor, admin) enforced; audit logs for create/update/delete.

US-010: As a user, I want to provide feedback (thumbs up/down) so that the system can improve.  
Acceptance: Feedback captured with query/session IDs; used for re-ranker training within 2 weeks.

## 6. Functional Requirements
### 6.1 Core Features
FR-001: Image-to-image search via embeddings and ANN.  
FR-002: Text-to-image search via shared embedding space.  
FR-003: Hybrid retrieval: dense vectors + BM25 on captions/metadata.  
FR-004: Re-ranking stage using cross-encoder or ITM model.  
FR-005: Object-level search with region embeddings and bounding boxes.  
FR-006: Metadata filters (tenant, brand, category, locale, time).  
FR-007: Ingestion pipeline: batch ETL + near-real-time delta upserts.  
FR-008: Deduplication via perceptual hashing; configurable thresholds.  
FR-009: Safety/content filtering.  
FR-010: Multi-tenant isolation and index partitioning.  
FR-011: Feedback capture and analytics.  
FR-012: Admin console for index status, metrics, and A/B control.  
FR-013: API and SDKs (JavaScript/TypeScript, Python).

### 6.2 Advanced Features
- AF-001: Query expansion and pseudo-relevance feedback.
- AF-002: Learned fusion weights for hybrid retrieval per tenant.
- AF-003: Domain adaptation/fine-tuning for specialized catalogs.
- AF-004: Active learning workflow with hard negative mining.
- AF-005: On-the-fly caption generation for uncaptioned images.
- AF-006: Multilingual text queries with multilingual CLIP/SigLIP.
- AF-007: Vector compression (OPQ + PQ) with recall-aware autoconfig.
- AF-008: Diversified bundles (MMR) and fairness constraints.

## 7. Non-Functional Requirements
### 7.1 Performance
- P50 ≤ 250 ms, P95 ≤ 500 ms, P99 ≤ 900 ms (1M images/shard, 768–1024-dim).
- Throughput ≥ 300 QPS/shard (CPU ANN) or ≥ 800 QPS/shard (GPU ANN).
- Index build: 1M images in ≤ 90 min with IVF-PQ; HNSW build in ≤ 120 min.

### 7.2 Reliability
- Uptime ≥ 99.5% monthly.
- Zero data loss RPO; RTO ≤ 30 min.
- Idempotent upserts; at-least-once ingestion with dedup guards.

### 7.3 Usability
- Drag-and-drop, paste-URL, responsive grid, keyboard navigation.
- Clear confidence and facets; one-click filters; dark mode.

### 7.4 Maintainability
- Modular services; typed interfaces; 80%+ unit test coverage.
- Infrastructure as code (Terraform); reproducible builds.
- Backward-compatible APIs for 12 months.

## 8. Technical Requirements
### 8.1 Technical Stack
- Backend: Python 3.11+, FastAPI 0.111+, Uvicorn 0.30+, Gunicorn 22+
- ML: PyTorch 2.4+, Hugging Face Transformers 4.44+, TorchVision 0.19+
- Vector Index: FAISS 1.8.0 (CPU/GPU), optional Qdrant 1.8+/Milvus 2.4+
- Search: OpenSearch/Elasticsearch 8.x (BM25, filters)
- Data: PostgreSQL 15+, Redis 7+ (cache), Kafka 3.7+ (events)
- Frontend: React 18+, Next.js 14+, TypeScript 5.x, TailwindCSS 3.x
- Orchestration: Docker, Kubernetes 1.29+, Helm, ArgoCD/GitHub Actions
- Cloud: AWS (S3, EKS, EC2, RDS), or GCP equivalents
- Observability: Prometheus, Grafana, OpenTelemetry, Loki/ELK
- Auth: OAuth2/OIDC (Auth0/AWS Cognito/Keycloak), JWT

### 8.2 AI/ML Components
- Embeddings: OpenCLIP ViT-L/14 336px (default), SigLIP large (optional), DINOv2 for image-image robustness.
- Re-ranker: BLIP-2 ITM (Image-Text Matching) or BLIP-ITM base; for image-image, CLIP high-res similarity or learned metric head.
- Object detector/region proposals: GroundingDINO/DETR; per-region embeddings.
- Captioning: BLIP/BLIP-2 caption model (for BM25).
- Safety: LAION NSFW classifier or OpenCLIP-based safety head; configurable.
- Losses (for fine-tuning): InfoNCE/contrastive; hard negative mining jobs.
- ANN: HNSW/IVF-PQ/IVF-OPQ-PQ; cosine similarity; L2-normalized embeddings.

## 9. System Architecture
### 9.1 High-Level Architecture (ASCII)
+-------------------+       +-----------------------+       +-----------------------+
|   Client (Web/M)  | <---> | API Gateway / Load    | <---> |  Auth (OIDC/JWT)      |
| React widget/SDK  |       | Balancer              |       |                       |
+---------+---------+       +----------+------------+       +-----------+-----------+
          |                               |                                |
          v                               v                                v
+---------+---------+       +------------+------------+       +-----------+-----------+
|   Search API      | <---- |   Re-ranker Service    | <---- |   Embedding Service   |
| FastAPI           |       | BLIP-ITM, MMR, fusion  |       | CLIP/SigLIP encoders  |
+----+--------------+       +------------+------------+       +-----------+-----------+
     |                                   |                                |
     v                                   v                                v
+----+--------------+       +------------+------------+       +-----------+-----------+
| Vector Store      | <---- |  Hybrid Search (BM25)   |       |  Metadata DB (PG)     |
| FAISS/Qdrant      |       | OpenSearch/Elasticsearch|       | Assets, tenants, ACLs |
+----+--------------+       +------------+------------+       +-----------+-----------+
     |                                   ^
     v                                   |
+----+--------------+                    |
|  Object Storage   |   +----------------+-----------+
|  S3/GCS (images)  |   |  Ingestion/ETL Pipeline   |
+----+--------------+   |  Kafka + Workers + Delta  |
     ^                  |  Index build & merge      |
     |                  +---------------------------+
     |                                 |
     +---------------------------+-----+
                                 v
                      +----------+-----------+
                      | Observability Stack |
                      | Metrics, Logs, APM  |
                      +---------------------+

### 9.2 Component Details
- Search API: Orchestrates retrieval; handles auth, filters, caching, and response shaping.
- Embedding Service: Stateless GPU service for online embedding of queries; batch for ingestion.
- Vector Store: ANN index per tenant/collection; supports sharding and delta indices.
- Re-ranker: Cross-encoder (image-text matching) and DINO/CLIP refinement; implements MMR/diversity and hybrid fusion.
- Hybrid Search: BM25 over captions/metadata; fuses with dense scores (e.g., weighted sum).
- Ingestion/ETL: Fetches images, computes pHash, captions, embeddings; trains IVF quantizers; builds and merges indices.
- Metadata DB: Stores asset metadata, regions, access controls, feedback.
- Object Storage: Original images and thumbnails.
- Observability: OpenTelemetry traces; Prometheus metrics for latency, recall proxy, index health.

### 9.3 Data Flow
1) Ingest: ETL pulls image -> preprocess -> pHash -> caption -> embed -> ann train (if needed) -> build index -> persist vectors+payload -> publish index manifest -> deploy.
2) Query (text): Text -> embed -> ANN top-N -> BM25 candidates -> fusion -> re-rank (cross-encoder) -> MMR -> safety -> filters -> response.
3) Query (image): Image -> preprocess -> embed -> ANN -> re-rank (image-image) -> MMR -> safety -> filters -> response.
4) Object search: Region proposals -> per-region embeddings at ingest; query uses region embedding; similar object retrieval via region index.
5) Feedback: User feedback stored; periodic jobs mine hard negatives and update re-ranker.

## 10. Data Model
### 10.1 Entity Relationships
- Tenant 1—N Collections
- Collection 1—N Assets (ImageAsset)
- Asset 1—N Regions (object bounding boxes)
- Asset 1—1 Embedding (global); Region 1—1 Embedding (local)
- Asset N—M Labels/Tags
- Query 1—N Results (SearchResult)
- User N—M Queries; User role in Tenant (RBAC)
- Feedback linked to Query and Result

### 10.2 Database Schema (PostgreSQL)
- tenants(id PK, name, created_at, plan, settings_json)
- users(id PK, email, name, tenant_id FK, role ENUM[viewer, editor, admin], created_at)
- collections(id PK, tenant_id FK, name, description, created_at)
- assets(id PK, collection_id FK, external_id, uri, mime_type, width, height, phash, caption, created_at, updated_at, safe ENUM[safe, flagged, unsafe], metadata JSONB, embedding_version)
- regions(id PK, asset_id FK, bbox_x, bbox_y, bbox_w, bbox_h, label, created_at, embedding_version)
- embeddings(id PK, ref_type ENUM[asset, region], ref_id FK, dim, norm, vector BLOB/bytea (optional if stored externally), index_id, created_at)
- indices(id PK, tenant_id, collection_id, type ENUM[faiss, qdrant], shard_id, version, status ENUM[ready, building, merging], params JSONB, created_at)
- queries(id PK, user_id, tenant_id, collection_id, query_type ENUM[text, image], query_text, query_image_uri, created_at, latency_ms)
- results(id PK, query_id FK, rank, asset_id, region_id, score, rerank_score, filters_json)
- feedback(id PK, query_id, asset_id, user_id, label ENUM[positive, negative], created_at)
- audit_logs(id PK, actor_id, action, entity, entity_id, diff JSONB, created_at)

Vectors may be stored in FAISS/Qdrant with payload including asset_id, region_id, tenant, collection, labels, timestamp, safe flag.

### 10.3 Data Flow Diagrams (ASCII - Ingest)
[Source] -> [ETL Worker] -> [Captioner] -> [Embedder] -> [ANN Trainer] -> [Index Builder] -> [Index Registry] -> [Serving Nodes]

### 10.4 Input Data & Dataset Requirements
- Image formats: JPEG, PNG, WebP; max 20 MB; min 64x64 px (recommended ≥224 px).
- Metadata: optional title, tags, category, brand, locale, timestamp.
- Ground truth for evaluation: Query->relevant set annotations or click logs.
- Datasets for benchmarking/fine-tuning: LAION subsets, COCO, Fashion/Products (as licensed), internal curated sets.
- Safety datasets for calibration.

## 11. API Specifications
### 11.1 REST Endpoints
- POST /v1/search/text
- POST /v1/search/image
- POST /v1/index/upsert
- POST /v1/index/delete
- GET /v1/index/status
- POST /v1/feedback
- GET /v1/health
- POST /v1/auth/token (if first-party)
- GET /v1/admin/metrics (admin scope)

### 11.2 Request/Response Examples
POST /v1/search/text
Request:
{
  "tenant_id": "t_123",
  "collection_id": "c_catalog",
  "query": "red floral summer dress",
  "top_k": 20,
  "filters": {"brand": ["Acme"], "safe": ["safe"]},
  "enable_rerank": true,
  "hybrid": {"bm25_weight": 0.3, "dense_weight": 0.7}
}
Response:
{
  "results": [
    {"asset_id":"a1","uri":"https://cdn/.../a1.jpg","score":0.82,"rerank_score":0.88,"metadata":{"brand":"Acme","category":"dress"}},
    {"asset_id":"a2","uri":"https://cdn/.../a2.jpg","score":0.80}
  ],
  "latency_ms": 312,
  "debug": {"stage_times_ms":{"embed":12,"ann":95,"bm25":60,"rerank":120,"mmr":15}}
}

POST /v1/search/image
Request (multipart/form-data):
- tenant_id: t_123
- collection_id: c_catalog
- image: file
- top_k: 20
- filters: {"category":["sneakers"]}
Response similar to text search.

POST /v1/index/upsert
{
  "tenant_id":"t_123",
  "collection_id":"c_catalog",
  "assets":[
    {"external_id":"ext-001","uri":"https://s3/.../img1.jpg","metadata":{"brand":"Acme","category":"dress"}}
  ],
  "options":{"async": true}
}
Response:
{"job_id":"job_789","status":"queued"}

POST /v1/feedback
{
  "tenant_id":"t_123",
  "query_id":"q_456",
  "asset_id":"a1",
  "label":"positive"
}
Response: {"status":"ok"}

### 11.3 Authentication
- OAuth2/OIDC with JWT Bearer tokens.
- Scopes: search:read, index:write, admin:read.
- Tenant isolation enforced by token claims and filters.

Code snippet (FastAPI search stub):
from fastapi import FastAPI, UploadFile, File, Depends
from pydantic import BaseModel
app = FastAPI()

class TextSearchReq(BaseModel):
    tenant_id: str
    collection_id: str
    query: str
    top_k: int = 20
    filters: dict | None = None
    enable_rerank: bool = True
    hybrid: dict | None = None

@app.post("/v1/search/text")
async def search_text(req: TextSearchReq, user=Depends(auth_dep)):
    emb = embed_text(req.query)  # CLIP/SigLIP
    candidates = ann_query(req.tenant_id, req.collection_id, emb, k=200, filters=req.filters)
    if req.hybrid:
        bm25 = bm25_query(req.query, filters=req.filters, k=200)
        candidates = fuse_dense_sparse(candidates, bm25, req.hybrid)
    ranked = rerank_if_enabled(candidates, enable=req.enable_rerank)
    return format_response(ranked, timings=True)

## 12. UI/UX Requirements
### 12.1 User Interface
- Search bar with text input; microphone optional for voice-to-text.
- Drag-and-drop image area; paste URL.
- Results: masonry/grid with lazy loading; hover actions (save, similar, details).
- Filters panel: brand, category, tags, date, safety.
- Object search mode: draw bounding box or auto-detected regions overlay.
- Admin console: ingestion jobs, index status, A/B toggles, metrics.

### 12.2 User Experience
- Immediate feedback: skeleton loading; partial results streamed if enabled.
- Keyboard shortcuts: up/down to navigate, enter to open, f to filter.
- Clear empty states and error messages with retry options.
- Multi-language UI; locale-aware.

### 12.3 Accessibility
- WCAG 2.1 AA: high-contrast mode, alt text from captions, ARIA roles.
- Keyboard-only operations; screen reader labels.
- Adjustable text size and reduced motion setting.

## 13. Security Requirements
### 13.1 Authentication
- OIDC/OAuth2 with PKCE for SPA; JWT validation at gateway and services.

### 13.2 Authorization
- RBAC per tenant/collection; attribute-based filters; least privilege.
- Signed URLs for object storage; CDN token auth.

### 13.3 Data Protection
- TLS 1.2+ in transit; SSE-S3 or CMEK at rest.
- Hashing/salting of sensitive identifiers; rotate secrets with Vault/SM.

### 13.4 Compliance
- GDPR/CCPA: data deletion APIs, consent tracking for feedback.
- Audit logging for administrative actions.
- DPA and SOC2-ready operational controls.

## 14. Performance Requirements
### 14.1 Response Times
- Text search: P95 ≤ 500 ms; Image search: P95 ≤ 650 ms (includes upload).
- Re-ranking budget: ≤ 150 ms for top-100 candidates on 1 GPU.

### 14.2 Throughput
- Target 1,000 QPS across 4 shards (250 QPS/shard) CPU; 3,000 QPS with GPU-assisted ANN.

### 14.3 Resource Usage
- Embedding GPU: ≤ 15 ms/text, ≤ 25 ms/image 336px per request (A10G).
- Memory per 1M vectors (768-dim, FP16+PQ): ≤ 2.5 GB index RAM.

## 15. Scalability Requirements
### 15.1 Horizontal Scaling
- Shard by tenant/collection; replica factor ≥ 2.
- Stateless services with HPA on QPS and latency.

### 15.2 Vertical Scaling
- GPU-enabled nodes for embedding/re-ranking; auto-scale up during peak.

### 15.3 Load Handling
- CDN for static assets; Redis for hot-result cache; request hedging; circuit breakers.
- Queue backpressure on ingestion; delta index for near-real-time writes.

## 16. Testing Strategy
### 16.1 Unit Testing
- 80%+ coverage for API, embedding wrappers, fusion, MMR, filters.
- Deterministic seeds for embedding tests.

### 16.2 Integration Testing
- End-to-end: ingest -> index -> query -> rerank.
- Multi-tenant isolation and auth scenarios.
- Vector store backends: FAISS and Qdrant matrix tests.

### 16.3 Performance Testing
- Locust/k6 load tests at target QPS; latency P50/P95/P99.
- Recall-vs-latency sweeps across efSearch, nprobe, PQ bits.

### 16.4 Security Testing
- SAST/DAST; OWASP ZAP; dependency scanning.
- Pen tests for auth bypass, injection, deserialization.

## 17. Deployment Strategy
### 17.1 Deployment Pipeline
- GitHub Actions: lint/test -> build Docker images -> push to registry -> Helm deploy to K8s (staging/prod).
- Model registry via MLflow; immutable model versions tagged.

### 17.2 Environments
- Dev: single-node, CPU FAISS, small dataset.
- Staging: 1–2 shards, GPU for re-rank.
- Prod: multi-AZ cluster, autoscaling nodes, managed DB.

### 17.3 Rollout Plan
- Blue-green for index swaps; canary 10% traffic for new models/config.
- A/B tests for embeddings and re-rankers with kill switch.

### 17.4 Rollback Procedures
- Index version pinning; revert to previous manifest.
- Model version rollback via registry; config flags to disable re-ranker.

## 18. Monitoring & Observability
### 18.1 Metrics
- API: QPS, error rates, latency percentiles.
- Retrieval: candidate set size, recall proxy (overlap with offline labels).
- Index: build times, memory footprint, PQ distortion, HNSW recall probes.
- ML: embedding drift (cosine norm distrib), safety classifier thresholds.
- Business: CTR, Success@k, zero-result rate.

### 18.2 Logging
- Structured JSON logs with correlation IDs (trace/span).
- Anonymize PII; redact tokens.

### 18.3 Alerting
- On-call alerts for elevated P95>600 ms, error rate>2%, index status not ready.
- Drift alert when KL divergence on embedding norms > threshold.

### 18.4 Dashboards
- Grafana: latency, throughput, index health, GPU utilization.
- Business analytics: A/B comparisons, CTR trends.

## 19. Risk Assessment
### 19.1 Technical Risks
- Model drift degrades quality.
- Index recall loss due to aggressive PQ.
- GPU shortages or cost spikes.
- Cross-encoder latency spikes under load.
- Ingestion backlogs affecting freshness.

### 19.2 Business Risks
- Content licensing and moderation issues.
- Vendor lock-in with managed vector DBs.
- Cost overruns due to unbounded growth.

### 19.3 Mitigation Strategies
- Continuous evaluation and drift detection; scheduled fine-tuning.
- Two-stage retrieval with re-ranking to recover quality.
- Multi-cloud or hybrid vector backend support; autoscaling guards.
- Rate limiting and priority queues; graceful degradation (disable rerank temporarily).
- Cost dashboards and quotas per tenant.

## 20. Timeline & Milestones
### 20.1 Phase Breakdown
- Phase 0 (Weeks 1–2): Requirements, data audit, PoC with FAISS + CLIP.
- Phase 1 (Weeks 3–8): MVP: text/image search, ingestion, FAISS HNSW, basic UI, safety filter, metrics.
- Phase 2 (Weeks 9–14): Re-ranker integration, hybrid BM25, object-level search, A/B testing, multi-tenancy, admin console.
- Phase 3 (Weeks 15–18): Hardening: scalability tests, security, SLA monitoring, documentation, SDKs.
- Phase 4 (Weeks 19–20): Beta with 2 tenants, feedback incorporation.
- GA (Week 24): Production readiness, 99.5% uptime SLO established.

### 20.2 Key Milestones
- M1 (Week 2): PoC Recall@10 ≥ 0.85 on curated set.
- M2 (Week 8): MVP live; P95 ≤ 600 ms; 1M images indexed.
- M3 (Week 14): Object search and re-ranker; Recall@10 ≥ 0.90.
- M4 (Week 18): Scale to 10M images; P95 ≤ 700 ms; 99.5% uptime for 30 days.
- GA (Week 24): Contracts ready; SLAs; SDKs; docs.

Estimated monthly cloud costs at 10M images, 500 QPS avg:
- Compute: 2x GPU (A10G/A100 split) ~$2.5–7k; CPU nodes ~$2–4k
- Storage: S3 10 TB ~$230; Egress/Transfer ~$500–2k
- DB/Vector: Managed Qdrant/Elasticsearch ~$1–3k
- Total: ~$6–16k/month depending on GPU choice and region

## 21. Success Metrics & KPIs
### 21.1 Measurable Targets
- Quality: Recall@10 ≥ 0.90; NDCG@10 ≥ 0.75; mAP@10 ≥ 0.60 on benchmarks.
- Latency: P95 ≤ 500 ms (1M), ≤ 650 ms (10M, sharded).
- Uptime: ≥ 99.5% monthly; error rate < 1%.
- Engagement: CTR uplift ≥ 15%; Success@10 ≥ 0.45 within 60 days post-launch.
- Freshness: New assets searchable ≤ 5 minutes; 95% delta merges < 30 minutes.
- Safety: ROC-AUC ≥ 0.98; FN < 0.5%.

## 22. Appendices & Glossary
### 22.1 Technical Background
- Multimodal embeddings map text and images into a shared vector space enabling cross-modal similarity via cosine similarity after L2 normalization.
- ANN accelerates approximate nearest neighbor search with structures such as HNSW (graph-based) or IVF-PQ (quantization-based).
- Two-stage retrieval improves quality: a fast bi-encoder first pass followed by a more expensive cross-encoder re-ranker.
- Hybrid retrieval combines BM25 (sparse) with dense vectors, often improving text-to-image alignment when captions/metadata exist.
- Perceptual hashing detects near-duplicates; helps reduce redundancy and improve diversity.
- Object-level indexing stores region embeddings for fine-grained retrieval.

### 22.2 References
- Radford et al., “Learning Transferable Visual Models From Natural Language Supervision” (CLIP)
- OpenCLIP: https://github.com/mlfoundations/open_clip
- FAISS: https://github.com/facebookresearch/faiss
- Qdrant: https://qdrant.tech
- Milvus: https://milvus.io
- BLIP/BLIP-2: https://github.com/salesforce/LAVIS
- GroundingDINO: https://github.com/IDEA-Research/GroundingDINO
- OpenSearch: https://opensearch.org

### 22.3 Glossary
- ANN: Approximate Nearest Neighbor, fast approximate similarity search.
- BM25: Ranking function for sparse keyword search.
- CLIP/OpenCLIP: Models aligning images and text in a shared embedding space.
- DINOv2: Self-supervised vision transformer robust for image similarity.
- Embedding: Dense vector representation of input (image or text).
- HNSW: Hierarchical Navigable Small World graph for fast ANN search.
- IVF-PQ: Inverted File with Product Quantization to compress vectors.
- MMR: Maximal Marginal Relevance for diversity in results.
- mAP/NDCG/Recall@k: Retrieval quality metrics.
- PQ/OPQ: (Optimized) Product Quantization methods for vector compression.
- Re-ranker: Second-stage model refining candidate rankings.
- Safety Filter: Classifier to exclude unsafe content.
- Tenant: Logical isolation unit for multi-tenant systems.

Repository structure
- aiml045_visual_search_engine/
  - README.md
  - notebooks/
    - evaluation.ipynb
    - ann_tuning.ipynb
    - hard_negative_mining.ipynb
  - src/
    - api/
      - main.py
      - auth.py
      - schemas.py
      - routers/
        - search.py
        - index.py
        - feedback.py
    - ml/
      - embeddings.py
      - reranker.py
      - captioner.py
      - detector.py
      - ann/
        - faiss_index.py
        - qdrant_client.py
      - safety.py
      - fusion.py
      - mmr.py
    - ingestion/
      - etl_worker.py
      - delta_merger.py
      - phash.py
      - jobs/
        - build_ivf_pq.py
        - train_quantizer.py
    - db/
      - models.py
      - repo.py
    - utils/
      - config.py
      - logging.py
  - tests/
    - unit/
    - integration/
    - performance/
  - configs/
    - default.yaml
    - prod.yaml
    - models/
      - clip_vitl14_336.yaml
      - blip_itm_base.yaml
  - data/
    - samples/
  - deployment/
    - helm/
    - docker/
      - Dockerfile.api
      - Dockerfile.ml
    - k8s/
      - api.yaml
      - ml.yaml
      - vectorstore.yaml

Sample config (YAML)
service:
  host: 0.0.0.0
  port: 8080
  workers: 4
ml:
  embedding_model: openclip_vitl14_336
  embedding_dim: 768
  device: cuda
  normalize: true
ann:
  backend: faiss
  method: ivf_pq
  nlist: 16384
  nprobe: 24
  pq_m: 64
  pq_bits: 8
reranker:
  enabled: true
  model: blip2_itm_base
hybrid:
  enabled: true
  dense_weight: 0.7
  bm25_weight: 0.3
safety:
  enabled: true
  threshold: 0.5

Code snippet (FAISS query)
import faiss
import numpy as np

def query_faiss(index: faiss.Index, qvec: np.ndarray, topk=200):
    qvec = qvec.astype('float32')
    faiss.normalize_L2(qvec)
    D, I = index.search(qvec, topk)
    return I[0].tolist(), D[0].tolist()

ASCII: Index sharding
+-----------+    +-----------+    +-----------+
| Shard 1   |    | Shard 2   |    | Shard N   |
| 2 replicas|    | 2 replicas|    | 2 replicas|
+-----+-----+    +-----+-----+    +-----+-----+
      \              |                 /
       \             |                /
        +------------+---------------+
                     |
               Query Router

Evaluation metrics targets
- Offline: Recall@10 ≥ 0.90, mAP@10 ≥ 0.60, NDCG@10 ≥ 0.75.
- Online: CTR uplift ≥ 15%, Success@10 ≥ 0.45.
- Latency: P95 ≤ 500 ms; Throughput ≥ 300 QPS/shard CPU, ≥ 800 QPS/shard GPU.
- Uptime: 99.5%+.

End of PRD.