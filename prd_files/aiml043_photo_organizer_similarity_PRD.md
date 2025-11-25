# Product Requirements Document (PRD)
# `Aiml043_Photo_Organizer_Similarity`

Project ID: aiml043
Category: Computer Vision, Similarity Search, Information Retrieval
Status: Draft for Review
Version: v1.0
Last Updated: 2025-11-25

## 1. Overview
### 1.1 Executive Summary
Aiml043_Photo_Organizer_Similarity is a photo organization system that automatically deduplicates, groups, and enables “find similar” search across personal and enterprise photo libraries. It leverages image embeddings, face and scene understanding, perceptual hashing, clustering, and metadata-aware ranking to deliver fast and accurate retrieval. The system combines approximate nearest neighbor (ANN) recall with re-ranking and supports hybrid search across text captions, OCR, and tags.

### 1.2 Document Purpose
This PRD defines product scope, requirements, architecture, data model, APIs, UI/UX, security, performance, testing, deployment, monitoring, risk, timeline, KPIs, and glossary to guide engineering, design, and product stakeholders from MVP through production.

### 1.3 Product Vision
Create a privacy-conscious, blazing-fast photo organizer that makes it effortless to:
- Remove duplicates and near-duplicates safely.
- Discover similar photos by content, people, places, and events.
- Build meaningful albums and stories with minimal manual work.
- Scale from single users to large organizations with millions of photos.

## 2. Problem Statement
### 2.1 Current Challenges
- Photo libraries are cluttered with duplicates (bursts, edits, screenshots), making storage and discovery painful.
- Users struggle to find visually similar images or consolidate shots of the same scene/person/event.
- Keyword-only search misses visual similarity; manual tagging doesn’t scale.
- Existing tools often lack accuracy, transparency, or performance at scale.

### 2.2 Impact Analysis
- Time wasted manually curating galleries.
- Increased storage cost from duplicates.
- Missed memories and reduced user satisfaction due to poor discovery.
- Low retention for photo apps lacking intelligent organization.

### 2.3 Opportunity
- Provide best-in-class similarity search with high precision, low latency, and trustable suggestions.
- Differentiate via hybrid visual + text search and event-aware clustering.
- Offer privacy-first infrastructure suitable for consumers and enterprises.

## 3. Goals and Objectives
### 3.1 Primary Goals
- Accurate deduplication and grouping at scale.
- “Find similar” search with high relevance and low latency.
- Event-based album suggestions combining time, GPS, and visual similarity.
- Face discovery and cluster labeling with user-controlled privacy.

### 3.2 Business Objectives
- Increase user retention by 15% via better discovery features.
- Reduce storage usage by 20% through dedupe recommendations.
- Offer a SaaS tier with per-seat pricing; enterprise API and admin controls.
- Reach 99.5% monthly uptime and <300 ms p95 similar-search latency.

### 3.3 Success Metrics
- Dedupe precision ≥ 95%, recall ≥ 85% (validated set).
- mAP@10 for “find similar” ≥ 0.60.
- Face clustering NMI ≥ 0.75 on internal benchmark.
- p95 search latency < 300 ms; p99 < 600 ms.
- 20%+ CTR on “find similar” suggestions; 70%+ acceptance on dedupe suggestions.

## 4. Target Users/Audience
### 4.1 Primary Users
- Consumers with large personal photo libraries.
- Photographers and creatives managing shoots.
- Enterprise DAM (Digital Asset Management) teams.

### 4.2 Secondary Users
- Product managers integrating visual search into apps.
- Customer support admins handling shared libraries.
- Data annotators involved in human-in-the-loop feedback.

### 4.3 User Personas
1) Maya Thompson, 34, Lifestyle Photographer
- Background: Shoots weddings and lifestyle sessions; 100k+ photos/year.
- Pain points: Thousands of near-duplicates from bursts; manual curation is slow; needs to find similar compositions fast.
- Goals: Rapid dedupe, reliable “find similar” to pick best shots; control over face grouping to deliver client albums quickly.

2) Alex Chen, 29, Tech-Savvy Consumer
- Background: 10 years of smartphone photos; travel and family; multi-cloud backups.
- Pain points: Hard to find “that one photo” by memory; messy duplicates of edits and screenshots.
- Goals: Smart albums by trips/events; simple cleanup; private and secure.

3) Priya Nair, 41, DAM Administrator (Enterprise)
- Background: Manages 10M+ marketing assets; compliance and permissions are critical.
- Pain points: Mixed-quality uploads, inconsistent metadata, duplicate brand assets.
- Goals: Scalable search and grouping; RBAC; audit logs; SLAs and monitoring.

## 5. User Stories
US-001: As a user, I want automatic detection of exact and near-duplicate photos so that I can safely reclaim storage.
- Acceptance: System lists duplicate sets with confidence scores; “merge/keep best” action; false-positive rate <5% on validation.

US-002: As a user, I want to find visually similar photos to a selected image so that I can pick the best composition or variant.
- Acceptance: Top-20 similar results return in <300 ms p95; relevance mAP@10 ≥ 0.60.

US-003: As a user, I want event-based grouping using time and location so that albums reflect trips and occasions.
- Acceptance: Events grouped with configurable time gaps; visually coherent clusters; users can split/merge events.

US-004: As a user, I want face discovery and labeling so that I can browse photos by person.
- Acceptance: Face clusters are >90% pure on internal benchmark; labeling propagates across cluster; optional per-user opt-out.

US-005: As a user, I want hybrid search by text + visual similarity so that captions and OCR help find images.
- Acceptance: Keyword filters narrow results; RRF or weighted fusion improves success rate by ≥10% over visual-only.

US-006: As an admin, I want per-tenant namespace isolation and RBAC so that enterprise data is secure.
- Acceptance: Separate indices and metadata per tenant; role-based access to actions and audit logs.

US-007: As a user, I want quality-aware ranking so that blurred/poorly exposed photos are de-prioritized.
- Acceptance: Quality score reduces poor images in top-20 by ≥50% relative to baseline.

US-008: As a user, I want transparent thresholds and controls so that I can tune aggressiveness of dedupe and grouping.
- Acceptance: UI sliders with preview; per-user overrides persist; “undo” available.

US-009: As an operator, I want background re-embedding and dual-index reads during migrations so that upgrades are safe.
- Acceptance: Shadow index built; dual-read compares; cutover toggled with feature flag; rollback documented.

## 6. Functional Requirements
### 6.1 Core Features
FR-001: Photo ingestion with idempotent content hash (SHA-256) and EXIF extraction.
FR-002: Embedding generation using pre-trained CLIP/SigLIP for scenes and InsightFace for faces.
FR-003: Perceptual hashing (pHash/dHash/aHash) for fast near-duplicate detection.
FR-004: ANN vector index (HNSW or IVF-PQ) for “find similar” search.
FR-005: Re-ranking pipeline with metadata-aware scoring and MMR for diversity.
FR-006: Dedup graph builder: edges above threshold; connected components to produce duplicate sets.
FR-007: Clustering: HDBSCAN/DBSCAN for density-based grouping; KMeans for scalable clustering; hierarchical clustering for album suggestions.
FR-008: Event detection: time-gap clustering with GPS proximity; fusion with visual similarity.
FR-009: Face workflow: detection, embedding, clustering per user; optional labeling and propagation.
FR-010: Hybrid search: combine keywords (captions, OCR, tags) with vector similarity via weighted fusion or RRF.
FR-011: Quality metrics (blur/sharpness/exposure) and NSFW/sensitive filters.
FR-012: Model/version lifecycle: versioned embedding spaces; background re-embedding; dual-index reads.
FR-013: API endpoints for upload, search, dedupe suggestions, cluster management, face labeling, hybrid search.
FR-014: UI for gallery, similar search, event albums, face clusters, dedupe review with safe-merge.

### 6.2 Advanced Features
FR-015: Active learning: collect user accept/reject signals to improve thresholds and re-ranking.
FR-016: Batch operations and rules (e.g., auto-archive low-quality near-duplicates).
FR-017: Admin console: per-tenant thresholds, RBAC, audit logs, usage metrics.
FR-018: Offline/edge pre-embedding SDK (optional) for privacy-first clients.
FR-019: Cross-modal re-ranker (lightweight cross-encoder) for top-N refinement.
FR-020: Cache popular neighbor sets and apply approximate batch queries for speed.

## 7. Non-Functional Requirements
### 7.1 Performance
- p95 search latency < 300 ms; p99 < 600 ms for top-20 results.
- Ingestion throughput ≥ 50 images/sec per worker; scalable horizontally.
- Re-embedding migration ≥ 1M images/day per cluster.

### 7.2 Reliability
- Uptime ≥ 99.5% monthly.
- Durable storage with multi-AZ replication.
- At-least-once processing with idempotency to avoid duplicates.

### 7.3 Usability
- Clear, reversible actions; preview before delete/merge.
- Keyboard shortcuts; tooltip intros; guided onboarding.

### 7.4 Maintainability
- Modular microservices; clear versioning; infrastructure as code.
- Tests with ≥ 80% unit coverage for core logic.
- Feature flags and safe migrations.

## 8. Technical Requirements
### 8.1 Technical Stack
- Backend: Python 3.11+, FastAPI 0.115+, Uvicorn 0.30+
- Workers: Python 3.11+, Celery 5.4+ or Ray 2.9+
- Vector DB/Index: FAISS 1.8+ (IVF-PQ/HNSW) and/or Qdrant 1.8+ (HNSW)
- Metadata DB: PostgreSQL 15+
- Search/Text: Elasticsearch/OpenSearch 8+ (BM25, ingest OCR/captions)
- Message Queue: Redis 7+ or RabbitMQ 3.12+
- Object Storage: S3-compatible store (versioned buckets)
- Frontend: React 18+, TypeScript 5+, Vite 5+
- Auth: OAuth2/OIDC (Auth0/Keycloak), JWT access tokens
- Infra: Docker, Kubernetes 1.29+, Helm, Terraform
- Monitoring: Prometheus, Grafana, Loki, OpenTelemetry
- CI/CD: GitHub Actions, Dependabot
- OCR (optional): Tesseract 5+ or PaddleOCR 2.7+

### 8.2 AI/ML Components
- Scene embeddings: OpenCLIP/CLIP ViT-B/32, SigLIP base; normalized vectors; cosine distance.
- Face embeddings: InsightFace/ArcFace models; per-tenant face clustering.
- Perceptual hashing: pHash, aHash, dHash; Hamming distance thresholds.
- Clustering: HDBSCAN/DBSCAN; KMeans/mini-batch KMeans; linkage for hierarchy.
- Re-ranking: lightweight cross-encoder or CLIP score refinement; MMR lambda configurable.
- Dimensionality reduction: OPQ/PQ for compression; optional PCA/UMAP for visualization.
- Threshold calibration: ROC/PR; per-model/per-tenant overrides.
- Evaluation: mAP@K, NMI/ARI, ROC-AUC/PR-AUC.

## 9. System Architecture
### 9.1 High-Level Architecture (ASCII diagram)
Users/Web/SDK
   |
   v
[API Gateway] --- [Auth/OIDC]
   |
   +--> [Ingestion Service] --> [Blob Storage (S3)] --> [EXIF/Metadata Extractor]
   |                                 |
   |                                 v
   |                             [Content Hash]
   |
   +--> [Embedding Service] (Scene, Face) --\
   |                                         \
   |                                          --> [Vector Index (FAISS/Qdrant)]
   |                                         /
   +--> [Hashing Service (pHash/aHash/dHash)]
   |
   +--> [Clustering Service] <--> [Metadata DB (Postgres)]
   |
   +--> [Event Detector] <---- [Time/GPS]
   |
   +--> [Search Service] --> [ANN Recall] -> [Re-ranker/MMR] -> [Results]
   |
   +--> [Hybrid Search] --> [Text Index (Elastic)] + [Vector Index] -> [Fusion]
   |
   +--> [Admin Console] [RBAC]
   |
   +--> [Monitoring/Logging]

Background:
[Re-Embed/Reindex Worker] -> [Shadow Index] -> [Dual Read] -> [Cutover/Rollback]
[Queue (Redis/RabbitMQ)] for asynchronous jobs

### 9.2 Component Details
- API Gateway: Rate limiting, auth enforcement, routing, request validation.
- Ingestion Service: Receives uploads/URLs; computes SHA-256, stores blobs, triggers pipelines.
- Metadata Extractor: EXIF, GPS, device/time parsing; OCR if enabled.
- Embedding Service: Batched GPU/CPU inference pipelines; model versioning; quantization optional.
- Hashing Service: Computes perceptual hashes; caches results.
- Vector Index: Tenant-isolated namespaces; HNSW or IVF-PQ; supports upserts and deletes.
- Clustering Service: Builds dedup graphs and HDBSCAN/DBSCAN clusters; event suggestion logic.
- Search Service: Two-stage retrieval (ANN → re-rank with metadata and MMR).
- Admin Console: Thresholds, RBAC, audits, migrations.
- Re-embed/Reindex Worker: Background migrations; dual-index with shadow builds.
- Observability: Centralized logs, metrics, traces.

### 9.3 Data Flow
1) Ingest: Client → API → Blob store; content hash computed; EXIF extracted.
2) Process: Enqueue job → Embedding + perceptual hashing; write to vector index and DB.
3) Dedup: Build similarity graph from pHash/Hamming + embedding cosine; connected components saved.
4) Clustering: Scene and face clustering; events from time/GPS and visual fusion.
5) Search: ANN recall top-N → re-rank with cross-encoder/metadata/MMR → results.
6) Hybrid: Keyword filter (Elastic) → ANN on filtered set → weighted fusion or RRF.
7) Migrations: Shadow index build; dual-read; cutover; rollback if regressions detected.

## 10. Data Model
### 10.1 Entity Relationships
- User (1) — (N) Photo
- Photo (1) — (N) Embedding (by model_version)
- Photo (1) — (0..N) Face (detected regions)
- Photo (N) — (N) Tag
- Photo (N) — (N) Cluster (scene/event/dedup clusters)
- Event (1) — (N) Photo
- DuplicateSet (1) — (N) Photo
- IndexVersion (1) — (N) Embedding
- Tenant (1) — (N) User; Tenant namespaces isolate data

### 10.2 Database Schema (PostgreSQL)
- tenants(id, name, created_at)
- users(id, tenant_id, email, role, created_at)
- photos(id UUID=sha256, tenant_id, user_id, blob_uri, mime_type, width, height, created_at, captured_at, gps_lat, gps_lng, device_make, device_model, quality_score, nsfw_flag, ocr_text, caption, phash, ahash, dhash)
- embeddings(id, photo_id, tenant_id, model_name, model_version, vector float[dim] or stored in vector DB only, created_at)
- faces(id, photo_id, bbox, embedding_ref, cluster_id nullable, created_at)
- clusters(id, tenant_id, type ENUM('scene','face','event','dedup'), label, metadata JSONB, created_at)
- cluster_members(cluster_id, photo_id, score, created_at)
- events(id, tenant_id, title, start_time, end_time, center_gps, created_at)
- duplicatesets(id, tenant_id, created_at)
- duplicateset_members(duplicateset_id, photo_id, confidence, quality_rank)
- tags(id, tenant_id, label)
- photo_tags(photo_id, tag_id)
- thresholds(tenant_id, model_name, model_version, dedup_thresh, similar_thresh, updated_at)
- index_versions(id, tenant_id, model_name, model_version, status ENUM('active','shadow','deprecated'), created_at)
- audit_logs(id, tenant_id, user_id, action, entity_type, entity_id, metadata JSONB, created_at)

Vector store (FAISS/Qdrant):
- Index per tenant and model_version; metadata: photo_id, captured_at, gps, tags, quality, face_ids.

### 10.3 Data Flow Diagrams (ASCII)
Ingest
Client -> API -> Blob -> EXIF -> Queue -> Embeddings/Hashes -> Vector DB + Postgres -> Clusters/Dedup

Search
Query Photo ID -> ANN Top-N -> Re-rank (metadata/cross-encoder/MMR) -> Result list

Hybrid
Text Query -> Text Index Filter -> ANN on subset -> Fusion (weighted/RRF) -> Results

### 10.4 Input Data & Dataset Requirements
- Image types: JPEG, PNG, HEIC/HEIF (convert to JPEG for inference if needed).
- EXIF parsing for timestamp, GPS, orientation; handle missing/incorrect metadata.
- Optional: captions from user or generated; OCR from screenshots.
- Ground truth for evaluation: curated pairs for duplicate/similar thresholds; labeled events; face identity test sets (opt-in and anonymized for evaluation).

## 11. API Specifications
### 11.1 REST Endpoints
- POST /v1/photos
  - Multipart upload or URL ingest; returns photo_id (sha256), status.
- GET /v1/photos/{photo_id}
  - Returns metadata, EXIF, quality, tags.
- GET /v1/photos/{photo_id}/similar?top_k=20&filter=...
  - Returns similar photos with scores and reasons (embedding, pHash).
- GET /v1/search?q=...&mode=hybrid&top_k=50
  - Hybrid text+visual search; supports filters (time range, GPS bbox, tags).
- GET /v1/dedup/suggestions?confidence_min=0.8
  - Returns duplicate sets with suggested keepers.
- POST /v1/dedup/resolve
  - Body: {duplicateset_id, action: 'merge'|'ignore', keep_photo_id}
- GET /v1/events
  - List suggested events; parameters: time_gap_minutes, gps_radius_meters.
- POST /v1/faces/label
  - Assign a label to a face cluster.
- GET /v1/faces/clusters
  - Paginated list of face clusters with representative thumbnails.
- POST /v1/reindex
  - Admin: start re-embedding for a tenant/model_version (shadow index).
- GET /v1/health, /v1/metrics
  - Health and Prometheus metrics.

### 11.2 Request/Response Examples
Upload
Request:
POST /v1/photos
Content-Type: multipart/form-data
file=@IMG_1234.jpg

Response:
{
  "photo_id": "a1b2c3...sha256",
  "status": "queued"
}

Find Similar
Request:
GET /v1/photos/a1b2c3.../similar?top_k=10

Response:
{
  "query_photo_id": "a1b2c3...",
  "top_k": 10,
  "results": [
    {"photo_id": "p2", "score": 0.92, "signals": {"cosine":0.94,"phash":0.88}, "explanations":["same scene","same event"]},
    {"photo_id": "p3", "score": 0.89, "signals": {"cosine":0.91,"phash":0.72}}
  ],
  "latency_ms": 147
}

Dedup Resolve
Request:
POST /v1/dedup/resolve
{
  "duplicateset_id": "dset_456",
  "action": "merge",
  "keep_photo_id": "p2"
}

Response:
{"status":"ok","removed":["p3","p4"],"kept":"p2","undo_token":"undo_789"}

### 11.3 Authentication
- OAuth2/OIDC with PKCE for web/mobile.
- JWT access tokens; scopes for read/write/admin.
- Per-tenant namespace enforced via claims.
- Optional API keys for service-to-service.

## 12. UI/UX Requirements
### 12.1 User Interface
- Gallery grid with “Similar” action on hover/context menu.
- Dedupe Review: grouped sets with side-by-side comparison, quality badges, resolution, filesize, capture time; choose “keep best” or manual pick.
- Event Albums: timeline with auto-clusters; split/merge; rename.
- Faces: unlabeled clusters; label suggestions; privacy toggles.
- Filters: time range slider, GPS map, tags, people, quality.

### 12.2 User Experience
- Instant feedback on search (spinner + p95 < 300 ms).
- Safe actions: Undo for merge/delete; trash/restore flow.
- Explanations for results (e.g., “same location within 20m”; “high visual similarity”).
- Accessibility-friendly keyboard navigation.

### 12.3 Accessibility
- WCAG 2.1 AA compliance.
- Alt text for thumbnails; focus states; color-contrast > 4.5:1.
- Screen reader labels for actions.

## 13. Security Requirements
### 13.1 Authentication
- OIDC, MFA optional for admin roles.
- Token lifetimes and refresh policies; token rotation.

### 13.2 Authorization
- RBAC: roles admin, editor, viewer.
- Per-tenant isolation; resource ownership checks.

### 13.3 Data Protection
- TLS 1.2+ in transit.
- Encryption at rest (KMS-managed keys).
- Least-privilege IAM; encrypted secrets management.
- Face data treated as sensitive; opt-in features; granular deletion.

### 13.4 Compliance
- GDPR/CCPA: data export/delete, purpose limitation, consent for face features.
- SOC 2-aligned controls; audit logging; retention policies.
- DPA and data residency options.

## 14. Performance Requirements
### 14.1 Response Times
- Similar search p95 < 300 ms; p99 < 600 ms.
- Dedupe suggestions page load < 1.5 s for 100 sets.
- Event list generation < 2 s p95 for 100k photos.

### 14.2 Throughput
- Ingestion ≥ 50 images/sec/worker; scalable to 10M images total per tenant.
- Re-embedding: ≥ 10 images/sec/GPU for ViT-B/32; parallelizable.

### 14.3 Resource Usage
- Embedding batch size auto-tuned; GPU utilization > 70%.
- Vector memory per 1M images (float32) ~ 512MB for 128-d PQ; use PQ/OPQ to reduce.

## 15. Scalability Requirements
### 15.1 Horizontal Scaling
- Stateless API pods; autoscale on CPU/RPS.
- Worker pool scaling based on queue depth.
- Vector DB sharded by tenant; replicas for read scaling.

### 15.2 Vertical Scaling
- GPU instances for embedding; memory-optimized nodes for FAISS build jobs.
- Increase HNSW M/efConstruction for quality as capacity allows.

### 15.3 Load Handling
- Backpressure via queue; rate limiting; circuit breakers.
- Cache neighbor sets for hot photos; CDN for thumbnails.

## 16. Testing Strategy
### 16.1 Unit Testing
- Embedding pipeline, hashing, thresholding, scoring functions.
- EXIF parsing and metadata normalization.
- Dataset validators for idempotency.

### 16.2 Integration Testing
- API + DB + Vector index end-to-end.
- Dual-index migration tests with shadow reads.
- Hybrid search fusion correctness.

### 16.3 Performance Testing
- ANN queries with Locust/k6; latency/throughput under load.
- Index build/re-embed timings; memory usage profiling.

### 16.4 Security Testing
- AuthZ bypass attempts; multi-tenant isolation tests.
- Static analysis, dependency scanning, container image scans.
- Pen tests and fuzzing for file ingestion.

## 17. Deployment Strategy
### 17.1 Deployment Pipeline
- GitHub Actions: lint → unit tests → build Docker → integration tests → security scan → sign → push → deploy via Helm.
- Separate pipelines for models and indices; model registry with version pinning.

### 17.2 Environments
- Dev: ephemeral preview environments per PR.
- Staging: full stack with anonymized data.
- Prod: multi-AZ; feature flags for rollouts.

### 17.3 Rollout Plan
- Canary 10% traffic → observe metrics → expand to 100%.
- Shadow index dual-read; compare score drift, latency; cutover on SLO pass.

### 17.4 Rollback Procedures
- Helm rollback; switch traffic to previous version.
- Revert index alias to previous active index.
- Invalidate caches; restore snapshots.

## 18. Monitoring & Observability
### 18.1 Metrics
- Latency: p50/p95/p99 for search, ingest.
- Quality: mAP@10, dedupe precision/recall, NMI/ARI for faces.
- Infra: CPU/GPU utilization, queue depth, index memory.
- Business: CTR on “find similar”, dedupe acceptance rate, active users.

### 18.2 Logging
- Structured JSON logs; request IDs; user/tenant context (no sensitive payloads).
- Sampling for high-volume endpoints.

### 18.3 Alerting
- On-call alerts for SLO violations, error rates, queue backlog.
- Anomaly detection on quality metrics drift.

### 18.4 Dashboards
- Grafana: API SLOs, worker throughput, vector index health, model version adoption.
- Product analytics: funnel for dedupe flow, search engagement.

## 19. Risk Assessment
### 19.1 Technical Risks
- Model drift reduces relevance.
- High memory usage in vector indices.
- Incorrect EXIF leading to poor event grouping.
- Face clustering bias or mislabeling.

### 19.2 Business Risks
- Privacy concerns about face data.
- Vendor lock-in with managed vector stores.
- Cost overruns from GPU/ storage growth.

### 19.3 Mitigation Strategies
- Continuous evaluation; A/B testing; threshold calibration with ROC/PR.
- PQ/OPQ compression; hierarchical indices; shard by tenant.
- Robust metadata fallbacks (visual-only events).
- Opt-in face features; transparency, easy deletion; periodic bias audits.
- Abstraction layer for vector DB; multi-cloud support; cost monitoring.

## 20. Timeline & Milestones
### 20.1 Phase Breakdown
- Phase 0 (Weeks 1-2): Requirements, design, dataset curation, infra scaffolding.
- Phase 1 (Weeks 3-6): Ingestion, embeddings, vector index, basic search API, UI gallery.
- Phase 2 (Weeks 7-10): Dedup graph, event detection, re-ranking, hybrid search.
- Phase 3 (Weeks 11-12): Face pipeline (opt-in), admin console, RBAC.
- Phase 4 (Weeks 13-14): Performance tuning, QA, security, compliance checks.
- Phase 5 (Weeks 15-16): Staging soak, canary, production launch.

### 20.2 Key Milestones
- M1 (Week 4): ANN search MVP (<500 ms p95).
- M2 (Week 8): Dedup + event grouping beta; mAP@10 ≥ 0.55.
- M3 (Week 12): Face labeling beta; NMI ≥ 0.70.
- M4 (Week 14): p95 search <300 ms; dedupe precision ≥ 95%.
- GA (Week 16): Uptime ≥ 99.5%; full admin/RBAC.

Estimated Costs (monthly, initial scale)
- Compute: $4–8k (mix of CPU/GPU nodes).
- Storage: $1–3k (blobs, indices, snapshots).
- Observability/Networking/CDN: $1k.
- Total: ~$6–12k/month at pilot scale.

## 21. Success Metrics & KPIs
### 21.1 Measurable Targets
- Dedupe precision ≥ 95%, recall ≥ 85% at GA.
- mAP@10 ≥ 0.60; hybrid search improves success rate by ≥ 10% vs visual-only.
- p95 search latency < 300 ms; uptime ≥ 99.5%.
- 20% reduction in storage from dedupe actions per active user.
- 15% increase in 30-day retention for users engaging with “find similar”.
- <1% rollback rate over 6 months of releases.

## 22. Appendices & Glossary
### 22.1 Technical Background
- Embeddings: Fixed-length vectors from pre-trained encoders capturing semantics; normalized for cosine similarity.
- Perceptual hashing: Looks for visually similar images via hash comparators robust to minor edits.
- ANN: Approximate search in high-dimensional spaces using HNSW or IVF-PQ for speed.
- Re-ranking and MMR: Adjusts initial recall using metadata-aware scoring and diversity control.
- Clustering: Density-based (HDBSCAN/DBSCAN) handles arbitrary shapes and noise; KMeans scales well.

### 22.2 References
- Radford et al., “Learning Transferable Visual Models From Natural Language Supervision” (CLIP).
- Chum et al., “Near-duplicate image detection”.
- HNSW: Malkov & Yashunin, “Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs”.
- HDBSCAN: Campello et al., “Hierarchical Density Estimates for Data Clustering, Visualization, and Outlier Detection”.
- MMR: Carbonell & Goldstein, “The Use of MMR, Diversity-Based Reranking for Reordering Documents”.

### 22.3 Glossary
- ANN: Approximate Nearest Neighbor search.
- CLIP/SigLIP: Vision-language models producing embeddings.
- Cosine similarity: Metric measuring angle between normalized vectors.
- dHash/aHash/pHash: Perceptual hashing methods for near-duplicate detection.
- HDBSCAN/DBSCAN: Density-based clustering algorithms.
- HNSW: Graph-based ANN index supporting fast search.
- KMeans: Centroid-based clustering algorithm.
- mAP@K: Mean Average Precision at K; retrieval quality metric.
- NMI/ARI: Clustering evaluation metrics.
- MMR: Maximal Marginal Relevance for diversification in ranking.
- OPQ/PQ: Quantization techniques for compressing vectors.
- RRF: Reciprocal Rank Fusion; combines ranked lists.
- EXIF: Metadata stored in image files (time, GPS, camera settings).

---------------------------------------
Repository Structure
- README.md
- notebooks/
  - 01_threshold_calibration.ipynb
  - 02_clustering_eval.ipynb
  - 03_ann_benchmark.ipynb
- src/
  - api/
    - main.py
    - routers/
      - photos.py
      - search.py
      - dedup.py
      - faces.py
      - admin.py
  - services/
    - ingestion.py
    - exif.py
    - embeddings.py
    - hashing.py
    - indexing.py
    - rerank.py
    - clustering.py
    - events.py
    - hybrid_search.py
    - quality.py
  - workers/
    - tasks.py
  - models/
    - clip_loader.py
    - insightface_loader.py
  - utils/
    - config.py
    - logging.py
    - metrics.py
  - eval/
    - retrieval_metrics.py
    - clustering_metrics.py
- tests/
  - unit/
  - integration/
  - performance/
- configs/
  - default.yaml
  - thresholds.yaml
- data/
  - samples/
- infra/
  - helm/
  - terraform/

Config Sample (configs/default.yaml)
server:
  host: 0.0.0.0
  port: 8080
auth:
  provider: oidc
  audience: photo-app
storage:
  s3_bucket: photos
  region: us-east-1
database:
  postgres_url: postgresql://...
vector_index:
  backend: qdrant
  distance: cosine
  shard_by: tenant
  hnsw:
    m: 32
    ef_construct: 200
    ef_search: 128
models:
  scene_encoder: clip_vit_b32
  face_encoder: insightface_glint360k
  model_version: v1.0
retrieval:
  top_k: 50
  mmr_lambda: 0.3
  reranker: clip_light
thresholds:
  dedup: 0.92
  similar: 0.80

API Code Snippet (FastAPI)
from fastapi import FastAPI, UploadFile, Depends
from src.services import ingestion, search
app = FastAPI()

@app.post("/v1/photos")
async def upload_photo(file: UploadFile, user=Depends(auth_user)):
    pid = await ingestion.handle_upload(file, user)
    return {"photo_id": pid, "status": "queued"}

@app.get("/v1/photos/{photo_id}/similar")
async def similar(photo_id: str, top_k: int = 20, user=Depends(auth_user)):
    res, latency = await search.find_similar(photo_id, top_k, user.tenant_id)
    return {"query_photo_id": photo_id, "top_k": top_k, "results": res, "latency_ms": latency}

Embedding Extraction Snippet (Python)
import torch
from PIL import Image
from src.models.clip_loader import load_clip

model, preprocess = load_clip("ViT-B/32", device="cuda")
def encode_image(path):
    img = preprocess(Image.open(path)).unsqueeze(0).to("cuda")
    with torch.no_grad():
        emb = model.encode_image(img)
    return torch.nn.functional.normalize(emb, dim=-1).cpu().numpy()

Two-Stage Retrieval Pseudocode
def two_stage_retrieval(query_id, tenant, topk=100, final_k=20):
    ann_candidates = vector_index.search(tenant, query_id, topk)
    re_ranked = reranker.score(query_id, ann_candidates)
    diversified = mmr(re_ranked, lambda_=0.3, k=final_k)
    return diversified

Evaluation Metrics Code (Python)
def map_at_k(queries, ground_truth, k=10):
    # queries: dict q -> [results]
    # ground_truth: dict q -> set(relevant)
    import numpy as np
    ap_list = []
    for q, results in queries.items():
        rel = ground_truth.get(q, set())
        hits, score = 0, 0.0
        for i, r in enumerate(results[:k], 1):
            if r in rel:
                hits += 1
                score += hits / i
        if rel:
            ap_list.append(score / min(len(rel), k))
    return float(np.mean(ap_list)) if ap_list else 0.0

Specific Metrics Targets
- Retrieval mAP@10 ≥ 0.60
- Dedupe ROC-AUC ≥ 0.98; operating point precision ≥ 95%
- Face cluster NMI ≥ 0.75
- p95 latency < 300 ms; p99 < 600 ms
- Uptime ≥ 99.5%

Threshold Calibration Procedure
- Build validation pairs: exact duplicates, near-duplicates, semantically similar, dissimilar.
- Compute cosine similarities and Hamming distances.
- Plot ROC and PR curves; choose operating points optimizing F1 with precision priority.
- Maintain per-model_version thresholds; allow per-tenant overrides.

Reindex Strategy
- Build shadow index with model_version vNew.
- Dual-read: compare overlap@K, score drift, latency; log deltas.
- If deltas within SLO, switch alias to vNew; keep old index for 7 days for rollback.

By adhering to this PRD, Aiml043_Photo_Organizer_Similarity delivers a robust, scalable, and privacy-conscious system for photo deduplication, grouping, and similarity search, ready for consumer and enterprise adoption.