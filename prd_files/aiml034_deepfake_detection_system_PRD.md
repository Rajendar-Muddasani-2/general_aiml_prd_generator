# Product Requirements Document (PRD)
# `Aiml034_Deepfake_Detection_System`

Project ID: aiml034  
Category: AI/ML – Computer Vision, Audio, Multimodal  
Status: Draft for Review  
Version: v1.0  
Last Updated: 2025-11-25

## 1. Overview
### 1.1 Executive Summary
Aiml034_Deepfake_Detection_System is a cloud-native, multimodal AI platform to detect synthetic/manipulated media across image, video, and audio. It combines spatial, temporal, frequency-forensic, and audio models into an ensemble with late fusion and calibrated scoring. It supports real-time streaming inference, batch analysis, explainability overlays, and a human-in-the-loop review workflow. Target users include social platforms, newsrooms, financial institutions, and trust & safety teams.

### 1.2 Document Purpose
This PRD defines the product requirements, functional and non-functional specifications, architecture, data model, APIs, UI/UX, security, deployment, testing, and success criteria to guide engineering, data science, and product teams through build, validation, and launch.

### 1.3 Product Vision
Deliver trustworthy, fast, and transparent deepfake detection at scale to help organizations protect users and brands, mitigate misinformation, and enable responsible media provenance. The system provides high accuracy (>90% across diverse datasets), low latency (<500 ms per frame/clip window), robust cross-domain generalization, and actionable explanations to power automated and human decisions.

## 2. Problem Statement
### 2.1 Current Challenges
- Rapid proliferation of synthetic media (face swaps, reenactment, lip-sync, AI-generated speech).
- Poor generalization across platforms/codecs; heavy compression and post-processing obfuscate cues.
- Lack of transparent, explainable signals for moderation, compliance, and appeals.
- Limited tools for real-time detection in streaming contexts.
- Evolving manipulation methods lead to model drift and reduced performance over time.

### 2.2 Impact Analysis
- Misinformation and reputational risk.
- Fraud in financial services using cloned voices and manipulated KYC videos.
- Increased moderation costs and latency without reliable automation.
- Legal/regulatory exposure without auditability and due process.

### 2.3 Opportunity
- Provide an enterprise-grade detection service with strong cross-dataset generalization, multimodal coverage, and explainability.
- Offer APIs, SDKs, and UI for rapid integration, plus governance and audit tooling.
- Differentiate with multimodal fusion, frequency-forensics branch, and AV-consistency checker.

## 3. Goals and Objectives
### 3.1 Primary Goals
- Detect manipulated media (image/video/audio) with calibrated confidence and human-readable rationale.
- Support real-time and batch workflows via REST APIs, SDKs, and UI.
- Maintain robustness against compression and platform artifacts.
- Provide human-in-the-loop review tools and audit logs.

### 3.2 Business Objectives
- Reduce false positives by 40% vs. baseline detectors through ensemble and calibration.
- Achieve 99.5% service uptime SLA.
- Time-to-first-detection <200 ms for live streams; <500 ms per 16-frame window for video clips (p95).
- Scale to 1 million media analyses/day within 3 months of GA.

### 3.3 Success Metrics
- Model: ROC-AUC ≥ 0.95; EER ≤ 10%; F1 ≥ 0.90 on held-out cross-dataset validation.
- Ops: p95 API latency <500 ms per window; 99.5% uptime; <0.1% job failure rate.
- Product: 90% of analyst users rate explanations as helpful; reduce manual review time by 30%.

## 4. Target Users/Audience
### 4.1 Primary Users
- Trust & Safety Analysts and Moderators.
- Fraud Prevention/AML Teams.
- Newsroom fact-checkers and investigative journalists.
- Platform/API integrators and backend engineers.

### 4.2 Secondary Users
- Legal/Compliance teams.
- Security Operations (SOC).
- Product Managers and Data Scientists evaluating model performance.

### 4.3 User Personas
1) Name: Maya Patel  
   Role: Trust & Safety Analyst at a social platform  
   Background: 6 years in content moderation, uses internal dashboards and policy playbooks.  
   Goals: Rapidly triage flagged videos, minimize false positives, provide appeal-ready evidence.  
   Pain Points: Inconsistent signals, opaque detector output, heavy backlog during spikes.  
   Needs: Confidence scores, visual/audio explanations, batch actions, robust audit logs.

2) Name: Carlos Nguyen  
   Role: Fraud Prevention Lead at a fintech  
   Background: 8 years in risk analytics; oversees KYC and transaction verification pipelines.  
   Goals: Automatically screen onboarding videos and support calls for manipulated identities/voices.  
   Pain Points: False negatives leading to account takeovers; poor integration with existing APIs.  
   Needs: Streaming detection API, webhook callbacks, thresholds per use-case, data retention controls.

3) Name: Evelyn Brooks  
   Role: Senior Editor at a news outlet  
   Background: 12 years in investigative journalism, coordinates fact-checking.  
   Goals: Verify user-submitted footage rapidly under deadline, document provenance.  
   Pain Points: Lack of transparent rationale, difficulty explaining decisions to the public.  
   Needs: Explainability overlays, provenance metadata (C2PA), exportable reports with citations.

4) Name: Liam O’Connor  
   Role: Backend Engineer integrating detection APIs  
   Background: 5 years building microservices; comfortable with Python/Node.  
   Goals: Simple, stable API; SDKs; predictable performance; sandbox environment.  
   Pain Points: Complex auth, inconsistent schemas, inadequate rate-limit feedback.  
   Needs: OpenAPI spec, sample code, idempotency keys, clear error codes.

## 5. User Stories
US-001: As a moderator, I want to upload a video and receive a deepfake probability and explanations so that I can decide if it violates policy.  
Acceptance: Upload via UI returns a job ID; within SLA, result includes label (real/fake), score [0,1], saliency visualization, temporal attention plot.

US-002: As a fraud analyst, I want a streaming API to score live audio so that I can block real-time attacks.  
Acceptance: gRPC/WebSocket or chunked REST supports 1s frames; latency <200 ms per second of audio; returns rolling risk score and voice-clone likelihood.

US-003: As an engineer, I want to call a REST endpoint with a media URL so that I can batch-process content.  
Acceptance: POST /v1/detect accepts URL; asynchronous job; webhook callback on completion with signed payload.

US-004: As an analyst, I want to override thresholds and add notes so that policy context is preserved.  
Acceptance: UI supports per-domain thresholds; notes stored; changes logged in audit trail.

US-005: As a PM, I want drift monitoring so that retraining triggers when AUROC drops.  
Acceptance: Metrics dashboard shows AUROC/EER by domain; automatic ticket when drift threshold crossed.

US-006: As a journalist, I want an exportable PDF/JSON report so that I can share findings with editors.  
Acceptance: Report includes inputs, scores, explanations, model version, timestamp, and provenance metadata.

US-007: As a security lead, I want role-based access so that sensitive media is restricted.  
Acceptance: RBAC with least privilege; actions logged.

US-008: As a data scientist, I want to review hard negatives to improve the model.  
Acceptance: Active learning queue of false positives/negatives; export to training set with labels and reasons.

US-009: As a platform admin, I want SSO integration so that users use corporate identity.  
Acceptance: OIDC SSO with SCIM provisioning; JWT with scopes.

US-010: As a developer, I want SDKs so that integration is faster.  
Acceptance: Python and Node SDKs published; examples for image/video/audio.

## 6. Functional Requirements
### 6.1 Core Features
FR-001: Media ingestion via upload, URL fetch, or streaming.  
FR-002: Preprocessing pipeline: video frame extraction, face detection/landmarks, alignment/cropping; audio VAD and resampling.  
FR-003: Multimodal model ensemble: spatial (CNN/ViT), temporal (3D CNN/transformer), frequency-residual branch, audio models (spectrogram CNN, wav2vec2/HuBERT), AV-consistency checker.  
FR-004: Late fusion and calibration: temperature scaling/Platt; domain-aware thresholds.  
FR-005: Output: label, probability score, per-branch scores, confidence intervals, and explanations (Grad-CAM, frequency heatmaps, temporal attention).  
FR-006: Real-time streaming inference with incremental windowing.  
FR-007: Asynchronous job processing with webhooks and polling.  
FR-008: Human-in-the-loop review queue with annotations, notes, and decision outcomes.  
FR-009: Reporting and export (PDF/JSON) including model version and provenance metadata (C2PA awareness).  
FR-010: Admin: model/version management, threshold profiles, API key management, RBAC.  
FR-011: Audit logging of all actions and inference events.  
FR-012: SDKs for Python and Node.js.

### 6.2 Advanced Features
- Domain adaptation and open-set detection for unseen manipulation methods.
- Adversarial defenses and JPEG/codec invariance augmentation toggles.
- On-edge inference option with quantized models; privacy-preserving local processing mode.
- Browser extension for quick URL checks using server-side analysis.
- Content authenticity score combining detector output with available provenance signals (e.g., C2PA claim validation).
- Active learning/Hard negative mining from live traffic.
- Batch pipelines with configurable augmentations.
- Per-source calibration and thresholding (platform/codec specific).

## 7. Non-Functional Requirements
### 7.1 Performance
- p95 latency: <500 ms per 16-frame window or per image; <200 ms per 1s audio chunk (streaming).
- Startup cold latency for serverless endpoints: <2 s; warm <100 ms.
- Model loading time: <3 s per model shard.

### 7.2 Reliability
- Uptime: 99.5% monthly.  
- Job success rate: >99.9% (excluding client errors).  
- Exactly-once processing with idempotency keys for media ingestion.

### 7.3 Usability
- Onboarding time <30 minutes with API keys and SDK examples.  
- Explanations legible to non-technical analysts.  
- Keyboard navigation and screen reader support.

### 7.4 Maintainability
- Modular microservices; clear interfaces.  
- Model registry and versioning; reproducible training via MLflow.  
- IaC with Helm; CI/CD automated tests and linting.

## 8. Technical Requirements
### 8.1 Technical Stack
- Language/Runtime: Python 3.11+, Node.js 20+, TypeScript 5+, Go 1.22 (optional for streaming gateway).  
- Backend: FastAPI 0.115+, Uvicorn 0.30+, Celery 5.4+, Redis 7.2, Kafka 3.6 (events).  
- ML: PyTorch 2.3+, TorchVision 0.18+, TorchAudio 2.3+, Transformers 4.44+, timm 1.0+, OpenCV 4.9+, ffmpeg 6.1+.  
- Inference: ONNX Runtime 1.19+, TorchScript, TensorRT (optional), Optimum.  
- Frontend: React 18+, Next.js 14+, TailwindCSS 3+, Recharts.  
- Storage: PostgreSQL 15+, S3-compatible object storage (e.g., AWS S3/MinIO), ElasticSearch/OpenSearch 2.x for logs/search.  
- Orchestration: Docker 25+, Kubernetes 1.30+, Helm 3.14+.  
- MLOps: MLflow 2.14+, Weights & Biases (optional), DVC 3.x, Great Expectations 1.x.  
- Auth: OAuth2.1/OIDC, Keycloak/Auth0.  
- Monitoring: Prometheus 2.53+, Grafana 11+, OpenTelemetry 1.8+.

### 8.2 AI/ML Components
- Spatial branch: EfficientNet-B4, XceptionNet, and ViT/DeiT models trained on patch-level artifacts.  
- Temporal branch: TimeSformer-B, SlowFast, and R(2+1)D on 16–64 frame windows capturing blink/pose/motion inconsistencies.  
- Frequency branch: CNN on high-pass/DCT residuals for upsampling/compression artifacts.  
- Audio branch: Spectrogram CNN (ResNet34), wav2vec2-base, HuBERT for cloned voice; prosody/phoneme duration features.  
- AV-consistency: Audio-visual transformer with cross-attention for lip-sync and phoneme-viseme mismatches.  
- Fusion: Stacking ensemble with logistic regression/MLP; temperature scaling for calibration.  
- Training: Focal loss, label smoothing, mixup/cutmix, curriculum (easy→hard), adversarial hard-negative mining, domain adaptation.  
- Datasets: FaceForensics++, DFDC, Celeb-DF, DeeperForensics, WildDeepfake, synthetic data generation pipeline.  
- Metrics: Accuracy, Precision/Recall/F1, ROC-AUC, PR-AUC, EER, calibration error (ECE), DET curves.

## 9. System Architecture
### 9.1 High-Level Architecture (ASCII diagram)
Clients (Web UI, SDKs, Integrations)
        |
   [API Gateway]
        |
  +-----+-------------------------------+
  |             Backend                 |
  |  - Auth Service (OIDC, RBAC)       |
  |  - Ingestion Service                |
  |  - Job Orchestrator (Celery/Kafka) |
  |  - Detection Service (Inference)   |
  |  - Fusion & Calibration            |
  |  - Explanations Service            |
  |  - Reporting & Export              |
  +-----+-----------+------------------+
        |           |
   [PostgreSQL]   [Object Storage (S3)]
        |           |
     [Metadata]   [Media Blobs, Artifacts]
        |
   [Model Registry (MLflow)]
        |
   [Training Pipelines] <---- [Data Lake / DVC]
        |
   [Monitoring & Drift Detection]
        |
   [Analytics (Prometheus/Grafana/ELK)]

### 9.2 Component Details
- API Gateway: Terminates TLS, rate limits, routes to microservices; supports REST and WebSockets.  
- Ingestion: Validates media, deduplicates via perceptual hashes, stores to object storage, enqueues job.  
- Detection: Preprocessing (ffmpeg/OpenCV, face alignment, VAD) then model inference using GPU accelerator; supports batch and streaming.  
- Fusion & Calibration: Combines branch scores; applies temperature scaling; domain-specific thresholds.  
- Explanations: Generates Grad-CAM/frequency heatmaps/temporal attention plots; overlays for UI.  
- Reporting: Aggregates results, compiles reports, optional C2PA validation.  
- Model Registry: Tracks versions, signatures, metrics, lineage.  
- Training Pipelines: Data curation, augmentation, training, validation, packaging, and deployment.  
- Monitoring: Metrics, logs, traces; drift detection by domain and trigger retraining.

### 9.3 Data Flow
1) Client uploads media or provides URL/stream.  
2) Ingestion validates, stores blob, creates Job.  
3) Preprocessing produces frames/audio chunks and aligned crops.  
4) Branch models run inference; partial outputs emitted for streaming.  
5) Fusion produces final score and label; calibration applied.  
6) Explanations generated; artifacts stored.  
7) Results persisted; callbacks/webhooks fired.  
8) Review UI presents result; feedback may be used for active learning.  
9) Metrics/logs streamed to monitoring; drift signals feed MLOps.

## 10. Data Model
### 10.1 Entity Relationships
- User (1—N) APIKeys  
- User (1—N) Reviews  
- Media (1—N) Jobs  
- Job (1—1) InferenceResult  
- Job (1—N) BranchScores  
- InferenceResult (1—N) ExplanationArtifacts  
- ModelVersion (1—N) InferenceResult  
- Dataset (1—N) Samples  
- AuditLog linked to User and Action targets (Media/Job/Settings)

### 10.2 Database Schema (PostgreSQL)
- users(id PK, email, name, role, org_id, created_at)  
- api_keys(id PK, user_id FK, key_hash, scopes, created_at, revoked_at)  
- media(id PK, owner_id FK, uri, content_type, hash, size_bytes, created_at, retention_ttl)  
- jobs(id PK, media_id FK, status, priority, created_at, started_at, finished_at, error)  
- inference_results(id PK, job_id FK, label, score, confidence_low, confidence_high, model_version_id FK, threshold_profile_id FK)  
- branch_scores(id PK, inference_result_id FK, branch_name, score, details JSONB)  
- explanation_artifacts(id PK, inference_result_id FK, type, s3_path, preview_uri, metadata JSONB)  
- model_versions(id PK, name, version, registry_uri, metrics JSONB, created_at)  
- threshold_profiles(id PK, name, domain, params JSONB)  
- reviews(id PK, job_id FK, reviewer_id FK, decision, notes, created_at)  
- audit_logs(id PK, actor_id FK, action, target_type, target_id, details JSONB, created_at)  
- webhooks(id PK, org_id FK, url, secret, events, created_at, active)  
- datasets(id PK, name, source, license, split_info JSONB, created_at)

### 10.3 Data Flow Diagrams
- Ingestion: Client -> API -> Object Storage (media) -> Job row -> Queue -> Detection service.  
- Inference: Detection -> Preproc outputs (temp storage) -> Models -> BranchScores -> Fusion -> InferenceResult + Explanations -> Object Storage.  
- Review: UI -> API -> fetch InferenceResult + Artifacts; Review saved -> AuditLog.

### 10.4 Input Data & Dataset Requirements
- Video: mp4, webm, mov, mkv; 240p–1080p; up to 10 min per job (configurable).  
- Image: jpg, png, webp; up to 20 MB.  
- Audio: wav, mp3, m4a, ogg; 8–48 kHz; mono/stereo.  
- Datasets: FaceForensics++, DFDC, Celeb-DF, DeeperForensics, WildDeepfake; ensure licensing compliance.  
- Augmentations: Re-encoding at varying bitrates, resizing, cropping, Gaussian noise/blur, color jitter, packet loss simulation, compression artifacts.  
- Labels: binary (real/fake) with manipulation type metadata when available.

## 11. API Specifications
### 11.1 REST Endpoints
- POST /v1/media  
  - Upload media (multipart) or by URL. Returns media_id.
- POST /v1/detect  
  - Body: { media_id or url, mode: "image"|"video"|"audio"|"auto", callback_url?, threshold_profile? }  
  - Returns job_id.
- GET /v1/jobs/{job_id}  
  - Returns status and result if complete.
- GET /v1/reports/{job_id}  
  - Returns JSON report; ?format=pdf for binary.
- GET /v1/models  
  - List deployed model versions and metrics.
- POST /v1/feedback  
  - Submit human label for a job: {job_id, label, notes}.
- POST /v1/webhooks  
  - Register webhook with events: ["job.completed","job.failed"].
- GET /v1/health  
  - Liveness/readiness probes.
- Auth: OAuth2 with scopes: detect:read, detect:write, admin:*.

Streaming (optional):  
- WebSocket /v1/stream/detect (video/audio chunks; returns rolling scores).  
- gRPC interface for low-latency streaming.

### 11.2 Request/Response Examples
curl upload:
curl -X POST https://api.deepfake.example.com/v1/media \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/path/video.mp4"

Response:
{ "media_id": "med_123", "uri": "s3://bucket/med_123" }

curl detect:
curl -X POST https://api.deepfake.example.com/v1/detect \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"media_id":"med_123","mode":"video","callback_url":"https://app.example.com/hook"}'

Response:
{ "job_id": "job_456", "status": "queued" }

curl job status:
curl https://api.deepfake.example.com/v1/jobs/job_456 \
  -H "Authorization: Bearer $TOKEN"

Response:
{
  "job_id": "job_456",
  "status": "completed",
  "result": {
    "label": "fake",
    "score": 0.93,
    "confidence_interval": [0.90, 0.95],
    "per_branch": [
      {"branch":"spatial_vit","score":0.88},
      {"branch":"temporal_timesformer","score":0.94},
      {"branch":"frequency_residual","score":0.96},
      {"branch":"audio_w2v2","score":0.15},
      {"branch":"av_consistency","score":0.91}
    ],
    "threshold_profile": "default_video",
    "model_version": "ensemble_v1.3.2",
    "explanations": {
      "saliency_map": "https://cdn.example.com/artifacts/job_456/frame_42_cam.png",
      "frequency_heatmap": "https://cdn.example.com/artifacts/job_456/frame_42_freq.png",
      "temporal_attention": "https://cdn.example.com/artifacts/job_456/attn.json"
    }
  }
}

Webhook payload (HMAC-SHA256 signed):
{
  "event": "job.completed",
  "job_id": "job_456",
  "timestamp": "2025-11-25T12:00:00Z",
  "signature": "hex...",
  "result": { ... }
}

### 11.3 Authentication
- OAuth2/OIDC with JWT access tokens (RS256).  
- API keys for service-to-service with scoped permissions and IP allowlisting.  
- HMAC verification for webhooks.  
- Support SSO (SAML/OIDC) for UI users and SCIM for provisioning.

## 12. UI/UX Requirements
### 12.1 User Interface
- Dashboard: job statuses, filters by label/score/date, domain calibration selector.  
- Detail View: video player with overlays (saliency/frequency maps), frame timeline with attention peaks, audio waveform with clone likelihood, AV desync markers.  
- Review Panel: decision controls, notes, thresholds override, reason codes.  
- Admin: model versions, threshold profiles, API keys, webhooks, audit logs.  
- Reports: export button for JSON/PDF.

### 12.2 User Experience
- Drag-and-drop upload; URL paste; progress feedback.  
- Real-time stream view shows rolling score; color-coded risk.  
- Explanations toggleable; tooltips explaining signals.  
- Keyboard shortcuts for navigation; bulk actions for batch reviews.

### 12.3 Accessibility
- WCAG 2.1 AA: color contrast, alt text, focus indicators, ARIA roles.  
- Captions and transcripts for tutorial content.  
- No reliance on color alone; haptic options where applicable.

## 13. Security Requirements
### 13.1 Authentication
- OIDC with MFA enforcement and session timeouts; refresh token rotation.

### 13.2 Authorization
- RBAC: roles (viewer, analyst, admin).  
- Resource scoping by organization; per-API key scopes.

### 13.3 Data Protection
- TLS 1.3 in transit; AES-256 at rest.  
- Encrypted object storage; server-side encryption with KMS.  
- Optional client-side encryption for sensitive media.  
- PII handling: data minimization, configurable retention TTLs, secure deletion.  
- Private-by-default: media not exposed publicly; signed URLs for artifacts.

### 13.4 Compliance
- SOC 2 Type II, ISO 27001 alignment.  
- GDPR/CCPA readiness: DSR workflows, consent and purpose limitation.  
- Audit logs immutable and retained per policy.  
- Vulnerability scanning and penetration testing.

## 14. Performance Requirements
### 14.1 Response Times
- Image inference p95 <300 ms per image.  
- Video window (16 frames at 224p crop) p95 <500 ms.  
- Audio 1s chunk p95 <200 ms.  
- Report generation <2 s.

### 14.2 Throughput
- 50 images/sec per inference node; 100 video windows/sec per GPU; 120 audio chunks/sec per node (reference A10-class GPU).  
- Streaming: up to 500 concurrent streams per cluster (scalable).

### 14.3 Resource Usage
- GPU memory per model shard: 2–8 GB; CPU preproc 1–2 cores per job.  
- Target cost per image <$0.002; per video minute <$0.02 at scale.

## 15. Scalability Requirements
### 15.1 Horizontal Scaling
- Kubernetes HPA/VPA based on GPU/CPU utilization and queue depth.  
- Stateless services for rapid scale-out; sharded model servers.

### 15.2 Vertical Scaling
- Larger GPU instances for batch jobs; memory-optimized nodes for training.

### 15.3 Load Handling
- Autoscale on QPS and stream count; backpressure with queue priorities.  
- Multi-region active-active deployment for HA.

## 16. Testing Strategy
### 16.1 Unit Testing
- Coverage >85% for preprocessing, fusion, calibration, API logic.  
- Synthetic fixtures for images/audio/video.

### 16.2 Integration Testing
- End-to-end pipeline with sample media; contract tests for APIs; idempotency tests.  
- Dataset stratified splits; prevent leakage.

### 16.3 Performance Testing
- Load tests with JMeter/Locust; latency and throughput under stress.  
- GPU warm/cold benchmarks; streaming soak tests.

### 16.4 Security Testing
- Static analysis (Bandit, Semgrep), dependency scanning.  
- DAST; secrets scanning; webhook signature verification tests.  
- RBAC and multi-tenant isolation tests.

## 17. Deployment Strategy
### 17.1 Deployment Pipeline
- GitHub Actions/GitLab CI: lint, test, build, containerize, scan, sign images (cosign), push to registry, Helm deploy.  
- MLflow model registry gates promotion to staging/prod.

### 17.2 Environments
- Dev (shared), Staging (mirrors prod), Prod (multi-region).  
- Sandbox environment for external integrators.

### 17.3 Rollout Plan
- Canary 5% traffic for 24h; monitor KPIs; progressive rollout to 100%.  
- Feature flags for new branches/models; gradual enablement.

### 17.4 Rollback Procedures
- Helm rollback to previous release; model version pinning.  
- Blue/green switch with traffic drain; database schema backward compatibility.

## 18. Monitoring & Observability
### 18.1 Metrics
- Model: AUROC, EER, F1, calibration error; per-domain breakdowns.  
- System: latency (p50/p95/p99), throughput, GPU/CPU utilization, queue depth.  
- Business: detection rate, false positive/negative rates, review time, conversion of appeals.  
- Drift: PSI on input distributions; performance by source/platform.

### 18.2 Logging
- Structured JSON logs with correlation IDs and trace IDs (OpenTelemetry).  
- Redact PII; sampling strategies for high volume.

### 18.3 Alerting
- On-call alerts for SLO breaches, queue backlog, error rate spikes.  
- Drift alerts when AUROC drops >3% week-over-week.

### 18.4 Dashboards
- Grafana: system health, latency, GPU usage.  
- Model performance board: AUROC/EER by domain/dataset and over time.  
- Ops: webhook success rates, job SLA compliance.

## 19. Risk Assessment
### 19.1 Technical Risks
- Concept drift due to new manipulation techniques.  
- Adversarial attacks targeting detector weaknesses.  
- Cross-domain generalization gaps; overfitting to known datasets.  
- Real-time constraints under peak loads.

### 19.2 Business Risks
- False positives causing user friction and reputational harm.  
- Regulatory scrutiny around automated decision-making.  
- Dataset licensing/compliance issues.  
- Vendor lock-in for cloud/GPU resources.

### 19.3 Mitigation Strategies
- Continuous hard-negative mining and active learning loop.  
- Open-set detection thresholds and abstain option; human review.  
- Cross-dataset validation and domain adaptation strategies.  
- Multi-cloud readiness; infrastructure as code; cost monitoring.  
- Transparency via explanations and appeals workflow.

## 20. Timeline & Milestones
### 20.1 Phase breakdown
- Phase 0 (2 weeks): Requirements, architecture, data governance.  
- Phase 1 (4 weeks): Data pipeline, dataset curation, baseline models (spatial/audio).  
- Phase 2 (6 weeks): Temporal, frequency, AV-consistency branches; fusion/calibration.  
- Phase 3 (4 weeks): APIs, UI, explainability overlays, reporting.  
- Phase 4 (4 weeks): Performance optimization, streaming, scalability, security hardening.  
- Phase 5 (2 weeks): Beta, feedback, bug bash, documentation.  
Total: ~18 weeks to GA.

### 20.2 Key Milestones
- M1: Data pipeline MVP complete (W4).  
- M2: Ensemble v1 achieves AUROC ≥0.94 on cross-dataset (W10).  
- M3: API v1 and UI alpha (W12).  
- M4: Streaming inference p95 <200 ms per audio chunk (W16).  
- M5: GA readiness review and compliance sign-off (W18).

Estimated Costs (first 6 months):
- Training: ~1,500 GPU-hours (~$25–$45k depending on provider).  
- Inference infra: ~$15k/month at 1M/day volume.  
- Storage/logging: ~$3k/month.  
- Personnel: 6 FTE (2 ML, 2 backend, 1 frontend, 1 DevOps).

## 21. Success Metrics & KPIs
### 21.1 Measurable targets
- Accuracy/F1 ≥ 0.90; ROC-AUC ≥ 0.95; EER ≤ 10% across cross-dataset validation.  
- Calibration ECE ≤ 0.05.  
- p95 latency: image <300 ms; video window <500 ms; audio chunk <200 ms.  
- Uptime ≥ 99.5%; job failure rate <0.1%.  
- False positive reduction ≥ 40% vs. baseline.  
- Analyst review time reduced by ≥ 30%.  
- Integration NPS ≥ 60 from developers.  
- Drift MTTR ≤ 2 weeks to restore KPIs after degradation.

## 22. Appendices & Glossary
### 22.1 Technical Background
- Forensic signals: frequency-domain artifacts (DCT/FFT), upsampling traces, color channel inconsistencies, boundary blending, rPPG cues, PRNU-like residuals, compression mismatch and noise patterns.  
- Temporal cues: blink rate anomalies, micro-expressions, head pose trajectories.  
- Multimodal consistency: lip-sync alignment and phoneme-viseme timing.  
- Calibration: temperature scaling to align predicted probabilities with empirical likelihood.  
- Open-set recognition: abstain when score uncertainty is high or OOD detected.

### 22.2 References
- FaceForensics++: https://github.com/ondyari/FaceForensics  
- DFDC: https://ai.facebook.com/datasets/dfdc/  
- Celeb-DF: https://github.com/yuezunli/celeb-deepfakeforensics  
- DeeperForensics: https://github.com/EndlessSora/DeeperForensics-1.0  
- WildDeepfake: https://github.com/deepfakeinthewild/deepfake-in-the-wild  
- XceptionNet: https://arxiv.org/abs/1610.02357  
- EfficientNet: https://arxiv.org/abs/1905.11946  
- TimeSformer: https://arxiv.org/abs/2102.05095  
- wav2vec 2.0: https://arxiv.org/abs/2006.11477  
- HuBERT: https://arxiv.org/abs/2106.07447  
- C2PA: https://c2pa.org/

### 22.3 Glossary
- AV-consistency: Agreement between audio and visual modalities (e.g., lip movements vs spoken phonemes).  
- Calibration: Adjusting model outputs to improve probability accuracy.  
- EER (Equal Error Rate): Operating point where false acceptance equals false rejection.  
- Grad-CAM: Technique to visualize important regions in inputs driving a CNN’s decision.  
- OOD (Out-of-Distribution): Inputs not represented in training distributions.  
- PRNU: Photo-response non-uniformity, sensor-level noise pattern useful for forensics.  
- ROC-AUC: Area under the receiver operating characteristic curve.

Repository Structure (proposed):
- notebooks/  
  - 01_data_exploration.ipynb  
  - 02_training_spatial.ipynb  
  - 03_training_temporal.ipynb  
  - 04_training_audio_av.ipynb  
  - 05_fusion_calibration.ipynb
- src/  
  - api/ (FastAPI routers)  
  - inference/ (preproc, models, fusion)  
  - streaming/ (WebSocket/gRPC)  
  - training/ (datasets, augmentations, trainers)  
  - explainability/ (gradcam, freq_maps, attention)  
  - utils/  
- configs/  
  - default.yaml  
  - thresholds/  
  - models/
- tests/  
  - unit/  
  - integration/  
  - performance/
- data/ (DVC tracked pointers)  
- scripts/ (deploy, eval, benchmarking)  
- mlruns/ (MLflow)  
- docs/ (OpenAPI, usage guides)

Sample Config (YAML):
service:
  max_concurrent_jobs: 1000
  retention_days: 30
models:
  spatial:
    name: efficientnet_b4
    input_size: 380
  temporal:
    name: timesformer_b
    window: 32
  frequency:
    name: freq_resnet18
  audio:
    name: wav2vec2_base
  fusion:
    type: logistic_regression
calibration:
  method: temperature_scaling
  per_domain: true
thresholds:
  default_video: 0.7
  default_audio: 0.75

Sample FastAPI Snippet (Python):
from fastapi import FastAPI, UploadFile, File, Depends
from pydantic import BaseModel
app = FastAPI()

class DetectRequest(BaseModel):
    media_id: str | None = None
    url: str | None = None
    mode: str = "auto"
    callback_url: str | None = None

@app.post("/v1/detect")
async def detect(req: DetectRequest, user=Depends(auth_guard)):
    job_id = enqueue_detection(req, user)
    return {"job_id": job_id, "status": "queued"}

This PRD specifies the complete scope, architecture, and execution plan to deliver Aiml034_Deepfake_Detection_System with high accuracy, low latency, explainability, and robust operations suitable for enterprise-grade deployment.