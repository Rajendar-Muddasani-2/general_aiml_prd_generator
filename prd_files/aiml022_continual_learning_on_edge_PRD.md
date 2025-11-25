# Product Requirements Document (PRD)
# `Aiml022_Continual_Learning_On_Edge`

Project ID: Aiml022_Continual_Learning_On_Edge
Category: General AI/ML – Continual/Online Learning on Edge
Status: Draft for Review
Version: 1.0
Last Updated: 2025-11-25

## 1. Overview
### 1.1 Executive Summary
Aiml022_Continual_Learning_On_Edge delivers a production-ready platform for continual learning on edge devices (mobile, embedded, gateways). It enables models to adapt to new data distributions in real time without centralizing sensitive data. The system combines resource-aware online learning, rehearsal buffers, drift detection, optional federated learning, and privacy-preserving techniques to update models on-device while maintaining performance guarantees, auditability, and safety controls.

### 1.2 Document Purpose
Define product requirements for engineering, ML, and product teams to build, deploy, and operate a scalable edge continual learning platform, including features, UX, technical architecture, APIs, security, testing, deployment, and KPIs.

### 1.3 Product Vision
Empower organizations to deliver intelligent, adaptive applications that continuously improve on-device while preserving privacy and resource budgets. Provide a unified MLOps control plane to orchestrate policies, model versions, telemetry, and federated aggregation—achieving robust accuracy under drift, low-latency inference (<500 ms), and 99.5%+ service uptime.

## 2. Problem Statement
### 2.1 Current Challenges
- Models degrade under concept drift/domain shift.
- Centralized retraining pipelines are slow, costly, and risky for privacy.
- Edge devices vary widely in compute, memory, and power budgets.
- Catastrophic forgetting during online updates.
- Lack of standardized on-device policies for updates, rollbacks, and telemetry.
- Inefficient communication for federated learning with intermittent connectivity.
- Limited visibility into edge adaptation quality, fairness, and privacy budgets.

### 2.2 Impact Analysis
- Reduced model accuracy leads to poor user experience and revenue loss.
- High data transfer costs and privacy concerns deter data centralization.
- Operational complexity increases support effort and time-to-fix.
- Compliance risks due to handling personal or sensitive data incorrectly.

### 2.3 Opportunity
- Build a safe, configurable continual learning stack enabling:
  - On-device adaptation with rehearsal and regularization to limit forgetting.
  - Privacy-preserving learning (differential privacy, federated, secure aggregation).
  - Automatic drift detection and active learning triggers.
  - Lightweight updates via adapters/LoRA and prototype/kNN heads.
  - Central control plane for visibility, governance, and experimentation.

## 3. Goals and Objectives
### 3.1 Primary Goals
- Deliver an SDK for on-device continual learning with:
  - Rehearsal memory, EWC-style regularization, knowledge distillation.
  - Drift/OOD detection and uncertainty calibration.
  - Adapter/LoRA-based parameter-efficient fine-tuning.
- Provide a cloud control plane:
  - Model registry, policy engine, telemetry, A/B testing, federated training.
  - Rollback, versioning, and OTA updates.
- Ensure privacy and compliance by default.

### 3.2 Business Objectives
- Reduce time-to-adapt to drift from months to hours/days.
- Lower data transfer/storage costs by ≥50% via on-device learning.
- Enable privacy-first deployments to unlock regulated markets.
- Improve user engagement/conversion metrics through better personalization.

### 3.3 Success Metrics
- Model accuracy: >90% on rolling window; forgetting ≤5%.
- Median on-device inference latency: <500 ms; on-device update epochs within budget.
- Uptime: ≥99.5% control plane.
- Privacy: DP epsilon per user ≤ 4 over 90 days where DP enabled.
- Federated round completion rate ≥95% with straggler tolerance.
- A/B improvement: ≥5% uplift in target KPI for adaptive variants.

## 4. Target Users/Audience
### 4.1 Primary Users
- Edge ML Engineers
- MLOps/Platform Engineers
- Data Scientists/ML Researchers

### 4.2 Secondary Users
- Product Managers
- Privacy/Compliance Officers
- Field Operations/SREs

### 4.3 User Personas
1) Name: Priya Nair, Edge ML Engineer
- Background: 7 years building mobile CV models for consumer apps; proficient in PyTorch, ONNX Runtime.
- Goals: Deploy adaptive models that keep up with changing scenes without cloud retraining; maintain battery and latency targets.
- Pain Points: Catastrophic forgetting; complex OTA updates; limited observability; constraints on device compute.
- Needs: SDK with rehearsal + regularization; policy-based updates; drift alerts; easy rollback.

2) Name: Lucas Martinez, Data Scientist
- Background: NLP personalization for voice assistants; experience with LoRA and prompt-tuning.
- Goals: Rapidly add new intents and user-specific vocabulary on-device.
- Pain Points: Privacy concerns over voice data; lack of standardized evaluation for continual learning; lab-to-prod gap.
- Needs: Few-shot adaptation with adapters; rolling-window evaluation metrics; federated option with DP.

3) Name: Mei Chen, Privacy & Compliance Officer
- Background: Legal/compliance in data protection frameworks (GDPR/CCPA).
- Goals: Ensure data stays on-device; audit trails and DP guarantees.
- Pain Points: Opaque model updates; unclear privacy budgets; insufficient user consent controls.
- Needs: Configurable DP; consent flows; audit logs; data purge and TTL policies.

4) Name: Aaron Price, SRE/Platform Engineer
- Background: Operates K8s, observability stacks; owns SLA/SLO.
- Goals: Reliable deployment, monitoring, and rollbacks; efficient scaling.
- Pain Points: Unpredictable edge connectivity; telemetry noise; federated stragglers.
- Needs: Robust pipeline, retries, backoff; alerting; hierarchical aggregation; device health dashboards.

## 5. User Stories
US-001: As an Edge ML Engineer, I want an on-device rehearsal buffer so that the model retains performance on previous classes.
- Acceptance: Buffer supports reservoir sampling; configurable cap (e.g., 500–5,000 examples); persistence across restarts; encryption at rest.

US-002: As a Data Scientist, I want to enable adapter-based updates (LoRA) so that fine-tuning fits within tight resource budgets.
- Acceptance: Adapter training runs within CPU-only budget of 2W average; achieves ≥85% of full fine-tune accuracy on validation.

US-003: As a PM, I want drift alerts when accuracy drops >5% over a 7-day rolling window so that we can trigger adaptation or rollback.
- Acceptance: Drift detector (ADWIN/DDM) flags events; alert in dashboard and webhook; recommended action shown.

US-004: As a Privacy Officer, I need DP-SGD enabled with per-user epsilon tracking so that privacy budgets are auditable.
- Acceptance: Per-user epsilon and delta reported; alarms if budget exceeds policy; DP can be toggled per segment.

US-005: As an SRE, I want OTA rollback to prior model version within 5 minutes so that we minimize bad impact from regressions.
- Acceptance: One-click rollback; device acknowledgment; monitoring sync.

US-006: As a Data Scientist, I want prototype/kNN heads for rapid class addition so that I can deploy new labels without full retraining.
- Acceptance: Add class via API; kNN head integrates with vector store; class live within 10 minutes; accuracy ≥80% for 5-shot.

US-007: As an Edge Engineer, I want uncertainty calibration so that active learning queries target low-confidence examples.
- Acceptance: Temperature scaling improves ECE by ≥30% from baseline; active learning budget respected.

US-008: As a Platform Engineer, I want federated training with secure aggregation so that no raw gradients are deanonymized.
- Acceptance: Secure aggregation protocol; client sampling; compressed updates; per-round completion ≥90% with dropouts.

US-009: As a PM, I want A/B testing for adaptive vs. static so that we can quantify uplift.
- Acceptance: Randomized assignment, guardrails; statistical significance reporting; segment breakdowns.

US-010: As a User, I want control to opt-in/out and purge local data so that I maintain privacy.
- Acceptance: In-app controls; purge command clears buffers and indices; audit log updated.

## 6. Functional Requirements
### 6.1 Core Features
FR-001: On-device continual learning engine with modes: task-, class-, domain-incremental.
FR-002: Catastrophic forgetting mitigation: EWC/SI regularization; rehearsal buffer with reservoir/coreset; knowledge distillation (LwF).
FR-003: Lightweight adaptation: Adapters/LoRA/QLoRA; selective layer freezing; prototype/kNN head on top of frozen backbone.
FR-004: Drift detection: ADWIN/DDM; OOD detection via energy score/Mahalanobis; uncertainty calibration via temperature scaling.
FR-005: Privacy: DP-SGD with clipping/noise; per-user privacy budgets; local-only personalization layers.
FR-006: Optional federated learning: FedAvg/FedProx; client sampling; secure aggregation; compressed updates (Top-k, quantization).
FR-007: Vector memory: On-device ANN (HNSW) for rehearsal and few-shot classification; eviction policies (LRU/age/importance).
FR-008: Policy engine: Update schedules/budgets; network/battery constraints; consent gating; guardrails and rollbacks.
FR-009: Telemetry: Metrics, events, drift signals, update outcomes; offline caching; batched uploads.
FR-010: Control plane: Model registry, experiment manager, A/B testing, dashboard, APIs.
FR-011: OTA delivery: Model/adapters/configs; differential/patch updates; rollback.
FR-012: Data governance: PII filtering before embedding; TTL, purge, and audit trail.

### 6.2 Advanced Features
- FR-013: Generative replay via small generative head to augment rehearsal when buffer is small.
- FR-014: Resource-aware training: Mixed precision, gradient checkpointing, update sparsification, micro-batching.
- FR-015: Hybrid retrieval (semantic + keyword/metadata); re-ranking with lightweight cross-encoder.
- FR-016: Hierarchical federated aggregation (edge gateways aggregate local devices).
- FR-017: Rolling-window evaluation and automated retraining triggers.
- FR-018: Battery/thermal-aware throttling and scheduled updates (e.g., idle, charging, Wi-Fi).
- FR-019: Fairness monitoring across segments; bias drift alerts.

## 7. Non-Functional Requirements
### 7.1 Performance
- Inference latency (P50): <500 ms; (P90): <800 ms on target devices.
- On-device update duration: policy-configured; e.g., max 2 minutes burst or 10 minutes/day cumulative.
- Telemetry ingestion API P95: <200 ms.

### 7.2 Reliability
- Control plane uptime: ≥99.5% monthly.
- OTA success rate: ≥98% with retries/backoff.
- Federated round completion: ≥95% with straggler tolerance.

### 7.3 Usability
- Dashboard tasks accomplished in ≤3 clicks (assign policy; rollback; view drift).
- SDK integration in <1 day with sample apps.

### 7.4 Maintainability
- Clean modular code structure, 85% unit test coverage for core components.
- Backwards-compatible APIs (semantic versioning).

## 8. Technical Requirements
### 8.1 Technical Stack
- Languages: Python 3.11+, TypeScript 5.6+
- Backend: FastAPI 0.115+, Uvicorn 0.30+, PostgreSQL 15+, Redis 7+, Kafka 3.6+, MinIO (S3-compatible) RELEASE.2024+
- Frontend: React 18.3+, Next.js 14+, Material UI 6+
- ML: PyTorch 2.4+, TorchVision/TorchAudio; ONNX Runtime 1.18+; Optimum/PEFT 0.13+; FAISS 1.8+
- Privacy: Opacus 1.4+ (DP-SGD)
- Federated: Flower 1.8+ or FedML 0.8+
- Observability: OpenTelemetry 1.26+, Prometheus 2.53+, Grafana 11+, ELK/OpenSearch 2.13+
- Packaging: Docker 26+, Kubernetes 1.30+, Helm 3.14+, Argo CD 2.11+
- CI/CD: GitHub Actions, pytest, tox, pre-commit

### 8.2 AI/ML Components
- Backbones: MobileNetV3/EfficientNet-Lite (vision), DistilBERT/MiniLM (NLP), small Temporal CNN/Transformer (time-series).
- Heads: Prototype/kNN classifier; softmax classifier.
- Adaptation: Adapters/LoRA/QLoRA with rank configurable (r=4–16).
- Losses: Cross-entropy, contrastive (self-supervised), EWC penalty, distillation loss.
- Optimizers: AdamW, SGD with momentum; cosine decay; EMA optional.
- Drift/OOD: ADWIN, DDM; energy score-based OOD; temperature scaling for calibration.

## 9. System Architecture
### 9.1 High-Level Architecture (ASCII)
Edge Device(s)                    Cloud Control Plane
+------------------+              +------------------------------------+
| On-device App    |              | API Gateway (FastAPI)              |
|  - Inference     | <----TLS---->|  - Authn/Authz (OAuth2/OIDC)       |
|  - Continual     |              |  - Device Registry                 |
|    Learning SDK  |              |                                    |
|  - Vector Memory |----TLS-----> | Telemetry Ingest (Kafka)           |
|  - Drift/OOD     |              |  - Stream Processing (Flink/Spark) |
|  - Privacy/DP    |              |                                    |
|  - OTA Updater   | <----TLS---->| Model Registry (S3+Postgres)       |
|  - Policy Agent  |              |  - Versioning/Artifacts            |
+------------------+              |                                    |
         ^                        | Federated Coordinator (Flower)     |
         |                        |  - Secure Aggregation              |
  A/B, Policies                   |  - Client Sampling                 |
         |                        |                                    |
         v                        | Monitoring (Prom, Grafana, ELK)    |
+------------------+              +------------------------------------+

### 9.2 Component Details
- On-device SDK: Inference wrapper, online trainer, rehearsal buffer, OOD/drift detector, vector index (HNSW), OTA client, policy executor, privacy/DP module.
- Control plane services:
  - API Gateway: REST APIs, device auth, rate limiting.
  - Model Registry: Store models, adapters, configs; metadata (semver, tags).
  - Telemetry Ingest: Kafka topics for metrics/events; processors compute drift, rolling accuracy.
  - Federated Coordinator: Schedules rounds, aggregates updates, enforces secure aggregation and compression.
  - Experiment Manager: A/B assignments and guardrails.
  - Policy Engine: Schedules updates based on device status and constraints.
  - Observability: Metrics/logs/traces pipelines and dashboards.

### 9.3 Data Flow
1) Inference -> predictions + confidence -> local telemetry -> batched upload.
2) Drift detection triggers local adaptation using rehearsal + EWC/LoRA within policy budgets.
3) Optional active learning requests labels (human-in-the-loop), updates memory.
4) Periodically, federated rounds request model delta; secure aggregate; new global model available.
5) OTA client fetches new model/adapters/config; validates; applies; can rollback.
6) Control plane computes rolling metrics, alerts, and suggests actions.

## 10. Data Model
### 10.1 Entity Relationships
- Organization 1—N Device
- Device N—N ModelVersion (via Assignment)
- ModelVersion 1—N Artifact (model, adapter, config)
- Device 1—N TelemetryEvent
- Device 1—N DriftEvent
- Device 1—N ReplayBufferIndexEntry
- Experiment 1—N Assignment
- FederatedRound 1—N ClientReport
- User 1—N AuditLog
- Policy 1—N Device

### 10.2 Database Schema (PostgreSQL)
Table: users
- id (uuid, pk), email (unique), role (enum: admin, engineer, scientist, privacy, viewer), org_id (fk), created_at

Table: organizations
- id (uuid, pk), name, created_at

Table: devices
- id (uuid, pk), org_id (fk), platform (enum: android, ios, linux, other), app_version, sdk_version, status, last_seen_at, dp_enabled (bool), privacy_epsilon (float), battery_level (int)

Table: models
- id (uuid, pk), name, task_type (vision,nlp,time_series), backbone, created_by, created_at

Table: model_versions
- id (uuid, pk), model_id (fk), version (semver), artifact_uri, adapter_uri, config_uri, checksum, created_at, status (active, deprecated)

Table: assignments
- id (uuid, pk), device_id (fk), model_version_id (fk), experiment_id (fk nullable), assigned_at, rollout_policy

Table: telemetry_events
- id (bigserial), device_id (fk), timestamp, event_type (inference, update, drift, error), payload (jsonb), metrics (jsonb), session_id

Table: drift_events
- id (bigserial), device_id (fk), timestamp, detector (adwin, ddm, energy), severity (low/med/high), window_stats (jsonb)

Table: policies
- id (uuid, pk), name, config (jsonb) [e.g., update windows, budget, consent required], created_at, org_id (fk)

Table: federated_rounds
- id (bigserial), model_version_id (fk), round_no, aggregator (fedavg,fedprox), started_at, completed_at, metrics (jsonb)

Table: client_reports
- id (bigserial), federated_round_id (fk), device_id (fk), update_uri, compressed (bool), dp_epsilon (float), duration_ms, status

Table: audit_logs
- id (bigserial), user_id (fk), device_id (fk nullable), action, details (jsonb), timestamp

Table: vector_index_entries
- id (bigserial), device_id (fk), embedding (bytea pq/scalar quantized), label, metadata (jsonb), created_at, ttl_at

### 10.3 Data Flow Diagrams (ASCII)
[Inference]
Input -> Backbone -> Head -> Prediction -> Telemetry queue -> Upload

[Online Update]
New sample -> OOD? -> if in-domain -> store in buffer -> compute gradients (adapter params) -> apply -> update vector index -> log

[Federated]
Scheduler -> Sample devices -> Send task -> Receive compressed updates -> Secure aggregate -> New global -> Publish -> OTA

### 10.4 Input Data & Dataset Requirements
- Supports image frames, text utterances, and time-series sensor data.
- On-device preprocessing/augmentation; deduplication and label drift tracking.
- Optional self/weak supervision: contrastive pairs, BYOL/SimSiam-style augmentations.
- Metadata required: timestamp, source, label confidence, consent flag.

## 11. API Specifications
### 11.1 REST Endpoints
- POST /v1/auth/token: obtain OAuth2 access token
- GET /v1/devices: list devices
- POST /v1/devices/register: register a device (mTLS required)
- GET /v1/models: list models
- POST /v1/models: create model
- POST /v1/models/{model_id}/versions: upload a new version
- POST /v1/assignments: assign model version to device(s)
- GET /v1/policies: list policies
- POST /v1/policies: create/update policy
- POST /v1/telemetry: ingest telemetry batch
- GET /v1/drift/events: query drift events
- POST /v1/federated/rounds: start round
- POST /v1/federated/rounds/{round_id}/reports: upload client report
- POST /v1/ota/rollback: rollback devices to previous version
- POST /v1/privacy/purge: purge device local data (via command)

### 11.2 Request/Response Examples
Example: Assign model version
Request (JSON):
{
  "device_ids": ["3f7c...","a4b1..."],
  "model_version_id": "8f21...",
  "rollout_policy": {"percent": 10, "canary": true, "max_error_rate": 0.05}
}
Response:
{"assignment_id":"c7a9...","status":"scheduled"}

Example: Telemetry ingest
Request:
{
  "device_id":"3f7c...",
  "events":[
    {"timestamp":"2025-11-25T12:00:00Z","event_type":"inference","metrics":{"latency_ms":132,"conf":0.82}},
    {"timestamp":"2025-11-25T12:00:05Z","event_type":"drift","payload":{"detector":"adwin","severity":"high"}}
  ]
}
Response:
{"accepted":2}

Example: Federated report upload (multipart)
- fields: round_id, device_id, dp_epsilon, duration_ms
- file: update.bin (compressed)

### 11.3 Authentication
- OAuth2/OIDC for users and services; roles/permissions (RBAC).
- mTLS for device registration and OTA; device tokens rotated every 30 days.
- JWT with short-lived access; refresh via secure channel.

## 12. UI/UX Requirements
### 12.1 User Interface
- Pages: Devices, Models & Versions, Experiments (A/B), Policies, Federated Rounds, Drift & Alerts, Privacy & Audit, Settings.
- Device detail: status, assigned version, local metrics, privacy epsilon, buffer utilization, last drift events.
- Model version detail: artifacts, adapters, metrics, rollout status, forget/transfer metrics.

### 12.2 User Experience
- Clear wizards to set update budgets (time/compute/network) and consent requirements.
- Contextual help for continual learning metrics (forgetting, transfer).
- One-click rollback and federated round restart.

### 12.3 Accessibility
- WCAG 2.1 AA compliance: keyboard navigation, contrast, ARIA labels.
- Localized UI strings; timezone and locale support.

## 13. Security Requirements
### 13.1 Authentication
- OIDC integration (e.g., Okta/Azure AD); MFA; device mTLS.

### 13.2 Authorization
- RBAC with least privilege. Roles: admin, engineer, scientist, privacy, viewer.
- Fine-grained permissions: device control, policy management, data export.

### 13.3 Data Protection
- TLS 1.3 in transit; AES-256 encryption at rest (server and on-device storage).
- Key rotation every 90 days; HSM/KMS-backed keys.
- DP-SGD privacy budgets; PII filtering pre-embedding; audit logs immutable.

### 13.4 Compliance
- GDPR, CCPA, SOC 2 readiness: data minimization, consent tracking, right to be forgotten via purge API.
- Data residency configurations per org.

## 14. Performance Requirements
### 14.1 Response Times
- API P95 <200 ms under 1k RPS; P99 <400 ms.
- Dashboard load P95 <2 s.

### 14.2 Throughput
- Telemetry ingest sustained 10k events/sec; burst 50k/sec.
- Federated coordinator supports 10k active devices per round per cluster.

### 14.3 Resource Usage
- On-device memory for SDK <150 MB including vector index (configurable).
- CPU load average for background updates <20% during policy window; thermal throttling aware.

## 15. Scalability Requirements
### 15.1 Horizontal Scaling
- Kubernetes HPA for API and ingest services; Kafka partitions scale-out.
- Sharded registries and caching (Redis) for model artifacts.

### 15.2 Vertical Scaling
- Federated aggregator nodes with more CPU/RAM as device counts grow.
- Model registry storage expansion via S3-compatible tiering.

### 15.3 Load Handling
- Backpressure via Kafka; retries with exponential backoff; idempotent endpoints.
- Rate limiting per device and per org.

## 16. Testing Strategy
### 16.1 Unit Testing
- SDK components: rehearsal buffer, EWC penalty, OOD detection, vector index ops.
- Backend: API handlers, policy enforcement, registry.

### 16.2 Integration Testing
- End-to-end with Docker Compose: API + Kafka + Postgres + MinIO.
- Simulated devices perform updates and telemetry.

### 16.3 Performance Testing
- Locust/Gatling for API; producer load tests for Kafka; on-device microbenchmarks for latency and update duration.

### 16.4 Security Testing
- SAST/DAST; dependency scanning; mTLS validation; authz tests; DP leakage testing via membership inference attack simulations.

## 17. Deployment Strategy
### 17.1 Deployment Pipeline
- GitHub Actions: lint, unit tests, build Docker images, SBOM, push to registry.
- Argo CD for GitOps-based deploys to staging/prod.

### 17.2 Environments
- Dev (shared), Staging (prod-parity), Production (multi-region).
- Edge sandbox fleet for pre-production field tests.

### 17.3 Rollout Plan
- Canary: 1% -> 10% -> 50% -> 100% with guardrails (error rate, drift severity).
- Feature flags for enabling on-device updates per cohort.

### 17.4 Rollback Procedures
- Automatic: if guardrails breach; rollback within 5 minutes.
- Manual: one-click via dashboard; audit logged.

## 18. Monitoring & Observability
### 18.1 Metrics
- Edge: latency, accuracy proxies, update duration, buffer size, battery impact.
- Control plane: API latency/error rates, telemetry lag, federated round durations.
- CL metrics: forgetting, forward/backward transfer, intransigence, ECE.

### 18.2 Logging
- Structured JSON logs; device logs sampled and redacted; correlation IDs.

### 18.3 Alerting
- Pager alerts on SLA breaches; drift severity high; federated failures; DP budget overrun.
- Webhooks/Slack for product-level alerts.

### 18.4 Dashboards
- Executive overview (uptime, adoption).
- ML quality (rolling accuracy, forgetting, drift).
- Federated rounds (participation, times, metrics).
- Privacy (epsilon histograms, purge events).

## 19. Risk Assessment
### 19.1 Technical Risks
- Catastrophic forgetting under extreme drift.
- On-device resource exhaustion (memory/thermal).
- Update instability and version skew across devices.
- Federated stragglers and non-IID client drift.
- DP noise harming utility.

### 19.2 Business Risks
- Regulatory non-compliance without robust controls.
- Poor user experience due to battery drain or performance regressions.
- Insufficient uplift vs. static models.

### 19.3 Mitigation Strategies
- Combine rehearsal + EWC + distillation; kNN fallback.
- Strict update budgets; thermal/battery-aware scheduling.
- Strong versioning and rollback; compatibility shims for embeddings.
- Hierarchical aggregation and client sampling; proximal regularization (FedProx).
- Adaptive DP (per-layer clipping); segment-based tuning of noise.

## 20. Timeline & Milestones
### 20.1 Phase breakdown
- Phase 0 (Weeks 1–2): Requirements finalization, architecture design.
- Phase 1 (Weeks 3–8): SDK core (inference wrapper, rehearsal, EWC, OOD, LoRA); basic dashboard and APIs.
- Phase 2 (Weeks 9–14): Policy engine, OTA, telemetry pipeline, vector memory, calibration; initial staging tests.
- Phase 3 (Weeks 15–18): Federated coordinator with secure aggregation; DP-SGD integration; A/B testing module.
- Phase 4 (Weeks 19–22): Scalability hardening, performance tuning, multi-platform SDK packages.
- Phase 5 (Weeks 23–24): Security audit, compliance review, docs/training; GA release.

### 20.2 Key Milestones
- M1 (Week 4): On-device rehearsal + EWC demo hits ≤5% forgetting on synthetic stream.
- M2 (Week 8): P50 inference <500 ms on reference device; OTA update end-to-end.
- M3 (Week 14): Drift alerts + auto-adapt policies; vector memory few-shot add class live.
- M4 (Week 18): Federated round across 1k simulated devices; secure aggregation.
- M5 (Week 22): Load tests pass 10k events/s ingest; API P95 <200 ms.
- GA (Week 24): 99.5% uptime SLO readiness; docs complete.

Estimated Cost (6 months):
- Team: 1 PM, 3 ML Eng, 3 Backend Eng, 1 Frontend, 1 SRE, 0.5 Privacy = ~$1.8M fully loaded.
- Cloud: ~$8k/month staging+prod; test devices/fleet sims: ~$3k.

## 21. Success Metrics & KPIs
### 21.1 Measurable targets
- Accuracy: >90% rolling window across target tasks; forgetting ≤5%.
- Latency: P50 <500 ms; P90 <800 ms on target devices.
- Uptime: Control plane ≥99.5%.
- Privacy: DP epsilon ≤4 per user per 90 days where enabled.
- Adoption: ≥500 active devices in first quarter; ≥10 orgs enabled.
- Federated: ≥95% round completion; <10% straggler penalty on wall time.
- Business: ≥5% uplift in target KPI (e.g., CTR or task success) for adaptive cohort.

## 22. Appendices & Glossary
### 22.1 Technical Background
- Continual learning settings: task-, class-, domain-incremental.
- Forgetting mitigation: EWC/SI, rehearsal buffers, LwF distillation, parameter isolation methods.
- Lightweight updates: adapters/LoRA/QLoRA; kNN/prototype heads; prompt/prefix tuning for NLP.
- Drift/OOD: ADWIN, DDM; energy-based OOD; temperature scaling for calibration.
- Privacy: DP-SGD with clipping; federated learning with secure aggregation; local-only personalization.
- Vector memory: On-device ANN (HNSW); product/scalar quantization; eviction policies.

### 22.2 References
- Kirkpatrick et al., “Overcoming catastrophic forgetting (EWC).”
- Lopez-Paz & Ranzato, “Gradient episodic memory.”
- Rebuffi et al., “iCaRL: Incremental Classifier and Representation Learning.”
- Li & Hoiem, “Learning without Forgetting (LwF).”
- Hu et al., “LoRA: Low-Rank Adaptation of Large Language Models.”
- Bifet & Gavalda, “ADWIN: Adaptive Windowing.”
- Goodfellow et al., “Deep learning” – optimization and regularization chapters.
- Flower, Opacus, FAISS official docs.

### 22.3 Glossary
- Continual Learning: Training paradigm where the model updates continuously with new data without full retraining.
- Catastrophic Forgetting: Loss of performance on previously learned tasks when learning new ones.
- Rehearsal Buffer: Memory of prior samples used during updates to retain old knowledge.
- EWC (Elastic Weight Consolidation): Regularization penalizing changes to important parameters.
- LoRA/QLoRA: Parameter-efficient fine-tuning via low-rank updates (optionally quantized).
- OOD (Out-of-Distribution): Inputs drawn from a different distribution than training data.
- Differential Privacy (DP): Framework to bound privacy leakage during learning.
- Federated Learning: Training across devices without centralizing raw data.
- Prototype/kNN Head: Classifier using nearest neighbors in embedding space.
- Secure Aggregation: Protocol to combine client updates without revealing individual contributions.

-----------------------------------------
Repository Structure
- notebooks/
  - experiments_continual_learning.ipynb
  - rehearsal_vs_ewc_ablation.ipynb
- src/
  - sdk/
    - inference.py
    - trainer_online.py
    - rehearsal_buffer.py
    - drift/
      - adwin.py
      - ddm.py
      - ood_energy.py
    - privacy/
      - dp_sgd.py
      - consent.py
    - adapters/
      - lora.py
      - qlora.py
    - vector_memory/
      - hnsw_index.py
      - quantization.py
    - policy/
      - agent.py
      - scheduler.py
    - ota/
      - client.py
    - utils/
      - metrics.py
      - calibration.py
  - server/
    - app.py (FastAPI)
    - routers/
      - devices.py
      - models.py
      - policies.py
      - telemetry.py
      - federated.py
      - auth.py
    - services/
      - registry.py
      - assignments.py
      - drift_service.py
      - ab_testing.py
      - secure_agg.py
    - models/ (ORM)
      - schemas.py
      - db.py
  - ml/
    - backbones/
      - mobilenetv3.py
      - distilbert.py
    - heads/
      - knn.py
      - softmax.py
    - loss/
      - ewc.py
      - distill.py
- tests/
  - unit/
  - integration/
  - perf/
- configs/
  - sdk_config.yaml
  - policy_defaults.yaml
  - server_config.yaml
- data/
  - sample_images/
  - sample_text/
- scripts/
  - run_federated_sim.py
  - load_test_kafka.py
- deployment/
  - docker/
  - helm/

Sample SDK Config (configs/sdk_config.yaml)
device:
  id: "auto"
  dp_enabled: true
  privacy_epsilon_budget: 4.0
inference:
  batch_size: 1
  fp16: true
continual_learning:
  mode: "class_incremental"
  rehearsal:
    max_items: 2000
    strategy: "reservoir"
  regularization:
    ewc_lambda: 0.4
  adapters:
    lora:
      rank: 8
      alpha: 16
      dropout: 0.05
update_budget:
  max_minutes_per_day: 10
  schedule: ["22:00-06:00"]
drift_detection:
  method: "adwin"
  sensitivity: "medium"
vector_memory:
  index: "hnsw"
  dim: 256
  max_items: 5000
  quantization: "pq"
telemetry:
  batch_max: 100
  upload_interval_sec: 60

Example On-Device Training Snippet (Python)
from sdk.trainer_online import OnlineTrainer
from sdk.adapters.lora import LoRAAdapter
from sdk.rehearsal_buffer import RehearsalBuffer
from sdk.drift.adwin import ADWIN
from sdk.utils.metrics import rolling_accuracy

buffer = RehearsalBuffer(max_items=2000, strategy="reservoir", encrypted=True)
adapter = LoRAAdapter(target_modules=["attn","ffn"], rank=8)
detector = ADWIN(delta=0.002)

trainer = OnlineTrainer(backbone="mobilenetv3_small", head="knn", adapter=adapter,
                        rehearsal_buffer=buffer, ewc_lambda=0.4, mixed_precision=True)

for x, y in streaming_data():
    pred, conf = trainer.infer(x)
    if detector.update(1 if pred==y else 0) and detector.is_change_detected():
        trainer.update(x, y)  # triggers bounded update per policy
    trainer.telemetry.log_inference(latency_ms=trainer.last_latency_ms, conf=conf)

REST API with FastAPI (server/app.py excerpt)
from fastapi import FastAPI, Depends
from routers import devices, models, policies, telemetry, federated, auth

app = FastAPI(title="aiml022 Control Plane", version="1.0")
app.include_router(auth.router, prefix="/v1/auth")
app.include_router(devices.router, prefix="/v1/devices")
app.include_router(models.router, prefix="/v1/models")
app.include_router(policies.router, prefix="/v1/policies")
app.include_router(telemetry.router, prefix="/v1/telemetry")
app.include_router(federated.router, prefix="/v1/federated")

ASCII Architecture: Data Flow Detail
+Device SDK+
 [Capture] -> [Preprocess] -> [Backbone] -> [Head] -> [Pred]
             -> [OOD/Drift] -> {Adapt? yes} -> [Adapters Update]
             -> [Rehearsal Memory Update]
             -> [Telemetry Buffer] -> [Batch Upload]

+Cloud+
 [API] -> [Kafka] -> [Stream Proc] -> [Metrics/Drift Alerts] -> [Policy Actions]
 [Registry] <-> [Artifacts Store]
 [Federated Coord] <-> [Clients]
 [Monitoring] -> [Dashboards/Alerts]

Service SLOs
- API Gateway: 99.5% uptime, P95 <200 ms
- Telemetry Ingest: 99.5% uptime, lag <5s P95
- Federated Coordinator: rounds success ≥95%

This PRD specifies a complete, privacy-first continual learning platform for edge environments with robust ML capabilities, APIs, architecture, and operational rigor, achieving targeted accuracy, latency, and reliability outcomes.