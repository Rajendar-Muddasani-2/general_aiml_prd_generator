# Product Requirements Document (PRD) / # aiml035_pipeline_hazard_detection

Project ID: aiml035
Category: General AI/ML – Multimodal Hazard Detection
Status: Draft for Review
Version: 1.0
Last Updated: 2025-11-25

## 1. Overview
### 1.1 Executive Summary
aiml035_pipeline_hazard_detection delivers an end-to-end multimodal AI/ML system to detect and localize hazards along critical pipeline infrastructure. It fuses streaming time-series sensors (pressure/flow telemetry, acoustic/vibration, temperature, gas concentration) with computer vision (UAV/ground camera feeds, thermal/multispectral imagery) to provide real-time detection, uncertainty-aware alerting, and actionable insights. It supports edge inference for low-latency events, cloud-based analytics, a rule-ML hybrid engine to reduce false alarms, and MLOps for continuous improvement.

### 1.2 Document Purpose
Define product scope, requirements, technical architecture, data/model strategy, APIs, UI/UX, security, performance, deployment, and success criteria to guide engineering, data science, and operations through delivery.

### 1.3 Product Vision
Provide a reliable, explainable, and scalable AI platform for early hazard detection on pipelines using multimodal fusion, with measurable improvements in time-to-detect, precision/recall, and operator workload reduction, deployable across edge and cloud environments.

## 2. Problem Statement
### 2.1 Current Challenges
- Delayed detection of leaks, ruptures, corrosion, ground movement, and third-party intrusions.
- High false alarm rates from threshold-only monitoring, overwhelming operators.
- Data silos across sensors and visual inspections; limited fusion across modalities.
- Sparse labels and extreme class imbalance for hazardous events.
- Inconsistent model lifecycle and drift management in production.
- Limited explainability and uncertainty quantification for safety-critical decisions.

### 2.2 Impact Analysis
- Increased response times and incident severity due to late detection.
- Operational inefficiencies from manual triage and duplicate alarms.
- Underutilized sensor investments without integrated analytics.
- Compliance and reputational risks from missed or delayed detection.

### 2.3 Opportunity
- Multimodal AI and hybrid rule-ML can materially improve detection precision/recall.
- Edge inference and robust MLOps reduce latency and improve availability.
- Self-supervised learning and digital twins leverage unlabeled data for better generalization.
- Transparent explainability and confidence estimates enhance operator trust.

## 3. Goals and Objectives
### 3.1 Primary Goals
- Real-time hazard detection with calibrated uncertainty and prioritized alerts.
- Multimodal fusion of sensor time-series and vision streams.
- Edge + cloud architecture with robust offline buffering and synchronization.
- Continuous evaluation, drift detection, and automated retraining triggers.

### 3.2 Business Objectives
- Reduce false alarms by 50% while improving recall by 20% within 6 months.
- Decrease median time-to-detect (TTD) by 40%.
- Achieve >99.5% service uptime and <500 ms edge inference latency.
- Accelerate deployment across multiple regions with standardized MLOps.

### 3.3 Success Metrics
- PR-AUC > 0.90 on validation and shadow deployments.
- Precision >= 85%, Recall >= 90% on critical hazard class; F1 >= 0.875.
- False alarm rate <= 0.1 per sensor per day at operating point.
- Median TTD <= 30 seconds for fast-onset hazards; <= 10 minutes for slow leaks.
- Alert deduplication reduces duplicates by >= 60%.
- MTTA (mean time to acknowledge) reduced by >= 30%.

## 4. Target Users/Audience
### 4.1 Primary Users
- Control room operators and field responders needing actionable, low-noise alerts.
- Reliability and integrity engineers analyzing trends and root causes.
- Data scientists and MLOps engineers managing models and monitoring performance.

### 4.2 Secondary Users
- Operations leadership reviewing KPIs and compliance.
- UAV operators and inspectors leveraging visual analytics.
- IT/OT security teams monitoring system health and access.

### 4.3 User Personas
- Persona 1: Alex Moreno, Control Room Operator
  - Background: 8 years in pipeline operations, shifts managing alarms and dispatch.
  - Pain points: Alarm fatigue, late detection, lack of context for prioritization.
  - Goals: Trustworthy alerts with clear severity, location, and cause hints; fast triage; transparent confidence.
- Persona 2: Priya Desai, Reliability Engineer
  - Background: Mechanical/industrial engineer, focuses on integrity and maintenance planning.
  - Pain points: Disparate data sources; manual trend analysis; unclear root causes.
  - Goals: Unified dashboard, historical analytics, explainability on model decisions, exported reports.
- Persona 3: Marco Nguyen, MLOps Engineer
  - Background: 6 years in data engineering and model ops across streaming systems.
  - Pain points: Ad-hoc deployments, lack of drift monitoring, manual threshold tuning.
  - Goals: Automated CI/CD, feature store, model registry, drift and health metrics, safe rollouts.
- Persona 4: Sara Kim, UAV Program Lead
  - Background: Runs periodic UAV inspections and visual data management.
  - Pain points: Slow manual review; inconsistent labeling; limited feedback into ML.
  - Goals: Assisted annotation, active learning suggestions, thermal/multispectral fusion insights.

## 5. User Stories
- US-001: As an operator, I want prioritized alerts with severity and confidence so that I can respond quickly to high-risk events.
  - Acceptance: Alerts show severity (Low/Medium/High/Critical), confidence (0-1), estimated location, and top contributing signals; response within 1 s of trigger at edge.
- US-002: As an operator, I want alert deduplication and grouping so that I don’t receive multiple alerts for the same incident.
  - Acceptance: Alerts within a configurable spatiotemporal window are merged; dedup rate > 60%.
- US-003: As a reliability engineer, I want explainability visualizations so that I can understand model rationale.
  - Acceptance: Per-alert SHAP/LIME summaries, spectrogram attributions, and rule hits available within 5 s in UI.
- US-004: As an MLOps engineer, I want automated drift monitoring so that I can trigger retraining when needed.
  - Acceptance: PSI/KS statistics computed daily; thresholds configurable; retraining job initiated with approval workflow.
- US-005: As a UAV lead, I want batch processing of flight imagery so that hazards are flagged for review.
  - Acceptance: Batch job processes images/video, returns detections and segmentation masks; mAP@0.5 >= 0.6 baseline.
- US-006: As a data scientist, I want a feature store so that streaming/batch features are consistent across training and inference.
  - Acceptance: Features documented, versioned, lineage tracked; latency for online features < 50 ms p95.
- US-007: As a site admin, I want RBAC/ABAC controls so that users see only authorized assets and data.
  - Acceptance: Role and attribute-based policies enforced; audit logs recorded.
- US-008: As an operator, I want feedback tools to mark alerts as true/false positive so that the model can improve.
  - Acceptance: One-click feedback in UI; stored with context; used in active learning pipeline.
- US-009: As a field responder, I want mobile-friendly alert summaries and navigation so that I can locate incidents.
  - Acceptance: Mobile UI renders alert details, map location, and image snippets; loads under 2 s on 4G.

## 6. Functional Requirements
### 6.1 Core Features
- FR-001 Multimodal ingestion: Stream ingest from sensors (MQTT/Kafka), video feeds (RTSP), and batch UAV uploads.
- FR-002 Time-series anomaly detection: Sliding windows with statistical, spectral, and learned features; change-point detection (CUSUM, BOCPD).
- FR-003 Spatiotemporal modeling: LSTM/GRU, TCN, and Transformer variants (Informer/Temporal Fusion Transformer) for long horizons.
- FR-004 Computer vision hazard detection: Object detection and segmentation for leaks, corrosion, cracks, and intrusions using CNN/Transformer backbones; supports thermal/multispectral.
- FR-005 Early/late fusion: Early fusion in feature space; late fusion via ensemble and Bayesian stacking with uncertainty aggregation.
- FR-006 Rule-ML hybrid engine: Engineering constraints and safety rules applied to ML scores for robust alarms and reduced false positives.
- FR-007 Uncertainty and calibration: Temperature scaling, conformal prediction; calibrated thresholds with cost sensitivities.
- FR-008 Alerting and deduplication: Ranking by severity, confidence, and impact; temporal/spatial clustering for deduplication.
- FR-009 Explainability: SHAP/LIME for time-series features; spectrogram attribution; vision saliency maps.
- FR-010 Edge inference: Optimized models (quantization/pruning), local buffering, store-and-forward under intermittent connectivity.
- FR-011 MLOps: Feature store, model registry, CI/CD for models and data pipelines, drift/health monitoring, shadow deployments.
- FR-012 Feedback loop: UI/API to capture operator feedback, active learning selection, and retraining integration.

### 6.2 Advanced Features
- FR-013 Self-supervised pretraining on unlabeled streams (contrastive, masked modeling).
- FR-014 Digital twin simulation for synthetic hazard injection; domain randomization for robustness.
- FR-015 Multi-target modeling: Distinguish sensor faults vs process anomalies; joint modeling for improved reliability.
- FR-016 Event localization: Particle filter tracking using pressure residuals and acoustic triangulation.
- FR-017 Auto-thresholding with extreme value theory (EVT) tail modeling.
- FR-018 Adaptive sampling and on-device preprocessing to reduce bandwidth.
- FR-019 Active learning for vision labels; semi-supervised segmentation with weak labels from inspection notes.
- FR-020 API integrations: Work management systems (create tickets), notifications (email/SMS/webhooks).

## 7. Non-Functional Requirements
### 7.1 Performance
- Edge inference latency: < 500 ms p95 per event window; cloud inference: < 800 ms p95.
- Throughput: Support 10k sensor messages/s per region and 200 concurrent video streams.
- Storage: Retain raw streams 30 days, features 180 days, alerts 2 years.

### 7.2 Reliability
- Uptime: >= 99.5% service uptime for critical paths.
- Data durability: >= 11 nines for archived alerts/labels; replicated storage.
- Backpressure/overflow handling with buffering and circuit breakers.

### 7.3 Usability
- Operator workflows require <= 3 clicks to triage an alert.
- Accessible, responsive UI; consistent visual language; contextual help.

### 7.4 Maintainability
- Modular services with clear interfaces; test coverage >= 80%.
- Infrastructure as code; automated dependency updates.
- Model and data lineage tracked end-to-end.

## 8. Technical Requirements
### 8.1 Technical Stack
- Languages: Python 3.11+, TypeScript 5.x
- Backend API: FastAPI 0.115+, Uvicorn 0.30+
- Stream processing: Apache Kafka 3.7+, Faust/ksqlDB; Apache Flink 1.19+ (optional)
- Feature store: Feast 0.40+
- Model training: PyTorch 2.3+, TorchVision 0.18+, TorchAudio 2.3+, Lightning 2.4+
- Transformers: Hugging Face Transformers 4.44+
- Change-point: ruptures 1.1+, bayesian-online-changepoint custom
- CV inference: ONNX Runtime 1.19+, TensorRT 10.x (edge NVIDIA)
- Serving: NVIDIA Triton Inference Server 24.08+ or TorchServe 0.11+
- Data: PostgreSQL 15+/TimescaleDB 2.14+; Redis 7.x; MinIO/S3
- Workflow: Airflow 2.9+ or Prefect 2.16+
- Frontend: React 18+, Vite 5+, Material UI 6+
- Containerization/Orchestration: Docker 24+, Kubernetes 1.29+
- Observability: Prometheus 2.53+, Grafana 11+, Loki 2.9+, OpenTelemetry 1.27+
- CI/CD: GitHub Actions; Terraform 1.8+; Helm 3.14+
- Auth: Keycloak 23+ (OIDC), OAuth2/JWT; mTLS with Envoy 1.30+

### 8.2 AI/ML Components
- Time-series models: TCN, LSTM/GRU, Informer/Temporal Fusion Transformer; baselines ARIMA/Prophet.
- Anomaly detection: Autoencoder/VAE, Isolation Forest, One-Class SVM; EVT tail modeling.
- Change-point: CUSUM, GLR, BOCPD; Kalman/particle filters for residual tracking.
- Vision: YOLOv8/Detectron2/Mask2Former; ViT/ConvNeXt backbones; thermal/multispectral fusion.
- Fusion: Early fusion via concatenated learned embeddings; late fusion via weighted ensemble and Bayesian model averaging.
- Calibration: Temperature scaling, isotonic regression; conformal prediction sets.
- Explainability: SHAP, LIME, Grad-CAM; spectrogram attribution.

## 9. System Architecture
### 9.1 High-Level Architecture (ASCII)
+-------------------+         +-------------------+         +---------------------+
| Edge Sensors      |         | Edge Video Feeds  |         | UAV Batch Uploads   |
| (pressure, vib,   |         | (RTSP, thermal)   |         | (images/videos)     |
+---------+---------+         +---------+---------+         +----------+----------+
          |                             |                               |
          v                             v                               v
+--------------------+      +--------------------+          +--------------------+
| Edge Gateway       |      | Edge Inference     |          | Batch Ingest API   |
| (buffer, preprocess|----->| (TS + CV models)   |<---------| (S3/MinIO)         |
+---------+----------+      +---------+----------+          +---------+----------+
          | Kafka/mqtt                 | Alerts/events                 |
          v                            v                               |
+----------------------+     +----------------------+       +----------------------+
| Cloud Ingestion      |---->| Stream Processing    |------>| Feature Store        |
| (Kafka/Flink)        |     | (windows, features)  |       | (online/offline)     |
+----------+-----------+     +----------+-----------+       +----------+-----------+
           |                              |                              |
           v                              v                              v
+----------------------+     +----------------------+       +----------------------+
| Model Serving        |<----| Fusion + Rule Engine |<----->| Model Registry       |
| (Triton/TorchServe)  |     | (scores, thresholds) |       | (MLflow)             |
+----------+-----------+     +----------+-----------+       +----------+-----------+
           |                              |                              |
           v                              v                              v
+----------------------+     +----------------------+       +----------------------+
| Alerts & API Layer   |<--->| UI/Web App           |<----->| Observability Stack  |
| (FastAPI/OIDC)       |     | (React)              |       | (Prom/Graf/Loki)     |
+----------------------+     +----------------------+       +----------------------+

### 9.2 Component Details
- Edge Gateway: Sensor normalization, resampling, compression, buffering; store-and-forward with local SQLite/Parquet.
- Edge Inference: Optimized models; adaptive sampling; on-device rules; local alerting and summary upload.
- Cloud Ingestion: Kafka topics per modality; schema registry; dead-letter queues.
- Stream Processing: Sliding windows; FFT/STFT; wavelets; cepstral/ACF; change-point statistics; feature computation to Feature Store.
- Model Serving: Real-time TS and CV model endpoints; fusion service aggregates outputs; ensembles with uncertainty.
- Rule Engine: Declarative rules (YAML) combining physics constraints and ML scores; severity mapping.
- Feature Store: Feast providing online (Redis) and offline (Parquet) stores.
- Model Registry: MLflow for model/version tracking, metrics, artifacts; A/B and shadow tags.
- API Layer: REST endpoints for ingestion, inference, alerts, feedback, models, and admin.
- UI: Operator dashboard, map view, timeline/spectrogram viewer, explainability panels, model/performance dashboards.
- Observability: Metrics, traces, logs; dashboards for latency, throughput, error rates, drift, TTD.

### 9.3 Data Flow
- Ingest -> Preprocess -> Feature Extraction -> Model Inference -> Fusion/Rules -> Alert Generation -> UI/Notifications -> Feedback -> Active Learning -> Retraining -> Deployment via CI/CD.

## 10. Data Model
### 10.1 Entity Relationships
- Site 1—N Device; Device 1—N Sensor; Sensor 1—N Stream; Stream 1—N Observation.
- Observation N—1 FeatureVector; FeatureVector N—1 ModelVersion.
- Event (hazard) 1—N Alert; Alert N—1 Event; Alert 1—N Feedback.
- ModelVersion N—N Evaluation; User N—N Role; User 1—N Feedback.

### 10.2 Database Schema (key fields)
- sites(id, name, region, lat, lon, attrs)
- devices(id, site_id, type, firmware, status)
- sensors(id, device_id, modality, unit, sampling_hz, status)
- streams(id, sensor_id, topic, codec, config_json)
- observations(id, stream_id, ts, value_json or blob)
- features(id, stream_id, ts_start, ts_end, vector, schema_ver)
- events(id, type, severity, status, site_id, geom, started_at, ended_at)
- alerts(id, event_id, generated_at, severity, confidence, message, evidence_ref)
- models(id, name, task, framework, owner)
- model_versions(id, model_id, version, uri, metrics_json, status)
- evaluations(id, model_version_id, dataset_id, metrics_json, created_at)
- users(id, email, name, org, attrs, created_at)
- roles(id, name, permissions)
- user_roles(user_id, role_id)
- feedback(id, alert_id, user_id, label, comment, created_at)
- datasets(id, name, modality, split, uri, schema, notes)
- audit_logs(id, ts, actor, action, entity, payload)

TimescaleDB hypertables for observations/features; PostGIS extension for geom.

### 10.3 Data Flow Diagrams (ASCII)
[Sensor Streams] -> [Windowing/Features] -> [TS Models] -> \
                                            [Fusion+Rules] -> [Alerts]
[Video Frames]  -> [CV Models]            -> /

[Alerts] -> [UI/Notifications] -> [Feedback] -> [Active Learning] -> [Retraining] -> [Registry] -> [Serving]

### 10.4 Input Data & Dataset Requirements
- Time-series
  - Modalities: pressure/flow telemetry, acoustic/vibration, temperature, gas concentration.
  - Sampling: 1–1000 Hz depending on sensor; synchronized to common clock with NTP/PTP.
  - Windows: 2–60 s rolling windows; overlaps 50–90%.
  - Labels: Hazard onset/offset, type, location (if known). Sparse; allow weak labels and event notes.
- Vision
  - Sources: Ground cameras (RTSP), UAV imagery (RGB, thermal, multispectral).
  - Formats: H.264/H.265 video; JPEG/PNG images; GeoTIFF for thermal/multispectral.
  - Annotations: Bounding boxes, polygons, and tags; support weak labels from inspection logs.
- Metadata
  - Topology map, device locations, calibration data, environmental context (weather/temperature).
- Data quality
  - Missing data handling, sensor health flags, outlier filters, synchronization tolerance < 50 ms for fusion.
- Privacy/compliance
  - Blur faces/license plates in vision; redact PII in logs.

## 11. API Specifications
### 11.1 REST Endpoints
- POST /api/v1/ingest/sensor
  - Ingest batched sensor observations.
- POST /api/v1/ingest/video/summary
  - Ingest edge-generated frame summaries or detections.
- POST /api/v1/inference/timeseries
  - Request synchronous anomaly scoring on a window.
- POST /api/v1/inference/vision
  - Upload image(s)/frames for hazard detection.
- GET /api/v1/alerts
  - List alerts with filters (time range, site, severity, status).
- POST /api/v1/alerts/ack
  - Acknowledge alert(s).
- POST /api/v1/feedback
  - Submit operator feedback for an alert.
- GET /api/v1/models
  - List models and versions with metrics.
- POST /api/v1/models/promote
  - Promote a model version to production (RBAC protected).
- GET /api/v1/metrics/health
  - Service health and uptime metrics.
- POST /api/v1/rules/validate
  - Validate uploaded rule configurations.

### 11.2 Request/Response Examples
- Example: POST /api/v1/inference/timeseries
  Request:
  {
    "stream_id": "str_123",
    "ts_start": "2025-11-25T10:00:00Z",
    "ts_end": "2025-11-25T10:00:10Z",
    "values": [0.12, 0.10, 0.15, ...],
    "features": null,
    "model_version": "tcn_v14"
  }
  Response:
  {
    "anomaly_score": 0.92,
    "confidence": 0.88,
    "calibrated_threshold": 0.75,
    "change_point_prob": 0.67,
    "explanations": {"top_features": [["rms", 0.21], ["fft_band_2", 0.17]]},
    "latency_ms": 184
  }

- Example: POST /api/v1/feedback
  Request:
  {
    "alert_id": "alt_456",
    "label": "true_positive",
    "comment": "Confirmed leak via field inspection"
  }
  Response:
  {"status": "ok", "message": "Feedback recorded"}

### 11.3 Authentication
- OAuth2/OIDC with JWT access tokens; short-lived tokens with refresh.
- mTLS for service-to-service; scope-based authorization.
- Rate limiting per client; API keys for machine clients when appropriate.

## 12. UI/UX Requirements
### 12.1 User Interface
- Dashboard: KPI tiles (TTD, PR-AUC, false alarm rate), active alerts, system health.
- Map view: Sites/devices; alerts overlaid with severity; click-through to detail.
- Alert detail: Timeline, spectrograms, salient features, saliency maps, rule hits, confidence, recommended actions.
- Video/CV viewer: Frame thumbnails, overlays (boxes/masks), thermal gradients.
- Model ops: Model/version list, metrics, drift charts, shadow vs prod comparisons.
- Feedback pane: One-click labels; comment box; audit trail.

### 12.2 User Experience
- Priority navigation to active incidents; keyboard shortcuts for triage.
- Contextual hints explaining confidence and severity.
- Batch operations for acknowledging/merging alerts.
- Offline-friendly mobile view for field responders.

### 12.3 Accessibility
- WCAG 2.1 AA compliance: High contrast, keyboard navigation, ARIA roles.
- Alt text and captions for media; adjustable font sizes.

## 13. Security Requirements
### 13.1 Authentication
- OIDC provider (Keycloak); MFA optional per policy; passwordless support (WebAuthn).

### 13.2 Authorization
- RBAC for roles (Operator, Engineer, Admin, Viewer); ABAC for site-based access.
- Policy-as-code (OPA) for fine-grained controls; just-in-time elevation with approvals.

### 13.3 Data Protection
- TLS 1.3 in transit; AES-256 at rest; envelope encryption for object storage.
- Secrets via Vault; regular rotation; no secrets in code/repos.
- PII minimization; CV redaction for faces/plates.

### 13.4 Compliance
- Align with SOC 2 Type II, ISO 27001, GDPR where applicable.
- Full audit trail of access, model changes, and rule updates.

## 14. Performance Requirements
### 14.1 Response Times
- Edge inference: < 500 ms p95 per window; cloud inference: < 800 ms p95.
- UI: Initial dashboard load < 2 s p90; alert detail < 1.5 s p90.

### 14.2 Throughput
- Ingest: >= 10k messages/s per region; burst to 50k messages/s for 5 minutes.
- CV: Process >= 200 concurrent streams with GPU autoscaling.

### 14.3 Resource Usage
- Edge footprint: <= 2 CPU cores, <= 4 GB RAM per node for base models.
- Cloud serving: GPU utilization target 60–80%; batch size autotuned.

## 15. Scalability Requirements
### 15.1 Horizontal Scaling
- Stateless API and serving layers scale via Kubernetes HPA.
- Kafka partitions scale with load; consumer groups for parallelism.

### 15.2 Vertical Scaling
- GPU node pools for CV models; CPU-optimized pools for streaming jobs.
- Edge devices support optional accelerator (Jetson, Intel NPU).

### 15.3 Load Handling
- Backpressure via Kafka; autoscale on lag and CPU/GPU; graceful degradation (reduced sampling rate) under load.

## 16. Testing Strategy
### 16.1 Unit Testing
- Python pytest with 85% coverage for feature extraction, model wrappers, fusion, rules.
- Frontend Jest/RTL tests for components.

### 16.2 Integration Testing
- Testcontainers for Kafka, Postgres, MinIO; contract tests for APIs; end-to-end pipelines with synthetic data.

### 16.3 Performance Testing
- Locust for API load; Kafka benchmark tooling; NVML metrics for GPU saturation tests.
- Scenario-based TTD and false alarm evaluation using replay harness.

### 16.4 Security Testing
- SAST/DAST in CI; dependency scanning; container image scanning; periodic pen tests.
- Secrets leakage detection; policy validation for RBAC/ABAC.

## 17. Deployment Strategy
### 17.1 Deployment Pipeline
- GitFlow with protected main; GitHub Actions CI (lint, test, build images).
- CD with Argo CD/Helm to dev/stage/prod; model registry promotion gates.
- IaC via Terraform for cloud resources; Helm charts versioned.

### 17.2 Environments
- Dev (shared minimal), Staging (prod-like, shadow data), Production (HA, multi-AZ).
- Edge: staged rollouts via device groups; OTA updates with signed artifacts.

### 17.3 Rollout Plan
- Phase 1: Shadow deployment alongside existing monitoring; no-alert mode; compare metrics.
- Phase 2: Limited canary (10% sites), with rollback guardrails.
- Phase 3: Global roll once KPIs met.

### 17.4 Rollback Procedures
- Blue/green switchback within 5 minutes; database migrations reversible; model version pinning; config snapshots.

## 18. Monitoring & Observability
### 18.1 Metrics
- System: CPU/GPU/memory, queue lag, request rates, error rates.
- Model: Precision/recall/PR-AUC by class and region, calibration error (ECE), TTD, drift metrics (PSI/KS), false alarm rate.
- Edge health: Buffer occupancy, last-sync time, local inference latency.

### 18.2 Logging
- Structured JSON logs; correlation IDs; PII redaction; log levels with dynamic controls.

### 18.3 Alerting
- SLO breaches (latency, errors), drift thresholds, data staleness, edge offline durations.
- PagerDuty/Slack/MS Teams integrations.

### 18.4 Dashboards
- Operations: Ingest/serving health, backlog, throughput.
- Model: Performance over time, drift by feature, calibration diagnostics.
- Edge: Fleet status, version compliance, connectivity.

## 19. Risk Assessment
### 19.1 Technical Risks
- Sparse labels causing poor supervised performance.
- Domain shift across regions and seasons.
- Edge connectivity/intermittent power affecting data completeness.
- Multimodal time sync drift leading to degraded fusion.

### 19.2 Business Risks
- Operator distrust if early versions generate too many false alarms.
- Integration complexity with existing systems and processes.
- Cost overruns for GPU-heavy CV processing.

### 19.3 Mitigation Strategies
- Use self-supervised pretraining and semi-supervised learning; digital twins for augmentation.
- Drift-aware monitoring and automatic threshold recalibration; domain adaptation.
- Edge buffering/store-and-forward; heartbeat monitoring.
- Time sync verification, cross-correlation checks, and resynchronization logic.

## 20. Timeline & Milestones
### 20.1 Phase Breakdown
- Phase 0 (2 weeks): Discovery, data audit, architecture finalization.
- Phase 1 (6 weeks): Ingestion, feature store, baseline TS anomaly detection (Isolation Forest, AE).
- Phase 2 (6 weeks): Spatiotemporal models (TCN/TFT), change-point integration, rule engine v1.
- Phase 3 (6 weeks): CV models for leaks/corrosion/cracks; edge inference prototype.
- Phase 4 (4 weeks): Fusion (early/late), uncertainty calibration, explainability.
- Phase 5 (4 weeks): MLOps (registry, CI/CD), drift monitoring, shadow deployment.
- Phase 6 (4 weeks): UI/UX, alerting workflows, feedback loop.
- Phase 7 (4 weeks): Performance hardening, canary rollout, SLOs, documentation.

Total: ~32 weeks (~8 months)

### 20.2 Key Milestones
- M1: Ingestion and feature store live (end Phase 1)
- M2: TS detection PR-AUC > 0.8 in offline eval (end Phase 2)
- M3: CV mAP@0.5 > 0.6 and edge inference < 500 ms (end Phase 3)
- M4: Fusion PR-AUC > 0.9 and false alarm rate <= target (end Phase 4)
- M5: MLOps and drift in place; shadow deployment (end Phase 5)
- M6: Operator UI beta and feedback integration (end Phase 6)
- M7: Canary deployment meets KPIs; production GA (end Phase 7)

Estimated Costs (rough order):
- Team (8 FTE avg): $220k/month blended => ~$1.76M over 8 months
- Cloud/GPU/Edge hardware and storage: ~$400k
- Contingency and licenses: ~$150k
Total: ~$2.31M

## 21. Success Metrics & KPIs
### 21.1 Measurable Targets
- Detection performance: PR-AUC > 0.90; F1 >= 0.875; MCC >= 0.7
- Latency: Edge < 500 ms p95; Cloud < 800 ms p95
- Uptime: >= 99.5% service uptime
- False alarms: <= 0.1 per sensor per day; alert dedup >= 60%
- TTD: <= 30 s fast-onset; <= 10 min slow leaks
- Operator outcomes: MTTA down by 30%; > 80% operator satisfaction survey
- Drift handling: Automatic threshold recalibration keeps precision within 10% of baseline over 3 months without retrain
- Label efficiency: 30% reduction in manual labeling via active learning
- Cost efficiency: GPU utilization 60–80%; cost per processed GB reduced by 20% over 3 months

## 22. Appendices & Glossary
### 22.1 Technical Background
- Multimodal fusion improves robustness: early fusion captures interactions; late fusion offers resilience to missing modalities.
- Time-series anomaly detection benefits from combining reconstruction error (autoencoders) with residual change-point tests (CUSUM/BOCPD).
- Transformers (Informer/TFT) capture long-range dependencies and handle covariates like weather/schedules.
- Uncertainty calibration with temperature scaling and conformal prediction helps align scores with risk.
- EVT-based thresholding better models tail behavior under extreme imbalance.
- Digital twins provide synthetic, labeled transients and failure modes for training and stress testing.

### 22.2 References
- Wu et al., Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting.
- Lim et al., Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting.
- Ruff et al., Deep One-Class Classification.
- Lundberg and Lee, A Unified Approach to Interpreting Model Predictions (SHAP).
- Siffer et al., Anomaly Detection in Streams with Extreme Value Theory.
- Adams and MacKay, Bayesian Online Changepoint Detection.

### 22.3 Glossary
- Anomaly Score: Numeric value indicating deviation from normal behavior.
- Change-point: Time at which a statistical property of a signal changes.
- Conformal Prediction: Technique to produce calibrated confidence sets for model predictions.
- Early Fusion: Combining multiple modalities at feature level before modeling.
- Late Fusion: Combining outputs from modality-specific models into a final decision.
- PR-AUC: Area under the precision-recall curve, robust under class imbalance.
- Shadow Deployment: Running a new model in parallel without impacting production decisions.
- TTD (Time-To-Detect): Time lag between true event onset and system alert.
- Uncertainty Calibration: Aligning predicted probabilities with observed frequencies.

Repository Structure (proposed)
- README.md
- notebooks/
  - exploration/
  - prototypes/
  - eval/
- src/
  - ingest/
    - kafka_consumer.py
    - mqtt_bridge.py
  - features/
    - ts_features.py
    - spectrograms.py
  - models/
    - ts/
      - tcn.py
      - transformer_tft.py
      - autoencoder.py
    - cv/
      - detector.py
      - segmenter.py
    - fusion/
      - late_fusion.py
      - calibration.py
    - changepoint/
      - cusum.py
      - bocpd.py
  - serving/
    - api.py
    - routers/
      - inference.py
      - alerts.py
      - feedback.py
  - rules/
    - engine.py
    - schemas.py
  - mlops/
    - registry.py
    - drift_monitor.py
    - retrain_pipeline.py
  - utils/
    - logging.py
    - config.py
- tests/
  - unit/
  - integration/
  - performance/
- configs/
  - app.yaml
  - rules.yaml
  - models.yaml
  - kafka.yaml
- data/
  - samples/
  - schemas/
- deployment/
  - helm/
  - terraform/
- ui/
  - web/

Sample Config (configs/app.yaml)
server:
  host: 0.0.0.0
  port: 8080
auth:
  oidc_issuer: https://auth.example.com/realms/main
  audience: aiml035-api
kafka:
  brokers: ["kafka-1:9092","kafka-2:9092"]
  topics:
    sensors: sensors.raw.v1
    features: sensors.features.v1
    alerts: alerts.v1
feature_store:
  provider: feast
  online_store: redis://redis:6379
  offline_store: s3://aiml035/features/
models:
  ts_default: tcn_v14
  cv_default: yolo_v9_fused
thresholds:
  anomaly: 0.8
  change_point: 0.6

Sample FastAPI Snippet (src/serving/api.py)
from fastapi import FastAPI, Depends
from routers import inference, alerts, feedback
from utils.auth import auth_dep

app = FastAPI(title="aiml035 API", version="1.0")

app.include_router(inference.router, prefix="/api/v1", dependencies=[Depends(auth_dep)])
app.include_router(alerts.router, prefix="/api/v1", dependencies=[Depends(auth_dep)])
app.include_router(feedback.router, prefix="/api/v1", dependencies=[Depends(auth_dep)])

Sample Rule (configs/rules.yaml)
- id: leak_pressure_drop
  description: Trigger when pressure residual drop aligns with acoustic spike
  conditions:
    - metric: ts.anomaly_score
      op: ">"
      value: 0.85
    - metric: cp.cusum
      op: ">"
      value: 5.0
    - metric: acoustic.fft_band_3
      op: ">"
      value: 2.5
  severity: "High"
  window_s: 10
  dedup_s: 120

Edge Inference Optimization (example)
- Quantization: Post-training INT8 for TCN and YOLO variants with <1% degradation.
- Pruning: 30% structured pruning on redundant channels.
- Runtime: ONNX Runtime with CUDA EP or TensorRT on supported hardware.

Evaluation Metrics and Reporting
- Offline: PR-AUC, ROC-AUC, F1, MCC, ECE; per-class and macro/micro averages.
- Online: Rolling 7/30-day PR-AUC, calibration drift, false alarm rate, TTD distributions.
- Vision: mAP@0.5, mAP@0.5:0.95; segmentation IoU; thermal-only vs fused comparisons.

This PRD defines the complete scope, architecture, and execution plan for aiml035_pipeline_hazard_detection, focusing on AI/ML, computer vision, NLP-free time-series modeling, multimodal fusion, edge/cloud deployment, robust MLOps, and operator-centric UX to deliver measurable improvements in safety and operational efficiency.