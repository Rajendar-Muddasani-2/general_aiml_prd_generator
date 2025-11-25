# Product Requirements Document (PRD)
# `Aiml021_Edge_Cloud_Collaborative_Inference`

Project ID: aiml021
Category: AI/ML Platform – Edge-Cloud Collaborative Inference
Status: Draft for Review
Version: v1.0
Last Updated: 2025-11-25

## 1. Overview
### 1.1 Executive Summary
aiml021 enables split and collaborative inference across edge devices and cloud services to meet stringent latency, accuracy, privacy, and cost targets under variable network and workload conditions. It supports dynamic model partitioning, confidence-based early exit, cascaded inference, adaptive offloading, feature compression, and SLA-aware orchestration. The solution ships as: (a) an Edge SDK/Agent for on-device inference and routing; (b) a Cloud Gateway with model serving, policy engine, dynamic batching, and monitoring; (c) an Operator Console for configuration, governance, and observability.

### 1.2 Document Purpose
Define product scope, requirements, architecture, APIs, data models, testing, deployment, and KPIs for aiml021. This PRD guides engineering, data science, product, and operations in building and operating the system.

### 1.3 Product Vision
Deliver a robust, privacy-preserving, and cost-optimized AI/ML inference platform that adapts in real-time to device capabilities and network conditions, seamlessly blending edge responsiveness with cloud precision to power modern multimodal applications.

## 2. Problem Statement
### 2.1 Current Challenges
- Single-location inference (edge-only or cloud-only) fails to meet mixed requirements across latency, accuracy, privacy, and cost.
- Variable network QoS leads to unpredictable latency and degraded user experience.
- Static routing and monolithic models waste resources under fluctuating loads.
- Lack of uncertainty awareness and calibration leads to poor escalation decisions.
- Limited observability of offload rates, costs, and quality over time.

### 2.2 Impact Analysis
- Slow or inconsistent response times degrade user engagement and conversion.
- Overuse of cloud resources increases operational costs.
- Underuse of edge compute results in unnecessary latency and bandwidth use.
- Poor escalation leads to accuracy regressions or privacy concerns.
- Inadequate monitoring risks SLA breaches and regulatory issues.

### 2.3 Opportunity
- Split computing with learned or heuristic offloading delivers optimal trade-offs per request.
- Early-exit and cascaded architectures reduce latency and bandwidth while preserving accuracy.
- Feature compression and privacy techniques reduce cost and risk.
- Comprehensive monitoring enables continuous optimization and governance.

## 3. Goals and Objectives
### 3.1 Primary Goals
- Provide a framework for split inference with dynamic offloading policies.
- Enable early-exit/branchy and cascaded pipelines with uncertainty-aware decisioning.
- Ensure privacy-preserving feature transport and adaptive compression.
- Offer SLA-aware orchestration with robust monitoring and governance.

### 3.2 Business Objectives
- Reduce average cloud compute cost per request by ≥30% vs cloud-only baseline.
- Improve p95 user-perceived latency by ≥40% vs cloud-only baseline.
- Achieve ≥99.5% monthly uptime for cloud components.
- Expand to 3+ key verticals (retail, smart camera analytics, voice assistants) within 2 quarters.

### 3.3 Success Metrics
- Accuracy: >90% top-1 for target CV tasks; >92 F1 for target NLP tasks.
- Latency: Edge-only p95 < 300 ms; Edge→Cloud p95 < 700 ms; p99 < 1200 ms.
- Offload rate: 30–60% depending on SLA; adaptive per environment.
- Cost per 1k requests: ≥30% lower vs cloud-only baseline.
- Privacy: 100% of offloaded features encrypted in transit (TLS 1.3); zero PII leakage in features per DLP scans.

## 4. Target Users/Audience
### 4.1 Primary Users
- ML Engineers and MLOps teams building and deploying edge-aware pipelines.
- Application Developers integrating inference into edge apps, kiosks, and IoT devices.
- Data Scientists designing models and offload policies.

### 4.2 Secondary Users
- Product Managers defining SLAs and privacy policies.
- SRE/DevOps for reliability, scaling, and cost management.
- Security and Compliance teams.

### 4.3 User Personas
1) Priya Menon – Senior ML Engineer
- Background: 7 years in computer vision; maintains on-device models for retail analytics.
- Pain points: Inconsistent latency in store networks; high cloud costs; limited visibility into offload decisions.
- Goals: Reliable sub-500 ms responses; fine control of offload thresholds; automated monitoring and rollback.

2) Alex Gomez – Edge App Developer
- Background: Builds Android/iOS apps for field technicians with on-device NLP and image capture.
- Pain points: Complex SDKs; version drift between edge and cloud; handling offline gracefully.
- Goals: Simple SDK; easy upgrades; offline fallback with replay queues.

3) Dana Schultz – MLOps/SRE Lead
- Background: Runs multi-cluster GPU serving; owns SLAs and observability.
- Pain points: Tail latency, GPU underutilization, noisy alerts, unpredictable costs.
- Goals: Dynamic batching; autoscaling; clear dashboards; cost caps with budgets and alerts.

4) Mei Chen – Data Privacy Officer
- Background: Oversees data handling policies and audits.
- Pain points: Risk of sensitive data in features; insufficient encryption/audit trails.
- Goals: Feature anonymization; strong encryption; detailed audit logs and access controls.

## 5. User Stories
US-001: As an ML Engineer, I want to configure a cut layer for my model so that early layers run on edge and later layers run in the cloud.
- Acceptance: Using console or config file, I can set layer index; telemetry shows activation size and latency impact.

US-002: As an App Developer, I want confidence-based early-exit on edge so that many requests return locally under 200 ms.
- Acceptance: Configure thresholds; see local exit rate ≥40% on validation; unit tests for threshold logic.

US-003: As MLOps, I want a policy engine that dynamically offloads based on network RTT, edge load, and predicted difficulty.
- Acceptance: Policy API returns decision within 5 ms; can A/B test policy versions; offload rate adapts to network changes.

US-004: As a Privacy Officer, I want features to be encrypted and anonymized before transport.
- Acceptance: TLS 1.3 enforced; optional differential noise; DLP scan of features shows no PII leakage.

US-005: As SRE, I want dynamic batching and autoscaling in the cloud so that p95 latency remains under target at 10k RPS.
- Acceptance: Load test shows p95 < 700 ms at 10k RPS with autoscaling; GPU utilization ≥60% average.

US-006: As ML Engineer, I want cascaded inference where a small edge model filters requests and a cloud model refines complex cases.
- Acceptance: End-to-end pipeline with configurable cascade; accuracy improves ≥3% at same cost versus edge-only.

US-007: As MLOps, I want robust offline mode with deferred cloud refinement.
- Acceptance: Replay queue preserves up to 24h of deferred requests; no data loss; reconciliation metrics exposed.

US-008: As Product Manager, I want dashboards for latency, accuracy, offload rate, and cost by region.
- Acceptance: Grafana dashboards with drilldowns; daily email summaries; alerts on SLA breaches.

US-009: As App Developer, I want a simple REST API to submit features to cloud and retrieve results.
- Acceptance: POST /v1/infer returns within SLA; documented schemas; SDK reference code.

US-010: As ML Engineer, I want shadow evaluation to validate new models/policies without impacting users.
- Acceptance: Shadow mode toggles from console; impact and deltas viewable; canary rollout supported.

## 6. Functional Requirements
### 6.1 Core Features
FR-001 Split Inference: Configure and execute model partitioning at a chosen cut layer.
FR-002 Early Exit: Support branchy networks with calibrated thresholds per exit.
FR-003 Dynamic Offloading: Policy engine with heuristic and learning-based modes (contextual bandits/RL).
FR-004 Cascaded Inference: Lightweight edge model + heavy cloud model with verification/reranking.
FR-005 Feature Compression: Quantization (int8), sparsification, entropy coding; tunable rate–distortion.
FR-006 Privacy Controls: Feature anonymization, TLS 1.3 encryption, optional differential privacy noise.
FR-007 Adaptive Batching: Cloud-side dynamic batching with deadline-aware scheduling.
FR-008 Uncertainty Estimation: Confidence, entropy, temperature scaling; selective classification.
FR-009 Caching/Memoization: Edge embedding/result caches with similarity thresholds and TTL.
FR-010 Resilience: Offline fallback, replay queues for deferred cloud refinement.
FR-011 Monitoring: Telemetry on latency, accuracy, offload rates, costs; drift detection; A/B testing.
FR-012 Operator Console: UI for configuration, deployments, policy tuning, and dashboards.
FR-013 SDK/Agent: Edge SDK for Android/iOS/Linux; provides inference graph execution and routing.

### 6.2 Advanced Features
- FR-014 Adaptive Cut-Layer Selection: Auto-tune cut layer based on bandwidth and device load.
- FR-015 Mixture-of-Experts Routing: Edge gating to select specialized cloud experts.
- FR-016 Multimodal Pipelines: Video keyframe detection, ROI extraction at edge; cloud global context fusion.
- FR-017 Knowledge Distillation: Cloud-to-edge distillation for continual improvement.
- FR-018 Canary/Shadow Policies: Safe rollout for models and policies with guardrails.
- FR-019 Cost-Aware Optimization: Policies that optimize latency–accuracy–cost–privacy objectives.

## 7. Non-Functional Requirements
### 7.1 Performance
- p95 latency: Edge-only < 300 ms; Edge→Cloud < 700 ms; p99 < 1200 ms.
- Throughput: ≥10k RPS across regions with autoscaling.
- Edge CPU usage: <60% on target devices for typical workloads; GPU/NPU use when available.

### 7.2 Reliability
- Uptime: ≥99.5% monthly for cloud services.
- Replay durability: 24h guaranteed retention on edge for deferred requests.
- Exactly-once processing for replay via idempotency keys.

### 7.3 Usability
- SDK integration < 2 hours for basic flow.
- Clear docs and examples; typed clients in Python/TypeScript.

### 7.4 Maintainability
- Modular services with well-defined APIs.
- CI/CD with >80% unit test coverage on core libraries.
- Backwards-compatible APIs with semantic versioning.

## 8. Technical Requirements
### 8.1 Technical Stack
- Edge SDK: Python 3.11+, C++17, Android (Kotlin), iOS (Swift), ONNX Runtime Mobile 1.18+, TensorFlow Lite 2.15+, PyTorch Mobile 2.3+.
- Cloud:
  - Inference Serving: NVIDIA Triton Inference Server 24.08+, TorchServe 0.9+, ONNX Runtime 1.18+.
  - API: FastAPI 0.111+ (Python 3.11+), Uvicorn 0.30+.
  - Message/Streaming: Kafka 3.6+.
  - Caching: Redis 7.2+.
  - Databases: PostgreSQL 15+ (metadata), TimescaleDB 2.13+ (metrics), MinIO/S3 (artifacts).
  - Orchestration: Kubernetes 1.29+, Helm 3.14+, KEDA 2.14+.
  - Observability: OpenTelemetry 1.27+, Prometheus 2.52+, Grafana 11+, Loki 2.9+, Jaeger 1.54+.
- Frontend: React 18+, TypeScript 5+, Node.js 20+, Vite 5+.

### 8.2 AI/ML Components
- CV models: MobileNetV3/EfficientNet-Lite (edge), ResNet50/EfficientNet-B3/YOLOv8-L (cloud).
- NLP models: DistilBERT (edge), BERT-base/DeBERTa-v3-base (cloud).
- Audio: Distil-Whisper (edge), Whisper-medium (cloud).
- Calibration: Temperature scaling, Platt scaling.
- Policy learning: Contextual bandits (LinUCB, Thompson Sampling), RL (DQN).
- Compression: Post-training quantization int8, pruning to target sparsity 30–70%, entropy coding (range coding).

## 9. System Architecture
### 9.1 High-Level Architecture (ASCII diagram)
 Users/App
   |
   v
+--------------------+           +-----------------------+
| Edge Device/Agent  |<--------->| Cloud Gateway (API)  |
| - Preprocess       |    TLS    | - AuthN/AuthZ        |
| - Edge Model       |           | - Policy Engine      |
| - Early Exit       |           | - Feature Router     |
| - Offload Client   |---------->| - Dynamic Batching   |
| - Cache/Replay     |  features | - Model Registry     |
+--------------------+           +----------+------------+
                                            |
                                            v
                                   +--------------------+
                                   | Model Serving      |
                                   | (Triton/TorchServe)|
                                   +---------+----------+
                                             |
                                             v
                                    +-------------------+
                                    | Observability     |
                                    | (Prom, Grafana,   |
                                    |  OTel, Jaeger)    |
                                    +-------------------+
                                             |
                                             v
                                   +--------------------+
                                   | Data Stores        |
                                   | (Postgres, Redis,  |
                                   |  S3/MinIO, TSDB)   |
                                   +--------------------+

### 9.2 Component Details
- Edge Agent/SDK: Executes local layers, computes uncertainty, performs early exit, compresses features, manages caches and replay queues, collects telemetry.
- Cloud Gateway: Authenticated endpoint for features/inference; consults Policy Engine for routing; schedules requests and batches; returns results.
- Policy Engine: Consumes signals (network RTT, device load, entropy, SLA); outputs offload decision, cut layer, compression rate; supports heuristic/bandit/RL policies.
- Model Serving: Hosts deep models; supports dynamic batching and GPU acceleration; returns predictions and confidences.
- Observability Stack: Metrics, traces, logs; drift detection; dashboards and alerts.
- Data Stores: Metadata (Postgres), feature/result cache (Redis), artifacts (S3/MinIO), time-series metrics (TimescaleDB).

### 9.3 Data Flow
1) Input arrives at Edge Agent (image/audio/text).
2) Edge preprocessing (denoise, tokenization), partial inference to cut layer.
3) If confidence above threshold at an exit, return local result; else compress features.
4) Send compressed features via TLS to Cloud Gateway; include context (SLA, deadlines, device profile).
5) Policy Engine may adjust batch/deadline and choose model/expert.
6) Model Serving processes features; returns refined output.
7) Gateway responds to Edge Agent; Agent returns to app; caches embeddings/results.
8) Telemetry sent asynchronously to observability pipeline.
9) Offline: replay queues hold requests until connectivity resumes; batched refinement upon reconnection.

## 10. Data Model
### 10.1 Entity Relationships
- Device has many Sessions.
- Session has many Requests.
- Request may have EdgeResult and/or CloudResult.
- Request has Telemetry records.
- Model has many Versions and Policies.
- Policy has Experiments (A/B groups) and Decisions.

### 10.2 Database Schema (PostgreSQL 15+; simplified)
- devices(id PK, org_id, device_type, cpu_info, accel_info, app_version, created_at)
- sessions(id PK, device_id FK, network_type, region, start_at, end_at)
- models(id PK, name, modality, created_at)
- model_versions(id PK, model_id FK, version, cut_layers JSONB, exits JSONB, artifacts_uri, created_at)
- policies(id PK, name, type ENUM(heuristic, bandit, rl), config JSONB, created_at)
- policy_experiments(id PK, policy_id FK, name, traffic_split JSONB, status, created_at)
- requests(id PK, session_id FK, request_ts, modality, sla_ms, route ENUM(edge, cloud, hybrid), status, deadline_ts, idempotency_key, offload_reason, compression_cfg JSONB)
- edge_results(id PK, request_id FK, model_version_id FK, result JSONB, confidence, latency_ms)
- cloud_results(id PK, request_id FK, model_version_id FK, result JSONB, confidence, latency_ms, batch_id)
- telemetry(id PK, request_id FK, metric_name, metric_value, ts)
- costs(id PK, request_id FK, cloud_cost_usd, egress_mb, cpu_ms, gpu_ms)

### 10.3 Data Flow Diagrams (ASCII)
[App/Input] -> [Edge Preprocess] -> [Edge Inference] -> {Early Exit?}
   Yes -> [Return Result] -> [Telemetry]
   No  -> [Compress Features] -> [TLS] -> [Cloud Gateway] -> [Policy+Batch]
         -> [Model Serving] -> [Result] -> [Edge] -> [Return] -> [Telemetry]

### 10.4 Input Data & Dataset Requirements
- CV: RGB images (up to 1080p), video frames at 15–30 FPS; JPEG/PNG; transform pipelines defined.
- NLP: UTF-8 text up to 2k tokens; tokenizers consistent across edge/cloud.
- Audio: 16 kHz mono WAV; VAD on edge recommended.
- Datasets for benchmarking include public corpora per modality; ensure licensing compliance.
- Annotations stored in S3/MinIO; versioned with DVC or LakeFS.

## 11. API Specifications
### 11.1 REST Endpoints (Cloud)
- POST /v1/infer
  - Description: Submit features or raw input for inference; supports hybrid mode.
  - Body: JSON with metadata + base64 tensor or signed URL.
- POST /v1/features
  - Description: Submit compressed intermediate features from edge split.
- GET /v1/models
  - Description: List available models/versions and supported cut layers/exits.
- POST /v1/policy/decide
  - Description: Get offload decision; inputs include RTT, edge load, entropy, SLA.
- POST /v1/telemetry
  - Description: Send metrics/logs.
- POST /v1/cache/get / /v1/cache/put
  - Description: Retrieve/store cached results by content hash.
- POST /v1/deploy/canary
  - Description: Configure canary/shadow deployments.

### 11.2 Request/Response Examples
Example: POST /v1/features
Request:
{
  "request_id": "a1b2c3",
  "device_id": "dev-123",
  "model": "cv_detector",
  "model_version": "1.4.2",
  "cut_layer": "layer_12",
  "compression": {"quant": "int8", "sparsity": 0.5, "codec": "range"},
  "sla_ms": 700,
  "context": {"rtt_ms": 85, "edge_load": 0.6},
  "features_b64": "AAABBB...=="
}
Response:
{
  "status": "ok",
  "route": "cloud",
  "result": {"labels": ["person"], "scores": [0.95], "boxes": [[12,34,200,400]]},
  "confidence": 0.95,
  "latency_ms": 220
}

Example: POST /v1/policy/decide
Request:
{
  "entropy": 0.38,
  "edge_confidence": 0.72,
  "rtt_ms": 120,
  "edge_load": 0.7,
  "deadline_ms": 600,
  "modality": "image",
  "cost_hint": {"budget_usd_per_1k": 1.5}
}
Response:
{
  "decision": "offload",
  "target": {"model": "cv_detector", "version": "1.4.2"},
  "cut_layer": "layer_12",
  "compression": {"quant": "int8", "sparsity": 0.4},
  "expected_latency_ms": 450,
  "confidence_threshold": 0.8
}

### 11.3 Authentication
- OAuth 2.0 client credentials for service-to-service.
- mTLS optional for edge–cloud.
- JWTs (RS256) with short TTL for devices.
- API keys for development only; not allowed in production.

## 12. UI/UX Requirements
### 12.1 User Interface
- Console sections:
  - Overview: SLAs, offload rates, latency heatmaps.
  - Models: Registry, versions, cut layers, exits, artifacts.
  - Policies: Heuristic/bandit/RL configs; canary/shadow controls.
  - Deployments: Regions, scaling, health.
  - Monitoring: Dashboards for latency/accuracy/cost/drift.
  - Settings: Auth, privacy, budgets, alerts.

### 12.2 User Experience
- Guided wizards for adding a model and defining split/exit thresholds.
- Live preview of activation sizes and estimated bandwidth at cut layers.
- Policy simulation tool with historical traces.

### 12.3 Accessibility
- WCAG 2.1 AA compliance.
- Keyboard navigation and ARIA labels.
- High-contrast themes.

## 13. Security Requirements
### 13.1 Authentication
- OAuth2, mTLS, JWT with rotation; device enrollment flow.
### 13.2 Authorization
- RBAC with roles: Admin, Operator, Observer.
- Resource scoping by org/project/region.
### 13.3 Data Protection
- TLS 1.3 in transit; AES-256 at rest.
- Secrets managed via Kubernetes Secrets + external vault.
- Feature anonymization pipeline; optional differential privacy noise.
### 13.4 Compliance
- SOC 2 Type II-ready controls; logging and audit trails.
- GDPR/CCPA alignment: data minimization, right to delete, purpose limitation.

## 14. Performance Requirements
### 14.1 Response Times
- Local early-exit: median < 120 ms, p95 < 300 ms.
- Offloaded requests: p95 < 700 ms, p99 < 1200 ms.
### 14.2 Throughput
- 10k RPS sustained with autoscaling in two regions.
### 14.3 Resource Usage
- Cloud GPU utilization target: 60–85%.
- Edge memory footprint: <200 MB for SDK + model weights where feasible; configurable.

## 15. Scalability Requirements
### 15.1 Horizontal Scaling
- KEDA-driven autoscaling of API/serving based on RPS and queue depth.
### 15.2 Vertical Scaling
- GPU types configurable; dynamic batch sizes per instance.
### 15.3 Load Handling
- Regional load balancing with geo-DNS; spillover to nearest healthy region.
- Deadline-aware scheduling; prioritize near-deadline requests.

## 16. Testing Strategy
### 16.1 Unit Testing
- >80% coverage for SDK routing, compression, and policy components.
### 16.2 Integration Testing
- End-to-end tests with simulated RTT, packet loss, and device load.
- Golden datasets across modalities; regression checks for accuracy and latency.
### 16.3 Performance Testing
- Load tests at 1k/5k/10k RPS; chaos testing (network partitions).
- Tail-latency focus; soak tests 24–72 hours.
### 16.4 Security Testing
- Static analysis (Bandit, ESLint), SAST/DAST, dependency scanning.
- Pen tests on API and console; mTLS certificate rotation drills.

## 17. Deployment Strategy
### 17.1 Deployment Pipeline
- GitHub Actions: lint → unit → build → container scan → integration → deploy to staging → canary → prod.
- IaC with Terraform and Helm.
### 17.2 Environments
- Dev, Staging, Prod across 2–3 regions; feature flags and policy experiments.
### 17.3 Rollout Plan
- Shadow mode for new policies/models; 5% canary → 25% → 50% → 100%.
- Automated rollback on SLA breach or error rates >2x baseline.
### 17.4 Rollback Procedures
- One-click rollback in console; artifact pinning to last known good.
- Replay of affected requests where applicable.

## 18. Monitoring & Observability
### 18.1 Metrics
- Latency: edge-only, edge→cloud, per-region, per-model (p50/p95/p99).
- Accuracy/F1 per modality; calibration ECE/MCE.
- Offload rate, early-exit rate, cache hit rate.
- Costs: GPU-hours, egress MB, cost per 1k requests.
- Drift: embedding distribution distance (e.g., KL/PSI).
### 18.2 Logging
- Structured JSON logs; request IDs; decision traces.
### 18.3 Alerting
- SLA breach alerts; error rate >1%; cost budget thresholds; drift thresholds.
### 18.4 Dashboards
- Grafana boards for latency, throughput, utilization, cost; policy impact panels.

## 19. Risk Assessment
### 19.1 Technical Risks
- Unstable networks leading to missed deadlines.
- Over-aggressive compression hurting accuracy.
- Model/policy regressions in production.
### 19.2 Business Risks
- Cloud cost overruns.
- Compliance findings on data handling.
- Vendor lock-in for serving stack.
### 19.3 Mitigation Strategies
- Offline mode, replay queues, local-only fallback.
- Rate–distortion sweeps, per-modality compression configs.
- A/B testing, canary, shadow; robust rollbacks.
- Multi-provider deploy patterns; portable model formats (ONNX).

## 20. Timeline & Milestones
### 20.1 Phase breakdown
- Phase 0 (2 weeks): Requirements, design, repo scaffolding.
- Phase 1 (6 weeks): Edge SDK v0.1, split inference MVP, cloud API, basic serving.
- Phase 2 (6 weeks): Early-exit, compression pipeline, dynamic batching, monitoring MVP.
- Phase 3 (6 weeks): Policy engine (heuristic + bandit), caching, offline/replay, console alpha.
- Phase 4 (4 weeks): Security hardening, canary/shadow, autoscaling, performance tuning.
- Phase 5 (4 weeks): GA readiness, documentation, SLO validation, cost optimization.

Total: ~28 weeks (approx. 6.5 months).

Budget rough order:
- Engineering: 6 FTE x 6.5 months ≈ 39 FTE-months.
- Cloud: ~$25k–$40k for dev/staging compute and storage.
- Tools/licenses: ~$5k.

### 20.2 Key Milestones
- M1: Split inference E2E (Week 8).
- M2: Early-exit + compression + basic dashboards (Week 14).
- M3: Policy engine + caching + offline resilience (Week 20).
- M4: Security/Compliance + autoscaling + p95 SLA met at 10k RPS (Week 24).
- GA: Documentation complete, >99.5% uptime across 30 days, cost targets met (Week 28).

## 21. Success Metrics & KPIs
### 21.1 Measurable targets
- Accuracy: >90% top-1 CV; >92 F1 NLP; <3% drop vs cloud-only under typical compression.
- Latency: p95 < 700 ms; p99 < 1200 ms for hybrid; local p95 < 300 ms.
- Cost: ≥30% reduction per 1k requests vs cloud-only.
- Availability: ≥99.5% monthly uptime.
- Early-exit rate: ≥40% for designated workloads without accuracy loss >1.5%.
- Offload policy win rate: ≥10% latency or cost improvement vs static baseline in A/B.
- Cache hit rate: ≥20% on repetitive workloads.

## 22. Appendices & Glossary
### 22.1 Technical Background
- Split computing enables moving only intermediate activations from edge to cloud, reducing bandwidth while leveraging powerful models.
- Early-exit/branchy networks provide anytime predictions with uncertainty-aware thresholds.
- Contextual bandits and RL optimize offloading decisions under multi-objective trade-offs.
- Feature compression (quantization, sparsity, entropy coding) balances rate–distortion for activations.
- Calibration techniques (temperature scaling) align confidence with accuracy, crucial for selective offloading.

### 22.2 References
- Teerapittayanon, Natthapon, et al. “BranchyNet: Fast inference via early exiting.” 2016.
- Kang, Y., et al. “Neurosurgeon: Collaborative intelligence between the cloud and mobile edge.” 2017.
- NVIDIA Triton Inference Server docs.
- ONNX Runtime and TensorRT documentation.
- OpenTelemetry, Prometheus, Grafana docs.

### 22.3 Glossary
- Split Inference: Running part of a model on edge and the rest in cloud.
- Cut Layer: The layer where the model is partitioned for feature offloading.
- Early Exit: Exiting inference early at intermediate layers when confidence is high.
- Cascaded Inference: Lightweight first-stage model triggers heavier second-stage processing only as needed.
- Uncertainty/Entropy: Measures of prediction confidence to guide offloading.
- Dynamic Batching: Aggregating requests to improve throughput without violating SLAs.
- Contextual Bandit: Learning algorithm for per-request routing decisions.
- Differential Privacy: Technique to protect privacy by adding noise to shared statistics/features.
- Calibration: Aligning model confidence with true likelihoods.

Repository Structure
- root/
  - README.md
  - notebooks/
    - 01_split_cut_layer_analysis.ipynb
    - 02_early_exit_thresholds.ipynb
    - 03_policy_bandit_simulation.ipynb
  - src/
    - edge_sdk/
      - python/
        - agent.py
        - routing.py
        - compression.py
        - cache.py
        - telemetry.py
      - android/
      - ios/
    - cloud/
      - api/
        - main.py
        - routers/
          - infer.py
          - policy.py
          - telemetry.py
      - policy_engine/
        - heuristic.py
        - bandit.py
        - rl_agent.py
      - serving/
        - triton_client.py
        - batching.py
      - observability/
        - otel.py
    - common/
      - schemas.py
      - auth.py
  - tests/
    - unit/
    - integration/
    - performance/
  - configs/
    - edge_config.yaml
    - cloud_config.yaml
    - policies/
      - heuristic_default.yaml
      - bandit_linucb.yaml
  - data/
    - samples/
  - deploy/
    - helm/
    - terraform/

Config Sample (edge_config.yaml)
device_id: "dev-123"
models:
  - name: "cv_detector"
    version: "1.4.2"
    cut_layer: "layer_12"
    exits:
      - name: "exit_1"
        threshold: 0.85
compression:
  quant: "int8"
  sparsity: 0.4
transport:
  endpoint: "https://api.collab-infer.cloud"
  auth: {type: "jwt", token_path: "/etc/creds/token"}
cache:
  enabled: true
  ttl_seconds: 3600
offline:
  replay_retention_hours: 24

API Code Snippet (FastAPI)
from fastapi import FastAPI, Depends
from pydantic import BaseModel

app = FastAPI()

class FeatureReq(BaseModel):
    request_id: str
    device_id: str
    model: str
    model_version: str
    cut_layer: str
    compression: dict
    sla_ms: int
    context: dict
    features_b64: str

@app.post("/v1/features")
async def features(req: FeatureReq):
    decision = await policy_engine.decide(req.context, req.sla_ms, req.model)
    batch = await scheduler.enqueue(req, decision)
    result = await serving.run(batch)
    return {"status": "ok", "route": "cloud", "result": result.payload, "confidence": result.conf, "latency_ms": result.latency}

Edge Routing Pseudocode
def route_request(input):
    x = preprocess(input)
    act, confs = run_to_cut_layer(x)
    if confs.max() >= threshold and meets_sla(local_latency_estimate()):
        return local_output(confs)
    features = compress(act, quant="int8", sparsity=0.4)
    if offline():
        replay_queue.enqueue(features)
        return fallback_output()
    resp = cloud_infer(features, sla_ms=target_sla)
    cache.store(hash(input), resp)
    return resp

Specific Metrics Targets
- >90% accuracy CV, >92 F1 NLP
- <500 ms average end-to-end latency; p95 < 700 ms hybrid
- 99.5% uptime monthly
- ≥30% cost reduction vs cloud-only
- ≥40% early-exit rate without >1.5% accuracy loss

This PRD defines the complete product scope and technical blueprint for aiml021_edge_cloud_collaborative_inference, enabling adaptive, privacy-preserving, and cost-efficient ML inference across edge and cloud.