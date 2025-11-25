# Product Requirements Document (PRD) / # `aiml019_model_compression_pipeline`

Project ID: aiml019_model_compression_pipeline
Category: AI/ML Platform – Model Optimization & MLOps
Status: Draft (for Review)
Version: v1.0
Last Updated: 2025-11-25

## 1. Overview
### 1.1 Executive Summary
aiml019_model_compression_pipeline is an automated, reproducible pipeline to compress machine learning models for deployment across cloud, edge, and mobile. It orchestrates and validates multiple compression techniques—quantization, pruning, low-rank factorization, weight clustering, knowledge distillation, and architecture optimizations—under accuracy, latency, and size constraints. The system performs multi-objective policy search, provides quantifiable quality/latency tradeoffs, and exports production-ready artifacts for common runtimes.

### 1.2 Document Purpose
This PRD defines requirements, scope, stakeholders, architecture, APIs, data model, testing, deployment, and success metrics for delivering aiml019_model_compression_pipeline as a self-serve product for ML engineers, data scientists, MLOps, and application developers.

### 1.3 Product Vision
Empower teams to ship efficient, affordable, and reliable AI by making model compression turnkey, measurable, and portable—preserving task quality while minimizing cost and latency.

## 2. Problem Statement
### 2.1 Current Challenges
- Large models incur high inference costs, latency, and carbon footprint.
- Manual compression is time-consuming, error-prone, and hard to reproduce.
- Fragmented tooling across frameworks; no standard export path to target runtimes.
- Lack of automated accuracy/regression guardrails and drift monitoring post-compression.
- Mixed-precision and structured pruning policies require deep expertise and repeated trial-and-error.

### 2.2 Impact Analysis
- Cloud spend increases 2–5x for unoptimized models at scale.
- Latency SLAs are missed, degrading user experience and conversion.
- Mobile/edge deployments often infeasible due to model size and memory limits.
- Compliance and reliability risks without standardized evaluation and reporting.

### 2.3 Opportunity
- Centralized pipeline enabling teams to reduce model size by 4–20x and latency by 2–10x with <1% accuracy drop.
- Provide consistent evaluation, reporting, and export across NLP, vision, speech, and tabular models.
- Reduce time-to-market and operational costs via automation and reproducibility.

## 3. Goals and Objectives
### 3.1 Primary Goals
- Automate end-to-end compression with PTQ, QAT, pruning, low-rank, clustering, distillation, and mixed-precision search.
- Maintain task performance within configured tolerances; default ≤1% relative metric drop.
- Export interoperable artifacts (PyTorch TorchScript, TensorFlow SavedModel/TFLite, ONNX) and runtime configs.
- Provide measurable reports including accuracy, latency, throughput, size, and calibration coverage.

### 3.2 Business Objectives
- Reduce inference cost by ≥40% for target workloads within 6 months.
- Enable new device classes by shrinking model binaries by ≥4x on average.
- Shorten model optimization lead time from weeks to hours.
- Achieve platform adoption across ≥6 product teams within first year.

### 3.3 Success Metrics
- Quality retention: ≥99% of baseline (e.g., top-1, F1, BLEU/ROUGE, WER/CER depending on task).
- Latency: P50 < 500 ms (server), < 80 ms (on-device tier where applicable); P95 < 800 ms.
- Availability: ≥99.5% uptime for the service.
- Compression ratio: ≥4x median across reference models.
- Time-to-result: < 6 hours for typical compression runs on a 1B-parameter baseline with calibration and fine-tune.
- User satisfaction (CSAT): ≥4.5/5 from internal users.

## 4. Target Users/Audience
### 4.1 Primary Users
- ML Engineers optimizing models for production.
- Data Scientists owning model quality and experimentation.
- MLOps Engineers managing deployment, observability, and reliability.

### 4.2 Secondary Users
- Backend Engineers integrating compressed models into services.
- Mobile/Edge Developers targeting constrained environments.
- Product Managers evaluating tradeoffs between quality and cost.

### 4.3 User Personas
1) Maya Chen – Senior ML Engineer
- Background: 6 years in NLP and recommendation models; PyTorch power user.
- Pain points: Manual quantization trials; inconsistent results across environments; struggling to keep latency SLAs while retaining F1.
- Goals: Automate mixed-precision and pruning; clear reports to justify tradeoffs; easy ONNX export.

2) Leo Alvarez – MLOps Engineer
- Background: Operates GPU clusters and CI/CD for models; Kubernetes, Terraform, Prometheus.
- Pain points: Drifting performance post-release; lack of standardized artifacts; manual rollback when accuracy dips.
- Goals: Reproducible jobs with config-as-code; guardrails and alerts; canary rollout and auto-rollback.

3) Priya Singh – Product Manager
- Background: Leads vision-based feature; responsible for cost and adoption metrics.
- Pain points: Rising inference costs; opaque quality impacts of compression; slow optimization cycles.
- Goals: Reduce cost >40% with minimal user-visible degradation; standardized dashboards and Cohort/segment quality views.

4) Alex Novak – Mobile Developer
- Background: On-device inference for iOS and Android using TFLite and Core ML wrappers.
- Pain points: App size constraints; variations in runtime operators; need for quantization-friendly models with minimal accuracy loss.
- Goals: Achieve sub-20 MB models with fast cold-start and compatible ops.

## 5. User Stories
- US-001: As an ML Engineer, I want to run PTQ with activation calibration so that I can quickly test 8-bit vs 4-bit tradeoffs.
  Acceptance: Given a baseline model and calibration dataset, pipeline runs PTQ with min-max/KL/percentile calibration and outputs accuracy/latency/size; results available via UI/API.

- US-002: As an ML Engineer, I want automated structured pruning with sensitivity analysis so that I can prune channels with minimal accuracy loss.
  Acceptance: System computes layer-wise sensitivity and produces candidate pruning masks; post-fine-tune accuracy within configured tolerance.

- US-003: As a Data Scientist, I want to train a student via knowledge distillation so that the model is smaller while preserving metrics.
  Acceptance: Teacher–student training runs with KL divergence and optional feature-map distillation; final student evaluated against baseline.

- US-004: As an MLOps Engineer, I want mixed-precision search so that bit-widths are allocated per layer under accuracy constraints.
  Acceptance: Policy search completes within budget and yields per-layer precision config; exported as runtime-ready artifact.

- US-005: As a Backend Engineer, I want an API to fetch the best compressed artifact for a model so that I can integrate it in a service.
  Acceptance: GET /models/{id}/best returns artifact URI, runtime, opset, and perf metadata.

- US-006: As a PM, I want a report comparing cost and quality before/after compression so that I can decide on rollout.
  Acceptance: Downloadable PDF/HTML report showing metrics, cost projections, and risk flags.

- US-007: As a Mobile Developer, I want TFLite export with operator compatibility checks so that I can ship to app stores safely.
  Acceptance: Export validates operators and quantization schema; emits compatibility report.

- US-008: As an SRE/MLOps, I want drift/fairness checks post-compression so that we avoid regressions for key cohorts.
  Acceptance: Segment-wise metrics reported; alerts on threshold breaches.

## 6. Functional Requirements
### 6.1 Core Features
- FR-001: Baseline capture (model weights, datasets, metrics, seeds, env).
- FR-002: PTQ with activation calibration (min-max, KL, percentile), per-tensor/per-channel, symmetric/asymmetric, cross-layer equalization, bias correction, BN folding.
- FR-003: QAT support with training hooks and fake-quant nodes.
- FR-004: Pruning (unstructured, structured channel/filter, block sparsity), sensitivity analysis, iterative pruning, fine-tuning.
- FR-005: Low-rank factorization (SVD, LoRA-style for linear; CP/Tucker for conv).
- FR-006: Weight sharing/clustering (k-means; Huffman-friendly codebook export).
- FR-007: Knowledge distillation (teacher–student; soft targets with KL; feature distillation; contrastive for encoders; multi-task heads).
- FR-008: Architecture optimizations (layer fusion, operator reordering, depthwise/grouped conv replacement, bottlenecking hidden dims, early-exit heads).
- FR-009: Mixed-precision search with multi-objective optimization (quality vs latency/size) and constraints.
- FR-010: Automated evaluation suite across task metrics (top-1/5, F1, BLEU/ROUGE, perplexity, WER/CER), latency/throughput, memory.
- FR-011: Robustness checks (distribution shift sensitivity, outlier robustness, numerical stability under low-bit).
- FR-012: Exporters (TorchScript, ONNX, TFLite) with runtime configs and opset validation.
- FR-013: Reporting (HTML/PDF), lineage and reproducibility (hashes, seeds, configs).
- FR-014: API and UI for job creation, monitoring, and artifact retrieval.
- FR-015: Project-level policies and reusable recipes (yaml-configurable).

### 6.2 Advanced Features
- AF-001: Progressive shrinking and cascaded inference planning (early-exit).
- AF-002: Calibration dataset representativeness scoring.
- AF-003: Auto hyperparameter tuning for fine-tune after pruning/quantization.
- AF-004: Cohort-based fairness metrics and post-compression drift monitoring.
- AF-005: Integration with hardware-accelerated runtimes (ONNX Runtime EPs, OpenVINO, TensorRT) when available.
- AF-006: Cost modeling: estimate monthly savings based on QPS and instance pricing.

## 7. Non-Functional Requirements
### 7.1 Performance
- Pipeline job orchestration must handle ≥100 concurrent jobs.
- Job completion for medium models (<1B params) within 6 hours on a 4x L4 or A10 instance class, given provided calibration and fine-tune budgets.

### 7.2 Reliability
- Service uptime ≥99.5%.
- Job retry with exponential backoff; at-least-once execution semantics; idempotent artifact writes.

### 7.3 Usability
- Clear defaults; wizards for common tasks; tooltips and docs in UI.
- Downloadable templates and example notebooks.

### 7.4 Maintainability
- Modular plugin architecture for compression passes.
- Config-as-code with versioned YAML; semantic versioning; extensive test coverage (≥80% lines).

## 8. Technical Requirements
### 8.1 Technical Stack
- Languages: Python 3.11+, TypeScript 5+, SQL.
- Frameworks:
  - PyTorch 2.3+ with torch.ao.quantization/torch.compile
  - TensorFlow 2.16+ with TensorFlow Model Optimization Toolkit (TFMOT)
  - ONNX 1.16+, ONNX Runtime 1.18+
  - Optional accelerators: OpenVINO 2024.4+, TensorRT 10.0+
- Orchestration: Kubernetes 1.30+, Argo Workflows 3.5+ or Celery 5.4+ with Redis 7+.
- API: FastAPI 0.115+ (Python), OpenAPI 3.1.
- UI: React 18+, Next.js 14+, Material UI 5+.
- Messaging: Kafka 3.6+ or Redis Streams.
- Storage:
  - PostgreSQL 15+ (metadata)
  - S3-compatible object store (artifacts, reports)
  - Redis 7+ (caching, task queues)
- Observability: OpenTelemetry 1.27+, Prometheus 2.53+, Grafana 11+.
- CI/CD: GitHub Actions, Docker 26+, Helm 3.15+, Terraform 1.9+.
- Auth: OAuth2/OIDC (Auth0/Okta), JWT.

### 8.2 AI/ML Components
- Quantization: PTQ, QAT, per-channel weights, mixed-precision, calibration (min-max/KL/percentile), cross-layer equalization.
- Pruning: magnitude-based, structured (channel/filter), block sparsity; iterative with fine-tuning.
- Low-rank: SVD/LoRA for linear, CP/Tucker for conv.
- Weight clustering: k-means with codebook export.
- Distillation: KL divergence to soft targets, FitNets/attention transfer, contrastive distillation for encoders.
- Policy search: Bayesian optimization, evolutionary search, or Optuna-based multi-objective search (accuracy, latency, size).
- Export: TorchScript, ONNX opset 17+, TFLite int8/int4 trials (where supported).
- Evaluation: Metric plugins for vision (top-1/5), NLP (F1, BLEU, ROUGE-L, perplexity), speech (WER/CER), tabular (AUC, F1).

## 9. System Architecture
### 9.1 High-Level Architecture (ASCII)
Users/UI/CLI
   |
   v
[API Gateway (FastAPI)] --(Auth/OIDC)--> [Orchestrator]
                                         |            \
                                         v             v
                                   [Job Queue]     [Policy Search Engine]
                                         |
                                         v
                             [Compression Workers Pool]
                             /      |       |       \
                            v       v       v        v
                        [PTQ]   [Pruning] [Distill] [Export]
                             \      |        |        /
                              v     v        v       v
                               [Evaluator & Reporter]
                                        |
                                        v
                         [Metrics DB]   [Artifact Store]
                               |                |
                               v                v
                           [Dashboards]     [Model Registry]

### 9.2 Component Details
- API Gateway: Receives job requests, validates configs, enforces RBAC.
- Orchestrator: Breaks jobs into stages, schedules on workers, tracks state, retries.
- Workers: Containerized executors running compression passes; GPU/CPU pools.
- Policy Search Engine: Multi-objective tuner for mixed-precision/pruning ratios.
- Evaluator: Runs standardized evaluation and robustness tests.
- Exporters: Produce artifacts for specified runtimes and validate operator compatibility.
- Artifact Store: Versioned model binaries, logs, and reports in S3-compatible storage.
- Metrics DB: PostgreSQL for metadata; Prometheus for runtime metrics.
- Dashboards: Grafana for system metrics; UI reports for model metrics.
- Model Registry: Stores model versions, signatures, and deployment recommendations.

### 9.3 Data Flow
1) Register baseline model and datasets.
2) Create compression job with constraints and target runtimes.
3) Orchestrator runs baseline capture (freeze env, seeds).
4) PTQ pass with calibration; quick evaluation.
5) Pruning pass with sensitivity analysis and fine-tune.
6) Distillation pass (optional student) + optional QAT.
7) Low-rank/clustering if size target unmet.
8) Mixed-precision policy search under constraints.
9) Full verification suite and robustness checks.
10) Export artifacts and generate reports.
11) Persist metrics and lineage; expose via API/UI.

## 10. Data Model
### 10.1 Entity Relationships
- Project 1—N Model
- Model 1—N ModelVersion
- ModelVersion 1—N CompressionJob
- CompressionJob 1—N JobStage (PTQ, pruning, etc.)
- CompressionJob 1—N Artifact
- Dataset can be linked to Project and used by many Jobs
- Policy/Recipe N—N CompressionJob
- MetricRun linked to CompressionJob and ModelVersion

### 10.2 Database Schema (PostgreSQL)
- projects(id PK, name, owner_id, created_at)
- models(id PK, project_id FK, name, task_type, framework, created_at)
- model_versions(id PK, model_id FK, version_tag, hash, path_uri, baseline_metrics JSONB, created_at)
- datasets(id PK, project_id FK, name, type, path_uri, size_bytes, schema JSONB, created_at)
- compression_jobs(id PK, model_version_id FK, status, created_by, policy_id FK, target_runtimes JSONB, constraints JSONB, started_at, completed_at)
- job_stages(id PK, job_id FK, name, status, logs_uri, metrics JSONB, started_at, completed_at)
- policies(id PK, project_id FK, name, config YAML, created_at)
- artifacts(id PK, job_id FK, type, runtime, opset, path_uri, size_bytes, checksum, created_at)
- metric_runs(id PK, job_id FK, dataset_id FK, metrics JSONB, latency_ms_p50, latency_ms_p95, throughput_qps, created_at)
- users(id PK, email, name, role, provider, created_at)
- api_tokens(id PK, user_id FK, token_hash, scopes, created_at, expires_at)
- reports(id PK, job_id FK, html_uri, pdf_uri, created_at)

Indexes: on (model_id, version_tag), (job_id, status), GIN on JSONB fields for metrics/constraints.

### 10.3 Data Flow Diagrams (ASCII)
[ModelVersion] + [Datasets] + [Policy]
        |
        v
 [CompressionJob] --> [JobStages: PTQ -> Prune -> Distill -> Search -> Verify -> Export]
        |                                                       |
        v                                                       v
    [Artifacts] <------------------------------------------- [MetricRuns]
        |
        v
     [Reports]

### 10.4 Input Data & Dataset Requirements
- Calibration dataset: 1k–10k representative samples; configurable loaders.
- Evaluation dataset: task-specific standard splits; support for stratified cohorts.
- Data contracts: schema, preprocessing steps, tokenizers/transforms versioned.
- Privacy: datasets must be free from PII unless encrypted and access-controlled.

## 11. API Specifications
### 11.1 REST Endpoints
- POST /projects
- GET /projects/{project_id}
- POST /models
- POST /models/{model_id}/versions
- GET /models/{model_id}/versions/{version_id}
- POST /jobs (create compression job)
- GET /jobs/{job_id}
- GET /jobs?status=running
- POST /policies
- GET /policies/{policy_id}
- GET /models/{model_id}/best?target_runtime=onnxruntime
- GET /artifacts/{artifact_id}
- GET /reports/{job_id}
- POST /tokens (admin only)

### 11.2 Request/Response Examples
- Create Job (cURL):
  curl -X POST https://api.example.com/jobs \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{
      "model_version_id": "mv_123",
      "policy_id": "pol_789",
      "target_runtimes": ["onnxruntime","tflite"],
      "constraints": {
        "max_metric_drop_pct": 1.0,
        "max_latency_ms_p50": 500,
        "min_compression_ratio": 4.0
      }
    }'

  Response:
  {
    "job_id": "job_456",
    "status": "queued",
    "estimated_completion_minutes": 240
  }

- Get Best Artifact:
  GET /models/m_001/best?target_runtime=onnxruntime
  Response:
  {
    "artifact_id": "art_999",
    "runtime": "onnxruntime",
    "opset": 18,
    "uri": "s3://bucket/models/m_001/artifacts/art_999.onnx",
    "size_mb": 38.4,
    "metrics": {
      "metric_name": "F1",
      "baseline": 0.914,
      "compressed": 0.909,
      "latency_ms_p50": 122,
      "throughput_qps": 82
    }
  }

### 11.3 Authentication
- OAuth2/OIDC with JWTs; scopes: read:*, write:*, admin:*.
- API tokens for service accounts; hashed at rest; short-lived recommended.
- Rate limiting per token and IP; default 1000 RPM.

## 12. UI/UX Requirements
### 12.1 User Interface
- Dashboard: active jobs, job health, completion ETA.
- Model Detail: versions, baselines, recommended artifacts.
- Job Wizard: select datasets, policies, constraints, runtimes.
- Reports: side-by-side baseline vs compressed charts; cohort metrics.
- Artifacts: download links, runtime compatibility badges.
- Policies: editor with YAML validation and templates.

### 12.2 User Experience
- Guided flows for common tasks (PTQ-only, Prune+Fine-tune, Distill).
- Inline explanations for methods and tradeoffs.
- Tooltips for metrics and constraints.
- Dark/light themes; persistent filters.

### 12.3 Accessibility
- WCAG 2.1 AA: keyboard navigation, ARIA labels, color contrast.
- Screen-reader friendly tables and charts.

## 13. Security Requirements
### 13.1 Authentication
- OIDC/OAuth2 with MFA enforcement where available.
- Service account tokens with least privilege.

### 13.2 Authorization
- RBAC: roles (viewer, editor, admin) scoped by project.
- Resource-level permissions for models, datasets, jobs, artifacts.

### 13.3 Data Protection
- TLS 1.2+ in transit; AES-256 at rest for object store.
- Secrets in vault (HashiCorp Vault or cloud KMS); no plaintext in logs.
- Signed URLs for artifact access; time-bound.

### 13.4 Compliance
- GDPR-ready (data subject rights), SOC 2-aligned controls.
- Audit logs for sensitive operations (download, delete, policy changes).

## 14. Performance Requirements
### 14.1 Response Times
- API P50 ≤ 150 ms, P95 ≤ 400 ms under 500 RPM.
- Job scheduling latency ≤ 5 s average.

### 14.2 Throughput
- Sustain ≥100 concurrent jobs; scale to 500 with autoscaling.
- Artifact uploads/downloads at ≥200 MB/s aggregate per region.

### 14.3 Resource Usage
- GPU/CPU pool autoscaling; target GPU utilization ≥70% during training/fine-tune.
- Memory backpressure protection; per-job limits configurable.

## 15. Scalability Requirements
### 15.1 Horizontal Scaling
- Stateless API replicas; sharded workers by capability (GPU vs CPU).
- Partitioned queues; multi-region read replicas for PostgreSQL.

### 15.2 Vertical Scaling
- Larger GPU types for distillation/QAT; CPU-optimized nodes for PTQ/export.

### 15.3 Load Handling
- Pre-warmed worker pools; adaptive batching for evaluation.
- Backoff and queuing for bursty submissions; fair scheduling per project.

## 16. Testing Strategy
### 16.1 Unit Testing
- ≥80% coverage for compression passes, exporters, evaluators.
- Deterministic seeds; golden file comparisons for small models.

### 16.2 Integration Testing
- End-to-end jobs on reference models (ResNet50, BERT-base, Conformer-small).
- Cross-framework export-and-load tests (Torch→ONNX→Runtime).

### 16.3 Performance Testing
- Latency/throughput benchmarks across runtimes and bit-widths.
- Stress tests for 500 concurrent jobs; soak tests 72 hours.

### 16.4 Security Testing
- Static analysis (Bandit), dependency scanning (Snyk).
- AuthZ tests for RBAC; audit log verification.
- Pen tests on API endpoints; secrets leak checks.

## 17. Deployment Strategy
### 17.1 Deployment Pipeline
- GitHub Actions: lint/test → build images → push to registry → Helm deploy to staging → integration tests → promote to prod.
- Infra as code with Terraform; environment configs versioned.

### 17.2 Environments
- Dev: ephemeral namespaces per PR.
- Staging: mirrors prod scale; synthetic datasets.
- Prod: multi-AZ; autoscaling enabled.

### 17.3 Rollout Plan
- Canary 10% traffic for API; shadow jobs for new worker images.
- Metric-based promotion: error rate <0.5%, job success >99%.

### 17.4 Rollback Procedures
- Helm rollback to previous chart; database migrations backward-compatible or gated.
- Artifact compatibility maintained via semantic versioning.

## 18. Monitoring & Observability
### 18.1 Metrics
- System: CPU/GPU utilization, queue depth, job success/failure rates.
- Performance: latency percentiles, throughput, memory footprint.
- Model: baseline vs compressed metrics, compression ratio, calibration coverage.
- Business: cost per 1k inferences, monthly savings estimate.

### 18.2 Logging
- Structured JSON logs with request IDs, job IDs.
- Log levels: INFO for state transitions, DEBUG for internals, ERROR for failures.
- PII redaction filters.

### 18.3 Alerting
- On-call alerts: API error spikes, job failure rate >2%, queue backlog >15 min.
- Model alerts: quality drop > threshold, drift detector triggers.

### 18.4 Dashboards
- Grafana boards for system and job metrics.
- UI reports with metric trends, cohort breakdowns, and export compatibility.

## 19. Risk Assessment
### 19.1 Technical Risks
- Incompatibility with certain operators in target runtimes.
- Accuracy collapse at low bit-widths due to distribution shifts.
- Long policy search times for large models.

### 19.2 Business Risks
- Underestimation of compute costs for fine-tuning.
- Low adoption due to perceived complexity.
- Rapidly evolving frameworks causing maintenance burden.

### 19.3 Mitigation Strategies
- Early operator validation; fallbacks and operator replacement.
- Progressive compression with guardrails and auto-abort thresholds.
- Budget-aware search; caching of sensitivity analyses; pinned dependencies and upgrade windows.

## 20. Timeline & Milestones
### 20.1 Phase Breakdown
- Phase 0 (Week 1–2): Requirements, design, and scaffolding; repo setup.
- Phase 1 (Week 3–6): Core passes (PTQ, pruning, export) and evaluation suite.
- Phase 2 (Week 7–9): Distillation, QAT, mixed-precision search.
- Phase 3 (Week 10–11): UI/UX, reporting, RBAC, observability.
- Phase 4 (Week 12): Hardening, docs, pilot rollout.

### 20.2 Key Milestones
- M1: Baseline capture + PTQ demo (Wk 4)
- M2: Structured pruning + fine-tune (Wk 6)
- M3: Distillation + QAT (Wk 9)
- M4: Mixed-precision search + exporters (Wk 10)
- M5: Reports + UI + APIs GA (Wk 12)

Estimated team: 5 FTE (2 ML Eng, 1 MLOps, 1 Backend, 1 Frontend).
Estimated compute cost during build: $8k–$15k/month.
Ongoing ops: $3k–$7k/month excluding user workloads.

## 21. Success Metrics & KPIs
### 21.1 Measurable Targets
- ≥4x median compression ratio across reference models.
- ≤1% relative drop in primary metric (or configurable).
- P50 latency reduction ≥2x; P95 reduction ≥1.5x.
- ≥99.5% service uptime.
- ≥6 internal teams adopting within 6 months; ≥20 active users/month.
- Time-to-first-result ≤2 hours for small/medium models (<400M params).

## 22. Appendices & Glossary
### 22.1 Technical Background
- Quantization: PTQ vs QAT; dynamic vs static; per-tensor vs per-channel; symmetric/asymmetric; calibration via min-max, KL, percentile; cross-layer equalization, bias correction.
- Pruning: magnitude, structured channel/filter, block sparsity; one-shot vs iterative; sensitivity analysis; fine-tuning for recovery.
- Factorization: SVD/LoRA for dense layers; CP/Tucker for convolutional tensors.
- Weight clustering: k-means tying; codebooks; entropy coding friendly distributions.
- Knowledge distillation: KL divergence to soft targets; FitNets/attention transfer; contrastive distillation for encoders; multi-task heads preservation.
- Architecture tweaks: BN folding, operator fusion, depthwise separable convs, bottlenecks, early-exit heads.
- Policy search: multi-objective (quality/latency/size); progressive shrinking; budget-aware search.
- Vector search parallels: PQ/OPQ codebooks, ANN indexing tradeoffs, recall-latency curves akin to accuracy-compression curves.

### 22.2 References
- Jacob et al., Quantization and Training of Neural Networks for Efficient Inference.
- He et al., Channel Pruning for Accelerating Very Deep Networks.
- Frankle & Carbin, Lottery Ticket Hypothesis.
- Hinton et al., Distilling the Knowledge in a Neural Network.
- Howard et al., MobileNetV2 (depthwise separable convs).
- TensorFlow Model Optimization Toolkit docs.
- PyTorch quantization docs.
- ONNX Runtime performance tuning guides.
- Optuna multi-objective optimization docs.

### 22.3 Glossary
- PTQ: Post-Training Quantization.
- QAT: Quantization-Aware Training.
- Per-channel: Independent scales per output channel of a weight tensor.
- Mixed-precision: Assigning different bit-widths per layer or tensor type.
- Sensitivity analysis: Estimating metric drop per layer compression to guide pruning.
- Knowledge distillation: Training a smaller student model to match a larger teacher.
- Calibration dataset: Small sample used to estimate activation ranges and scales.
- Compression ratio: Baseline size divided by compressed size.
- Throughput (QPS): Queries per second handled by the model service.
- Robustness: Stability under distribution shift and outliers.

-------------------------
Repository Structure
- notebooks/
  - 01_baseline_capture.ipynb
  - 02_ptq_calibration.ipynb
  - 03_pruning_sensitivity.ipynb
  - 04_distillation_qat.ipynb
  - 05_mixed_precision_search.ipynb
- src/
  - api/
    - main.py (FastAPI app)
    - routers/
      - projects.py
      - models.py
      - jobs.py
      - policies.py
      - artifacts.py
  - orchestrator/
    - scheduler.py
    - worker.py
    - stages/
      - ptq.py
      - pruning.py
      - distillation.py
      - lowrank.py
      - clustering.py
      - search.py
      - export.py
      - evaluate.py
  - ml/
    - datasets/
    - metrics/
    - exporters/
    - runtimes/
  - utils/
    - config.py
    - logging.py
    - registry.py
- configs/
  - policy_examples/
    - ptq_8bit.yaml
    - prune50_finetune.yaml
    - distill_bert_base_to_mini.yaml
    - mixed_precision_latency_500ms.yaml
- tests/
  - unit/
  - integration/
  - performance/
- data/
  - calibration_samples/
  - evaluation/
- scripts/
  - register_model.py
  - create_job.py
  - fetch_best.py
- infra/
  - helm/
  - terraform/

Config Sample (YAML)
policy:
  name: "prune_then_ptq"
  stages:
    - pruning:
        type: "structured_channel"
        target_sparsity: 0.5
        schedule: "iterative"
        sensitivity: "magnitude"
        finetune:
          epochs: 3
          lr: 3e-5
    - ptq:
        bits:
          weights: [8, 4]
          activations: [8]
        calibration:
          method: "kl"
          samples: 2048
        per_channel_weights: true
        symmetric: true
    - search:
        objective: ["accuracy", "latency_ms_p50", "size_mb"]
        constraints:
          max_metric_drop_pct: 1.0
          max_latency_ms_p50: 500
        budget:
          trials: 40
          max_hours: 2

Python API Usage
from client import Aiml019Client

client = Aiml019Client(base_url="https://api.example.com", token="...")

mv = client.register_model_version(
    model_id="m_001",
    path_uri="s3://bucket/baselines/bert-base.pt",
    task_type="text_classification",
    framework="pytorch",
    baseline_metrics={"F1": 0.914}
)

job = client.create_job(
    model_version_id=mv["id"],
    policy_id="pol_prune_ptq",
    target_runtimes=["onnxruntime"],
    constraints={"max_metric_drop_pct": 1.0, "max_latency_ms_p50": 500}
)

print("Job:", job["job_id"])

Exported Artifact Metadata (JSON)
{
  "artifact_id": "art_123",
  "runtime": "onnxruntime",
  "opset": 18,
  "compression": {
    "quantization": "int8_per_channel_weights_symmetric",
    "pruning": "structured_channel_50pct",
    "low_rank": "none"
  },
  "metrics": {
    "top1": {"baseline": 76.2, "compressed": 75.9},
    "latency_ms_p50": 118,
    "latency_ms_p95": 212,
    "throughput_qps": 85.4
  },
  "compatibility": {
    "operators_validated": true,
    "warnings": []
  }
}

ASCII Architecture Diagram (detailed)
+------------------+       +-------------------+       +------------------+
|   Web UI/CLI     | <---> |   API Gateway     | <---> |   Auth Provider  |
+------------------+       +-------------------+       +------------------+
           |                          |
           v                          v
     +-----------+             +-------------+        +---------------------+
     |  Metrics  | <---------> | Orchestrator| <----> | Policy Search Engine|
     |   DB      |             +-------------+        +---------------------+
           ^                         |
           |                         v
     +-----------+          +-------------------+
     | Dashboards|          |  Job Queue        |
     +-----------+          +--------+----------+
                                     |
                          +----------+----------+
                          | Compression Workers |
                          +----+-----+-----+----+
                               |     |     |
                              PTQ  Prune  Distill
                               \     |     /
                                \    |    /
                              +-------------+
                              | Evaluator   |
                              +------+------+
                                     |
                         +-----------+-----------+
                         |      Artifact Store   |
                         +-----------------------+

Performance Targets
- Primary metric retention ≥99% baseline.
- P50 latency < 500 ms (server), < 80 ms (on-device target tier).
- Service uptime ≥99.5%.
- Compression ratio ≥4x median; ≥2x throughput increase where CPU-bound.

Cost Model (illustrative)
- Typical run (BERT-base, PTQ+prune+finetune 3 epochs): ~12 GPU-hours on L4 → ~$15–$30.
- Distillation small student (5 epochs): 40–80 GPU-hours → ~$120–$300.
- Team monthly ops (50 jobs): $2k–$5k, depending on accelerators and datasets.

Data Governance
- Dataset access via signed URLs; lineage tracked per job.
- PIIs must be masked/preprocessed before upload; encryption enforced.

End of PRD.