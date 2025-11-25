# Product Requirements Document (PRD) / # `aiml033_out_of_distribution_detection`

Project ID: AIML-033
Category: AI/ML — Model Reliability & Safety
Status: Draft for Review
Version: 1.0.0
Last Updated: 2025-11-25

## 1. Overview
### 1.1 Executive Summary
aiml033_out_of_distribution_detection is a production-ready platform and SDK for detecting out-of-distribution (OOD) inputs across computer vision, NLP, audio, and tabular ML systems. It provides post-hoc OOD scoring, training-time techniques (Outlier Exposure), calibration and abstention policies, drift monitoring, and routing/guardrails to prevent overconfident incorrect predictions. It exposes REST APIs, a UI for configuration and monitoring, SDKs for Python/JS, and integrations for cloud deployment.

### 1.2 Document Purpose
Define requirements, architecture, APIs, metrics, and delivery plan for an OOD detection solution that integrates with existing ML inference pipelines to increase reliability, reduce risk, and improve user trust.

### 1.3 Product Vision
A unified OOD detection layer that:
- Works with any model (black-box or white-box) and any modality.
- Is easy to integrate (one-line SDK or REST).
- Provides measurable safety improvements via calibrated abstention, routing, and monitoring.
- Operates at scale with low latency and high availability.

## 2. Problem Statement
### 2.1 Current Challenges
- Models are overconfident on unfamiliar inputs.
- Lack of standardized OOD mechanisms across modalities and frameworks.
- Thresholds don’t transfer across domains; manual tuning is error-prone.
- Limited visibility into dataset shift/drift post-deployment.
- Compliance and risk teams need auditable guardrails.

### 2.2 Impact Analysis
- Reduced user trust due to incorrect high-confidence outputs.
- Elevated operational costs from unbounded manual review.
- Increased risk of policy violations and liability.
- Degraded downstream performance in RAG/routing systems.

### 2.3 Opportunity
Deliver a cross-domain, extensible OOD solution with proven methods (energy scores, Mahalanobis, ensembles, conformal prediction) and continuous monitoring that improves safety while preserving speed.

## 3. Goals and Objectives
### 3.1 Primary Goals
- Detect OOD inputs with AUROC ≥ 0.95 on standard benchmarks.
- Provide configurable abstention/routing with p95 latency < 500 ms end-to-end.
- Support both post-hoc and training-time OOD strategies.
- Offer centralized monitoring and drift alerts.

### 3.2 Business Objectives
- Reduce critical mispredictions by ≥ 60%.
- Lower manual review costs by ≥ 30%.
- Improve SLA reliability to 99.5% uptime.
- Enable compliance-ready audit trails.

### 3.3 Success Metrics
- OOD AUROC ≥ 0.95; FPR@95%TPR ≤ 20%.
- Calibration ECE ≤ 2.5% after temperature scaling.
- Risk-coverage AUC improvements ≥ 20% vs. baseline.
- p95 inference latency ≤ 500 ms; availability ≥ 99.5%.
- False accept rate at selected operating point ≤ 5% on validation.

## 4. Target Users/Audience
### 4.1 Primary Users
- ML Engineers integrating OOD detection into inference pipelines.
- Data Scientists calibrating thresholds and evaluating metrics.
- MLOps/Platform Engineers managing deployment and monitoring.

### 4.2 Secondary Users
- Product Managers defining abstention UX and routing.
- Risk/Compliance teams auditing safety mechanisms.
- LLM/RAG engineers configuring retrieval guardrails.

### 4.3 User Personas
1) Priya Nair — Senior ML Engineer
- Background: 7 years in CV/NLP production systems; PyTorch, FastAPI.
- Pain points: High-confidence failures in new geographies; long debug cycles.
- Goals: Drop-in OOD module; low latency; transparent metrics; CI/CD friendly.

2) Alex Chen — Data Scientist
- Background: Statistical ML; calibration and evaluation; scikit-learn expert.
- Pain points: Ad-hoc thresholding; lack of drift visibility; poor reproducibility.
- Goals: Robust threshold calibration; risk-coverage analysis; reproducible reports.

3) Jamie Rodriguez — MLOps Lead
- Background: Kubernetes, observability, SRE practices.
- Pain points: SLA breaches; no alerts on distribution shift; fragmented logs.
- Goals: 99.5% uptime; actionable alerts; scalable rollout; safe rollback.

4) Morgan Lee — LLM/RAG Engineer
- Background: Retrieval systems; vector databases; embeddings.
- Pain points: Hallucinations due to low similarity queries; weak guardrails.
- Goals: Embedding-space OOD filtering; conformal thresholds; routing decisions.

## 5. User Stories
US-001: As an ML Engineer, I want a REST API to score OOD likelihood for an input so that I can gate my model’s prediction.
- Acceptance: POST /v1/ood/score returns score, label (in/out), threshold_id, p95 latency < 200 ms for scoring (excluding model inference).

US-002: As a Data Scientist, I want to calibrate thresholds on a validation set so that I can control FPR to a target.
- Acceptance: POST /v1/ood/calibrate supports target FPR; returns threshold and ROC summary.

US-003: As an MLOps Lead, I want drift alerts when embedding distribution shifts so that I can mitigate risk.
- Acceptance: Alerts fire when PSI > 0.2 or Mahalanobis > configured threshold for 5-minute windows.

US-004: As a Product Manager, I want abstention/routing policies configurable per project so that UX remains consistent.
- Acceptance: UI supports rules: abstain, human-review, route-to-tool, fallback; audit log created.

US-005: As a DS, I want to compare OOD methods (MSP, ODIN, Energy, Mahalanobis, kNN) so that I choose best trade-off.
- Acceptance: Experiments view displays AUROC, FPR@95, latency, and ECE per method.

US-006: As a RAG Engineer, I want conformal retrieval thresholds so that low-similarity queries are rejected predictably.
- Acceptance: Nonconformity = 1 - max cosine; quantile-calibrated threshold provides ≤ 5% false accept on calibration set.

US-007: As a Security Officer, I want audit logs for decisions so that I can demonstrate compliance.
- Acceptance: Each OOD decision includes timestamp, features, score, threshold version, user/project, and outcome.

US-008: As an ML Engineer, I want SDK hooks to extract penultimate features so that feature-space OOD works with my model.
- Acceptance: Python SDK provides torch hook utilities and ONNX intermediate extraction.

US-009: As a DS, I want to run Outlier Exposure training on auxiliary data so that my classifier learns a reject boundary.
- Acceptance: Training pipeline supports OE; logs loss curves; validates improvements vs baseline.

## 6. Functional Requirements
### 6.1 Core Features
FR-001 OOD Scoring: Support MSP, ODIN (temperature + perturbation), Energy-based scores.
FR-002 Feature-space OOD: Mahalanobis, kNN density, Local Outlier Factor on penultimate-layer features.
FR-003 Uncertainty Estimation: Deep ensembles, MC Dropout; calibration via temperature scaling.
FR-004 Generative Scores: Likelihood ratio/WAIC for flows/VAEs (optional).
FR-005 Self-supervised features: Use SimCLR/CLIP-style embeddings; centroid/kNN scoring.
FR-006 Threshold Calibration: Quantile-based, ROC-based, and conformal.
FR-007 Abstention & Routing: Policy engine for reject/route/fallback with reason codes.
FR-008 Drift Monitoring: Track embedding distribution, MMD/PSI, FID-like stats; alerts.
FR-009 Datasets/Experiments: Benchmarking modules for CV and NLP tasks.
FR-010 SDKs: Python and JavaScript for easy integration with models and apps.
FR-011 UI Console: Configure thresholds, visualize ROC/PR, risk-coverage, drift dashboards.
FR-012 Audit & Reporting: Exportable reports; decision logs; model versions and thresholds.

### 6.2 Advanced Features
- FR-013 Outlier Exposure Training: Integrate auxiliary OOD data; margin/energy regularization; unknown class (optional).
- FR-014 Conformal Prediction: Risk-controlling thresholds and risk-coverage curves.
- FR-015 Index-aware Signals for Vector Search: HNSW expansion count, IVF/PQ residual diagnostics.
- FR-016 Human-in-the-loop Feedback: Label OOD/ID and refine thresholds online.
- FR-017 Multi-modal Support: CV, NLP, audio, tabular via unified embedding interface.
- FR-018 Router Integration: RAG routing to tools/web when OOD suspected.

## 7. Non-Functional Requirements
### 7.1 Performance
- Scoring p95 latency: ≤ 200 ms (excluding upstream model inference); end-to-end ≤ 500 ms.
- Throughput: ≥ 300 req/s per pod at batch=8 on GPU; graceful degradation to CPU ≤ 100 req/s.

### 7.2 Reliability
- Uptime: ≥ 99.5%.
- Zero data loss for decision logs; at-least-once processing on Kafka streams.

### 7.3 Usability
- UI tasks discoverable within 3 clicks.
- SDK integration in < 20 lines of code.

### 7.4 Maintainability
- 85% unit test coverage for core scoring and calibration.
- Backward-compatible APIs with semantic versioning.

## 8. Technical Requirements
### 8.1 Technical Stack
- Backend: Python 3.11+, FastAPI 0.115+, Uvicorn 0.30+, PyTorch 2.4+, TorchVision 0.19+, HuggingFace Transformers 4.45+, scikit-learn 1.5+, faiss 1.8+, PyTorch Lightning 2.4+.
- Inference: ONNX Runtime 1.19+, NVIDIA Triton (optional), CUDA 12.x.
- Frontend: React 18+, TypeScript 5+, Next.js 14+, TailwindCSS 3+.
- Data/Storage: PostgreSQL 15+, Redis 7+, Object Storage (S3/GCS), Kafka 3.6+.
- Orchestration: Docker, Kubernetes 1.29+, Helm 3+, Argo CD/GitHub Actions.
- Observability: Prometheus, Grafana, OpenTelemetry, Loki, Sentry.
- Auth: OAuth2/OIDC, JWT.

### 8.2 AI/ML Components
- Embedding backbones: ResNet/ViT for images; BERT/RoBERTa/Distil for text; CLIP for multi-modal; wav2vec2 for audio.
- OOD methods: MSP, ODIN, Energy-based, Mahalanobis, kNN/LOF, deep ensembles, MC Dropout, likelihood ratio/WAIC.
- Calibration: Temperature scaling, Dirichlet prior networks (optional), conformal thresholds.
- Drift: MMD, PSI, Mahalanobis on embedding mean/cov, Frechet distance-like stats.

## 9. System Architecture
### 9.1 High-Level Architecture (ASCII diagram)
Client/UI/SDK
    |
    v
+-------------------+        +--------------------+        +-------------------+
|  API Gateway      | -----> | OOD Scoring Svc    | <----> | Embedding Extract |
| (FastAPI)         |        | (methods/threshold)|        | (Backbone/ONNX)   |
+-------------------+        +--------------------+        +-------------------+
         |                               |                           |
         v                               v                           v
+-------------------+        +--------------------+        +-------------------+
| Policy Router     | -----> | Decision Logger    | -----> | PostgreSQL/Object |
| (abstain/route)   |        | + Kafka producer   |        | Storage           |
+-------------------+        +--------------------+        +-------------------+
         |
         v
+-------------------+        +--------------------+
| Drift Monitor     | <----> | Metrics/Alerts     |
| (batch/stream)    |        | (Prometheus/Alrt)  |
+-------------------+        +--------------------+

### 9.2 Component Details
- API Gateway: AuthN/Z, request validation, rate limiting, routing to scoring/calibration/monitoring endpoints.
- Embedding Extractor: Loads model, returns logits and features; supports GPU/CPU; caching of model weights.
- OOD Scoring Service: Implements methods, manages thresholds, returns scores/labels.
- Policy Router: Applies configured policies: accept, abstain, human-review, route to tool.
- Decision Logger: Writes immutable logs with scores, thresholds, reasons.
- Drift Monitor: Periodic/stream analysis of embeddings and scores; triggers alerts.
- Storage: PostgreSQL for metadata; object storage for datasets/models; Redis for hot thresholds; Kafka for event streaming.

### 9.3 Data Flow
- Training: Ingest ID dataset; optional OE with auxiliary OOD; train backbone/classifier; export embeddings and calibration set.
- Calibration: Compute scores on validation; pick threshold via target FPR or conformal; store ThresholdVersion.
- Serving: For each request, extract embedding/logits; compute OOD score; compare to threshold; apply policy; log decision.
- Monitoring: Aggregate recent embeddings; compute drift metrics; alert if thresholds exceeded.

## 10. Data Model
### 10.1 Entity Relationships
- Project 1—N ModelVersion
- ModelVersion 1—N ThresholdVersion
- InferenceRequest 1—1 OODDecision
- OODDecision N—1 ThresholdVersion
- DriftWindow 1—N DriftMetric
- User N—M Project (via ProjectRole)
- Feedback N—1 OODDecision

### 10.2 Database Schema (selected)
- users(id PK, email, name, role, created_at)
- projects(id PK, name, org_id, created_at)
- project_roles(id PK, project_id FK, user_id FK, role)
- model_versions(id PK, project_id FK, name, framework, uri, created_at)
- thresholds(id PK, model_version_id FK, method, parameters JSONB, metric_target, created_at, active bool)
- inference_requests(id PK, project_id FK, model_version_id FK, input_hash, modality, received_at)
- ood_decisions(id PK, inference_request_id FK, threshold_id FK, score float, label enum[in,out], reason, latency_ms, created_at)
- drift_windows(id PK, project_id FK, window_start, window_end)
- drift_metrics(id PK, drift_window_id FK, metric_name, value, threshold, status)
- audits(id PK, project_id FK, actor, action, payload JSONB, created_at)
- feedback(id PK, ood_decision_id FK, user_id FK, label enum[in,out], notes, created_at)

### 10.3 Data Flow Diagrams
[Client] -> [API] -> [Embedder] -> [OOD Scorer] -> [Policy Router] -> [Decision Log]
                                    ^                                   |
                                    |                                   v
                             [Threshold Store]                    [Feedback/Audit]

### 10.4 Input Data & Dataset Requirements
- ID training/validation sets with labels or pseudo-labels for embeddings.
- Auxiliary OOD for OE: broad coverage and domain-agnostic; ensure no label leakage.
- For benchmarks: image (CIFAR-10/100 ID vs SVHN, TinyImageNet, LSUN, ImageNet-O as OOD); text (intent classification datasets plus OOD corpora).
- Ensure balanced calibration set and hold-out OOD for evaluation to avoid spurious correlations.

## 11. API Specifications
### 11.1 REST Endpoints
- POST /v1/ood/score
  - Body: { project_id, model_version_id, method, input (URL|bytes|text), modality, params?, threshold_id? }
  - Returns: { score, label: in|out, threshold_id, method, reasons[], latency_ms, request_id }

- POST /v1/ood/calibrate
  - Body: { project_id, model_version_id, method, dataset_uri, target: { type: fpr|tpr|quantile, value }, conformal?: { alpha } }
  - Returns: { threshold_id, method, threshold_value, metrics: { auroc, fpr95, ece }, curve_uri }

- GET /v1/thresholds/{project_id}/{model_version_id}
  - Returns active and historical thresholds.

- POST /v1/router/policy
  - Body: { project_id, rules: [{ condition: { method, operator, value }, action: abstain|review|route, route_to? }] }
  - Returns: { policy_id, status }

- POST /v1/drift/ingest
  - Body: { project_id, embeddings: [[...]], scores: [...], window: { start, end } }
  - Returns: { window_id, computed_metrics: [...] }

- GET /v1/drift/status?project_id=...
  - Returns current drift metrics and alert status.

- POST /v1/feedback
  - Body: { request_id, label: in|out, notes }
  - Returns: { feedback_id }

- GET /v1/audit?project_id=...&from=...&to=...
  - Returns audit entries.

### 11.2 Request/Response Examples
Request:
curl -X POST https://api.example.com/v1/ood/score \
 -H "Authorization: Bearer <jwt>" \
 -H "Content-Type: application/json" \
 -d '{"project_id":"prj_123","model_version_id":"mod_v2","method":"energy","modality":"image","input":"https://.../cat.png"}'

Response:
{
  "score": -8.42,
  "label": "in",
  "threshold_id": "thr_789",
  "method": "energy",
  "reasons": ["below_threshold(-6.0)"],
  "latency_ms": 147,
  "request_id": "req_abcd"
}

### 11.3 Authentication
- OAuth2/OIDC with JWT bearer tokens.
- Project-scoped API keys (optional) with RBAC.
- mTLS between internal services.

## 12. UI/UX Requirements
### 12.1 User Interface
- Dashboards: AUROC, FPR@95, ECE, risk-coverage curves, drift metrics (PSI/MMD), latency.
- Threshold management: create/calibrate/activate; compare versions.
- Policy builder: condition-action rules; test sandbox.
- Experiment compare: multi-method leaderboard with metrics and cost/latency.
- Decision log explorer: filters, drill-down, export CSV/JSON.

### 12.2 User Experience
- Onboarding wizard: connect model, run baseline calibration, enable default policy.
- Contextual help and method tooltips.
- One-click “Validate on Benchmark” to reproduce report.

### 12.3 Accessibility
- WCAG 2.1 AA compliance.
- Keyboard navigation, ARIA labels, high-contrast mode.

## 13. Security Requirements
### 13.1 Authentication
- OIDC integration with major IdPs; short-lived JWTs; refresh tokens with rotation.

### 13.2 Authorization
- Project-level RBAC: Admin, Editor, Viewer, Auditor roles.
- Least-privilege for service accounts.

### 13.3 Data Protection
- TLS 1.2+ in transit; AES-256 at rest.
- PII minimization; data retention policies; encryption keys via KMS.

### 13.4 Compliance
- SOC 2 Type II process alignment.
- GDPR/CCPA support: data subject requests; audit logging.

## 14. Performance Requirements
### 14.1 Response Times
- OOD score endpoint: p50 ≤ 100 ms, p95 ≤ 200 ms, p99 ≤ 350 ms (scoring only).
- End-to-end with embedding extraction: p95 ≤ 500 ms on GPU.

### 14.2 Throughput
- ≥ 300 req/s/pod GPU; scale linearly to 3,000 req/s with 10 pods.

### 14.3 Resource Usage
- GPU memory footprint ≤ 4 GB for typical backbones (ViT-B/16, BERT-base).
- CPU mode: ≤ 2 vCPU and 4 GB RAM per pod at 100 req/s.

## 15. Scalability Requirements
### 15.1 Horizontal Scaling
- Stateless scoring pods with HPA based on QPS and latency KPIs.

### 15.2 Vertical Scaling
- Auto-select optimized runtime (ONNX/TensorRT) and batch sizes.

### 15.3 Load Handling
- Rate limiting per API key; circuit breakers; adaptive batching with max delay 20 ms.

## 16. Testing Strategy
### 16.1 Unit Testing
- Method correctness (MSP, ODIN, Energy, Mahalanobis).
- Threshold calculators; calibration math; serialization.

### 16.2 Integration Testing
- End-to-end scoring flow with embeddings and router.
- DB/Kafka integration; idempotent logging.

### 16.3 Performance Testing
- Load tests at 10x expected QPS; soak tests 24h.
- Latency under various payload sizes and batch configs.

### 16.4 Security Testing
- AuthZ bypass attempts; JWT tampering; OWASP API Top 10.
- Data leakage scans for logs and datasets.

## 17. Deployment Strategy
### 17.1 Deployment Pipeline
- GitHub Actions: lint, tests, security scans, build images, push to registry.
- Automated model cards and method metadata bundling.

### 17.2 Environments
- Dev → Staging → Production; isolated VPCs; separate secrets.

### 17.3 Rollout Plan
- Canary release to 5% traffic; monitor AUROC proxy and latency; ramp in 24–48h.

### 17.4 Rollback Procedures
- One-click Helm rollback; revert threshold versions; replay Kafka events as needed.

## 18. Monitoring & Observability
### 18.1 Metrics
- Service: QPS, latency p50/p95/p99, error rates, GPU/CPU utilization.
- OOD: score distributions, acceptance rates, AUROC on shadow labels, FPR@95 drift.
- Calibration: ECE trends; temperature parameters.
- Drift: PSI, MMD, Mahalanobis distance, embedding norms.

### 18.2 Logging
- Structured JSON with request_id; masked sensitive data.
- Decision logs with method, threshold, outcome, reasons.

### 18.3 Alerting
- SLO breach: latency or availability.
- Drift thresholds exceeded.
- Spike in abstentions or false accepts (if labels available).

### 18.4 Dashboards
- Grafana: service health, OOD performance, drift drill-down.
- Business KPIs: reduction in critical errors, review volume.

## 19. Risk Assessment
### 19.1 Technical Risks
- Thresholds not transferring across domains.
- Likelihood paradox in generative models.
- Overfitting to proxy OOD during calibration.
- Latency spikes on large models.

### 19.2 Business Risks
- Insufficient coverage leads to false security.
- Increased abstentions frustrate users.
- Compute costs for ensembles.

### 19.3 Mitigation Strategies
- Conformal thresholds; periodic recalibration.
- Use likelihood ratios/WAIC, avoid raw likelihoods.
- Diverse OE datasets and synthetic negatives.
- Adaptive batching and model distillation for speed.

## 20. Timeline & Milestones
### 20.1 Phase breakdown
- Phase 0 — Discovery & Design: 2 weeks
- Phase 1 — Prototype (MSP, Energy, Mahalanobis, UI basics): 4 weeks
- Phase 2 — MVP (calibration, router, drift, SDKs, APIs): 6 weeks
- Phase 3 — Hardening (security, performance, canary, docs): 4 weeks
- Phase 4 — Pilot with 2 teams (NLP + CV): 4 weeks
- Phase 5 — GA (SLA, billing, support): 4 weeks
Total: 24 weeks

### 20.2 Key Milestones
- M1: Architecture and repo scaffolding (end of week 2)
- M2: Core OOD methods passing AUROC ≥ 0.9 on benchmarks (week 6)
- M3: Calibration + router integrated, p95 ≤ 500 ms E2E (week 12)
- M4: Drift monitoring and alerts live (week 16)
- M5: Pilot success metrics hit (week 20)
- M6: GA release (week 24)

Estimated Costs (6 months):
- Team: 1 PM, 2 ML Eng, 1 DS, 2 MLE/Platform, 1 FE Eng, 1 SRE (~$900k total loaded)
- Infra: $6k–$12k/month GPU/CPU; staging/prod environments; monitoring stack

## 21. Success Metrics & KPIs
### 21.1 Measurable targets
- AUROC ≥ 0.95 on CIFAR10 vs SVHN/TinyImageNet; FPR@95 ≤ 20%.
- NLP intent OOD AUROC ≥ 0.92; false accept ≤ 5% at operating point.
- Calibration ECE ≤ 2.5% after temperature scaling.
- p95 E2E latency ≤ 500 ms; uptime ≥ 99.5%.
- ≥ 60% reduction in high-severity mispredictions.
- ≥ 30% reduction in manual review workload.

## 22. Appendices & Glossary
### 22.1 Technical Background
- OOD vs. open-set recognition vs. anomaly detection.
- Shift types: covariate (p(x)), label (p(y)), concept (p(y|x)).
- Post-hoc scoring:
  - MSP: max softmax; ODIN uses temperature + small perturbation to sharpen separation.
  - Energy: log-sum-exp of logits; often superior to MSP.
  - Feature-space: Mahalanobis distance to class centroids; kNN density; LOF.
  - Uncertainty: ensembles, MC Dropout for epistemic uncertainty.
  - Generative: raw likelihoods are unreliable; use likelihood ratios or WAIC.
- Training-time:
  - Outlier Exposure (OE) with auxiliary data; margin/energy penalties; unknown class.
- Calibration/abstention: temperature scaling; conformal prediction; risk-coverage analysis.

### 22.2 References
- Liu et al., “Energy-based Out-of-distribution Detection,” NeurIPS.
- Hendrycks & Gimpel, “A Baseline for Detecting Misclassified and Out-of-Distribution Examples.”
- Liang et al., “Enhancing the Reliability of Out-of-distribution Image Detection in Neural Networks” (ODIN).
- Lee et al., “A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks” (Mahalanobis).
- Nalisnick et al., “Do Deep Generative Models Know What They Don’t Know?” (likelihood paradox).
- conformal prediction resources and risk-coverage literature.

### 22.3 Glossary
- In-Distribution (ID): Inputs drawn from the same distribution as the training data.
- Out-of-Distribution (OOD): Inputs outside the training distribution support.
- AUROC: Area under ROC curve; threshold-agnostic separability.
- FPR@95%TPR: False Positive Rate at 95% True Positive Rate.
- ECE: Expected Calibration Error; measures confidence calibration.
- MSP: Maximum Softmax Probability.
- ODIN: OOD detection with temperature scaling and input perturbation.
- Energy Score: Negative energy from logits; lower (more negative) often indicates ID.
- Mahalanobis Distance: Distance using class covariance; applied to features.
- Conformal Prediction: Framework for finite-sample guarantees via nonconformity scores.
- PSI/MMD: Population Stability Index / Maximum Mean Discrepancy for drift.

Repository structure
- aiml033_out_of_distribution_detection/
  - README.md
  - configs/
    - default.yaml
    - methods/
      - energy.yaml
      - mahalanobis.yaml
      - odin.yaml
  - src/
    - api/
      - main.py
      - routers/
        - ood.py
        - calibrate.py
        - drift.py
        - policy.py
    - core/
      - embeddings.py
      - methods/
        - msp.py
        - energy.py
        - odin.py
        - mahalanobis.py
        - knn.py
        - lof.py
      - calibration/
        - temperature.py
        - conformal.py
        - roc_threshold.py
      - drift/
        - psi.py
        - mmd.py
        - mahalanobis_pop.py
      - router/
        - policy_engine.py
      - utils/
        - metrics.py
        - serialization.py
    - services/
      - threshold_store.py
      - decision_logger.py
      - drift_monitor.py
    - clients/
      - sdk_python/
        - __init__.py
        - client.py
      - sdk_js/
        - index.ts
  - notebooks/
    - calibration_demo.ipynb
    - benchmark_cifar.ipynb
    - nlp_intent_ood.ipynb
  - tests/
    - unit/
    - integration/
    - perf/
  - data/
    - samples/
  - scripts/
    - serve_local.sh
    - run_benchmarks.py
  - Dockerfile
  - pyproject.toml
  - package.json

Code snippets
1) Energy score (PyTorch):
def energy_score(logits, temperature=1.0):
    import torch
    return -temperature * torch.logsumexp(logits / temperature, dim=-1)

2) FastAPI endpoint:
from fastapi import FastAPI, Depends
from pydantic import BaseModel
app = FastAPI()

class ScoreReq(BaseModel):
    project_id: str
    model_version_id: str
    method: str
    modality: str
    input: str  # URL or text
    params: dict | None = None
    threshold_id: str | None = None

@app.post("/v1/ood/score")
def score(req: ScoreReq):
    # 1) fetch threshold; 2) extract embedding/logits; 3) compute score; 4) decide
    score_val = -6.8
    label = "out" if score_val > -6.0 else "in"
    return {"score": score_val, "label": label, "threshold_id": req.threshold_id or "auto", "method": req.method, "reasons": ["above_threshold(-6.0)"], "latency_ms": 120, "request_id": "req_123"}

3) Config example (YAML):
method: energy
parameters:
  temperature: 1.0
threshold:
  strategy: conformal
  alpha: 0.05
routing:
  rules:
    - condition: { method: energy, operator: ">", value: -6.0 }
      action: abstain

Benchmark targets
- CV: CIFAR-10 (ID) vs SVHN/TinyImageNet/LSUN/ImageNet-O (OOD) with AUROC ≥ 0.95.
- NLP: Banking77 (ID) with OOD news/web corpora AUROC ≥ 0.92.
- Embedding OOD for RAG: false accept ≤ 5% at 95% coverage using conformal thresholds.

Service SLOs
- Availability ≥ 99.5%.
- p95 latency ≤ 500 ms (end-to-end with embedding extraction).
- Drift alert detection within 5 minutes of threshold breach.

Integration guidance
- Black-box models: use MSP/Energy from logits; or embedding-only methods (kNN/Mahalanobis) using penultimate features.
- Vector search: threshold on max cosine; consider top-1 vs top-2 gap; use conformal calibration for guarantees.
- Routing: when OOD, abstain or route to tools/web; for RAG, switch to keyword retrieval or request clarification.

This PRD defines a complete, cloud-deployable OOD detection solution for robust, safe, and observable ML systems.