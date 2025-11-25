# Product Requirements Document (PRD) / `aiml018_regularization_for_robustness_testing`

Project ID: aiml018  
Category: AI/ML Platform - Robustness & Regularization  
Status: Draft for Review  
Version: 1.0  
Last Updated: 2025-11-25

## 1. Overview
### 1.1 Executive Summary
aiml018_regularization_for_robustness_testing is a platform and SDK to make ML models more robust and reliable through standardized regularization recipes and comprehensive robustness testing. It provides pluggable training regularizers (e.g., Mixup, TRADES, SAM, dropout), an evaluation harness (AutoAttack, corruption benchmarks, OOD detection, calibration), extensible APIs, a cloud-deployable service, and a UI to design, launch, compare, and report on robustness experiments across vision, NLP, and tabular models.

### 1.2 Document Purpose
This PRD specifies product scope, requirements, architecture, APIs, UI/UX, data model, testing, deployment, and KPIs to guide engineering, product, design, and ops to deliver a production-ready system.

### 1.3 Product Vision
Enable practitioners to achieve robustness by default. Make it trivial to apply state-of-the-art regularization and to audit models under adversarial, corruption, distribution shift, and OOD conditions with actionable, reproducible reports.

## 2. Problem Statement
### 2.1 Current Challenges
- Models overfit and fail under small perturbations and shifts.
- Robustness techniques are scattered across papers with non-trivial integration.
- Evaluation varies; results are irreproducible and not comparable across teams.
- Gaps in calibration and OOD handling lead to overconfident failures in production.
- Lack of UI and automation to design and manage robustness experiments.

### 2.2 Impact Analysis
- Unrobust models degrade user trust, increase support costs, and risk safety.
- Inconsistent evaluation slows deployment and compliance approvals.
- Extra engineering cycles to implement each technique from scratch.

### 2.3 Opportunity
Ship a unified robustness toolkit and service that:
- Improves robust accuracy and calibration with minimal code.
- Standardizes evaluation and reporting for governance and audits.
- Accelerates experimentation, leading to faster, safer deployments.

## 3. Goals and Objectives
### 3.1 Primary Goals
- Provide modular regularization recipes for training across modalities.
- Provide a standardized robustness evaluation suite.
- Offer a UI and API to configure, run, and compare experiments.
- Deliver reproducibility via configurations, artifacts, and seeded runs.

### 3.2 Business Objectives
- Reduce time-to-robust-model by 50%.
- Increase model reliability: +10–20% robust accuracy on standard benchmarks.
- Facilitate enterprise adoption with governance-grade reports and APIs.

### 3.3 Success Metrics
- Robust accuracy improvement vs baseline: +10% on CIFAR-10 under PGD ε=8/255.
- mCE (mean Corruption Error) reduction ≥ 20% on CIFAR-10-C or ImageNet-C.
- Calibration ECE ≤ 3% on in-domain and ≤ 5% under shift.
- OOD AUROC ≥ 0.90 on common OOD datasets (SVHN→CIFAR-10; Text OOD benchmarks).
- API P95 latency < 500 ms for report retrieval; training launch < 2 s.
- Service uptime ≥ 99.5%.

## 4. Target Users/Audience
### 4.1 Primary Users
- ML engineers and data scientists building and deploying models.
- Research engineers evaluating new robustness methods.
- MLOps teams monitoring and governing models.

### 4.2 Secondary Users
- Product managers needing interpretable robustness reports.
- Compliance and risk teams requiring audit trails.
- Educators teaching robust ML.

### 4.3 User Personas
- Persona 1: Alex Kim, ML Engineer
  - Background: 5 years in CV/NLP; proficient in PyTorch, FastAPI, Docker.
  - Pain Points: Re-implementing adversarial training per project; messy evaluation scripts; unstable results across seeds.
  - Goals: One-click robust training; reproducible evaluations; clear trade-off dashboards.
- Persona 2: Priya Mehta, MLOps Lead
  - Background: Manages training/inference infra on Kubernetes; observability expert.
  - Pain Points: Fragmented pipelines; lack of standardized metrics; audit requests take weeks.
  - Goals: Unified service with APIs; versioned configurations; governance-ready artifacts; SSO integration.
- Persona 3: Dr. Luis García, Research Scientist
  - Background: Robust ML research; publishes on DRO, certified robustness.
  - Pain Points: Integrating new techniques with baselines; heavy compute overhead; comparing across datasets.
  - Goals: Extensible SDK; reference implementations for TRADES, SAM, IBP; fair comparisons with dashboards.
- Persona 4: Jamie Lee, Product Manager
  - Background: Oversees AI features in a consumer app.
  - Pain Points: Hard to interpret robustness claims; unclear user impact.
  - Goals: Simple, visual summaries; business-relevant metrics; change impact over releases.

## 5. User Stories
- US-001: As an ML engineer, I want to select a regularization recipe (Mixup + SAM + Label Smoothing) from a template so that I can improve robustness without writing custom code.
  - Acceptance: User can pick template, adjust hyperparameters, and launch a training run; run logs and artifacts are saved; baseline and improved metrics are displayed side-by-side.
- US-002: As a research engineer, I want to run PGD and AutoAttack evaluations to compute robust accuracy at various Lp budgets so that I can quantify adversarial robustness.
  - Acceptance: Evaluation produces accuracy vs ε curves, stores metrics/artifacts, and generates a PDF/HTML report.
- US-003: As an MLOps lead, I want REST APIs to programmatically submit models and request robustness evaluations so that I can integrate into CI/CD.
  - Acceptance: Endpoints authenticated via OAuth2/API keys; responses include run IDs and status; webhooks notify on completion.
- US-004: As a data scientist, I want calibration and OOD metrics (ECE, NLL, Brier, AUROC) so that I can assess confidence under shift.
  - Acceptance: System computes metrics, draws reliability diagrams, and provides threshold recommendations.
- US-005: As a researcher, I want to evaluate on distribution shifts (WILDS loaders, worst-group accuracy) so that I can report worst-case risk.
  - Acceptance: System supports group annotations, computes worst-group accuracy and V-REx/IRM penalties if enabled.
- US-006: As a PM, I want a dashboard to compare experiments and see business impact so that I can make go/no-go decisions.
  - Acceptance: UI shows key KPIs, deltas vs baseline, and exportable summaries.
- US-007: As an engineer, I want certified robustness via randomized smoothing so that I can provide probabilistic guarantees.
  - Acceptance: Certificates computed for a subset; summary table shows certified radii coverage.

## 6. Functional Requirements
### 6.1 Core Features
- FR-001: Regularization Recipes Library
  - Includes: L2/weight decay (decoupled), L1, dropout/dropconnect, stochastic depth, noise injection, Mixup/Manifold Mixup, CutMix, AugMix, RandAugment, label smoothing, confidence penalty, entropy maximization, Huber/bi-tempered/focal losses, gradient/Jacobian penalties, VAT/consistency regularization, TRADES/MART, SAM, SWA/SWAG, cosine/one-cycle LR.
- FR-002: Distributional Robustness
  - DRO, group DRO, IRM, V-REx penalties configurable.
- FR-003: Adversarial Training/Eval
  - FGSM/PGD/CW/AutoAttack; robust accuracy curves across Lp norms and budgets.
- FR-004: Corruption Benchmarks
  - CIFAR-10-C/ImageNet-C style with mCE; severity sweeps; patch/occlusion tests.
- FR-005: OOD & Calibration
  - Energy scores, ODIN, Mahalanobis; AUROC/AUPR; temperature scaling; reliability diagrams; selective prediction coverage risk.
- FR-006: Certified Robustness
  - Randomized smoothing; IBP/CROWN-IBP hooks; Lipschitz constraints and spectral normalization.
- FR-007: Experiment Management
  - Projects, datasets, models, recipes, runs; versioned configs; seed management; artifact storage; comparison views.
- FR-008: API & UI
  - REST APIs; web UI for recipe builder, run launcher, dashboards; report export (PDF/HTML/JSON).
- FR-009: Observability
  - Logging, metrics, robust dashboards; alerts for regressions.

### 6.2 Advanced Features
- AF-001: Auto-Recipe Tuner
  - Bayesian/Hyperband tuning over regularization hyperparameters for target robustness metric.
- AF-002: Test-Time Adaptation/Augmentation
  - TTA, entropy minimization at inference; ablation studies.
- AF-003: Backdoor/Trigger Robustness Checks
  - Trigger pattern tests; anomaly scores.
- AF-004: Multi-modal Hooks
  - Extensible interfaces for text (token-level dropout, mixup in embedding space) and audio (SpecAugment, noise injection).

## 7. Non-Functional Requirements
### 7.1 Performance
- API P95 latency < 500 ms for metadata/report retrieval.
- Batch evaluation throughput ≥ 200 images/s on 1 x A10 for CIFAR-10; ≥ 30 images/s on 1 x A100 for ImageNet-sized inputs.

### 7.2 Reliability
- Uptime ≥ 99.5%.
- Retry with exponential backoff for artifact uploads and evaluation jobs.
- Idempotent run submission with client-provided dedup keys.

### 7.3 Usability
- Recipe templates; sensible defaults; inline docs.
- Exportable, shareable links and reports.

### 7.4 Maintainability
- Modular plugins for new techniques.
- 90% unit test coverage for core SDK; API contract tests; semantic versioning.

## 8. Technical Requirements
### 8.1 Technical Stack
- Language: Python 3.11+
- Frameworks: PyTorch 2.3+, torchvision 0.18+, timm 1.0+, Hugging Face Transformers 4.45+
- Backend: FastAPI 0.115+, Uvicorn 0.30+
- Frontend: React 18+, TypeScript 5+, Vite 5+
- Orchestration: Kubernetes 1.30+, Argo Workflows 3.5+ (or K8s Jobs)
- Storage: PostgreSQL 15+, Redis 7+ (queues/cache), MinIO/S3 for artifacts
- Experiment Tracking: MLflow 2.14+ (optional Weights & Biases 0.16+)
- Messaging: NATS 2.10+ or RabbitMQ 3.13+
- Monitoring: Prometheus 2.53+, Grafana 10+, OpenTelemetry 1.27+
- Auth: Keycloak 22+ or Auth0; OAuth2/OIDC
- Packaging: Docker, Helm 3.14+
- CUDA/cuDNN compatible with target GPUs for training jobs

### 8.2 AI/ML Components
- Training SDK: recipes, losses, schedulers, adversarial attacks, augmentations
- Evaluation harness: PGD/AutoAttack, corruption loader, OOD/calibration modules
- Certified robustness: randomized smoothing, IBP/CROWN-IBP adapters
- Dataset loaders: CIFAR, ImageNet, WILDS, custom dataset registry
- Metrics: robust accuracy, mCE, ECE/NLL/Brier, OOD AUROC/AUPR, worst-group accuracy, sharpness proxies

## 9. System Architecture
### 9.1 High-Level Architecture (ASCII)
+-------------------+        HTTPS        +-------------------+
|  Web UI (React)   | <-----------------> |  API (FastAPI)    |
+-------------------+                     +----------+--------+
                                                     |
                                                     | gRPC/HTTP
                                                     v
                                           +---------+---------+
                                           |  Orchestrator     |
                                           | (Argo/K8s Jobs)   |
                                           +----+---------+----+
                                                |         |
                                                |         |
                                       +--------v-+     +-v--------+
                                       | Trainer  |     | Evaluator|
                                       | Pods     |     | Pods     |
                                       +----+-----+     +----+-----+
                                            |                |
                                            | artifacts/     | metrics/
                                            v                v
                                    +-------+--------+  +---+---------+
                                    | Artifact Store |  | PostgreSQL  |
                                    | (S3/MinIO)     |  | + Redis     |
                                    +-------+--------+  +-------------+
                                            |
                                            v
                                       +----+-----+
                                       | MLflow   |
                                       +----------+

### 9.2 Component Details
- Web UI: SPA for recipe builder, run launcher, dashboards.
- API: REST endpoints for CRUD of projects, datasets, models, recipes, runs; auth; webhooks.
- Orchestrator: Schedules training/eval pods with GPU requirements; mounts datasets; passes configs.
- Trainer Pods: Execute training using SDK; log metrics to MLflow; upload checkpoints, configs.
- Evaluator Pods: Run robustness tests, generate reports.
- Storage: Postgres for metadata; Redis for queues/cache; S3/MinIO for artifacts.
- Observability: Prometheus scrapes services; Grafana dashboards; logs via OpenTelemetry.

### 9.3 Data Flow
1) User creates project/dataset/model and selects a recipe via UI or API.  
2) API persists config; orchestrator launches job.  
3) Trainer reads dataset, applies recipe, trains model.  
4) Model and artifacts stored; metrics logged.  
5) Evaluator runs adversarial/corruption/OOD/calibration; reports generated.  
6) UI displays results; reports exportable; webhooks notify completion.

## 10. Data Model
### 10.1 Entity Relationships
- User (1..*) Project
- Project (1..*) Dataset
- Project (1..*) Model
- Project (1..*) Recipe
- Recipe (1..*) ExperimentRun
- Model (1..*) ExperimentRun
- ExperimentRun (1..*) Metrics, (1..*) Artifacts
- Dataset may have GroupAnnotations for worst-group metrics
- APIKey belongs to User

### 10.2 Database Schema (PostgreSQL 15+)
- users
  - id (uuid, pk), email (text, unique), name (text), role (enum: admin, user, viewer), created_at (timestamptz)
- api_keys
  - id (uuid, pk), user_id (fk users.id), key_hash (text), scope (text[]), created_at, revoked (bool)
- projects
  - id (uuid, pk), name (text), description (text), owner_id (fk users), created_at
- datasets
  - id (uuid, pk), project_id (fk), name (text), type (enum: vision, nlp, tabular, audio), uri (text), schema (jsonb), group_annotations (jsonb), created_at
- models
  - id (uuid, pk), project_id (fk), name (text), framework (enum: pytorch, tf, sklearn, custom), uri (text), input_spec (jsonb), created_at
- recipes
  - id (uuid, pk), project_id (fk), name (text), spec (jsonb), created_at, version (int)
- experiment_runs
  - id (uuid, pk), project_id (fk), model_id (fk), recipe_id (fk), type (enum: train, evaluate), status (enum: queued, running, succeeded, failed, canceled), config (jsonb), seed (int), started_at, finished_at, worker_pod (text)
- metrics
  - id (uuid, pk), run_id (fk experiment_runs), name (text), value (double), context (jsonb), step (int), created_at
- artifacts
  - id (uuid, pk), run_id (fk), name (text), uri (text), type (enum: checkpoint, report_html, report_pdf, plots, logs, other), created_at
- webhooks
  - id (uuid, pk), project_id (fk), url (text), event (enum: run_completed, run_failed), secret (text), created_at

### 10.3 Data Flow Diagrams
[Create Run] User -> API -> DB (persist config) -> Orchestrator -> Trainer/Evaluator -> Artifact Store/MLflow -> DB update status -> Webhook

### 10.4 Input Data & Dataset Requirements
- Supported datasets: CIFAR-10/100, ImageNet, WILDS benchmarks, SST-2/IMDB (NLP), custom.
- Requirements:
  - Datasets must be versioned and immutable; URIs point to S3/HTTP/local PVC.
  - Provide train/val/test splits; optional group annotations for DRO.
  - Licensing compliance for public datasets.
- Input spec examples:
  - Vision: RGB images, size canonicalized (e.g., 224x224), normalization per dataset.
  - NLP: Tokenized text using HF tokenizers; max seq length configurable.
  - Tabular: CSV/Parquet with schema JSON; feature normalization config.

## 11. API Specifications
### 11.1 REST Endpoints (v1)
- POST /api/v1/projects
- GET /api/v1/projects/{project_id}
- POST /api/v1/projects/{project_id}/datasets
- POST /api/v1/projects/{project_id}/models
- POST /api/v1/projects/{project_id}/recipes
- GET /api/v1/projects/{project_id}/recipes
- POST /api/v1/projects/{project_id}/runs
- GET /api/v1/runs/{run_id}
- POST /api/v1/runs/{run_id}/cancel
- GET /api/v1/runs/{run_id}/artifacts
- GET /api/v1/runs/{run_id}/metrics
- POST /api/v1/evaluate  (shortcut to launch evaluation run on a model checkpoint)
- POST /api/v1/webhooks/test
- POST /api/v1/auth/token (OAuth2 Password/Client Credentials)
- GET /api/v1/recipes/templates

### 11.2 Request/Response Examples
- Create Recipe
Request:
POST /api/v1/projects/{project_id}/recipes
Content-Type: application/json
{
  "name": "mixup_sam_trades_v1",
  "spec": {
    "regularizers": {
      "mixup": {"alpha": 0.2, "prob": 1.0},
      "label_smoothing": {"epsilon": 0.05},
      "sam": {"rho": 0.05, "adaptive": true},
      "trades": {"beta": 6.0, "steps": 10, "step_size": 0.007}
    },
    "optimizer": {"type": "adamw", "lr": 3e-4, "weight_decay": 0.01},
    "scheduler": {"type": "cosine", "warmup_epochs": 5},
    "early_stopping": {"patience": 10},
    "seeds": [1,2,3]
  }
}
Response: 201
{"id": "rec_123", "version": 1}

- Launch Run
POST /api/v1/projects/{project_id}/runs
{
  "type": "train",
  "model_id": "mod_456",
  "recipe_id": "rec_123",
  "dataset_id": "ds_789",
  "config_overrides": {"epochs": 200, "batch_size": 256},
  "webhook_url": "https://hooks.example.com/robust"
}
Response: 202
{"run_id": "run_abcd", "status": "queued"}

- Python SDK snippet
from aiml018 import Client
c = Client(api_key="...", base_url="https://robust.example.com")
run = c.launch_train(project_id="proj1",
                     model_id="mod_456",
                     recipe_id="rec_123",
                     dataset_id="ds_789",
                     config_overrides={"epochs": 200, "batch_size": 256})
print(run.id)

### 11.3 Authentication
- OAuth2/OIDC with JWTs; scopes: projects:read, runs:write, admin:all.
- API keys for service-to-service; keys hashed in DB.
- mTLS optional for internal traffic.

## 12. UI/UX Requirements
### 12.1 User Interface
- Pages:
  - Dashboard: KPIs, recent runs, alerts.
  - Recipe Builder: Form/JSON editor; presets (e.g., “Adversarial (TRADES)”, “Flat Minima (SAM+SWA)”).
  - Experiment Launcher: Select model/dataset/recipe; resources; seeds.
  - Run Detail: Live logs, metrics charts (loss, accuracy, ECE), robust curves, corruption heatmaps, worst-group tables.
  - Reports: Download PDF/HTML; share links.
  - Datasets & Models: Registry with metadata, versioning.

### 12.2 User Experience
- Opinionated defaults with tooltips and links to docs.
- Comparison mode: select baseline and candidate; diff metrics.
- Wizard for dataset setup and group annotations.

### 12.3 Accessibility
- WCAG 2.1 AA compliance: keyboard navigation, ARIA labels, color contrast.
- Alt text for charts; downloadable CSV for tables.

## 13. Security Requirements
### 13.1 Authentication
- SSO via OIDC provider; MFA optional.
- Token expiration and rotation.

### 13.2 Authorization
- Role-based access control (RBAC): admin, editor, viewer.
- Project-level permissions; dataset/model ownership.

### 13.3 Data Protection
- TLS 1.2+ in transit; server-side encryption for S3 artifacts (AES-256).
- Parameterized queries; secrets managed via Kubernetes Secrets/HashiCorp Vault.

### 13.4 Compliance
- Audit logs for all changes and run executions.
- Data retention policies configurable per project.

## 14. Performance Requirements
### 14.1 Response Times
- API P50 < 100 ms, P95 < 500 ms for metadata; report downloads start < 1 s.

### 14.2 Throughput
- Support 100 concurrent users; 200 concurrent runs across a 10-node GPU cluster.

### 14.3 Resource Usage
- Trainer/Evaluator pods request/limit configs set by job size; autoscaling based on GPU/CPU usage.

## 15. Scalability Requirements
### 15.1 Horizontal Scaling
- Scale API instances behind a load balancer; stateless services; Redis/Postgres HA.
- Job queue workers autoscale on backlog length.

### 15.2 Vertical Scaling
- Larger GPU types for heavy models; configurable per run.

### 15.3 Load Handling
- Backpressure: queue limits; fair scheduling per project quotas.
- Rate limiting and burst control per API key.

## 16. Testing Strategy
### 16.1 Unit Testing
- SDK functions (mixup, TRADES loss, SAM step) with determinism tests.
- Metrics calculators (ECE, AUROC) against known fixtures.

### 16.2 Integration Testing
- End-to-end run: launch train on small dataset (CIFAR-10 subset), then eval; assert artifacts and metrics exist.
- API contract tests via schemathesis.

### 16.3 Performance Testing
- Load tests with k6/Locust; ensure latency SLOs.
- Training throughput benchmarks on target GPUs.

### 16.4 Security Testing
- Static analysis (Bandit), dependency scanning.
- AuthZ tests for RBAC; simulate token misuse.

## 17. Deployment Strategy
### 17.1 Deployment Pipeline
- GitHub Actions:
  - Lint/test -> Build Docker images -> Push to registry
  - Helm chart version bump -> Staging deploy -> Integration tests -> Manual approval -> Prod deploy

### 17.2 Environments
- Dev: single-node, ephemeral datasets.
- Staging: mirrors prod; synthetic datasets for tests.
- Prod: multi-node GPU cluster, HA Postgres/Redis, S3 or MinIO.

### 17.3 Rollout Plan
- Blue/Green for API; canary 10% traffic for 24 hours.
- Jobs use versioned images; pin per run.

### 17.4 Rollback Procedures
- Helm rollback to previous release.
- Preserve DB migrations with down scripts.
- Maintain N-2 images and charts.

## 18. Monitoring & Observability
### 18.1 Metrics
- System: CPU/GPU utilization, memory, queue depth, job success rate.
- API: RPS, latency (P50/P95/P99), error rates.
- ML: training/validation accuracy, robust accuracy by ε, mCE, ECE, NLL, Brier, OOD AUROC, worst-group accuracy, sharpness proxy (Hessian trace approx).

### 18.2 Logging
- Structured JSON logs with request IDs and run IDs.
- Log sampling for high-volume workers.

### 18.3 Alerting
- Pager alerts: API error rate > 2% over 5 min; job failure rate > 10% over 30 min.
- ML alerts: regression in robust accuracy > 5% vs baseline on main branch.

### 18.4 Dashboards
- Grafana dashboards for infra and ML metrics.
- Per-project dashboards with KPIs and recent trends.

## 19. Risk Assessment
### 19.1 Technical Risks
- High compute cost for adversarial training/eval.
- Instability of certain techniques (e.g., TRADES) without careful tuning.
- Dataset licensing and storage size.

### 19.2 Business Risks
- Adoption barrier if integration is complex.
- User confusion due to many options/hyperparameters.
- Overpromising guarantees where only empirical robustness is provided.

### 19.3 Mitigation Strategies
- Provide lightweight baselines and progressive enhancement.
- Strong defaults and templates; auto-tuning option.
- Clear documentation on guarantees vs empirical results.
- Cost controls: budget caps and preemption.

## 20. Timeline & Milestones
### 20.1 Phase breakdown
- Phase 0 (Week 1): Requirements finalization; design review.
- Phase 1 (Weeks 2–5): Core SDK (recipes, attacks, metrics), API MVP, dataset/model registry.
- Phase 2 (Weeks 6–8): Evaluation harness (corruptions, OOD, calibration), UI MVP (dashboard, launcher).
- Phase 3 (Weeks 9–10): Advanced features (TRADES, SAM, DRO, randomized smoothing), reports, observability.
- Phase 4 (Weeks 11–12): Hardening, docs, security review, staging soak, production launch.

### 20.2 Key Milestones
- M1: SDK v0.1 with Mixup/CutMix/AugMix and PGD evaluation (Week 3)
- M2: REST API + UI MVP (Week 5)
- M3: Full evaluation suite + reports (Week 8)
- M4: Advanced features + certified robustness (Week 10)
- GA: v1.0 production release (Week 12)

Estimated Engineering Effort: 3–4 engineers, 12 weeks  
Estimated Cloud Cost (monthly at steady state): $6k–$12k (depends on GPU usage)

## 21. Success Metrics & KPIs
### 21.1 Measurable targets
- Technical:
  - +10% robust accuracy vs baseline on CIFAR-10 with PGD ε=8/255 within 2 weeks of adoption.
  - mCE reduction ≥ 20% on CIFAR-10-C or ImageNet-C.
  - ECE ≤ 3% in-domain, ≤ 5% under shift after temperature scaling.
  - OOD AUROC ≥ 0.90 on at least two benchmark pairs.
  - Report generation time < 2 minutes for 10k-sample eval.
- Product:
  - ≥ 10 active projects in first quarter.
  - ≥ 50 robustness runs/week by end of quarter 2.
  - User satisfaction (CSAT) ≥ 4.5/5 for UI/API usability.

## 22. Appendices & Glossary
### 22.1 Technical Background
- Regularization families:
  - Parameter-level: L2/weight decay, L1 sparsity, spectral normalization, Lipschitz/Parseval constraints, early stopping.
  - Architecture-level: dropout, dropconnect, stochastic depth, noise injection.
  - Data-level: Mixup, Manifold Mixup, CutMix, AugMix, RandAugment, adversarial augmentation, style/color jitter.
  - Objective-level: label smoothing, confidence penalty, entropy maximization, robust losses (Huber/bi-tempered/focal), gradient/Jacobian penalties, VAT, TRADES/MART.
  - Optimization-level: SAM, SWA/SWAG, cosine/one-cycle schedules.
  - Distributional: DRO, group DRO, IRM, V-REx.
  - Certified: randomized smoothing, IBP/CROWN-IBP.
- Evaluation:
  - Adversarial: FGSM, PGD, CW, AutoAttack; robust accuracy vs ε under Lp norms.
  - Corruptions: datasets with noise/blur/weather/digital; mCE; severity sweeps.
  - Distribution shift: WILDS loaders; worst-group accuracy; spurious correlation tests.
  - OOD/Calibration: energy/ODIN/Mahalanobis; AUROC/AUPR; ECE/NLL/Brier; reliability diagrams.

### 22.2 References
- TRADES: Zhang et al., “Theoretically Principled Trade-off between Robustness and Accuracy.”
- SAM: Foret et al., “Sharpness-Aware Minimization.”
- AugMix: Hendrycks et al., “AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty.”
- Mixup: Zhang et al., “mixup: Beyond Empirical Risk Minimization.”
- AutoAttack: Croce & Hein, “Reliable Evaluation of Adversarial Robustness with AutoAttack.”
- ODIN: Liang et al., “Enhancing The Reliability of Out-of-distribution Image Detection.”
- Energy OOD: Liu et al., “Energy-based Out-of-distribution Detection.”
- Randomized Smoothing: Cohen et al., “Certified Adversarial Robustness via Randomized Smoothing.”
- WILDS: Koh et al., “WILDS: A Benchmark of in-the-Wild Distribution Shifts.”

### 22.3 Glossary
- Robust Accuracy: Accuracy under specified perturbations or attacks.
- mCE: Mean Corruption Error; normalized error across corruption types and severities.
- ECE: Expected Calibration Error; measures confidence calibration.
- OOD: Out-of-Distribution detection; identifying inputs not from training distribution.
- DRO: Distributionally Robust Optimization; optimizes for worst-case distribution.
- IRM/V-REx: Invariance-based regularization techniques for domain generalization.
- PGD/FGSM/CW: Adversarial attack methods.
- TRADES/MART: Adversarial training objectives.
- SAM/SWA: Optimization techniques to favor flatter minima.
- IBP/CROWN-IBP: Methods for interval bound propagation and certified robustness.

Repository Structure
- root/
  - README.md
  - pyproject.toml
  - requirements.txt
  - src/
    - aiml018/
      - api/
        - main.py
        - routers/
          - projects.py
          - datasets.py
          - models.py
          - recipes.py
          - runs.py
      - sdk/
        - training/
          - recipes/
            - mixup.py
            - cutmix.py
            - augmix.py
            - trades.py
            - sam.py
            - dro.py
          - losses/
            - focal.py
            - bi_tempered.py
            - huber.py
          - optim/
            - schedulers.py
            - swa.py
          - attacks/
            - fgsm.py
            - pgd.py
            - cw.py
            - autoattack.py
          - certified/
            - smoothing.py
            - ibp.py
        - evaluation/
          - corruption/
          - ood/
          - calibration/
          - reports/
      - orchestration/
        - argo_templates/
        - k8s_jobs.py
      - data/
        - loaders/
          - cifar.py
          - imagenet.py
          - wilds.py
          - text/
            - sst2.py
            - imdb.py
      - utils/
        - metrics.py
        - logging.py
        - config.py
  - configs/
    - recipes/
      - mixup_sam_trades.yaml
      - dro_irm.yaml
    - datasets/
      - cifar10.yaml
  - notebooks/
    - 01_baselines.ipynb
    - 02_robustness_eval.ipynb
    - 03_certified_robustness.ipynb
  - tests/
    - unit/
    - integration/
    - e2e/
  - data/ (gitignored)
  - deploy/
    - helm/
    - docker/
  - docs/

Config Sample (YAML)
experiment:
  name: "cifar10_resnet50_mixup_sam_trades"
  seed: 42
  epochs: 200
  batch_size: 256
dataset:
  name: "CIFAR-10"
  loader: "cifar"
  root_uri: "s3://datasets/cifar10"
model:
  arch: "resnet50"
  pretrained: false
optimizer:
  type: "adamw"
  lr: 0.0003
  weight_decay: 0.01
scheduler:
  type: "cosine"
  warmup_epochs: 5
regularizers:
  mixup: {alpha: 0.2, prob: 1.0}
  label_smoothing: {epsilon: 0.05}
  sam: {rho: 0.05, adaptive: true}
  trades: {beta: 6.0, steps: 10, step_size: 0.007}
evaluation:
  adversarial:
    autoattack: {eps: 0.031, norm: "Linf"}
    pgd: {eps: 0.031, steps: 50, step_size: 0.007}
  corruptions:
    dataset: "CIFAR-10-C"
    severities: [1,2,3,4,5]
  ood:
    methods: ["energy", "odin", "mahalanobis"]
  calibration:
    temperature_scaling: true

API Code Snippet (FastAPI)
from fastapi import FastAPI, Depends
from pydantic import BaseModel

app = FastAPI()

class RunCreate(BaseModel):
    type: str
    model_id: str
    recipe_id: str
    dataset_id: str
    config_overrides: dict | None = None

@app.post("/api/v1/projects/{project_id}/runs")
def create_run(project_id: str, req: RunCreate, user=Depends(auth_user)):
    run_id = orchestrator.submit_run(project_id, req)
    return {"run_id": run_id, "status": "queued"}

Performance Targets and Budgets
- Compute estimates for CIFAR-10 ResNet-50 with TRADES+SAM:
  - Training: ~20–30 GPU-hours on A100 (repeated seeds x3 => 60–90 GPU-hours).
  - Evaluation: PGD/AutoAttack ~2–4 GPU-hours per model.
  - Monthly budget controls: project-level GPU-hour caps; notify at 80/100% thresholds.

End of PRD.