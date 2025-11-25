# Product Requirements Document (PRD)
# `Aiml041_Feature_Engineering_Leakage_Thermal_Metric_Predictor`

Project ID: Aiml041_Feature_Engineering_Leakage_Thermal_Metric_Predictor
Category: General AI/ML – Time-series Regression & Feature Engineering
Status: Draft for Review
Version: 1.0
Last Updated: 2025-11-25

## 1. Overview
### 1.1 Executive Summary
Aiml041_Feature_Engineering_Leakage_Thermal_Metric_Predictor is a machine learning system to predict thermal metrics (e.g., temperature, thermal load, cooling capacity utilization) from multivariate time-series and contextual inputs while explicitly preventing data leakage. The product provides a leakage-safe feature engineering pipeline, model training and selection, probabilistic predictions with uncertainty estimates, explainability, and production-grade serving APIs with monitoring and drift detection. It targets environments like data centers, industrial equipment, HVAC systems, and battery packs where accurate thermal predictions inform control, planning, and safety.

### 1.2 Document Purpose
This PRD defines the scope, requirements, architecture, data model, APIs, UX, performance, security, testing, deployment, monitoring, risks, and milestones for delivering a production-ready, leakage-safe thermal metric prediction platform.

### 1.3 Product Vision
Deliver a high-accuracy, low-latency, trustworthy thermal prediction platform that:
- Automates robust, leakage-safe feature engineering for time-series.
- Provides strong baseline models and advanced sequence models.
- Offers interval predictions and explainability for operational trust.
- Is easy to deploy via cloud-native APIs and a web console.
- Ensures continuous reliability through monitoring and retraining.

## 2. Problem Statement
### 2.1 Current Challenges
- Models trained with inadvertent target/data leakage yield inflated offline accuracy but fail in production.
- Heterogeneous, multivariate inputs require complex, time-aware feature engineering.
- Existing toolchains lack standardized leakage-guarded pipelines and evaluation practices.
- Operational deployments often omit uncertainty, explainability, and drift monitoring.

### 2.2 Impact Analysis
- Leakage leads to overconfident predictions, unsafe control decisions, and financial loss from energy inefficiency or downtime.
- Manual feature engineering is slow, error-prone, and irreproducible.
- Lack of monitoring erodes trust and slows adoption.

### 2.3 Opportunity
Provide a turnkey system with:
- Reusable, leakage-safe feature transformers and validation templates.
- Fast, accurate models and probabilistic forecasts.
- Enterprise-grade serving, monitoring, and governance.
- Reduced total cost of ownership via automation and templates.

## 3. Goals and Objectives
### 3.1 Primary Goals
- Prevent data/target leakage through end-to-end pipelines and evaluations.
- Achieve strong predictive accuracy for thermal metrics with uncertainty estimates.
- Provide explainability and model governance artifacts.
- Offer production-ready APIs and UI for training, inference, and monitoring.

### 3.2 Business Objectives
- Reduce model development time by 50% through reusable components.
- Improve operational thermal prediction accuracy by ≥20% vs. customer baselines.
- Lower energy or cooling costs via better forecasting-driven control.
- Enable new revenue via managed inference and model monitoring.

### 3.3 Success Metrics
- Accuracy: ≥20% reduction in RMSE vs. baseline; MAPE ≤10% on validation.
- Latency: Online prediction P50 < 300 ms, P95 < 500 ms with warm cache.
- Uptime: ≥99.5% API availability.
- Coverage: 90% prediction intervals achieve 88–92% empirical coverage.
- Drift detection: Time-to-detection for significant drift ≤ 24 hours.
- Leakage tests: 100% of pipelines pass leakage unit and integration tests.

## 4. Target Users/Audience
### 4.1 Primary Users
- Data scientists building thermal prediction models.
- ML engineers deploying and maintaining inference APIs.
- Operations engineers needing reliable thermal forecasts and alerts.

### 4.2 Secondary Users
- Product managers tracking KPI improvements (accuracy, latency, coverage).
- Site reliability engineers monitoring system health.
- Compliance/governance stakeholders reviewing model cards and lineage.

### 4.3 User Personas
- Persona 1: Dr. Aisha Patel, Senior Data Scientist
  - Background: PhD in Statistics, 8 years in time-series forecasting. Familiar with Python, scikit-learn, LightGBM, SHAP.
  - Pain Points: Time-aware feature engineering is error-prone; leakage is hard to guard against; tuning deep models is costly.
  - Goals: Fast experiments with safe pipelines, transparent evaluation, and reliable uncertainty estimates.
- Persona 2: Marco Rossi, ML Engineer
  - Background: Software engineer turned MLOps, 6 years with FastAPI, Kubernetes, Terraform, observability stacks.
  - Pain Points: Packaging feature pipelines with models is brittle; model drift and versioning often ad-hoc; tight latency SLAs.
  - Goals: One-click deployment, robust model registry, autoscaling serving, strong monitoring.
- Persona 3: Linh Tran, Operations Lead
  - Background: Runs data center operations; KPI-driven; less ML-savvy.
  - Pain Points: Unpredictable forecasts; little visibility into model confidence; incidents due to stale models.
  - Goals: Reliable forecasts with clear confidence, simple dashboards, timely alerts, and impact reports.

## 5. User Stories
- US-001: As a data scientist, I want time-based train/validation/test splitting utilities so that I avoid target leakage. Acceptance: Splits prohibit future data in training folds; verified by unit tests.
- US-002: As a data scientist, I want rolling and lag feature generators bound to the pipeline so that transforms fit only on train and apply consistently. Acceptance: sklearn-compatible transformer with fit/transform separation; leakage tests pass.
- US-003: As a data scientist, I want multi-window aggregations (5/30/120 minutes) so that the model captures short/long-term dynamics. Acceptance: Features computed without peeking; configurable windows.
- US-004: As a data scientist, I want robust scaling/winsorization to handle outliers. Acceptance: RobustScaler, clipping with train-only quantiles.
- US-005: As a data scientist, I want interaction features and categorical encoders with leakage-safe target encoding. Acceptance: Target encoding uses out-of-fold statistics only.
- US-006: As an ML engineer, I want a model registry to version artifacts and metadata. Acceptance: Register, promote, and rollback models via API/UI; lineage tracked.
- US-007: As a data scientist, I want quantile regression and conformal prediction so that I get valid uncertainty intervals. Acceptance: Empirical coverage within ±2% of target on validation.
- US-008: As an operator, I want an API to get predictions with confidence intervals. Acceptance: /predict returns point and intervals <500 ms P95.
- US-009: As a data scientist, I want SHAP explanations and partial dependence plots to understand drivers. Acceptance: UI displays global/local explanations for chosen datasets.
- US-010: As an SRE, I want monitoring of latency, error rate, drift, and calibration. Acceptance: Dashboards with alerts for threshold breaches.
- US-011: As a PM, I want model cards summarizing data, metrics, fairness, and limitations. Acceptance: Auto-generated model cards in registry.
- US-012: As an ML engineer, I want nested CV with Bayesian optimization to robustly tune models. Acceptance: Configurable HPO with time-aware CV; reproducible seeds.
- US-013: As a data scientist, I want residual slicing to identify error hot spots. Acceptance: Automatic reports by context buckets.
- US-014: As an admin, I want RBAC and audit logs. Acceptance: Roles (viewer/editor/admin) enforced; actions logged.
- US-015: As a user, I want a web console for uploading data, configuring features, running training, and deploying. Acceptance: React app with guided flows and validation.

## 6. Functional Requirements
### 6.1 Core Features (FR-001 …)
- FR-001: Time-aware dataset splitters (expanding window, rolling origin, group-based).
- FR-002: Leakage-safe feature pipeline with fit/transform separation for scalers, encoders, temporal aggregations.
- FR-003: Temporal feature generators (lags, rolling mean/std/min/max, EMA, trend/seasonality, cooldown/warmup, cumulative/rolling deltas).
- FR-004: Multi-window aggregations (configurable windows; default 5/30/120 minutes).
- FR-005: Robust preprocessing (winsorization/clipping; log/sqrt transforms; RobustScaler).
- FR-006: Categorical encoding (one-hot, target encoding with out-of-fold, learnable embeddings for deep models).
- FR-007: Baseline modeling (Ridge/Lasso/ElasticNet, RandomForest, XGBoost, LightGBM, CatBoost).
- FR-008: Sequence modeling (LSTM/GRU/TCN, Temporal Fusion Transformer) with mask-aware batching.
- FR-009: Loss functions and evaluation (MAE, RMSE, MAPE/SMAPE, pinball loss; cross-time and context-bucket evaluation).
- FR-010: Uncertainty (quantile regression; conformal prediction; Monte Carlo dropout).
- FR-011: Explainability (SHAP global/local; permutation importance; PDP/ICE; monotonic constraints support).
- FR-012: Hyperparameter optimization (random, early-stopped grid, Bayesian with Optuna) using nested CV.
- FR-013: Model registry (artifacts, metrics, lineage, model cards).
- FR-014: Serving API (REST) for /predict, /explain, /health, /metrics.
- FR-015: Drift monitoring (data drift, concept drift, calibration, coverage).
- FR-016: UI for dataset management, feature configuration, training runs, deployment, dashboards.
- FR-017: Residual slicing and error attribution reports.
- FR-018: Shadow deployment, canary rollout, and rollback.
- FR-019: Audit logging and RBAC.
- FR-020: Export/import of pipelines and models as versioned bundles.

### 6.2 Advanced Features
- AF-001: Automated feature discovery and ablation to identify high-impact features safely.
- AF-002: Stability tests across retrains (feature importance stability, monotonicity checks).
- AF-003: Auto-calibration of intervals with conformal wrappers and online recalibration.
- AF-004: Batch scoring jobs with backfills and scheduled retraining.
- AF-005: Integration with feature store (optional) for real-time features.

## 7. Non-Functional Requirements
### 7.1 Performance
- Inference latency: P50 < 300 ms, P95 < 500 ms for single prediction with preloaded model.
- Throughput: ≥200 RPS per replica for gradient boosting models; ≥50 RPS per replica for deep models.
- Training time: Baseline model training ≤30 minutes on 8 vCPU/32GB for 10M rows.

### 7.2 Reliability
- Availability: ≥99.5% monthly for inference API.
- Error rate: <0.1% 5xx responses.
- Durable storage of artifacts and logs with multi-AZ redundancy.

### 7.3 Usability
- Clear templates and guardrails; UI validation for leakage pitfalls.
- Documentation with examples and notebooks.

### 7.4 Maintainability
- Modular codebase; 80% unit test coverage for core libs.
- CI/CD with linting, typing (mypy), and automated tests.

## 8. Technical Requirements
### 8.1 Technical Stack
- Languages: Python 3.11+, TypeScript 5+
- Backend: FastAPI 0.115+, Uvicorn, Gunicorn, Pydantic v2
- Frontend: React 18+, Next.js 14+, Material UI 5+
- ML: scikit-learn 1.5+, LightGBM 4.4+, XGBoost 2.1+, CatBoost 1.2+, PyTorch 2.4+, PyTorch Lightning 2.4+, statsmodels 0.14+, shap 0.45+
- HPO: Optuna 3.6+
- Data: Pandas 2.2+, Polars 1.6+, Dask 2024.10+, PyArrow 17+
- Workflow: Airflow 2.10+ or Prefect 2.16+
- Storage: PostgreSQL 15+, Redis 7+, Object store (S3-compatible), MinIO (dev)
- Messaging (optional): Kafka 3.7+
- Observability: Prometheus 2.54+, Grafana 11+, OpenTelemetry 1.27+, Loki 2.9+
- Model Registry: MLflow 2.16+
- Infrastructure: Docker, Kubernetes 1.30+, Helm, Terraform 1.8+
- Auth: OAuth2/OIDC (Auth0/Keycloak)

### 8.2 AI/ML Components
- Feature pipeline library (custom sklearn-compatible transformers).
- Model training service with time-aware CV and nested CV.
- Model selection and ensembling.
- Uncertainty modules (quantile models, conformal prediction).
- Explainability service (SHAP, permutation).
- Drift detection (data and concept) and calibration tracking.

## 9. System Architecture
### 9.1 High-Level Architecture (ASCII diagram)
+-------------------+      +---------------------+      +-------------------+
|   Data Sources    | ---> |  Ingestion Service  | ---> |  Object Storage   |
| (sensors, logs,   |      | (batch/stream)      |      | (datasets/artifs) |
+-------------------+      +----------+----------+      +-----+-------------+
                                      |                       |
                                      v                       v
                           +----------+-----------+   +-------+--------+
                           |  Feature Engineering |   |   DB (Postgres)|
                           |   Pipeline Service   |   | (metadata, runs)|
                           +----------+-----------+   +-------+--------+
                                      |                       |
                                      v                       |
                           +----------+-----------+           |
                           |   Training & HPO     |<----------+
                           | (time-aware CV)      |
                           +----------+-----------+
                                      |
                                      v
                           +----------+-----------+
                           |   Model Registry     |
                           +----------+-----------+
                                      |
                                      v
+-------------------+       +---------+----------+       +-----------------+
|  Web Console      | <---- |  Inference API     | <---- |  Feature Store* |
|  (React)          |       |  (FastAPI)         |       |  or Redis Cache |
+-------------------+       +---------+----------+       +-----------------+
                                      |
                                      v
                           +----------+-----------+
                           | Monitoring & Alerts  |
                           | (Prometheus/Grafana) |
                           +----------------------+
*optional

### 9.2 Component Details
- Ingestion: Validates schema, timestamps, and partitions; writes to object storage; triggers lineage.
- Feature Engineering: Leakage-safe transformers; pipelines persisted with model.
- Training & HPO: Builds baselines and deep models; runs nested CV; logs metrics to registry.
- Model Registry: Stores artifacts, metrics, model cards, signatures; supports promotion and rollback.
- Inference API: Loads selected model; performs online feature transforms; returns predictions and intervals; exposes /metrics for Prometheus.
- Web Console: Dataset management, feature config, training orchestration, deployment control, dashboards.
- Monitoring: Latency, errors, drift, calibration, coverage; alert routing.

### 9.3 Data Flow
1) Data arrives (batch/stream) -> stored in object storage with metadata in Postgres.
2) Feature service generates train/test features with time-aware transforms (fit on train only).
3) Training runs nested CV + HPO; best model registered with artifacts and model card.
4) Deployment promotes a model to production; Inference API serves /predict with online features.
5) Monitoring tracks performance, drift, and calibration; triggers retraining if thresholds breached.

## 10. Data Model
### 10.1 Entity Relationships
- Organization (1) — (N) Project
- Project (1) — (N) Dataset
- Dataset (1) — (N) FeatureSet
- FeatureSet (1) — (N) TrainingRun
- TrainingRun (1) — (N) ModelVersion
- ModelVersion (1) — (N) Deployment
- Deployment (1) — (N) Prediction
- ModelVersion (1) — (N) ExplainabilityReport
- Alerts linked to Project/Deployment

### 10.2 Database Schema (key tables/columns)
- users: id, email, name, role, org_id, created_at
- organizations: id, name, created_at
- projects: id, org_id, name, description, created_at
- datasets: id, project_id, uri, schema_json, rows, timestamps, created_at
- featuresets: id, dataset_id, config_yaml, version, created_at
- training_runs: id, featureset_id, params_json, metrics_json, started_at, ended_at, status, seed
- model_versions: id, training_run_id, registry_uri, signature_json, metrics_json, card_uri, created_at, stage (none/staging/production/archived)
- deployments: id, model_version_id, endpoint, replicas, config_json, created_at, status
- predictions: id, deployment_id, request_json, response_json, latency_ms, created_at
- explainability_reports: id, model_version_id, shap_uri, summary_json, created_at
- alerts: id, project_id, type, severity, message, created_at, status
- audit_logs: id, user_id, action, target_type, target_id, metadata_json, created_at

### 10.3 Data Flow Diagrams (ASCII)
[Raw Data] --> [Validation] --> [Object Storage] --> [Feature Pipeline (fit on train)] --> [Train/Val/Test]
[Train/Val/Test] --> [HPO + CV] --> [Best Model] --> [Registry] --> [Deployment] --> [Inference API]
[Inference API] --> [Monitoring] --> [Alerts/Drift] --> [Retrain Trigger]

### 10.4 Input Data & Dataset Requirements
- Required fields:
  - timestamp (ISO 8601), unique per entity/time.
  - target: thermal_metric (continuous).
  - features: multivariate numeric (e.g., power draw, ambient temp, humidity, airflow rate, CPU load proxy, fan RPM) and categorical context (location, device type, operating mode).
- Sampling: regular preferred; handle irregular with forward fill and mask features.
- Minimum history: ≥120 minutes per entity for default windows.
- Missing data: imputation with leakage-safe historical statistics; mask indicators.
- Size limits: single file ≤50 GB; distributed processing optional via Dask.
- Quality: monotonic timestamp; no duplicate timestamps per entity/context.

## 11. API Specifications
### 11.1 REST Endpoints
- POST /v1/datasets/upload
  - Uploads dataset metadata and location.
- POST /v1/features/preview
  - Returns preview of engineered features on a sample, using leakage-safe transform with provided split reference.
- POST /v1/train
  - Starts a training run with specified feature config and model search space.
- GET /v1/models
  - Lists registered models with metrics.
- POST /v1/deploy
  - Promotes a model version to an endpoint.
- POST /v1/predict
  - Returns point prediction and intervals for latest observation(s).
- POST /v1/explain
  - Returns SHAP values for given observations.
- GET /v1/health
  - Liveness/readiness.
- GET /v1/metrics
  - Prometheus metrics.
- GET /v1/drift
  - Drift and calibration summary.

### 11.2 Request/Response Examples
- POST /v1/predict request:
{
  "model_id": "projA:model:1.3.0",
  "instances": [
    {
      "entity_id": "unit-42",
      "timestamp": "2025-10-15T12:30:00Z",
      "features": {
        "ambient_temp": 27.4,
        "humidity": 0.45,
        "power_draw": 3.8,
        "airflow": 120.0,
        "mode": "eco",
        "location": "dc-west-1"
      }
    }
  ],
  "return_intervals": true,
  "interval_alpha": 0.1
}
- Response:
{
  "model_id": "projA:model:1.3.0",
  "predictions": [
    {
      "entity_id": "unit-42",
      "timestamp": "2025-10-15T12:30:00Z",
      "thermal_metric": 65.2,
      "interval": { "lower": 61.7, "upper": 68.9, "alpha": 0.1 },
      "latency_ms": 142
    }
  ]
}

- POST /v1/train request:
{
  "project_id": "projA",
  "dataset_id": "ds_202510",
  "feature_config_uri": "s3://bucket/configs/feat_cfg.yaml",
  "split": {
    "method": "time",
    "train_end": "2025-08-31T23:59:59Z",
    "val_end": "2025-09-30T23:59:59Z"
  },
  "models": ["lightgbm", "catboost", "tft"],
  "hpo": { "method": "bayes", "trials": 60, "seed": 42 },
  "objective": "rmse",
  "quantiles": [0.1, 0.5, 0.9]
}
- Response:
{ "training_run_id": "tr_000345", "status": "submitted" }

### 11.3 Authentication
- OAuth2/OIDC with JWT bearer tokens.
- Scopes: dataset:write, train:run, model:read, deploy:write, predict:invoke, explain:invoke, admin.
- API keys supported for machine-to-machine with scope limitations.
- All endpoints require HTTPS/TLS 1.2+.

## 12. UI/UX Requirements
### 12.1 User Interface
- React dashboard with sections:
  - Datasets: upload, schema, validation results.
  - Features: configure windows, lags, encoders; leakage checks.
  - Training: launch runs, view HPO progress, metrics, residual slicing.
  - Models: registry view, model cards, compare versions.
  - Deployments: promote/canary/rollback; endpoint status.
  - Monitoring: latency/throughput, drift, calibration, coverage dashboards.
  - Explainability: SHAP summaries, PDP/ICE explorers.

### 12.2 User Experience
- Guided wizards for split configuration and leakage guardrails.
- Contextual tooltips and warnings (e.g., future timestamp usage).
- Exportable reports (PDF/HTML) for stakeholders.

### 12.3 Accessibility
- WCAG 2.1 AA: keyboard navigation, color contrast, ARIA labels.
- Screen reader friendly charts with alt descriptions.

## 13. Security Requirements
### 13.1 Authentication
- OIDC/OAuth2, MFA optional, SSO integration.

### 13.2 Authorization
- RBAC with project-level scoping; least privilege.
- Audit logs for data access, training, deployment, and prediction calls.

### 13.3 Data Protection
- Encryption in transit (TLS 1.2+) and at rest (AES-256 on storage).
- Row-level PII avoidance; if present, tokenize or hash; configurable retention policies.

### 13.4 Compliance
- Alignment with SOC 2 and ISO 27001 controls for access, change management, logging.
- GDPR-friendly data management (export/delete on request).

## 14. Performance Requirements
### 14.1 Response Times
- /predict: P50 < 300 ms, P95 < 500 ms per instance or small batch (≤64).
- /explain: SHAP for single instance < 2 s (approximate methods); batch offline.

### 14.2 Throughput
- Gradient boosting: ≥200 RPS per pod with 2 vCPU.
- Deep models: ≥50 RPS per pod with 2 vCPU and CPU-only; GPU optional.

### 14.3 Resource Usage
- Memory footprint per model: ≤1.5 GB typical (boosting); ≤4 GB (deep).
- CPU load target: ≤70% at steady state; autoscale at 60% utilization.

## 15. Scalability Requirements
### 15.1 Horizontal Scaling
- Kubernetes HPA on CPU/RPS.
- Stateless inference pods with model weights loaded from object storage on startup; Redis cache for hot features.

### 15.2 Vertical Scaling
- Adjustable pod resources; GPU nodes for deep models when needed.

### 15.3 Load Handling
- N+1 redundancy; rolling upgrades; rate limiting and backpressure; circuit breakers.

## 16. Testing Strategy
### 16.1 Unit Testing
- Coverage ≥80% for transformers, splitters, evaluators, and inference logic.
- Property-based tests for leakage (no future info in train transforms).

### 16.2 Integration Testing
- End-to-end: ingest → feature pipeline → train → deploy → predict → monitor.
- Time-based CV correctness using synthetic datasets.

### 16.3 Performance Testing
- Load tests for /predict (Locust/K6).
- Model warm-start latency tests; cache effectiveness.

### 16.4 Security Testing
- Static analysis (Bandit), dependency scanning.
- AuthZ tests for RBAC; fuzzing inputs; secrets scanning.

## 17. Deployment Strategy
### 17.1 Deployment Pipeline
- GitHub Actions: lint, type-check, unit tests, build Docker, push to registry, integration tests on staging, manual approval to prod.
- IaC via Terraform; Helm charts for services.

### 17.2 Environments
- Dev: local MinIO/Postgres/Redis; mock auth.
- Staging: full stack with synthetic data.
- Production: multi-AZ, autoscaling, managed secrets.

### 17.3 Rollout Plan
- Shadow deployment first; compare residuals.
- Canary: 10% → 50% → 100% over 24–48 hours with automated rollback on SLO breach.

### 17.4 Rollback Procedures
- One-click revert to previous model version or app image.
- Automated rollback if latency/error/drift thresholds breached.

## 18. Monitoring & Observability
### 18.1 Metrics
- System: CPU, memory, RPS, latency histograms, error rates.
- Model: RMSE/MAE online (against delayed truth if available), calibration error, interval coverage, drift (PSI, KL).
- Data: missingness rate, schema violations.

### 18.2 Logging
- Structured JSON logs; correlation IDs; request/response sampling.
- Sensitive fields redacted.

### 18.3 Alerting
- Thresholds: latency P95, error rate >0.5%, drift score > configured limit, coverage < target − 3%.
- PagerDuty/Slack integration.

### 18.4 Dashboards
- Grafana dashboards for system and model metrics.
- Explainer dashboards: SHAP drift across versions.

## 19. Risk Assessment
### 19.1 Technical Risks
- Leakage due to custom transforms outside pipeline.
- Concept drift degrading accuracy.
- High variance from small datasets.
- Deep model training cost/instability.

### 19.2 Business Risks
- User adoption hindered by complexity.
- Latency SLAs unmet for complex models.
- Data quality issues from external systems.

### 19.3 Mitigation Strategies
- Enforce pipeline-bound transforms; static checks and tests.
- Scheduled retraining and drift-triggered retrains.
- Start with boosting baselines; scale to deep only when needed.
- Provide templates, wizards, and documentation.

## 20. Timeline & Milestones
### 20.1 Phase breakdown
- Phase 0 (Week 1): Requirements finalization, architecture design.
- Phase 1 (Weeks 2–5): Feature pipeline library, splitters, leakage tests; baseline models.
- Phase 2 (Weeks 6–8): HPO + nested CV; uncertainty (quantiles, conformal); explainability.
- Phase 3 (Weeks 9–10): Model registry integration; API scaffolding; UI basics.
- Phase 4 (Weeks 11–12): Monitoring (metrics, drift, calibration); dashboards; alerts.
- Phase 5 (Weeks 13–14): Deployment (shadow/canary), scaling; security hardening; RBAC.
- Phase 6 (Weeks 15–16): Performance tuning; documentation; GA readiness.

### 20.2 Key Milestones
- M1: Leakage-safe pipeline MVP (Week 5)
- M2: Probabilistic predictions + explainability (Week 8)
- M3: Serving API + Registry integration (Week 10)
- M4: Monitoring & Drift dashboards (Week 12)
- M5: Production canary success (Week 15)
- GA: 16 weeks

Estimated Costs (3 months execution window):
- Cloud: $6,000–$12,000 (training + staging + prod moderate load)
- Team: 1 DS, 1 MLE, 1 FE, 1 BE, 0.5 SRE ≈ $180k–$240k for 4 months

## 21. Success Metrics & KPIs
### 21.1 Measurable targets
- Accuracy: ≥20% RMSE improvement vs. baseline linear model; MAPE ≤10% on validation.
- Calibration: 90% interval coverage within 88–92%.
- Latency: /predict P95 < 500 ms; uptime ≥99.5%.
- Adoption: ≥5 active projects in first quarter; ≥20 training runs/month.
- Reliability: <0.1% 5xx; drift detection within 24 hours of significant change.
- Engineering: 80% test coverage; <2% flaky tests.

## 22. Appendices & Glossary
### 22.1 Technical Background
- Leakage prevention:
  - Use time-based or group-based splits.
  - Pipeline-bound transformations: fit on train only, transform on val/test.
  - Rolling features computed with right-closed windows excluding current target horizon.
  - Target encoding via out-of-fold means; never compute on full dataset pre-split.
  - Nested CV for tuning to avoid overfitting on validation.
- Feature engineering:
  - Temporal lags (t-1, t-5, t-30, etc.), rolling stats (mean/std/min/max), EMAs, trend/seasonality (sin/cos time-of-day/week), cooldown/warmup flags.
  - Robustness: winsorization at train quantiles (e.g., 1–99%), log/sqrt transforms.
  - Interactions: ratios, products; polynomial terms with regularization.
- Modeling:
  - Start with LightGBM/CatBoost; consider TCN/TFT for long sequences.
  - Quantile regression and conformal for uncertainty.
  - Explainability: SHAP; monotonic constraints for stability.
- Monitoring:
  - Drift (PSI/KS), calibration (expected vs. empirical), residual slicing by context.

### 22.2 References
- Hyndman & Athanasopoulos, Forecasting: Principles and Practice.
- Ribeiro et al., "Why Should I Trust You?": LIME/SHAP literature.
- Conformal prediction tutorials and surveys.
- scikit-learn, LightGBM, CatBoost, XGBoost, PyTorch docs.
- Optuna documentation; MLflow model registry best practices.

### 22.3 Glossary
- Data leakage: Using information in training that would not be available at prediction time.
- Target leakage: Leakage arising when target-derived info contaminates features.
- Time-based split: Train/validation/test partition by time to prevent future peeking.
- Rolling window: Moving window aggregation over past observations.
- Quantile regression: Predict specific quantiles of target distribution.
- Conformal prediction: Framework to produce valid prediction intervals.
- Calibration: Agreement between predicted intervals and empirical frequencies.
- Drift: Change in data distribution (data drift) or target relationship (concept drift).
- SHAP: Shapley Additive Explanations for feature importance.
- PDP/ICE: Partial Dependence/Individual Conditional Expectation curves.
- HPO: Hyperparameter optimization (e.g., Bayesian optimization).
- RMSE/MAE/MAPE: Error metrics for regression performance.

Repository Structure
- root/
  - notebooks/
    - 01_eda.ipynb
    - 02_feature_engineering_leakage_checks.ipynb
    - 03_model_baselines.ipynb
    - 04_probabilistic_predictions.ipynb
  - src/
    - aiml041/
      - __init__.py
      - data_ingestion.py
      - splitting.py
      - features/
        - temporal.py
        - robust.py
        - categorical.py
        - interactions.py
      - models/
        - baselines.py
        - sequences.py
        - uncertainty.py
        - explainability.py
      - evaluation/
        - metrics.py
        - residual_slicing.py
      - registry.py
      - serving/
        - api.py
        - schema.py
        - inference.py
        - monitoring.py
      - utils/
        - io.py
        - config.py
        - logging.py
  - tests/
    - test_splitting.py
    - test_features_temporal.py
    - test_leakage_guards.py
    - test_models_baselines.py
    - test_uncertainty.py
    - test_api.py
  - configs/
    - feature_config.yaml
    - training_config.yaml
    - deployment.yaml
  - data/
    - sample_raw.csv
    - sample_processed.parquet
  - Makefile
  - pyproject.toml
  - Dockerfile
  - README.md

Sample Configs
- configs/feature_config.yaml
split:
  method: time
  train_end: "2025-08-31T23:59:59Z"
  val_end: "2025-09-30T23:59:59Z"
entity_id: "unit_id"
timestamp: "timestamp"
target: "thermal_metric"
temporal_features:
  lags: [1, 5, 30, 60]
  rolling:
    - {window: 5, stats: ["mean", "std", "min", "max"]}
    - {window: 30, stats: ["mean", "std"]}
    - {window: 120, stats: ["mean"]}
  ema:
    - {span: 12}
trend_seasonality:
  dow: true
  tod_sin_cos: true
robust:
  winsorize: {lower_q: 0.01, upper_q: 0.99}
  scaler: "robust"
categoricals:
  one_hot: ["mode", "location"]
  target_encoding:
    columns: ["device_type"]
    cv_folds: 5
interactions:
  ratios: [["power_draw", "airflow"]]
  products: [["ambient_temp", "humidity"]]

- configs/training_config.yaml
models:
  lightgbm:
    params:
      objective: "regression"
      n_estimators: 1500
      learning_rate: 0.03
      num_leaves: 64
      feature_fraction: 0.8
      bagging_fraction: 0.8
      bagging_freq: 1
    hpo_space:
      num_leaves: [31, 255]
      min_data_in_leaf: [20, 200]
      learning_rate: [0.005, 0.1]
  catboost:
    params:
      depth: 8
      iterations: 1500
      learning_rate: 0.03
  tcn:
    params:
      layers: [32, 64, 64]
      kernel_size: 3
      dropout: 0.1
quantiles: [0.1, 0.5, 0.9]
cv:
  method: "expanding_window"
  folds: 5
  gap: 1
objective: "rmse"
seed: 42
early_stopping_rounds: 100

Sample Code Snippets
- Leakage-safe temporal feature transformer (simplified):
class RollingFeatureTransformer:
    def __init__(self, lags, rolling_windows, entity_col, time_col, target_col):
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.entity_col = entity_col
        self.time_col = time_col
        self.target_col = target_col
        self.fitted_ = False

    def fit(self, df_train):
        # No future-derived statistics; only store schema and training quantiles if needed
        self.fitted_ = True
        return self

    def transform(self, df):
        # Assumes df is sorted by entity_col, time_col
        import pandas as pd
        df = df.sort_values([self.entity_col, self.time_col]).copy()
        for L in self.lags:
            df[f"{self.target_col}_lag_{L}"] = df.groupby(self.entity_col)[self.target_col].shift(L)
        for w in self.rolling_windows:
            for stat in ["mean", "std", "min", "max"]:
                if stat not in w["stats"]:
                    continue
                col = f"{self.target_col}_roll{w['window']}_{stat}"
                df[col] = getattr(
                    df.groupby(self.entity_col)[self.target_col].shift(1).rolling(w["window"]),
                    stat
                )()
        return df

- FastAPI predict endpoint:
from fastapi import FastAPI, Depends
from pydantic import BaseModel
import joblib, time

app = FastAPI()
model = joblib.load("/models/current/model.pkl")
pipeline = joblib.load("/models/current/pipeline.pkl")

class Instance(BaseModel):
    entity_id: str
    timestamp: str
    features: dict

class PredictRequest(BaseModel):
    model_id: str
    instances: list[Instance]
    return_intervals: bool = True
    interval_alpha: float = 0.1

@app.post("/v1/predict")
def predict(req: PredictRequest):
    t0 = time.time()
    X = pipeline.online_transform(req.instances)  # custom method for online features
    yhat, intervals = model.predict_with_intervals(X, alpha=req.interval_alpha)
    latency = int((time.time() - t0) * 1000)
    return {"model_id": req.model_id, "predictions": [
        {"entity_id": inst.entity_id, "timestamp": inst.timestamp,
         "thermal_metric": float(yh), "interval": intervals[i], "latency_ms": latency}
        for i, (inst, yh) in enumerate(zip(req.instances, yhat))
    ]}

- Conformal prediction wrapper (conceptual):
class ConformalRegressor:
    def __init__(self, base, alpha=0.1):
        self.base = base
        self.alpha = alpha
        self.cal_scores_ = None

    def fit(self, X_train, y_train, X_val, y_val):
        self.base.fit(X_train, y_train)
        residuals = abs(y_val - self.base.predict(X_val))
        self.cal_scores_ = np.quantile(residuals, 1 - self.alpha)

    def predict_with_intervals(self, X):
        y = self.base.predict(X)
        q = self.cal_scores_
        return y, [{"lower": float(y[i] - q), "upper": float(y[i] + q), "alpha": self.alpha} for i in range(len(y))]

Service SLOs
- Accuracy improvement ≥20% RMSE vs. linear baseline on holdout.
- /predict uptime ≥99.5%.
- P95 latency <500 ms for typical payloads.

End of PRD.