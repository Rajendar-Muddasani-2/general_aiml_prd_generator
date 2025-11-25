# Product Requirements Document (PRD)
# `Aiml036_Bias_Detection_And_Mitigation`

Project ID: aiml036
Category: AI/ML – Fairness, Responsible AI, Model Governance
Status: Draft
Version: 1.0.0
Last Updated: 2025-11-25

## 1. Overview
### 1.1 Executive Summary
aiml036 is a platform and toolkit for detecting, quantifying, mitigating, and monitoring bias in AI/ML systems across NLP, computer vision, tabular, and multimodal use cases. It provides automated subgroup discovery, comprehensive fairness metrics with statistical significance, counterfactual and causal probes, and a suite of pre-/in-/post-processing mitigation techniques. The system integrates into training/inference pipelines via Python SDK, CLI, and REST APIs, and supports continuous monitoring and governance (model cards, datasheets, audit trails, approvals).

### 1.2 Document Purpose
This PRD defines scope, requirements, architecture, data model, APIs, UI/UX, security, performance, deployment, and success metrics for aiml036 to align engineering, product, and compliance stakeholders.

### 1.3 Product Vision
Make fairness-by-design practical: a developer-first and governance-ready platform that helps teams reliably improve worst-group performance and reduce harmful disparities without sacrificing transparency, reproducibility, or operational efficiency.

## 2. Problem Statement
### 2.1 Current Challenges
- Teams lack standardized pipelines to detect and quantify bias across protected attributes and intersections.
- Fairness metrics are fragmented; significance and confidence intervals are rarely computed.
- Bias mitigation is ad hoc; unclear trade-offs and unstable results across datasets and releases.
- Limited tooling for counterfactual fairness, attribution disparity, and calibration parity.
- Sparse monitoring of subgroup drift and outcome disparities in production.
- Compliance pressure requires auditability, approvals, and documentation.

### 2.2 Impact Analysis
- Business: Reputational risk, legal exposure, revenue loss from unfair decisions, and customer churn.
- Technical: Models generalize poorly to underrepresented cohorts; unstable performance across releases.
- Operational: Manual, slow audits; no baseline governance; difficulty reproducing results.

### 2.3 Opportunity
- Provide a unified, statistically rigorous fairness platform to detect and mitigate bias, improve worst-group accuracy, and maintain fairness SLAs via monitoring and approvals.
- Enable objective, traceable choices across fairness–accuracy–calibration trade-offs.

## 3. Goals and Objectives
### 3.1 Primary Goals
- Deliver end-to-end bias detection, mitigation, and monitoring for tabular, NLP, and vision tasks.
- Provide rigorous fairness evaluation: subgroup metrics, intersections, bootstrapped CIs, permutation tests.
- Offer mitigation recipes: reweighting, GroupDRO, adversarial debiasing, fairness-constrained optimization, post-hoc thresholding.
- Enable continuous monitoring and governance: dashboards, alerts, model cards, datasheets, approvals.

### 3.2 Business Objectives
- Reduce time-to-audit by 70%.
- Improve worst-group accuracy by ≥10% with ≤2% overall accuracy drop.
- Achieve 99.5% uptime for APIs and dashboards.
- Accelerate compliance sign-off and reduce incidents by 50%.

### 3.3 Success Metrics
- >90% audit coverage of defined cohorts in production models.
- <500 ms p95 latency for audit metric retrieval; <2 s p95 for on-demand subgroup report generation.
- ≥95% reproducibility rate (hash-identical runs given same seed/data/version).
- ≥30% reduction in statistically significant disparities (e.g., TPR parity gap) post-mitigation.

## 4. Target Users/Audience
### 4.1 Primary Users
- ML Engineers and Data Scientists
- Responsible AI/Fairness Leads
- MLOps/Platform Engineers

### 4.2 Secondary Users
- Product Managers
- Compliance, Legal, and Risk Officers
- Domain Experts and Annotators

### 4.3 User Personas
- Persona 1: Maya Kapoor – Senior ML Engineer
  - Background: 7 years in ML; builds classification and ranking systems (NLP and tabular).
  - Pain points: Lacks standardized fairness pipeline; manual slicing and metrics; difficulty tuning thresholds per group.
  - Goals: Automate fairness evaluations; integrate mitigation into training with CI/CD; ensure stable worst-group risk.

- Persona 2: Luis Ortega – Responsible AI Lead
  - Background: Ethics/compliance and data science hybrid; sets policy; runs reviews.
  - Pain points: Inconsistent reports; no confidence intervals; limited documentation; no approvals workflow.
  - Goals: Comparable metrics across teams; clear evidence of statistical significance; model cards; approval traces.

- Persona 3: Chen Wang – MLOps Engineer
  - Background: Platform/infra; Kubernetes, observability, secrets, and deployment pipelines.
  - Pain points: Difficult to productionize fairness checks; ad hoc batch jobs; limited alerting and dashboards.
  - Goals: Robust APIs, scalable workers, monitoring, SLOs; seamless integration with existing observability stack.

- Persona 4: Alisha Brown – Product Manager
  - Background: Owns a decisioning product; cares about user trust and KPIs.
  - Pain points: Trade-offs unclear; no single pane of glass; can’t quantify business impact of fairness changes.
  - Goals: Visualize Pareto frontiers; scenario plan; track fairness KPIs and customer impact.

## 5. User Stories
- US-001: As an ML Engineer, I want to run an automated bias audit on a model’s predictions by specified sensitive attributes so that I can identify disparities early.
  - Acceptance: API accepts predictions, labels, and group data; returns metrics with 95% bootstrap CIs and significance tests.

- US-002: As a Responsible AI Lead, I want to evaluate intersectional subgroups so that hidden disparities are surfaced.
  - Acceptance: System computes metrics for intersections (e.g., gender x age) with minimum sample controls.

- US-003: As an ML Engineer, I want to try pre-/in-/post-processing mitigation techniques and compare outcomes so that I can pick the best trade-off.
  - Acceptance: UI shows side-by-side metrics, Pareto frontier (accuracy vs fairness), and recommended policy.

- US-004: As an MLOps Engineer, I want continuous monitoring of subgroup metrics in production so that I get alerts when disparities exceed thresholds.
  - Acceptance: Configurable alert rules; notifications via email/Slack/Webhooks; links to dashboards.

- US-005: As a Compliance Officer, I want downloadable model cards and datasheets so that audits can be satisfied.
  - Acceptance: PDF/JSON export includes metrics, CIs, datasets used, mitigation steps, and approvals.

- US-006: As a Data Scientist, I want counterfactual fairness tests so that I can measure sensitivity to protected attributes.
  - Acceptance: Supports minimal-edit perturbations for text and attribute edits for images; reports effect sizes.

- US-007: As a PM, I want to see the impact of mitigation on business KPIs so that I can make a go/no-go decision.
  - Acceptance: Dashboard connects fairness metrics to cost/benefit assumptions, with scenario toggles.

## 6. Functional Requirements
### 6.1 Core Features
- FR-001 Audit Pipeline: Upload predictions/labels and group data; compute fairness metrics with bootstrapped CIs and permutation tests.
- FR-002 Cohort & Intersection Analysis: Automatic slice discovery and manual subgroup definitions; sample-size checks.
- FR-003 Metrics Library: Demographic parity, disparate impact ratio, equal opportunity, equalized odds, predictive parity, calibration within groups, error-rate balance, subgroup AUC, worst-group accuracy/risk.
- FR-004 Visualization: ROC/PR per group, threshold sweeps, calibration curves by group, Simpson’s paradox checks, disparity heatmaps.
- FR-005 Bias Probes: WEAT/SEAT (embeddings), StereoSet/CrowS-Pairs for NLP, template perturbations, fairness stress tests.
- FR-006 Mitigation Suite:
  - Pre: reweighting/resampling, label noise correction, data repair, counterfactual augmentation, representation balancing, deduping.
  - In: adversarial debiasing, fairness-constrained optimization (Lagrangian), GroupDRO, MI/MMD penalties, INLP, cost-sensitive/focal loss, monotonic constraints.
  - Post: group-wise threshold optimization (equalized odds/opportunity), reject option, calibrated equalized odds, per-group calibration.
- FR-007 Threshold Optimizer: Optimize metrics subject to constraints; support multi-objective trade-offs.
- FR-008 Monitoring & Alerts: Drift and disparity monitoring; alerting; cohort health tracking.
- FR-009 Governance: Model cards, datasheets, approval workflows, audit logs, red-teaming workflows.
- FR-010 SDK/CLI/API: Python SDK, CLI, REST for seamless integration.
- FR-011 Reproducibility: Run artifacts, seeds, environment hashes, dataset versioning.
- FR-012 Report Export: PDF/HTML/JSON with charts and appendices.

### 6.2 Advanced Features
- FR-013 Counterfactual Fairness via Causal Graphs (optional DAG input).
- FR-014 Attribution Disparity: SHAP/Captum-based per-group feature attributions and gap analysis.
- FR-015 Active Learning: Suggests data acquisition to close cohort gaps.
- FR-016 Auto-Mitigation Recommendation: Recommender ranks mitigation candidates by expected uplift and cost.
- FR-017 Red-Teaming Integration: Human-in-the-loop adversarial data collection with feedback loops.

## 7. Non-Functional Requirements
### 7.1 Performance
- p95 API GET metric latency < 500 ms; p95 audit job submission < 300 ms; batch audit completion within SLA (configurable).
### 7.2 Reliability
- 99.5% uptime monthly; job retry with exponential backoff; idempotent endpoints.
### 7.3 Usability
- Accessible UI; clear tooltips for metrics; templated reports; guided setup wizards.
### 7.4 Maintainability
- Modular services; typed SDK; semantic versioning; strong test coverage (>85% lines; >90% critical modules).

## 8. Technical Requirements
### 8.1 Technical Stack
- Backend: Python 3.11+, FastAPI 0.115+, Pydantic 2.x, SQLAlchemy 2.x, Celery 5.x
- ML: PyTorch 2.3+, torchvision 0.18+, scikit-learn 1.5+, transformers 4.44+, SHAP 0.45+, Captum 0.7+, Fairlearn 0.9+, AIF360 0.6+
- Data: PostgreSQL 15+, Redis 7+, MinIO/S3, Kafka 3.x (optional), Parquet/Arrow
- Orchestration: Prefect 2.x
- Frontend: React 18+, TypeScript 5+, Vite, TailwindCSS
- Infra: Docker, Kubernetes 1.28+, Helm, Terraform, Prometheus, Grafana, OpenTelemetry, ELK/Opensearch
- Auth: OIDC/OAuth2.1, Keycloak/Okta/Auth0
- Packaging: Poetry or pip-tools; pre-commit hooks; ruff/mypy/black

### 8.2 AI/ML Components
- Fairness metric engine with bootstrap/permutation modules (Numba-accelerated where possible)
- Mitigation trainers: GroupDRO, adversarial debiasing (gradient reversal), Lagrangian-constrained optimizer
- Threshold optimization solver (cvxpy 1.4+ / scipy optimize)
- Probe suite: WEAT/SEAT, StereoSet/CrowS-Pairs loaders and evaluators
- Counterfactual augmentation for text (template edits) and images (attribute toggles via pretrained attribute editors where available)
- Monitoring analyzers for drift/disparity (population stability, PSI, KS, EMD) and fairness metric deltas

## 9. System Architecture
### 9.1 High-Level Architecture
ASCII diagram:
[Browser/UI] --HTTPS--> [API Gateway] --> [FastAPI App]
                                   |            \
                                   |             -> [Auth Service (OIDC)]
                                   |             -> [Metric Engine Service]
                                   |             -> [Mitigation Service]
                                   |             -> [Report/Model Card Service]
                                   |             -> [Monitoring Service]
                                   |             -> [Scheduler (Prefect)]
                                   v
                              [Message Bus/Queue (Redis/Kafka)]
                                   |
                              [Worker Pool (Celery)]
                                   |
                        +----------+-----------+
                        |                      |
                  [PostgreSQL]            [Object Store (S3/MinIO)]
                        |                      |
                     [Metadata]           [Artifacts, datasets, reports]
                        |
                   [Observability]
            (Prometheus, OTel, Grafana, Logs)

### 9.2 Component Details
- API Service: Validates requests, triggers jobs, streams metrics, serves dashboards.
- Metric Engine: Computes fairness metrics, bootstrap CIs, permutation p-values; parallelized.
- Mitigation Service: Trains mitigation strategies; exposes models or policies; logs trade-offs.
- Monitoring Service: Periodic jobs ingest production outcomes; computes cohort health; triggers alerts.
- Report Service: Generates model cards/datasheets; stores in object store.
- Scheduler/Workers: Orchestrate batch audits, heavy computations, and report generation.
- Auth: OIDC provider integration; RBAC/ABAC enforcement.
- Storage: PostgreSQL for metadata; object store for artefacts and datasets.

### 9.3 Data Flow
1. User submits audit job via SDK/API with dataset references and predictions.
2. API stores metadata; schedules job to workers.
3. Metric Engine loads data from object store; computes metrics and CIs; persists results.
4. Optional mitigation run; results logged.
5. Report Service compiles results; artifacts stored; model card generated.
6. Monitoring process ingests production logs, computes disparities, and triggers alerts.

## 10. Data Model
### 10.1 Entity Relationships
- Project 1—N ModelVersion
- ModelVersion 1—N AuditRun
- AuditRun 1—N MetricResult
- Project 1—N Dataset
- Dataset 1—N SensitiveAttribute
- SensitiveAttribute N—N GroupDefinition (including intersections)
- AuditRun 1—N ProbeResult (WEAT/etc.)
- AuditRun 1—N MitigationRun
- MitigationRun 1—1 ThresholdPolicy
- Project 1—N AlertRule; AlertRule 1—N AlertEvent
- User N—N Role; Project 1—N Approval

### 10.2 Database Schema (selected tables)
- projects(id, name, description, owner_user_id, created_at, updated_at)
- datasets(id, project_id, uri, schema_hash, row_count, created_at)
- sensitive_attributes(id, dataset_id, name, type, values_json, is_protected_flag)
- group_definitions(id, project_id, name, expression_json, min_samples)
- models(id, project_id, name, task_type, framework, created_at)
- model_versions(id, model_id, version, artifact_uri, git_commit, params_json, created_at)
- audit_runs(id, model_version_id, dataset_id, status, config_json, seed, started_at, finished_at)
- metric_results(id, audit_run_id, group_id, metric_name, value, ci_low, ci_high, p_value, sample_size)
- probe_results(id, audit_run_id, probe_name, score, ci_low, ci_high, details_json)
- mitigation_runs(id, audit_run_id, method, params_json, baseline_metrics_json, mitigated_metrics_json, artifact_uri, started_at, finished_at)
- threshold_policies(id, mitigation_run_id, policy_json, constraints_json)
- alerts_rules(id, project_id, name, condition_json, channels_json, enabled)
- alerts_events(id, rule_id, occurred_at, payload_json, status)
- approvals(id, project_id, approver_user_id, audit_run_id, decision, notes, decided_at)
- users(id, email, name, org, created_at)
- roles(id, name, permissions_json)
- audit_logs(id, user_id, action, entity_type, entity_id, at, diff_json)

### 10.3 Data Flow Diagrams
- Inputs: predictions.csv, labels.csv, features.parquet, groups.json
- Process: metric engine -> bootstrap -> significance -> aggregation -> storage
- Outputs: metrics.json, report.pdf, model_card.json, mitigation_artifacts/

### 10.4 Input Data & Dataset Requirements
- Tabular: numerical/categorical features; labels (binary/multi-class/regression); sensitive attributes or proxies.
- NLP: text fields; tokenization handled via transformers; optional embeddings.
- CV: image URIs; optional attribute annotations; embeddings via pretrained encoders.
- Sensitive attributes: explicit or inferred proxies (optional), with documentation.
- Minimum sample thresholds per group/intersection; configurable (e.g., >=50 instances).

## 11. API Specifications
### 11.1 REST Endpoints (v1)
- POST /api/v1/projects
- GET /api/v1/projects/{project_id}
- POST /api/v1/projects/{project_id}/datasets
- POST /api/v1/projects/{project_id}/models
- POST /api/v1/projects/{project_id}/models/{model_id}/versions
- POST /api/v1/audits
- GET /api/v1/audits/{audit_id}
- GET /api/v1/audits/{audit_id}/metrics
- POST /api/v1/mitigations
- GET /api/v1/mitigations/{mitigation_id}
- POST /api/v1/thresholds/optimize
- GET /api/v1/reports/{audit_id}
- POST /api/v1/monitoring/ingest
- POST /api/v1/alerts/rules
- GET /api/v1/alerts/events
- POST /api/v1/approvals
- GET /api/v1/modelcards/{model_version_id}

### 11.2 Request/Response Examples
- Example: Submit audit
Request:
POST /api/v1/audits
Content-Type: application/json
{
  "project_id": "prj_123",
  "model_version_id": "modv_456",
  "dataset_id": "ds_789",
  "task_type": "binary_classification",
  "sensitive_attributes": ["gender", "age_bucket"],
  "intersections": true,
  "metrics": ["demographic_parity", "equal_opportunity", "equalized_odds", "calibration_by_group", "worst_group_accuracy"],
  "min_group_size": 50,
  "bootstrap_samples": 1000,
  "seed": 42
}

Response:
{
  "audit_id": "audit_abc",
  "status": "queued",
  "estimated_completion_sec": 120
}

- Example: Optimize thresholds
POST /api/v1/thresholds/optimize
{
  "audit_id": "audit_abc",
  "constraints": {"equal_opportunity_diff_max": 0.02},
  "objective": "maximize_accuracy",
  "groups": ["gender", "age_bucket"]
}
Response:
{
  "policy_id": "pol_123",
  "thresholds": {"global": 0.51, "gender:female": 0.47, "gender:male": 0.53},
  "expected_metrics": {"accuracy": 0.918, "equal_opportunity_diff": 0.018}
}

### 11.3 Authentication
- OAuth2/OIDC with JWT bearer tokens; scopes for read/write/admin.
- Service accounts for CI/CD; API keys optional with RBAC limits.

## 12. UI/UX Requirements
### 12.1 User Interface
- Pages: Projects, Datasets, Models & Versions, Audits, Mitigations, Monitoring, Alerts, Reports, Settings.
- Visualizations: disparity heatmaps, ROC/PR by group, calibration curves, threshold sweeps, Pareto charts.
- Wizards: “Run Audit,” “Run Mitigation,” “Set Alerts,” “Generate Model Card.”

### 12.2 User Experience
- Guided explanations for fairness metrics; hover tooltips; info modals on trade-offs.
- Compare baseline vs mitigated runs; pin and annotate charts.
- Downloadable artifacts: CSV/JSON/PDF.

### 12.3 Accessibility
- WCAG 2.1 AA compliance: keyboard navigation, ARIA labels, high-contrast mode, screen-reader support.

## 13. Security Requirements
### 13.1 Authentication
- OIDC/OAuth2.1; passwordless SSO; MFA enforcement optional.
### 13.2 Authorization
- RBAC with roles: viewer, contributor, maintainer, admin; optional ABAC (project-level).
### 13.3 Data Protection
- TLS 1.2+; encryption at rest (KMS-managed); per-tenant isolation; PII minimization and masking for sensitive attributes.
### 13.4 Compliance
- GDPR/CCPA support (data subject requests, retention policies), SOC 2/ISO 27001 alignment; audit logs immutable storage.

## 14. Performance Requirements
### 14.1 Response Times
- p95: <500 ms for metric reads; <2 s for chart generation; <5 s for model card generation (cached).
### 14.2 Throughput
- Sustain 100 RPS for read endpoints; 10 concurrent audit submissions per project without degradation.
### 14.3 Resource Usage
- Metric engine uses vectorized ops and batched IO; autoscale workers; GPU optional for deep models.

## 15. Scalability Requirements
### 15.1 Horizontal Scaling
- Scale API and workers via Kubernetes HPA; stateless services; Redis/Kafka partitioning.
### 15.2 Vertical Scaling
- Allow GPU nodes for adversarial debiasing and GroupDRO.
### 15.3 Load Handling
- Queue backpressure; priority lanes for interactive tasks; rate limiting per tenant.

## 16. Testing Strategy
### 16.1 Unit Testing
- Coverage >85%; property-based tests for metrics; numerical stability checks.
### 16.2 Integration Testing
- E2E tests with sample datasets (tabular/NLP/CV); golden snapshots; contract tests for APIs.
### 16.3 Performance Testing
- Load tests (k6/Locust); bootstrap CI performance benchmarking; memory profiling.
### 16.4 Security Testing
- SAST/DAST, dependency scanning, secrets scanning, RBAC bypass tests; pen tests pre-GA.

## 17. Deployment Strategy
### 17.1 Deployment Pipeline
- GitHub Actions: lint/test/build; image scan; Helm chart deploy; blue/green or canary.
### 17.2 Environments
- Dev, Staging, Prod; separate credentials and data stores; feature flags.
### 17.3 Rollout Plan
- MVP private beta (2–3 teams); staged rollout with SLOs; ramp alerts thresholds.
### 17.4 Rollback Procedures
- Helm rollback, DB migration down scripts, artifact retention with version pinning.

## 18. Monitoring & Observability
### 18.1 Metrics
- System: latency, error rate, job duration, worker queue depth.
- Fairness: disparity metrics over time, worst-group accuracy, drift scores per cohort.
### 18.2 Logging
- Structured JSON logs; correlation IDs; audit trails with immutable sink.
### 18.3 Alerting
- On-call alerts for SLO breaches; fairness alerts to stakeholders; Slack/Email/Webhooks.
### 18.4 Dashboards
- Grafana: API performance, worker throughput, fairness KPIs, cohort health.

## 19. Risk Assessment
### 19.1 Technical Risks
- High variance in small groups; mitigation instability; false positives in drift/disparity.
- Integration complexity across diverse model types.
### 19.2 Business Risks
- Misinterpretation of metrics; over-optimizing fairness at cost of business value; compliance expectations evolving.
### 19.3 Mitigation Strategies
- Minimum sample safeguards; bootstrap CIs; educate via UI; sandbox to evaluate trade-offs; versioned policies; advisory reports.

## 20. Timeline & Milestones
### 20.1 Phase breakdown
- Phase 0 (2 weeks): Discovery, detailed design, user interviews.
- Phase 1 (6 weeks): Core audit pipeline, metrics, SDK, basic UI, reports.
- Phase 2 (6 weeks): Mitigation suite (pre/in/post), threshold optimizer, Pareto visualization.
- Phase 3 (4 weeks): Monitoring, alerts, model cards/datasheets, governance workflows.
- Phase 4 (3 weeks): Performance hardening, security, accessibility, documentation.
- Phase 5 (2 weeks): Beta with pilot teams; feedback and fixes.
- Phase 6 (2 weeks): GA release, enablement, success measurement.

Total: ~25 weeks.

### 20.2 Key Milestones
- M1: Audit MVP with CIs and permutation tests (Week 8)
- M2: Mitigation suite + threshold optimizer (Week 14)
- M3: Monitoring + alerts + model cards (Week 18)
- M4: Perf/security hardening (Week 22)
- M5: GA (Week 25)

Estimated Costs (monthly, pilot scale):
- Cloud infra: $3k–$8k
- Engineering team (5–6 FTE): budgeted operationally
- Tools/licenses: ~$500–$2k

## 21. Success Metrics & KPIs
### 21.1 Measurable targets
- Uptime: ≥99.5%
- Latency: p95 metric read <500 ms
- Audit coverage: ≥90% of active models
- Fairness improvement: ≥30% reduction in key disparity gaps; worst-group accuracy +10%
- Reproducibility: ≥95% identical reruns
- Time-to-audit: reduce by ≥70%
- Incidents related to bias: -50% within 6 months of adoption

## 22. Appendices & Glossary
### 22.1 Technical Background
- Bias taxonomy: sampling, label, measurement, historical, aggregation, representation, deployment/interaction; proxies for sensitive attributes.
- Fairness metrics: demographic/statistical parity, disparate impact ratio, equal opportunity, equalized odds, predictive parity, calibration within groups, error-rate balance, subgroup AUC, worst-group accuracy/risk.
- Evaluation: slice/cohort metrics with bootstrapped confidence intervals and permutation tests; ROC/PR per group; threshold sweeps; calibration curves; Simpson’s paradox checks; counterfactual fairness tests; attribution disparity (SHAP/IG); residual analysis and uplift by group.
- Mitigation: pre-processing (reweighting/resampling, label noise correction, data repair, counterfactual augmentation, balancing, dedup/near-duplicate control); in-processing (adversarial debiasing, Lagrangian constraints, GroupDRO, MI/MMD penalties, INLP, cost-sensitive/focal loss, monotonicity); post-processing (group-wise thresholds, reject option, calibrated equalized odds, per-group calibration).
- Monitoring & governance: drift/disparity monitoring; active learning; model cards/datasheets; red-teaming; incident response; feedback loops.
- Trade-offs: accuracy–fairness Pareto frontier; calibration vs equalized odds; privacy–fairness; robustness vs fairness.

### 22.2 References
- Hardt et al. (2016) Equality of Opportunity in Supervised Learning
- Kleinberg et al. (2017) Inherent Trade-Offs in the Fair Determination of Risk Scores
- Agarwal et al. (2018) A Reductions Approach to Fair Classification
- Sagawa et al. (2020) Distributionally Robust Neural Networks
- Zhang et al. (2018) Mitigating Unwanted Bias with Adversarial Learning
- Ravfogel et al. (2020) INLP
- WEAT/SEAT papers; StereoSet; CrowS-Pairs
- Fairlearn, AIF360 documentation

### 22.3 Glossary
- Protected attribute: Attribute such as gender, race, age that requires special consideration.
- Proxy variable: Variable correlated with a protected attribute.
- Group fairness: Fair outcomes across groups.
- Individual fairness: Similar individuals receive similar outcomes.
- Demographic parity: Outcome independence from protected attribute.
- Equalized odds: Equal TPR and FPR across groups.
- Equal opportunity: Equal TPR across groups.
- Predictive parity: Equal PPV across groups.
- Calibration within groups: Predicted probabilities match outcomes per group.
- Worst-group accuracy: Accuracy of the group with the lowest performance.
- GroupDRO: Training that minimizes worst-case group risk.
- Adversarial debiasing: Learning that removes information about protected attributes via adversary.
- INLP: Iterative null-space projection to remove protected attribute information.
- Bootstrap: Resampling method to estimate uncertainty.
- Permutation test: Non-parametric significance test via label shuffling.

Repository Structure
- notebooks/
  - 01_quickstart_audit.ipynb
  - 02_mitigation_playbook.ipynb
  - 03_threshold_optimizer.ipynb
- src/
  - aiml036/
    - api/
    - metrics/
    - mitigation/
    - probes/
    - monitoring/
    - reporting/
    - sdk/
    - utils/
- tests/
  - unit/
  - integration/
  - e2e/
- configs/
  - default_config.yaml
  - alert_policies.yaml
- data/
  - samples/
    - tabular/
    - nlp/
    - vision/
- docker/
- helm/
- scripts/

Sample Config (YAML)
audit:
  project_id: prj_123
  model_version_id: modv_456
  dataset_uri: s3://bucket/data/test.parquet
  task_type: binary_classification
  sensitive_attributes: [gender, age_bucket]
  intersections: true
  metrics: [demographic_parity, equal_opportunity, equalized_odds, calibration_by_group, worst_group_accuracy]
  bootstrap_samples: 1000
  min_group_size: 50
  seed: 42

Python SDK Example
from aiml036.sdk import Client

client = Client(base_url="https://fair.example.com", token="...")

audit = client.audits.create(
    project_id="prj_123",
    model_version_id="modv_456",
    dataset_id="ds_789",
    task_type="binary_classification",
    sensitive_attributes=["gender", "age_bucket"],
    intersections=True,
    metrics=["demographic_parity", "equal_opportunity", "equalized_odds", "calibration_by_group", "worst_group_accuracy"],
    bootstrap_samples=1000,
    min_group_size=50,
    seed=42
)
audit.wait()  # poll until complete
metrics = client.audits.metrics(audit_id=audit.id)
print(metrics.summary())

Threshold Optimizer (Python snippet)
from aiml036.sdk import Thresholds

policy = Thresholds.optimize(
    audit_id=audit.id,
    constraints={"equal_opportunity_diff_max": 0.02},
    objective="maximize_accuracy"
)
print(policy.thresholds)

Example Metric Computation (pseudo)
from aiml036.metrics import metrics as M

res = M.demographic_parity(y_pred, groups)
ci = M.bootstrap_ci(res, n=1000, seed=42)
p = M.permutation_test(y_pred, y_true, groups, metric="tpr_gap")

Notes
- Set strict content policies: disallow use of protected attributes at inference if policy requires; allow per-group thresholds if permitted by policy and law.
- Document fairness-privacy interactions; support anonymized group evaluation where feasible.

This PRD defines a complete solution aimed at delivering a rigorous, developer-friendly, and governance-ready bias detection and mitigation platform with clear performance targets (e.g., >90% accuracy on representative tasks, <500 ms latency for metric reads, 99.5% uptime) and practical deployment patterns across modern AI/ML stacks.