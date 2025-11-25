# Product Requirements Document (PRD) / # aiml026_a_b_testing_infrastructure_ml

Project ID: aiml026_a_b_testing_infrastructure_ml
Category: General AI/ML Platform — A/B Testing and Experimentation Infrastructure
Status: Draft for Review
Version: v1.0
Last Updated: 2025-11-25

## 1. Overview
### 1.1 Executive Summary
We will build a robust A/B/n testing and experimentation platform purpose-built for machine learning applications, with deterministic assignment, feature flags, exposure logging, metrics-as-code, near-real-time analysis, and a statistical engine supporting both frequentist and Bayesian inference. The platform integrates tightly with ML infrastructure (feature stores, model registries, model serving) to enable safe canary and shadow deployments, fairness guardrails, variance reduction using covariates, counterfactual policy evaluation, and automated rollback. This platform will improve the speed, rigor, and safety of deploying and iterating on ML models across NLP, computer vision, ranking, and recommendation use cases.

### 1.2 Document Purpose
Define the comprehensive product, technical, and operational requirements for the end-to-end ML experimentation infrastructure. This PRD is the single source of truth for product scope, architecture, APIs, UI/UX, security, performance, testing, deployment, monitoring, and governance.

### 1.3 Product Vision
Democratize trustworthy ML experimentation with a modern, cloud-native platform that makes it easy for teams to design, run, and learn from controlled experiments—reducing risk, accelerating iteration, and improving model and product outcomes.

## 2. Problem Statement
### 2.1 Current Challenges
- Fragmented experimentation with ad-hoc scripts and dashboards; manual analysis is slow and error-prone.
- Inconsistent assignment and exposure logging lead to sample ratio mismatch (SRM) and biased results.
- Lack of near-real-time metrics and automated guardrails delays detection of regressions.
- Difficult to coordinate model versions, feature flags, and gradual rollout strategies.
- Limited ML-specific evaluation (e.g., IPS/DR, interleaving, uplift, fairness).
- Weak governance: collision detection, exclusion groups, and long-term holdouts are lacking.

### 2.2 Impact Analysis
- Slower time-to-value for ML deployments.
- Increased risk of negative impact to key metrics due to unsafe rollouts.
- Reduced trust in experimentation results due to bias, SRM, and p-hacking.
- Higher operational costs due to rework, outages, and manual processes.

### 2.3 Opportunity
Provide a unified experimentation platform that:
- Ensures rigorous, unbiased experimentation at scale.
- Enables ML-aware analysis and governance.
- Dramatically shortens feedback cycles while increasing safety and compliance.

## 3. Goals and Objectives
### 3.1 Primary Goals
- Deterministic assignment service with consistent hashing, mutually exclusive layers, targeted cohorts, and trigger conditions.
- Feature flags and model gating with canary, shadow, and progressive rollout policies.
- Client/server SDKs for consistent exposure logging and cross-device identity resolution.
- Nearline metrics pipeline with metrics-as-code and OLAP-backed reporting.
- Statistical engine with SRM detection, CUPED/CUPAC, sequential testing, FDR control, Bayesian/frequentist options.
- ML-specific modules: counterfactual evaluation (IPS/DR), interleaving for ranking, uplift and heterogeneous treatment effects, fairness guardrails.
- Governance: exclusion groups, long-term holdouts, privacy, audit trails, and compliance logs.
- Deep integration with model registry and feature store; offline-to-online correlation.

### 3.2 Business Objectives
- Reduce time-to-decision for experiments by 70%.
- Decrease incident rate in ML rollouts by 60% via automated guardrails and rollbacks.
- Improve model ROI by 10–20% via faster, safer iteration and ML-specific evaluation.

### 3.3 Success Metrics
- >99% deterministic assignment stability.
- <500 ms p95 assignment API latency.
- <2 minutes end-to-end metrics freshness lag for nearline aggregations.
- 99.5% platform uptime SLO.
- 90% of experiments pass pre-flight power/MDE checks.
- 100% of experiments subject to SRM detection and guardrails.

## 4. Target Users/Audience
### 4.1 Primary Users
- ML Engineers and Data Scientists running online experiments.
- Product Analysts defining metrics and powering decision-making.
- ML Platform Engineers integrating model serving and rollout control.

### 4.2 Secondary Users
- Product Managers reviewing experiment results and decisions.
- Data Engineers managing pipelines and data quality.
- Compliance/Privacy Officers auditing experiment policies and data handling.
- SRE/DevOps monitoring performance and availability.

### 4.3 User Personas
- Persona 1: Dr. Priya Raman — Senior ML Engineer
  - Background: PhD in ML; owns ranking models for a recommendation system.
  - Pain points: Slow iteration, difficulty validating offline metrics online; fear of regressions.
  - Goals: Rapid, safe rollouts with interleaving and IPS/DR to estimate policy impact; automated rollback if guardrails fail.

- Persona 2: Luis Ortega — Product Analyst
  - Background: MSc Statistics; responsible for experiment analysis and dashboards.
  - Pain points: Manual metric definitions; inconsistent unit-of-analysis (user vs session) leading to confusion; SRM often unnoticed.
  - Goals: Metrics-as-code, robust SRM checks, CUPED for variance reduction, clear pre-registration and automated reports.

- Persona 3: Mei Chen — ML Platform Engineer
  - Background: Backend + ML infra; owns model serving, feature store, and registry.
  - Pain points: Complexity of routing traffic across model versions; ensuring feature parity; coordinating rollout with flags.
  - Goals: Reliable assignment API, feature flags, shadow/canary, integration with model registry and feature store, and auditability.

- Persona 4: Sofia Martinez — Privacy & Compliance Lead
  - Background: Legal/compliance; ensures privacy by design.
  - Pain points: Lack of visibility into data flows, retention, and consent.
  - Goals: Data minimization, encryption, consent-aware SDKs, audit trails, deletion workflows.

## 5. User Stories
- US-001: As an ML Engineer, I want deterministic bucketing so that users see consistent treatments across sessions.
  - Acceptance: Given a stable user_id, the same experiment returns the same variant 99.99% of the time across calls.

- US-002: As an Analyst, I want SRM detection so that I can stop invalid experiments early.
  - Acceptance: The platform flags SRM if observed allocation deviates beyond configured thresholds (p<0.01), alerts Slack/email, and pauses analysis.

- US-003: As an ML Engineer, I want to run canary and shadow deployments so that I can validate latency/cost/accuracy before full rollout.
  - Acceptance: Traffic slicing configurable (e.g., 1%, 5%, 10%); real-time health SLOs trigger rollback within 5 minutes of breach.

- US-004: As an Analyst, I want metrics-as-code so that metric definitions are versioned, tested, and reusable.
  - Acceptance: Metrics are defined in YAML/DSL, validated on CI, and used by both nearline and batch pipelines.

- US-005: As a Product Manager, I want automated experiment reports so that I can make decisions quickly.
  - Acceptance: Daily reports include power/MDE, primary/guardrail metrics, variance reduction, p-values/posteriors, and clear recommendations.

- US-006: As a Platform Engineer, I want identity resolution across devices so that exposures and outcomes are correctly attributed.
  - Acceptance: Identity graph merges anonymous and logged-in events with de-duplication and attribution windows.

- US-007: As a Data Scientist, I want counterfactual evaluation (IPS/DR) so that I can estimate model policy uplift before deployment.
  - Acceptance: Provide IPS/DR estimators with confidence intervals; flag high-variance cases; integrate with feature store covariates.

- US-008: As a Compliance Lead, I want audit trails and data minimization so that we meet regulatory obligations.
  - Acceptance: Every access and change is logged with user, timestamp, before/after; PII is minimized, tokenized, or hashed.

- US-009: As a Ranker Owner, I want interleaving tests so that I can detect ranking quality differences faster.
  - Acceptance: Balanced interleaving with click modeling; significance tests; UI to configure and analyze.

- US-010: As an Analyst, I want multiple testing control so that I can run many metrics without inflating false discoveries.
  - Acceptance: Benjamini–Hochberg FDR and family-wise error options; clearly annotated results.

## 6. Functional Requirements
### 6.1 Core Features
- FR-001: Experiment Assignment Service
  - Deterministic bucketing via consistent hashing.
  - Support mutually exclusive layers; targeted cohorts; exposure-at-event vs exposure-at-view triggers.
- FR-002: Feature Flags & Model Gating
  - Runtime toggles, canary/shadow, gradual rollout, per-request overrides, and kill switches.
- FR-003: SDKs (Web, Mobile, Server)
  - Capture exposure, eligibility, treatment, metadata; handle de-duplication, retries; respect consent.
- FR-004: Identity & Attribution
  - Cross-device identity resolution; de-duplication windows; attribution windows; opt-in/opt-out.
- FR-005: Metrics-as-Code
  - DSL to define metrics (ratios, retention, latency pXX); tiers (primary, secondary, guardrail); unit-of-analysis control.
- FR-006: Data Pipeline
  - Streaming ingestion (Kafka/Kinesis), exactly-once processing (Flink/Spark), late event handling, near-real-time aggregations, OLAP storage.
- FR-007: Statistical Engine
  - SRM detection, CUPED/CUPAC, sequential testing (alpha-spending), multiple testing control (Benjamini–Hochberg), frequentist and Bayesian options, cluster-robust SEs.
- FR-008: ML Evaluation Modules
  - Offline-to-online correlation, counterfactual IPS/DR, interleaving, uplift modeling, heterogeneous treatment effects, fairness guardrails.
- FR-009: Lifecycle Automation
  - Power/MDE calculators, pre-registration, auto-stopping criteria, anomaly detection, auto rollback, and report generation.
- FR-010: Governance & Safety
  - Exclusion groups, long-term holdouts, privacy controls, differential privacy (optional), compliance logging, audit trails.
- FR-011: ML Infra Integration
  - Model registry/versioning, feature store snapshots for covariates, real-time feature parity checks, shadow mirroring, latency/cost monitoring.

### 6.2 Advanced Features
- FR-012: Multi-armed Bandits (epsilon-greedy, Thompson sampling) for exploration under guardrails.
- FR-013: Collision Detection across experiments and layers.
- FR-014: Auto-metric lineage: track dependencies between metrics and upstream features.
- FR-015: Backfill and replay pipeline for schema evolution or missed data.
- FR-016: Fairness analysis: demographic parity, equalized odds; configurable sensitive attributes with privacy protections.

## 7. Non-Functional Requirements
### 7.1 Performance
- Assignment API p95 latency < 200 ms (in-region), p99 < 500 ms.
- Nearline metric freshness: 2–5 minutes end-to-end.
- SDK overhead on page load < 50 ms; mobile CPU < 2% during event capture bursts.

### 7.2 Reliability
- 99.5% monthly uptime SLO; 99.9% for assignment API.
- Exactly-once semantics for streaming with end-to-end idempotency.
- Data loss < 0.01% per month.

### 7.3 Usability
- Self-serve UI for experiment creation in < 5 minutes.
- Clear warnings for SRM, low power, collisions, and privacy risks.

### 7.4 Maintainability
- Modular codebase, typed APIs, strong CI/CD with 85%+ unit test coverage on core modules.
- Backward-compatible API changes with versioning.

## 8. Technical Requirements
### 8.1 Technical Stack
- Languages: Python 3.11+, TypeScript 5+, SQL (ANSI), Scala 2.13+ (optional for Flink jobs).
- Backend: FastAPI 0.115+, gRPC 1.63+ (internal services), Node.js 20+ for UI backend.
- Frontend: React 18+, Next.js 14+, Material UI 6+.
- SDKs: JavaScript (ES2022), Python, Kotlin/Android, Swift/iOS, Java.
- Streaming: Apache Kafka 3.6+ or AWS Kinesis; Schema Registry (Confluent 7+).
- Stream Processing: Apache Flink 1.18+ or Spark Structured Streaming 3.5+.
- Storage:
  - Metadata: PostgreSQL 15+.
  - Events/OLAP: ClickHouse 24.8+ or Apache Druid 27+; Data Lake: S3/GCS with Parquet.
  - Query: Trino/Presto 435+.
- Feature Store: Feast 0.34+ (or equivalent).
- Model Registry: MLflow 2.14+.
- Orchestration: Airflow 2.9+ or Dagster 1.7+.
- Infra: Kubernetes 1.30+, Helm 3.14+, Terraform 1.9+.
- Observability: OpenTelemetry 1.27+, Prometheus 2.54+, Grafana 11+.
- Auth: OAuth2/OpenID Connect, Keycloak 23+ or Auth0.
- Secrets: HashiCorp Vault 1.15+.

### 8.2 AI/ML Components
- Statistical engine: SciPy 1.13+, StatsModels 0.14+, scikit-learn 1.5+, PyMC 5+; R integration optional via rpy2.
- IPS/DR and uplift: EconML 0.15+ or DoWhy 0.12+.
- Interleaving: custom module with balanced and team-draft variants.
- Fairness: AIF360 0.6+ (optional), Fairlearn 0.10+.

## 9. System Architecture
### 9.1 High-Level Architecture (ASCII)
[Clients SDKs] --HTTP/gRPC--> [Assignment Service + Feature Flags] --exposures--> [Event Ingestion (Kafka)]
                               |                                         \
                               |                                          -> [Outcome Events (Kafka)]
                               v
                    [Model Serving + Registry + Feature Store]
                               |
                               v
                 [Stream Processing (Flink/Spark)]
                               |
                               v
        [Nearline Aggregations (ClickHouse/Druid) + OLAP (Trino)]
                               |
                               v
            [Stat Engine + Metrics Service + Reports Generator]
                               |
                               v
                        [UI (React) + APIs (FastAPI)]

Auxiliary:
- [Identity Service] <-> [Assignment Service], [Ingestion]
- [Governance/Audit] <-> [All Services]
- [Monitoring/Alerting] <-> [All Services]

### 9.2 Component Details
- Assignment Service: Stateless compute with Redis cache; consistent hashing ensures deterministic buckets; supports mutual exclusion layers; per-request context.
- Feature Flags: Traffic policies with canary/shadow, overrides, and kill switches; integrates with model endpoints.
- Identity Service: Anonymous ID stitching to user ID; handles opt-in/out and consent.
- Ingestion: SDKs send exposures and outcomes; exactly-once via idempotent keys.
- Stream Processing: Deduplication, late event handling, windowed aggregations, CUPED covariate join from feature store snapshots.
- Metrics Service: Metrics-as-code interpreter, lineage tracking, freshness checks.
- Stat Engine: Frequentist and Bayesian inference, SRM, sequential testing, FDR control, cluster-robust SEs.
- Reports: Automated narratives, power/MDE, guardrail health, recommendations.
- UI: Experiment config, feature flag management, live metrics, reports, governance console.

### 9.3 Data Flow
1) Client requests assignment -> Assignment Service computes variant -> returns variant + metadata.
2) Client logs exposure and outcomes -> Event Ingestion -> Kafka topics partitioned by experiment and user.
3) Stream Processing consumes events -> dedup -> attribution -> joins covariates -> computes nearline aggregates -> writes to ClickHouse.
4) Metrics Service exposes OLAP queries -> Stat Engine runs inference -> UI renders dashboards.
5) Model gating decisions (canary/shadow) call Feature Flags; anomalies trigger rollback via orchestration.

## 10. Data Model
### 10.1 Entity Relationships
- Experiment (1) — (n) Variant
- Experiment (1) — (n) Assignment (by user/session)
- UserIdentity (1) — (n) Assignment, (n) Exposure, (n) OutcomeEvent
- Experiment (1) — (n) Metrics (via MetricDefinition)
- FeatureFlag (1) — (n) RolloutRule
- Experiment (1) — (1) Report (latest), (n) Report (history)
- ModelVersion (1) — (n) Experiment (optional)
- Cohort/Layer define eligibility and exclusivity constraints.

### 10.2 Database Schema (key tables)
- experiments
  - id (uuid PK), key (text unique), name (text), owner (text), layer_id (uuid), status (enum: draft, running, paused, completed), start_ts (timestamptz), end_ts (timestamptz), prereg_doc (jsonb), power_plan (jsonb), created_at, updated_at
- variants
  - id (uuid PK), experiment_id (uuid FK), name (text), allocation (float), model_version_id (uuid FK nullable), flag_payload (jsonb)
- layers
  - id (uuid PK), key (text unique), mutually_exclusive (bool), description (text)
- feature_flags
  - id (uuid PK), key (text unique), status (enum), default_variant (text), kill_switch (bool), created_at, updated_at
- rollout_rules
  - id (uuid PK), flag_id (uuid FK), type (enum: canary, shadow, gradual), percentage (float), conditions (jsonb), start_ts, end_ts
- user_identities
  - user_id (text PK), anon_ids (jsonb), attributes (jsonb), consent_state (jsonb), updated_at
- assignments (OLTP)
  - assignment_id (uuid PK), experiment_id (uuid FK), user_id (text), variant_id (uuid FK), assigned_at (timestamptz), exposure_trigger (enum), hash_key (text), request_context (jsonb)
- exposures (event/OLAP)
  - event_id (uuid PK), assignment_id (uuid), user_id (text), experiment_id (uuid), variant_id (uuid), ts (timestamptz), device_id (text), session_id (text), metadata (jsonb)
- outcome_events (event/OLAP)
  - event_id (uuid PK), user_id (text), session_id (text), ts (timestamptz), event_name (text), value (float), currency (text), metadata (jsonb)
- metric_definitions
  - id (uuid PK), key (text unique), dsl (jsonb), tier (enum: primary, secondary, guardrail), unit (enum: user, session, request), owner (text), version (int), created_at
- aggregates (OLAP)
  - grain (text: experiment-variant-unit-day), keys (jsonb), metrics (jsonb), ts (timestamptz)
- reports
  - id (uuid PK), experiment_id (uuid), content (jsonb), generated_at (timestamptz), author (text)
- audits
  - id (uuid PK), actor (text), entity (text), entity_id (uuid), action (text), prev (jsonb), next (jsonb), ts (timestamptz), ip (inet)

### 10.3 Data Flow Diagrams (ASCII)
[SDK] -> [Ingestion API] -> [Kafka topic: exposures]
[SDK] -> [Ingestion API] -> [Kafka topic: outcome_events]
[Kafka] -> [Flink: dedup + join covariates] -> [ClickHouse aggregates]
[ClickHouse] -> [Stat Engine] -> [Reports/UI]

### 10.4 Input Data & Dataset Requirements
- Identity fields: user_id (hashed or tokenized), device_id, session_id.
- Context: region, platform, app version, cohort attributes.
- Outcomes: conversions, revenue, retention, latency percentiles, errors, model scores.
- Covariates: pre-experiment behavior and features from feature store snapshots for CUPED/CUPAC.

## 11. API Specifications
### 11.1 REST Endpoints
- POST /v1/assignments
  - Body: { experimentKey, userId, sessionId, context, trigger: "view"|"event" }
  - Response: { experimentId, variant, reasons, hashKey, exposureToken }
- POST /v1/exposures
  - Body: { exposureToken, userId, ts, metadata }
  - Response: { status: "ok" }
- POST /v1/events/outcome
  - Body: { userId, sessionId, eventName, value, ts, metadata }
  - Response: { status: "ok" }
- POST /v1/flags/evaluate
  - Body: { flagKey, userId, context }
  - Response: { variant, payload, rule, killSwitch }
- POST /v1/experiments
  - Body: experiment config (JSON/DSL); returns experimentId
- GET /v1/experiments/{id}/summary
  - Response: status, allocations, SRM status, key metrics, guardrails
- POST /v1/metrics
  - Body: { key, dsl, tier, unit }
- GET /v1/reports/{experimentId}
  - Response: JSON with metrics, inference, recommendations
- POST /v1/overrides
  - Body: { userId|sessionId, experimentKey|flagKey, variant, ttl }

### 11.2 Request/Response Examples
Request:
POST /v1/assignments
{
  "experimentKey": "ranker_v4_vs_v5",
  "userId": "u_2f01c1",
  "sessionId": "s_998",
  "context": {"platform":"web","country":"US"},
  "trigger": "view"
}

Response:
{
  "experimentId": "e9b5c2c7-...-42",
  "variant": "v5",
  "reasons": ["eligible","layer:rankers","hash:0.73<0.75 allocation"],
  "hashKey": "ranker_v4_vs_v5:u_2f01c1",
  "exposureToken": "exp_tok_ae91..."
}

Metrics DSL example (YAML):
metric:
  key: "ctr_clicks_per_impression"
  unit: "session"
  tier: "primary"
  definition:
    numerator:
      event: "click"
      filter: {"position":"<=5"}
    denominator:
      event: "impression"
    aggregation: "ratio"
    window: {"type":"session"}

### 11.3 Authentication
- OAuth2/OIDC with JWT bearer tokens for user-facing APIs.
- Service-to-service: mTLS + JWT service accounts.
- Scopes: experiments:read/write, metrics:read/write, ingestion:write, flags:evaluate, reports:read.
- Rate limiting and WAF at API gateway.

## 12. UI/UX Requirements
### 12.1 User Interface
- Experiment creation wizard: define key, layer, variants, allocations, eligibility, triggers, metrics, power/MDE.
- Feature flags console: define flags, rules, canary/shadow policies, overrides.
- Live monitoring: traffic allocations, SRM status, guardrail metrics, latency/cost dashboards.
- Reports: narratives with primary/secondary/guardrail results, variance reduction, multiple testing annotations, recommendations.
- Governance center: exclusion groups, holdouts, audit logs, consent configurations.

### 12.2 User Experience
- Contextual guidance for SRM, power analysis, and when to use bandits vs fixed-split.
- Safe defaults: equal allocation, guardrail set, CUPED enabled when covariates available.
- Undo/rollback buttons with clear blast radius.

### 12.3 Accessibility
- WCAG 2.1 AA compliance.
- Keyboard navigation, ARIA labels, high-contrast themes.

## 13. Security Requirements
### 13.1 Authentication
- OIDC-compliant SSO; MFA for admin roles.
- API keys for ingestion endpoints with rotation and scoping.

### 13.2 Authorization
- RBAC: Admin, Analyst, ML Engineer, Platform Engineer, Read-only, Compliance.
- Resource scoping by team/project.

### 13.3 Data Protection
- Encryption in transit (TLS 1.2+) and at rest (KMS-managed).
- Pseudonymization of user identifiers; salted hashing for long-term storage.
- Data minimization and purpose limitation; consent propagation from SDKs.
- Optional differential privacy for sensitive metric reporting.

### 13.4 Compliance
- Audit logging for all configuration changes and data access.
- Data retention policies with deletion workflows.
- DPIA templates and consent management integration.

## 14. Performance Requirements
### 14.1 Response Times
- Assignment and flag evaluation p95 < 200 ms; p99 < 500 ms.
- Report generation under 60 seconds for 10M event datasets (cached aggregates).

### 14.2 Throughput
- Ingestion: 50K events/sec sustained; burst 200K/sec.
- Stream processors scale to 30 partitions per topic minimum.

### 14.3 Resource Usage
- SDK memory < 200 KB web, < 500 KB mobile.
- Stream jobs CPU utilization target 60–70% under peak.

## 15. Scalability Requirements
### 15.1 Horizontal Scaling
- Stateless assignment pods with HPA on QPS/latency.
- Kafka partitions scaled based on throughput; Flink parallelism auto-adjusted.
- ClickHouse sharded/replicated clusters; Trino worker autoscaling.

### 15.2 Vertical Scaling
- Increase memory/CPU for stat engine and OLAP queries during large analyses.

### 15.3 Load Handling
- Backpressure in ingestion; queue depth monitoring; prioritized guardrail metrics computation.

## 16. Testing Strategy
### 16.1 Unit Testing
- 85%+ coverage for assignment hashing, SDK logic, metrics DSL parser, stat functions.

### 16.2 Integration Testing
- End-to-end tests: assignment -> exposure -> ingestion -> aggregation -> inference.
- Contract tests for SDKs and APIs; schema evolution tests.

### 16.3 Performance Testing
- Load testing for ingestion and assignment with 200K events/sec; latency SLO validation.

### 16.4 Security Testing
- Static analysis (SAST), dependency scans, DAST on public endpoints.
- Pen testing pre-GA; secrets scanning in CI.

## 17. Deployment Strategy
### 17.1 Deployment Pipeline
- GitHub Actions: lint, test, build; Docker images; SBOM.
- Argo CD for GitOps to Kubernetes environments.
- Blue/green for APIs; canary for stateful services.

### 17.2 Environments
- Dev, Staging, Prod with isolated Kafka clusters and databases.
- Sandbox workspace for training and offline evaluation.

### 17.3 Rollout Plan
- Internal alpha (4 weeks), beta to two product teams (8 weeks), GA (after SLO and scale tests).

### 17.4 Rollback Procedures
- One-click rollback of service deployments.
- Feature flag kill switches and experiment pause/resume.
- Automated rollback if guardrail SLO breaches persist > 5 minutes.

## 18. Monitoring & Observability
### 18.1 Metrics
- Application: QPS, latency p50/p95/p99, error rates, cache hit rates.
- Data: ingestion throughput, consumer lag, dedup rate, late events rate.
- Experiment: SRM alerts, CUPED usage, variance reduction achieved.
- Business: conversion, revenue per user, retention.

### 18.2 Logging
- Structured JSON logs; correlation IDs; OpenTelemetry traces.

### 18.3 Alerting
- PagerDuty/Slack for SLO breaches; anomaly detection on metric drifts.

### 18.4 Dashboards
- Grafana: assignment performance, pipeline health, OLAP query performance, experiment status.

## 19. Risk Assessment
### 19.1 Technical Risks
- High variance in IPS/DR estimates -> mitigated via clipping, self-normalized IPS, DR with strong models.
- Late/duplicate events -> mitigated via watermarking, idempotent keys.
- Misconfigured eligibility causing collisions -> mitigated via collision detection and dry-run checks.

### 19.2 Business Risks
- Misinterpretation of results -> mitigated via pre-registration and education.
- Privacy incidents -> mitigated via minimization, encryption, consent, audits.

### 19.3 Mitigation Strategies
- Guardrails and auto-rollback; sequential testing; strong defaults; change management approvals for high-risk flags.

## 20. Timeline & Milestones
### 20.1 Phase breakdown
- Phase 0 (2 weeks): Discovery, design, PRD sign-off.
- Phase 1 (6 weeks): Assignment service, SDK MVPs (JS/Python), exposure logging, feature flags.
- Phase 2 (6 weeks): Streaming pipeline (Kafka+Flink), metrics-as-code, OLAP store, basic dashboards.
- Phase 3 (6 weeks): Statistical engine (SRM, CUPED, sequential, BH), reports, governance/audit.
- Phase 4 (4 weeks): ML modules (IPS/DR, interleaving, uplift, fairness), model registry/feature store integration.
- Phase 5 (4 weeks): Hardening, performance, security review, GA.

Total: 28 weeks.

### 20.2 Key Milestones
- M1: Assignment API p95 < 200 ms (Week 6)
- M2: Nearline metrics freshness < 5 min (Week 12)
- M3: CUPED variance reduction demonstrated > 20% on pilot (Week 18)
- M4: Auto-rollback live (Week 22)
- M5: GA readiness SLOs met (Week 28)

Estimated team: 7–9 FTE (2 BE, 2 Data Eng, 1–2 ML Eng, 1 FE, 1 SRE/DevOps, 1 PM).
Rough cloud costs at GA scale: $25k–$60k/month depending on throughput and OLAP size.

## 21. Success Metrics & KPIs
### 21.1 Measurable targets
- Platform uptime: ≥99.5%; Assignment API ≥99.9%.
- Assignment latency: p95 ≤200 ms; p99 ≤500 ms.
- Metrics freshness: ≤5 minutes for 95% of aggregates.
- SRM detection coverage: 100% experiments.
- Variance reduction: median ≥15% with CUPED/CUPAC enabled.
- Experiment decision lead time: ≤24 hours from start to first actionable report.
- Rollback automation: ≥90% of guardrail breaches auto-rolled back within 5 minutes.

## 22. Appendices & Glossary
### 22.1 Technical Background
- Consistent hashing with stable salts per layer ensures deterministic assignments and stable splits across deployments.
- CUPED uses pre-experiment covariates to reduce variance; CUPAC extends to aggregated covariates from feature stores.
- Sequential testing with alpha-spending (e.g., Pocock or O’Brien–Fleming) limits Type I error under peeking.
- IPS/DR provide counterfactual estimates for policy changes using logged propensities; self-normalized and clipped variants reduce variance.
- Interleaving increases sensitivity for ranking model comparisons by mixing results and observing user preferences.

### 22.2 References
- Deng et al., “Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data.”
- Russo, Van Roy, “An Information-Theoretic Analysis of Thompson Sampling.”
- Imbens, Rubin, “Causal Inference in Statistics, Social, and Biomedical Sciences.”
- Kohavi et al., “Trustworthy Online Controlled Experiments.”
- Schnabel et al., “Unbiased Learning-to-Rank with Biased Feedback.”
- Benjamini, Hochberg, “Controlling the False Discovery Rate.”

### 22.3 Glossary
- A/B/n testing: Controlled randomization across multiple variants.
- Assignment: The process of placing a unit (user/session) into a variant.
- CUPED/CUPAC: Variance reduction using covariates.
- Exposure: The point at which a unit is considered treated for intent-to-test analysis.
- Guardrail metric: Metric that must not regress beyond thresholds.
- Interleaving: Method to compare ranking models by mixing results.
- IPS/DR: Inverse Propensity Scoring / Doubly Robust estimators for counterfactual evaluation.
- Layer: A group of mutually exclusive experiments.
- MDE: Minimum Detectable Effect for power planning.
- SRM: Sample Ratio Mismatch indicating assignment or logging issues.

Repository Structure
- repo/
  - README.md
  - notebooks/
    - exploration/
    - offline_eval_ips_dr.ipynb
    - interleaving_sensitivity.ipynb
  - src/
    - api/
      - main.py
      - routers/
        - assignments.py
        - exposures.py
        - flags.py
        - metrics.py
        - reports.py
    - assignment/
      - hasher.py
      - layers.py
      - eligibility.py
      - overrides.py
    - sdk/
      - js/
      - python/
      - ios/
      - android/
    - ingestion/
      - producers/
      - consumers/
    - pipelines/
      - flink_jobs/
      - batch/
    - stats/
      - srm.py
      - cuped.py
      - sequential.py
      - fdr.py
      - bayesian.py
      - ips_dr.py
      - interleaving.py
      - uplift.py
      - fairness.py
    - ui/
      - web/
  - configs/
    - application.yaml
    - metrics/
      - ctr.yaml
      - revenue_per_user.yaml
  - tests/
    - unit/
    - integration/
    - load/
  - infra/
    - k8s/
    - helm/
    - terraform/
  - data/
    - schemas/
    - samples/
  - scripts/

Config Samples
application.yaml:
server:
  port: 8080
  cors: ["https://app.example.com"]
assignment:
  salt: "global_salt_v1"
  cache_ttl_seconds: 300
kafka:
  brokers: "kafka-1:9092,kafka-2:9092"
  topics:
    exposures: "exp_exposures_v1"
    outcomes: "exp_outcomes_v1"
olap:
  clickhouse:
    url: "http://clickhouse:8123"
auth:
  oidc_issuer: "https://auth.example.com"
  audience: "exp-platform"

API Snippet (FastAPI)
from fastapi import FastAPI, Depends
from pydantic import BaseModel

app = FastAPI()

class AssignmentRequest(BaseModel):
    experimentKey: str
    userId: str
    sessionId: str | None = None
    context: dict | None = None
    trigger: str = "view"

@app.post("/v1/assignments")
def assign(req: AssignmentRequest):
    variant, hash_key, reasons = compute_assignment(req)
    token = issue_exposure_token(req.userId, req.experimentKey, variant)
    return {"experimentId": lookup_id(req.experimentKey),
            "variant": variant,
            "reasons": reasons,
            "hashKey": hash_key,
            "exposureToken": token}

Specific Metrics Targets
- Assignment API: p95 < 200 ms; p99 < 500 ms.
- Metrics pipeline freshness: < 5 minutes.
- Uptime: 99.5% platform; 99.9% assignment API.
- Data loss: < 0.01%/month.
- Variance reduction: median ≥ 15% when covariates available.

End of PRD.