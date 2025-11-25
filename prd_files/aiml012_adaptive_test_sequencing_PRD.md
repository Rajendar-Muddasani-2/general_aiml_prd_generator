# Product Requirements Document (PRD)
# `Aiml012_Adaptive_Test_Sequencing`

Project ID: aiml012  
Category: AI/ML – Adaptive Assessment and Sequencing  
Status: Draft for Review  
Version: 1.0.0  
Last Updated: 2025-11-25

## 1. Overview
### 1.1 Executive Summary
Aiml012_Adaptive_Test_Sequencing is an AI-driven engine that dynamically selects the next-best question for a learner during an assessment or practice session. It combines student state modeling (IRT, knowledge tracing) with sequential decision algorithms (contextual bandits, reinforcement learning) and uncertainty-aware stopping rules to reduce test length while maintaining or improving measurement accuracy, engagement, and fairness. The product exposes real-time APIs for adaptive sequencing, admin tooling for item bank and policy configuration, and analytics dashboards for monitoring and evaluation. It is deployable as a cloud-native microservice.

### 1.2 Document Purpose
This PRD defines the product vision, functional and non-functional requirements, technical architecture, data model, APIs, UI/UX, security, performance, deployment, testing, monitoring, risks, and timelines for delivering the aiml012 adaptive test sequencing system.

### 1.3 Product Vision
Deliver accurate, fair, and explainable adaptive assessments that personalize question sequences in real time, minimize time-to-diagnosis, and support diverse learning contexts (K-12, higher education, professional certification, corporate training) through a robust ML platform.

## 2. Problem Statement
### 2.1 Current Challenges
- Fixed-form assessments are long, not personalized, and fail to maintain engagement.
- Existing adaptive engines often lack transparency, fairness auditing, or support for multi-objective goals (precision vs. length vs. coverage).
- Cold start for new learners and new items results in poor early recommendations.
- Models drift over time due to shifting populations or content updates.
- Limited tooling for A/B testing, off-policy evaluation, and simulation before live rollout.

### 2.2 Impact Analysis
- Excessive test length increases operational costs and lowers learner satisfaction.
- Inaccurate or biased assessments can harm outcomes and trust.
- Slow inference hurts UX and completion rates.
- Inability to audit and explain sequencing decisions impedes regulatory and institutional adoption.

### 2.3 Opportunity
- Use modern student state modeling (IRT, BKT/SAKT/AKT) and decision policies (UCB, Thompson Sampling, EIG) to optimize next-question selection.
- Introduce principled uncertainty quantification and stopping rules to shorten tests.
- Implement fairness, exposure control, and blueprint constraints to ensure coverage and equity.
- Provide visibility with “why this question next” explanations, dashboards, and thorough offline/online evaluation pipelines.

## 3. Goals and Objectives
### 3.1 Primary Goals
- Accurate estimation of learner ability/skill mastery with fewer questions.
- Real-time next-item recommendations with low latency.
- Configurable multi-objective policies balancing precision, length, engagement, and content coverage.
- Built-in fairness auditing and interpretable decision traces.

### 3.2 Business Objectives
- Reduce average assessment length by ≥40% while maintaining or improving measurement precision.
- Increase completion rates by ≥15% through improved engagement.
- Offer enterprise-grade reliability (99.5%+ uptime) and governance features to win institutional contracts.
- Provide A/B testing and analytics to demonstrate ROI to customers.

### 3.3 Success Metrics
- Ability estimation RMSE vs. ground-truth benchmarks: ≤0.35 on standardized scale or ≥0.90 correlation.
- Average test length reduction: ≥40%.
- Inference latency (p95): ≤300 ms for next-item selection; p99 ≤500 ms.
- Calibration error (ECE): ≤0.05.
- Fairness parity metrics (e.g., delta in RMSE across protected attributes): ≤5% relative difference.
- Uptime: ≥99.5%.
- Model/Policy explainability coverage: ≥95% of recommendations with human-readable rationale.

## 4. Target Users/Audience
### 4.1 Primary Users
- Assessment product managers and administrators
- Learning platform engineers integrating adaptive assessments
- Educators/instructors monitoring learner progress
- Learners taking adaptive assessments

### 4.2 Secondary Users
- Data scientists/ML engineers tuning models and policies
- Psychometricians responsible for content calibration
- Compliance and ethics officers auditing fairness and privacy
- Customer success/ops teams monitoring SLAs

### 4.3 User Personas
1) Persona: Maya Chen – Assessment Admin  
- Background: Director of Assessment at an EdTech company; 10 years in psychometrics and product operations.  
- Goals: Configure item blueprints, set difficulty distributions, enforce exposure limits, view reliability dashboards.  
- Pain Points: Hard to translate psychometric constraints into system configurations; needs clear audits and compliance reports; limited bandwidth.  
- Success: Streamlined configuration UI, policy templates, one-click audit reports, minimal maintenance.

2) Persona: Luis Ortega – ML Engineer  
- Background: ML engineer at a learning platform; strong Python, PyTorch, and MLOps experience.  
- Goals: Integrate real-time next-item API, manage model versions, run offline simulations and A/B tests, ensure p95 latency targets.  
- Pain Points: Unclear APIs, fragile data pipelines, difficulty reproducing experiments and debugging live issues.  
- Success: Clear SDKs, MLflow tracking, reproducible pipelines, robust observability and canary rollouts.

3) Persona: Dr. Amina Rahman – University Instructor  
- Background: Teaches statistics and data science; manages weekly quizzes and semester exams.  
- Goals: Short, accurate assessments that keep students engaged and provide diagnostic insights by skill.  
- Pain Points: Students complain about test length; wants fairness and transparent rationale of sequencing; needs LMS integration.  
- Success: Improved completion rates, reliable mastery reports, simple LMS plugin.

4) Persona: Jordan Lee – Learner  
- Background: Working professional pursuing certification; time-constrained, studies on mobile.  
- Goals: Quickly assess readiness, avoid unnecessary questions, receive fair and relevant items.  
- Pain Points: Long tests, confusing difficulty jumps, lack of clarity on progress.  
- Success: Short, adaptive flows with progress indicators and helpful explanations.

## 5. User Stories
- US-001: As an Admin, I want to create item blueprints with skill coverage and difficulty bands so that the adaptive engine respects content constraints.  
  Acceptance: Can create/edit blueprint; validation for coverage; stored version; used by engine in live sessions.

- US-002: As an Admin, I want to set exposure control limits per item so that no single item appears too frequently.  
  Acceptance: Exposure caps configurable; engine respects caps with measurable adherence >99%.

- US-003: As an Engineer, I want a real-time API to request the next item so that I can serve adaptive sequences in my app.  
  Acceptance: POST /v1/next-item returns item within 300 ms p95; includes rationale and constraints satisfied.

- US-004: As a Learner, I want the test to stop once sufficient confidence is reached so that I don’t answer unnecessary questions.  
  Acceptance: Stopping triggers when posterior variance or entropy thresholds met; UX shows reason for completion.

- US-005: As a Data Scientist, I want to train IRT and knowledge tracing models and compare them offline so that I can select the best model.  
  Acceptance: Training pipelines with MLflow tracking; evaluation dashboards; model registry with promotion workflow.

- US-006: As an Admin, I want fairness reports across demographic groups so that I can ensure equitable outcomes.  
  Acceptance: Reports include parity in error, length, and exposure; alerts when thresholds exceeded.

- US-007: As an Engineer, I want to simulate policies with synthetic students so that I can estimate regret and test length before going live.  
  Acceptance: Simulator supports parameterized student models; reports regret, length, accuracy; runs at scale.

- US-008: As an Instructor, I want skill-level mastery reports so that I can remediate specific areas.  
  Acceptance: Reports include mastery probabilities with confidence; export to CSV/PDF.

- US-009: As an Admin, I want to configure stopping rules and multi-objective weights so that I can tailor the assessment.  
  Acceptance: UI to set thresholds and weights; validation; audit-trail and versioning.

- US-010: As an Engineer, I want robust authentication and RBAC so that sensitive data is protected.  
  Acceptance: OAuth2/JWT; roles for Admin/Instructor/Engineer/Learner; audit logs for actions.

- US-011: As a Product Manager, I want A/B and bandit experiments so that I can incrementally improve outcomes.  
  Acceptance: Experiment assignment API; significance reporting; automatic traffic reallocation for bandits.

- US-012: As a Learner, I want mobile-friendly UI and accessible components so that I can test anywhere.  
  Acceptance: WCAG 2.1 AA compliance; responsive design validated across devices.

## 6. Functional Requirements
### 6.1 Core Features
- FR-001: Item Bank Management: CRUD for items, skills, metadata, difficulty parameters, content tags.
- FR-002: Blueprint & Constraints: Define skill coverage targets, difficulty bands, exposure caps, and fairness constraints.
- FR-003: Student State Modeling: Support IRT (1PL/2PL/3PL), BKT/SAKT/AKT, deep-IRT hybrids; maintain per-learner posteriors.
- FR-004: Policy Engine: Contextual bandits (UCB, Thompson Sampling), expected information gain, Fisher information, BALD; constraint-aware selection.
- FR-005: Stopping Rules: Posterior variance/entropy thresholds, marginal value of information, PAC-style bounds.
- FR-006: Real-Time Sequencing API: Next-item selection within SLA; rationale/explanation included.
- FR-007: Response Ingestion: Record answers, hints, attempts, response times; incremental model updates.
- FR-008: Cold Start Handling: Hierarchical priors, meta-learning initialization; similarity-based warm starts.
- FR-009: Online Learning & Drift: Sliding windows, drift detection, scheduled recalibration.
- FR-010: Evaluation Suite: Offline simulation, IPS/doubly robust off-policy evaluation, A/B framework, regret analysis.
- FR-011: Explainability: “Why this question next” text with attention/feature attributions and constraints satisfied.
- FR-012: Reporting: Mastery reports, test summaries, calibration, fairness reports.
- FR-013: Admin UI: Configuration, policy editing, model selection, experiment setup, monitoring dashboards.
- FR-014: Audit & Versioning: Versioned policies, models, blueprints; immutable audit logs.

### 6.2 Advanced Features
- FR-015: Federated Learning option for privacy-preserving training.
- FR-016: Differential privacy noise addition for aggregated analytics.
- FR-017: Multi-language content support and multilingual NLP tagging.
- FR-018: Real-time exposure balancing via probabilistic throttling.
- FR-019: Skill graph modeling with prerequisite relations and curriculum-aware sequencing.
- FR-020: Cost-aware selection (e.g., time-on-task constraints).

## 7. Non-Functional Requirements
### 7.1 Performance
- Next-item API p95 ≤300 ms; p99 ≤500 ms.
- Throughput: ≥2,000 RPS sustained; burst to 5,000 RPS for peak events with auto-scaling.
- Training: daily batch retrains complete within 4 hours for 50M interaction records.

### 7.2 Reliability
- Uptime ≥99.5% monthly; zero data loss RPO, 15-min RTO.
- Multi-AZ deployment; blue/green or canary rollouts.

### 7.3 Usability
- Admin workflows ≤3 clicks to common tasks; onboarding <1 hour.
- WCAG 2.1 AA accessibility.

### 7.4 Maintainability
- Code coverage ≥80%; modular microservices; API semver and deprecation policy.
- Infrastructure as Code (Terraform/Helm); automated linting and type checks.

## 8. Technical Requirements
### 8.1 Technical Stack
- Backend: Python 3.11+, FastAPI 0.111+, Uvicorn 0.30+, SQLAlchemy 2.0+.
- ML: PyTorch 2.4+, PyTorch Lightning 2.4+, scikit-learn 1.5+, XGBoost 2.1+.
- NLP/CV (optional): Hugging Face Transformers 4.45+, spaCy 3.7+, OpenCV 4.10+.
- Data: PostgreSQL 15+, Redis 7+, Kafka 3.7+ (optional), S3-compatible object storage.
- Orchestration: Kubernetes 1.30+, Docker, Helm 3.14+, Ray 2.30+ for distributed training.
- Feature Store: Feast 0.44+ (optional).
- Experiment Tracking: MLflow 2.16+.
- Frontend: React 18+, TypeScript 5+, Vite 5+, Chakra UI or Material UI.
- Auth: OAuth2/OIDC (Auth0/Keycloak), JWT, OPA for policy.
- Observability: Prometheus, Grafana, OpenTelemetry 1.8+, Loki.

### 8.2 AI/ML Components
- IRT estimators (1PL/2PL/3PL) with Bayesian or MLE fitting; calibration routines.
- Knowledge tracing: BKT, LSTM-based DKT, Transformer-based SAKT/AKT.
- Deep-IRT hybrids: joint embedding of items/skills with discrimination parameters.
- Decision policies: UCB1/2, LinUCB, Thompson Sampling (Gaussian/Beta-Bernoulli), EIG/Fisher/BALD.
- Stopping: posterior variance/entropy thresholds and VOI-based rules.
- Cold start: hierarchical priors, meta-learning (MAML-style) for rapid adaptation.
- Drift: ADWIN or PSI-based detection; sliding window retraining.

## 9. System Architecture
### 9.1 High-Level Architecture (ASCII diagram)
+---------------------------+           +---------------------------+
|        Web/Mobile UI      | <-------> |      API Gateway          |
+------------+--------------+           +-------------+-------------+
             |                                        |
             v                                        v
+------------+--------------+           +-------------+-------------+
|    Sequencing Service     | <-------> |     Model & Policy        |
|  (Next-item, Stopping)    |           |      Service (ML)         |
+------------+--------------+           +-------------+-------------+
             |                                        |
             v                                        v
+------------+--------------+           +-------------+-------------+
|  Feature Store / Cache    | <-------> |  Database (Postgres)      |
|      (Redis/Feast)        |           |  + Object Storage (S3)     |
+------------+--------------+           +-------------+-------------+
             |                                        |
             v                                        v
+------------+--------------+           +-------------+-------------+
|  Event Bus (Kafka)        | --------> |  Analytics & Monitoring   |
|  (responses, metrics)     |           | (Prometheus/Grafana/MLflow)|
+---------------------------+           +---------------------------+

### 9.2 Component Details
- API Gateway: JWT auth, rate limiting; routes requests to services.
- Sequencing Service: Orchestrates policy selection, applies constraints and stopping; returns next item and rationale.
- Model & Policy Service: Hosts model endpoints (state update, inference); retrieves parameters from model registry; exposes policy scoring APIs.
- Feature Store/Cache: Low-latency retrieval of learner/item features and embeddings; Redis for hot features.
- Database: Stores items, skills, interactions, sessions, blueprints, policies, model metadata.
- Object Storage: Training datasets, model artifacts, logs.
- Event Bus: Streams interactions and decisions for analytics and model updates.
- Analytics & Monitoring: Metrics, logs, traces; MLflow for experiments and model registry.

### 9.3 Data Flow
1) UI requests next item -> API Gateway -> Sequencing Service.  
2) Sequencing fetches learner state from cache/DB; queries Model & Policy Service for candidate item scores.  
3) Constraints (blueprints, exposure, fairness) applied; select item; log rationale; return item to UI.  
4) Learner submits response -> API -> Sequencing -> Model updates state (online update) -> events emitted to Kafka.  
5) Batch jobs consume events for retraining and calibration; results stored in MLflow and model registry.  
6) Monitoring consumes metrics, triggers alerts if SLAs/fairness thresholds breached.

## 10. Data Model
### 10.1 Entity Relationships
- User (1..n) Session
- Session (1..n) Interaction
- Item (n..m) Skill (via ItemSkill)
- PolicyConfig (1..n) Session
- ModelVersion (1..n) Session
- Blueprint (1..n) Session
- ExposureLog per Item and time window

### 10.2 Database Schema (PostgreSQL)
- users: id (uuid), role (enum), demographics (jsonb, optional), created_at
- sessions: id (uuid), user_id (uuid), status (enum: active/completed), policy_config_id, model_version_id, blueprint_id, started_at, completed_at, final_estimate (jsonb)
- items: id (uuid), content_ref (text), difficulty (float, nullable), discrimination (float, nullable), guessing (float, nullable), metadata (jsonb), active (bool)
- skills: id (uuid), name (text), description (text), parent_skill_id (uuid, nullable)
- item_skills: item_id (uuid), skill_id (uuid), weight (float)
- interactions: id (uuid), session_id (uuid), user_id (uuid), item_id (uuid), response (jsonb), correct (bool), response_time_ms (int), hints_used (int), attempts (int), timestamp
- policies: id (uuid), name (text), type (enum: ucb, thompson, eig, custom), config (jsonb), version (int), active (bool), created_by, created_at
- blueprints: id (uuid), name (text), config (jsonb), version (int), active (bool)
- exposure_limits: item_id (uuid), period (interval), cap (int), rolling_count (int), updated_at
- model_versions: id (uuid), name (text), type (enum: irt, bkt, sakt, akt, hybrid), uri (text), metrics (jsonb), status (enum: staged, prod, archived), created_at
- explanations: id (uuid), session_id (uuid), item_id (uuid), policy_id (uuid), rationale (jsonb), timestamp
- audits: id (uuid), actor (uuid), action (text), entity_type (text), entity_id (uuid), before (jsonb), after (jsonb), timestamp
- experiments: id (uuid), name (text), variants (jsonb), traffic_split (jsonb), status (enum), start_at, end_at
- fairness_reports: id (uuid), report (jsonb), created_at

Indexes: users.id, sessions.user_id, interactions.session_id, interactions.user_id, item_skills.item_id, model_versions.status, exposure_limits.item_id, policies.active, blueprints.active.

### 10.3 Data Flow Diagrams
- Training Data Flow:
  Interactions -> ETL (feature extraction, anonymization) -> Object Storage -> Training Jobs (Ray) -> MLflow registry -> Model Versions -> Deployment.
- Inference Data Flow:
  UI -> Next-item API -> Cache/DB -> Policy/Model -> Selection -> Explanation -> Response -> Update state -> Stream to Kafka -> Metrics.

### 10.4 Input Data & Dataset Requirements
- Item data: text/content refs, skill tags, estimated parameters (difficulty/discrimination/guessing), optional media.
- Historical interactions: (user_id/session_id/timestamp/item_id/response/correct/response_time/attempts/hints_used).
- Optional demographics (for fairness auditing) with explicit consent.
- Volume: up to 200M interactions; items up to 100k; skills up to 5k.
- Data quality: missing values handling, content validity checks, uniqueness, leakage prevention.
- Privacy: PII hashed; consent flags maintained; opt-out respected.

## 11. API Specifications
### 11.1 REST Endpoints
- POST /v1/sessions
  Creates a new adaptive session for a user.
- POST /v1/next-item
  Returns next item ID and rationale for an active session.
- POST /v1/submit-response
  Submits a response to an item and updates the learner state.
- GET /v1/sessions/{session_id}/report
  Returns mastery/ability estimates and summary.
- POST /v1/admin/items
  Create or update items.
- GET /v1/admin/items/{item_id}
  Retrieve item metadata.
- POST /v1/admin/blueprints
  Create/update blueprint configuration.
- POST /v1/admin/policies
  Create/update policy configuration.
- GET /v1/models
  List model versions and metrics.
- POST /v1/experiments
  Create A/B or bandit experiment.
- GET /v1/audit
  Fetch audit logs (admin only).

### 11.2 Request/Response Examples
1) Create Session
Request:
{
  "user_id": "8b2e6a3d-1f2c-4f4e-a2f5-6a1b0f7c9a00",
  "policy_config_id": "b4e8...",
  "model_version_id": "m_irt_v12",
  "blueprint_id": "bp_2025_01",
  "metadata": {
    "exam_name": "Stats 101 Midterm",
    "time_limit_min": 45
  }
}
Response:
{
  "session_id": "s_12345",
  "status": "active",
  "started_at": "2025-11-25T10:15:00Z"
}

2) Next Item
Request:
{
  "session_id": "s_12345"
}
Response:
{
  "item_id": "item_9876",
  "constraints_satisfied": ["skill_coverage", "exposure_cap", "difficulty_band"],
  "rationale": {
    "policy": "thompson",
    "expected_information_gain": 0.42,
    "predicted_correct_prob": 0.61,
    "uncertainty": 0.28
  },
  "stopping_recommended": false,
  "latency_ms": 132
}

3) Submit Response
Request:
{
  "session_id": "s_12345",
  "item_id": "item_9876",
  "response": {"choice": "B", "text": null},
  "correct": true,
  "response_time_ms": 22000,
  "hints_used": 0,
  "attempts": 1
}
Response:
{
  "accepted": true,
  "updated_estimate": {"theta": 0.65, "variance": 0.18},
  "stopping_recommended": true,
  "stopping_reason": "posterior_variance_below_threshold"
}

4) Report
GET /v1/sessions/s_12345/report
Response:
{
  "session_id": "s_12345",
  "status": "completed",
  "ability_estimate": {"theta": 0.68, "ci95": [0.45, 0.91]},
  "skill_mastery": [
    {"skill_id": "algebra", "p_mastery": 0.84},
    {"skill_id": "probability", "p_mastery": 0.62}
  ],
  "length": 14,
  "calibration": {"ece": 0.03},
  "fairness_notes": [],
  "completed_at": "2025-11-25T10:48:21Z"
}

### 11.3 Authentication
- OAuth2/OIDC flows with JWT bearer tokens.
- Scopes: read:session, write:session, admin:items, admin:policies, admin:reports.
- Optional fine-grained authorization via OPA policies.

## 12. UI/UX Requirements
### 12.1 User Interface
- Learner UI: minimal, responsive; progress indicator; clear question display; accessible controls; rationale tooltip on demand.
- Admin UI: item bank manager, blueprint editor, policy builder (sliders for weights), experiments setup, monitoring dashboards, audit log viewer.
- Instructor UI: class and individual mastery reports, test summaries, export tools.

### 12.2 User Experience
- Clear guidance; “why this question next” optional explanation; consistent difficulty progression to avoid whiplash.
- Error handling with graceful retries; loading skeletons; no blocking spinners >1s without feedback.

### 12.3 Accessibility
- WCAG 2.1 AA: keyboard navigation, ARIA labels, color contrast, screen-reader support.
- Localization and right-to-left support for languages where applicable.

## 13. Security Requirements
### 13.1 Authentication
- OIDC integration; MFA optional for admins.
- Short-lived access tokens; refresh tokens with rotation.

### 13.2 Authorization
- RBAC with least privilege; admin, instructor, engineer, learner roles.
- OPA-based policy for advanced constraints (e.g., PII field restrictions).

### 13.3 Data Protection
- Encryption: TLS 1.3 in transit; AES-256 at rest.
- Secrets in vault (e.g., HashiCorp Vault).
- Data minimization and retention policies; PII segregation; anonymization for analytics.

### 13.4 Compliance
- GDPR/CCPA for data subject rights; FERPA for educational records where applicable.
- DPA templates; audit logs immutable; DPIA maintained.
- Optional differential privacy and federated learning modes for high-sensitivity deployments.

## 14. Performance Requirements
### 14.1 Response Times
- Next-item p95 ≤300 ms; p99 ≤500 ms.
- Submit-response p95 ≤250 ms.

### 14.2 Throughput
- Sustain ≥2,000 RPS; auto-scale to 5,000 RPS bursts within 60 seconds.

### 14.3 Resource Usage
- CPU target utilization 60–70%; GPU utilization >50% during training; Redis cache hit rate ≥95%.

## 15. Scalability Requirements
### 15.1 Horizontal Scaling
- Stateless API pods behind load balancer; HPA based on CPU/RPS/latency.
- Redis cluster with sharding; Kafka partitions scaled to throughput.

### 15.2 Vertical Scaling
- Model serving pods with tunable CPU/memory; optional GPUs for deep KT models.

### 15.3 Load Handling
- Circuit breakers; backpressure via queue; autoscaling policies; graceful degradation to simpler policies under extreme load.

## 16. Testing Strategy
### 16.1 Unit Testing
- 80%+ coverage; model unit tests for IRT and KT components; policy selection edge cases; config validation.

### 16.2 Integration Testing
- End-to-end tests: create session -> next-item -> submit -> stop -> report.
- Contract tests for APIs; schema validation.

### 16.3 Performance Testing
- Load and stress tests with synthetic traffic; latency SLO validation; cache warm/cold scenarios.

### 16.4 Security Testing
- Static analysis (Bandit), dependency scanning; DAST; pen tests; authz bypass attempts; audit log tamper-evidence checks.

## 17. Deployment Strategy
### 17.1 Deployment Pipeline
- GitHub Actions: lint, tests, build Docker, push to registry, Helm deploy.
- ML pipeline: MLflow for experiments; model registry; CI to validate metrics and push to staging.

### 17.2 Environments
- Dev -> Staging -> Prod; separate cloud accounts; separate databases; anonymized data in non-prod.

### 17.3 Rollout Plan
- Canary 5% -> 25% -> 100% with automated rollback on SLO breaches.
- A/B gating for new policies/models.

### 17.4 Rollback Procedures
- Helm rollback to prior release; feature flags to disable new policy; revert model version via registry; restore DB snapshot if required.

## 18. Monitoring & Observability
### 18.1 Metrics
- Latency p50/p95/p99 (next-item, submit-response).
- Error rates, timeouts, saturation.
- Test length distribution; ability RMSE (offline), calibration (ECE).
- Regret estimates vs. oracle in simulation; exposure cap violations.
- Fairness parity metrics; drift detection signals.
- Cache hit rate; DB query latency; queue lag.
- Uptime and SLO compliance.

### 18.2 Logging
- Structured JSON logs with correlation IDs; PII redaction.
- Decision logs with rationale and constraints matched.

### 18.3 Alerting
- Pager alerts on latency SLO breach >10 minutes, error rate >2%, uptime dips.
- Warnings for fairness metric drifts and exposure violations.

### 18.4 Dashboards
- Service health; policy effectiveness; experiment outcomes; fairness and calibration; resource utilization.

## 19. Risk Assessment
### 19.1 Technical Risks
- Model drift degrades accuracy.
- Cold start mis-sequencing early items.
- Overfitting to historic data; poor generalization.
- Latency spikes under peak load.
- Constraint conflicts causing infeasible selections.

### 19.2 Business Risks
- Regulatory/compliance concerns slow adoption.
- Insufficient item bank quality/coverage reduces performance.
- Stakeholder resistance due to low transparency.
- Data sparsity for niche subjects.

### 19.3 Mitigation Strategies
- Regular recalibration; drift detection and scheduled retraining.
- Hierarchical priors and meta-learning for cold start.
- Extensive simulation and A/B testing; conservative rollout.
- Capacity planning and cache optimization.
- Constraint solver fallback strategies; conflict detection UI.

## 20. Timeline & Milestones
### 20.1 Phase breakdown
- Phase 0 (2 weeks): Discovery, detailed spec, data audit.
- Phase 1 (6 weeks): Core services (APIs, DB), item bank, session flow, basic IRT, Thompson Sampling policy, stopping rules, basic UI.
- Phase 2 (8 weeks): Knowledge tracing (SAKT/AKT), EIG policy, constraints/blueprints, exposure control, explanations, analytics.
- Phase 3 (4 weeks): Evaluation suite (simulator, off-policy), A/B framework, fairness dashboard, monitoring/alerting.
- Phase 4 (4 weeks): Hardening, scalability, security, compliance, documentation; canary rollout.

Total: 24 weeks.

Estimated Costs (6 months):
- Team: PM (0.5 FTE), Tech Lead (1 FTE), 3 Backend/ML Eng (3 FTE), Frontend (1 FTE), DevOps (0.5 FTE), Data Scientist (1 FTE), QA (0.5 FTE). Approx. $1.2M loaded.  
- Cloud/Tools: $80k (compute, storage, observability, CI/CD, pen tests).

### 20.2 Key Milestones
- M1 (Week 4): Session and next-item MVP live in staging.
- M2 (Week 8): IRT + Thompson with stopping; p95 ≤400 ms in staging.
- M3 (Week 14): SAKT/AKT + EIG; constraints and exposure control; explanations v1.
- M4 (Week 18): Simulator and off-policy evaluator; A/B framework; fairness reports.
- M5 (Week 24): Production canary; SLOs met; documentation complete.

## 21. Success Metrics & KPIs
### 21.1 Measurable targets
- Accuracy: ≥0.90 correlation between estimated and benchmark abilities; ECE ≤0.05.
- Efficiency: ≥40% reduction in average test length; ≥15% increase in completion rate.
- Performance: p95 latency ≤300 ms; uptime ≥99.5%.
- Fairness: ≤5% relative difference in RMSE and length across audited groups.
- Reliability: <0.5% exposure cap violations; <0.1% API error rate.
- Adoption: ≥3 pilot customers; ≥10k monthly active test sessions by quarter 2 post-launch.

## 22. Appendices & Glossary
### 22.1 Technical Background
- IRT: Latent trait models mapping item responses to ability estimates; supports discrimination and guessing parameters.  
- Knowledge Tracing: Sequence models estimating skill mastery over time; SAKT/AKT leverage attention for context-aware predictions.  
- Decision Policies: Contextual bandits (UCB, Thompson) and information-theoretic selection (EIG, BALD) for balancing exploration/exploitation.  
- Stopping: Use posterior variance, entropy, or value-of-information to end tests when confidence meets thresholds.  
- Off-Policy Evaluation: IPS and doubly robust methods to estimate policy performance from logged data without full deployment.  
- Fairness: Evaluate parity of accuracy, length, and exposure across groups; apply constraints or reweighting.  
- Privacy: Differential privacy for aggregated analytics; federated learning to keep raw data local.

### 22.2 References
- Rasch, G. “Probabilistic Models for Some Intelligence and Attainment Tests.”
- Baker, F. “The Basics of Item Response Theory.”
- Piech et al. “Deep Knowledge Tracing.” NIPS 2015.
- Pandey and Karypis. “A Self Attentive model for Knowledge Tracing.” 2019.
- Gal et al. “Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.” ICML 2016.
- Thomas, P., and Brunskill, E. “Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning.” ICML 2016.
- Kuleshov and Liang. “Calibrated Structured Prediction.” NIPS 2015.

### 22.3 Glossary
- IRT: Item Response Theory; statistical models for latent ability estimation.
- BKT/SAKT/AKT: Knowledge tracing variants using Bayesian and attention-based deep learning.
- EIG: Expected Information Gain; selects items maximizing expected reduction in uncertainty.
- BALD: Bayesian Active Learning by Disagreement; picks items that maximize mutual information.
- Contextual Bandit: A bandit setting with context features guiding action selection.
- Thompson Sampling: Bayesian randomized action selection proportional to probability of optimality.
- UCB: Upper Confidence Bound; picks action maximizing mean + uncertainty bonus.
- Calibration (ECE): Expected Calibration Error; measures alignment between predicted probabilities and observed outcomes.
- Exposure Control: Mechanisms to prevent overuse of specific items.
- Blueprint: Content constraints specifying skill coverage, difficulty bands, and distributions.

Repository Structure
- /
  - README.md
  - notebooks/
    - 01_eda.ipynb
    - 02_irt_training.ipynb
    - 03_kt_models.ipynb
    - 04_policy_simulator.ipynb
  - src/
    - api/
      - main.py
      - routes/
        - sessions.py
        - items.py
        - policies.py
        - reports.py
    - services/
      - sequencing_service.py
      - model_service.py
      - stopping_rules.py
      - constraints.py
      - explainability.py
    - ml/
      - irt/
        - fit_2pl.py
        - bayes_irt.py
      - kt/
        - sakt_model.py
        - akt_model.py
      - policy/
        - thompson.py
        - ucb.py
        - eig.py
      - eval/
        - simulator.py
        - offpolicy.py
    - data/
      - schemas.py
      - feature_store.py
    - infra/
      - config.py
      - logging.py
  - tests/
    - unit/
    - integration/
    - performance/
  - configs/
    - policy_default.yaml
    - stopping_rules.yaml
    - blueprint_example.yaml
  - data/ (gitignored)
  - deployment/
    - helm/
    - terraform/
  - mlflow/
  - Makefile
  - requirements.txt
  - docker/
    - Dockerfile.api
    - Dockerfile.training

Config Samples (YAML)
configs/policy_default.yaml
policy:
  name: "thompson_default"
  type: "thompson"
  weights:
    objective:
      precision: 0.6
      length: 0.2
      engagement: 0.1
      coverage: 0.1
  constraints:
    exposure_cap: 1000
    difficulty_band:
      min: -1.0
      max: 1.5
    blueprint_id: "bp_2025_01"
  exploration:
    min_probability: 0.05

configs/stopping_rules.yaml
stopping:
  posterior_variance_threshold: 0.20
  entropy_threshold: 0.35
  min_items: 8
  max_items: 30
  voi_delta_threshold: 0.01

API Code Snippet (FastAPI)
from fastapi import FastAPI, Depends
from pydantic import BaseModel
import time

app = FastAPI()

class NextItemRequest(BaseModel):
    session_id: str

class NextItemResponse(BaseModel):
    item_id: str
    constraints_satisfied: list[str]
    rationale: dict
    stopping_recommended: bool
    latency_ms: int

@app.post("/v1/next-item", response_model=NextItemResponse)
def next_item(req: NextItemRequest):
    t0 = time.time()
    # load state, score candidates, apply constraints
    item_id = "item_9876"
    constraints = ["skill_coverage", "exposure_cap"]
    rationale = {"policy": "thompson", "expected_information_gain": 0.42}
    stopping = False
    latency = int((time.time() - t0) * 1000)
    return NextItemResponse(
        item_id=item_id,
        constraints_satisfied=constraints,
        rationale=rationale,
        stopping_recommended=stopping,
        latency_ms=latency,
    )

Policy Snippet (Thompson Sampling pseudo-Python)
def thompson_select(candidates, posterior, constraints, rng):
    feasible = [c for c in candidates if constraints.satisfy(c)]
    samples = []
    for item in feasible:
        # Sample ability and item params (Gaussian approx)
        theta = rng.normal(posterior.mean, posterior.std)
        # Predict correct prob via 2PL
        p = sigmoid(item.disc * (theta - item.diff))
        # Sample reward from Beta posterior or Bernoulli with p
        r = rng.binomial(1, p)
        # Estimate information gain proxy (optional)
        utility = 0.7 * r + 0.3 * info_gain_approx(theta, item)
        samples.append((utility, item))
    return max(samples, key=lambda x: x[0])[1]

ASCII Sequence Diagram (Next Item Request)
User -> API Gateway -> Sequencing Service -> Model Service -> Cache/DB
 |       POST /next-item    |               |          |
 |------------------------->|               |          |
 |                          |  get state    |          |
 |                          |-------------> |          |
 |                          |    state      |          |
 |                          | <-------------|          |
 |                          |  score items  |--------->|
 |                          |  constraints  |<---------|
 |  item + rationale        |               |          |
 |<-------------------------|               |          |

Performance Targets Summary
- Ability RMSE ≤0.35; ECE ≤0.05.
- Test length reduced ≥40%.
- p95 next-item latency ≤300 ms; p99 ≤500 ms.
- Uptime ≥99.5%.

End of PRD.