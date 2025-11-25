# Product Requirements Document (PRD) / # aiml025_model_versioning_and_registry

Project ID: aiml025_model_versioning_and_registry
Category: AI/ML Platform – Model Lifecycle, Versioning, and Governance
Status: Draft
Version: 0.9
Last Updated: 2025-11-25

1. Overview
1.1 Executive Summary
- Build a unified model versioning and registry platform that supports immutable model versions with mutable aliases, full lineage and reproducibility, schema/signature validation, lifecycle governance, and safe promotion/rollout strategies across training and inference environments.
- The platform provides APIs, SDK/CLI, and a web UI to register, discover, compare, promote, and deprecate models and their artifacts (weights, tokenizers, preprocessors, containers) with strong security, attestation, and observability.
- Outcome: Faster, safer model iteration and deployment with auditability and zero-downtime cutovers.

1.2 Document Purpose
- Define product scope, requirements, architecture, APIs, security, performance, deployment, and success criteria for the model versioning and registry service.
- Serve as a source of truth for engineering, product, security, and MLOps stakeholders.

1.3 Product Vision
- A first-class, cloud-native registry that treats models as versioned, reproducible, and governed assets—enabling enterprise-grade ML at scale. It aligns with modern practices in data/feature versioning, vector index cutovers, and CI/CD-driven promotion to reduce operational risk and accelerate time-to-value.

2. Problem Statement
2.1 Current Challenges
- Ad-hoc model storage without immutable versions; difficult rollback.
- Missing lineage ties between models, datasets, code commits, and environments.
- Inconsistent input/output schema leading to runtime failures.
- Manual promotion to production; limited guardrails for performance, bias, or drift.
- Fragmented artifact storage and metadata with poor discoverability.
- Lack of standardized governance, access controls, and audit logs.

2.2 Impact Analysis
- Increased incidents during deployment, downtime from incompatible changes.
- Inability to reproduce results impairs compliance and trust.
- Slower iteration due to unclear “champion” model and manual approvals.
- Operational costs rise from duplicated efforts and failed rollouts.

2.3 Opportunity
- Centralize model lifecycle with immutable versions and alias-based cutovers.
- Automate validation, policy checks, and staging/production promotion.
- Provide consistent APIs/SDKs to accelerate developer productivity.
- Enhance observability and governance for compliance and enterprise readiness.

3. Goals and Objectives
3.1 Primary Goals
- Provide registry with immutable model versions and human-friendly aliases (staging, production, champion).
- Ensure end-to-end lineage and reproducibility (datasets, code, environment, seeds).
- Enforce schema/signature validation and backward compatibility checks.
- Enable safe rollout strategies (A/B, canary, shadow) and automatic rollback on KPI regressions.

3.2 Business Objectives
- Reduce model deployment lead time by 50%.
- Cut production model incidents by 60%.
- Improve compliance posture with auditable model governance.
- Increase model reuse and discovery across teams by 40%.

3.3 Success Metrics
- >95% of production models registered with full lineage and signatures.
- <500 ms p95 latency for registry read operations.
- 99.5% monthly uptime for registry service.
- 75% of promotions executed through automated policy gates.
- Mean rollback time < 10 minutes.

4. Target Users/Audience
4.1 Primary Users
- ML Engineers / Data Scientists
- MLOps / Platform Engineers
- Model Reviewers / Risk & Compliance

4.2 Secondary Users
- Product Managers
- SRE/Operations
- Security/IT

4.3 User Personas
- Persona 1: Priya Sharma, Senior ML Engineer
  - Background: 7 years in NLP and recommendation systems; uses PyTorch and Transformers.
  - Pain Points: Difficulty tracing which dataset/code produced a model; manual schema mismatches; slow rollbacks.
  - Goals: Fast, safe deploys; reproducible experiments; API-first registry with clear metrics and approvals.
- Persona 2: Alex Kim, MLOps Platform Engineer
  - Background: Kubernetes, CI/CD, cloud infrastructure; owns model serving and observability.
  - Pain Points: Lack of consistent artifact packaging/signing; fragile manual promotions; no unified audit trail.
  - Goals: Immutable versions, Sigstore attestation, role-based policies, GitOps-friendly manifests.
- Persona 3: Maria Gonzalez, Responsible AI Reviewer
  - Background: Data ethics, fairness metrics, governance frameworks.
  - Pain Points: Missing documentation/model cards, weak bias checks, no controlled approval workflows.
  - Goals: Enforce policy gates before promotion; maintain audit logs; ensure transparency and fairness threshold compliance.
- Persona 4: Jordan Lee, Product Manager
  - Background: Owns KPIs for feature performance and cost.
  - Pain Points: Unclear champion/challenger status; risky cutovers; unclear cost/latency trade-offs.
  - Goals: A/B and canary visibility; rollbacks on KPI regression; transparent metrics and cost reporting.

5. User Stories
- US-001 Register a Model
  - As a ML Engineer, I want to register a new model version with artifacts and metadata so that I can track it immutably and reproduce results.
  - Acceptance: POST /models/{name}/versions returns version_id with content-addressed checksum; artifacts stored; lineage links saved (dataset version, git SHA, hyperparams).
- US-002 Define Aliases
  - As a MLOps Engineer, I want to assign aliases (staging, production) to immutable versions so that I can switch traffic without downtime.
  - Acceptance: PUT /models/{name}/aliases/{alias} updates target atomically; audit log recorded.
- US-003 Schema Validation
  - As a ML Engineer, I want to define input/output signatures so that consumers can validate compatibility.
  - Acceptance: Uploading signature rejects incompatible changes unless override with approval; validation errors returned with diffs.
- US-004 Policy-Gated Promotion
  - As a Reviewer, I want to enforce thresholds (accuracy, bias, latency) before promotion to production.
  - Acceptance: Promotion request triggers automated checks; if all pass and approvals captured, alias updated, and event emitted.
- US-005 Rollback
  - As an SRE, I want to rollback production alias to prior version if KPIs regress.
  - Acceptance: One-click or API rollback updates alias in <1s and emits incident note; blast radius limited to configured percentage for canary.
- US-006 Multi-Format Artifacts
  - As a ML Engineer, I want to store PyTorch, ONNX, and tokenizer artifacts together with a manifest.
  - Acceptance: Registry accepts multi-part artifacts with types; retrieval returns manifest with references and checksums.
- US-007 Observability
  - As a PM, I want to compare model versions on key KPIs across datasets and time.
  - Acceptance: UI shows side-by-side metrics, test suites, and drift monitors; exportable as reports.
- US-008 Attestation and Signing
  - As Security, I want to require signed artifacts and attestations.
  - Acceptance: Unsigned uploads rejected per policy; signatures verified at promotion and retrieval.

6. Functional Requirements
6.1 Core Features
- FR-001 Immutable Model Versions: Content-addressed (SHA256) storage; semver (MAJOR.MINOR.PATCH+build).
- FR-002 Mutable Aliases: staging, production, champion, custom; atomic alias switch; audit logs.
- FR-003 Registry vs Artifact Store: Metadata in SQL DB; artifacts in S3-compatible store; presigned upload/download.
- FR-004 Lineage: Link to dataset/feature versions, training runs, code commit, environment, hyperparameters, seeds.
- FR-005 Signatures/Schemas: JSON schema for I/O; preprocessing/postprocessing versions; compatibility gates.
- FR-006 Lifecycle States: none -> staging -> production -> archived; retention and deprecation schedules.
- FR-007 Promotion/Rollout: A/B, canary, shadow, automated rollback on KPI thresholds.
- FR-008 Multi-Format Support: PyTorch, TensorFlow, ONNX, TorchScript, LoRA/PEFT adapters, quantization configs, tokenizers, prompt templates.
- FR-009 Observability Integration: Persist evaluation metrics, test results, drift monitors; comparison across versions.
- FR-010 Security & Governance: RBAC, signed artifacts, attestations, audit logs; approval workflows.
- FR-011 Interfaces: REST/gRPC APIs, SDK (Python), CLI; webhooks/events; GitOps manifests.
- FR-012 Idempotent Registration: Safe retries; deduplicate by checksum + semver.

6.2 Advanced Features
- FR-013 Policy Engine: Declarative YAML policies for performance, fairness, robustness, cost.
- FR-014 Model Cards: Auto-generated from metadata; editable with sections on intended use, risks, evaluation.
- FR-015 Feature/Vector Index Linkage: Record embedding model versions for RAG consistency; alias cutovers aligned to indices.
- FR-016 Cross-Region Replication: Geo-redundant artifacts and metadata; disaster recovery.
- FR-017 Blue/Green Serving Integration: Integrations with model serving gateways; traffic shaping APIs.
- FR-018 Signing/Verification: Sigstore/cosign integration with keyless OIDC; SLSA-style provenance.
- FR-019 SDK Auto-Validation: Client-side schema checks and env pinning prior to inference.
- FR-020 Cost/Latency Insights: Track per-inference cost and latency distributions per model version.

7. Non-Functional Requirements
7.1 Performance
- p95 registry read latency < 500 ms; write (register) < 2 s for metadata; artifact uploads limited by network.
- Alias switch atomicity < 1 s.
7.2 Reliability
- 99.5% monthly uptime; RPO ≤ 5 minutes; RTO ≤ 15 minutes.
7.3 Usability
- Web UI loads dashboards in < 2 s; accessible search and compare flows; clear promotion workflows.
7.4 Maintainability
- Modular microservices; 80%+ unit test coverage; API versioning; backward-compatible changes documented.

8. Technical Requirements
8.1 Technical Stack
- Backend: Python 3.11+, FastAPI 0.111+, gRPC (grpcio 1.64+)
- DB: PostgreSQL 15+
- Cache/Queue: Redis 7+; Kafka 3.6+ or NATS 2.10+ for events
- Object Storage: S3-compatible (AWS S3, MinIO) with SSE-KMS
- Auth: OIDC (Auth0/Okta/Azure AD), OAuth2.1, JWT
- Infra: Kubernetes 1.29+, Helm 3.14+, Argo CD 2.12+, KEDA 2.14+ (for autoscaling)
- CI/CD: GitHub Actions or Tekton; Cosign for signing
- Frontend: React 18+, TypeScript 5+, Vite 5+, Chakra UI or MUI 6+
- Observability: OpenTelemetry 1.27+, Prometheus, Grafana, Loki/ELK, Jaeger/Tempo
- Feature/Experiment Integration: MLflow Tracking 2.16+ (optional interop), Feast 0.36+ (optional)

8.2 AI/ML Components
- Format adapters: ONNX Runtime 1.19+, Torch 2.3+, TensorFlow 2.16+ for validation.
- Schema validation: Pydantic v2+, JSON Schema Draft 2020-12.
- Evaluation harness: scikit-learn 1.5+, Hugging Face Datasets 2.20+, robustness/bias libraries (Fairlearn 0.10+).
- Reproducibility: Conda env or requirements.txt pinning; Docker image digests.

9. System Architecture
9.1 High-Level Architecture (ASCII)
                +----------------------+
                |   Web UI (React)     |
                +----------+-----------+
                           |
                   HTTPS / OIDC
                           |
                +----------v-----------+         +------------------+
                |  API Gateway         +---------> Webhooks/Events  |
                |  (FastAPI, gRPC)     |         |  (Kafka/NATS)    |
                +----------+-----------+         +------------------+
                           |
       +-------------------+--------------------+
       |                                        |
+------v------+                         +-------v-------+
| Registry    |                         | Validation    |
| Service     |                         | & Policy Svc  |
| (metadata)  |                         | (eval, schema)|
+------+------+\                        +-------+-------+
       |       \                                |
       |        \                               |
+------v--+   +--v------+                +------v------+
| Postgres|   | Redis   |                | Model Serving|
| (metadata)| | Cache   |                | Gateways     |
+-----+---+   +----+----+                +------+------+
      |             |                           |
+-----v-------------v----+               +------v------+
| Object Storage (S3)     |               | Monitors    |
| Artifacts, Manifests    |               | & Dashboards|
+-------------------------+               +-------------+

9.2 Component Details
- API Gateway: Exposes REST/gRPC endpoints; performs authn/authz, rate limiting, request validation.
- Registry Service: CRUD for models/versions/aliases; stores lineage, metrics, approvals; issues presigned URLs.
- Validation & Policy Service: Executes schema checks, evaluation suites, bias/robustness and latency gates; writes results back.
- Object Storage: Immutable artifact blobs, manifests, signatures, model cards.
- Message Bus: Emits events for registrations, promotions, rollbacks; triggers CI/CD and serving updates.
- Serving Gateways: Integrations (e.g., KServe/Seldon/BentoML/Triton); support A/B, canary, shadow deployments.
- Observability: Collect metrics/logs/traces; dashboards for version comparisons and promotions.

9.3 Data Flow
- Register: Client -> API -> presigned upload -> object store; then lineage metadata -> Postgres; event emitted -> policy validation -> results stored.
- Promotion: Client -> API -> policy checks -> approvals -> alias update -> event -> serving gateway traffic update.
- Retrieval: Client -> API -> metadata from Postgres -> presigned download for artifacts.

10. Data Model
10.1 Entity Relationships
- Model (1) — (N) ModelVersion
- ModelVersion (1) — (N) Artifact
- ModelVersion (1) — (N) Metric
- Model (1) — (N) Alias (points to a ModelVersion)
- ModelVersion (1) — (1) Signature
- ModelVersion (1) — (N) PolicyCheck
- ModelVersion (1) — (N) Approval
- ModelVersion (N) — (1) DatasetVersion
- ModelVersion (N) — (1) TrainingRun
- ModelVersion (1) — (N) AuditLog
- Deployment (N) — (1) Alias

10.2 Database Schema (selected tables)
- models
  - id (uuid, pk), name (text, unique), owner_team (text), created_at
- model_versions
  - id (uuid, pk), model_id (fk), semver (text), build_id (text), checksum (sha256, unique), state (enum: none, staging, production, archived), created_by (user_id), created_at
  - code_commit (text), docker_image (text, digest), env_spec (jsonb), hyperparams (jsonb), seed (int)
  - dataset_version_id (fk), training_run_id (fk)
- aliases
  - id (uuid, pk), model_id (fk), alias (text, unique per model), target_version_id (fk), updated_by, updated_at
- artifacts
  - id (uuid, pk), version_id (fk), type (enum: pytorch, tf, onnx, tokenizer, preproc, postproc, container, lora, quant_cfg, prompt_template), uri (text), size_bytes, checksum, signature_ref (text), created_at
- signatures
  - id (uuid, pk), version_id (fk), input_schema (jsonb), output_schema (jsonb), preproc_version (text), postproc_version (text)
- metrics
  - id (uuid, pk), version_id (fk), dataset_label (text), metrics (jsonb: accuracy, f1, auroc, latency_ms, cost_usd), created_at
- approvals
  - id (uuid, pk), version_id (fk), approver (user_id), role (enum: owner, reviewer, risk), status (enum: pending, approved, rejected), comment (text), updated_at
- policy_checks
  - id (uuid, pk), version_id (fk), policy_name (text), status (pass/fail), details (jsonb), executed_at
- audit_logs
  - id (uuid, pk), resource_type (enum: model, version, alias), resource_id (uuid), action (text), actor (user_id), timestamp, diff (jsonb), ip
- deployments
  - id (uuid, pk), alias_id (fk), strategy (enum: ab, canary, shadow), config (jsonb), started_at, status

10.3 Data Flow Diagrams (ASCII)
Register Flow:
Client -> API: create version
API -> S3: presigned PUT
Client -> S3: upload artifacts
API -> DB: insert version, artifacts, lineage
API -> Bus: emit "version.created"
Policy Svc -> DB: insert policy_checks
Policy Svc -> Bus: emit "validation.completed"

Promotion Flow:
Client -> API: promote alias
API -> Policy Svc: ensure checks & approvals
API -> DB: update alias target
API -> Bus: emit "alias.updated"
Serving -> Fetch alias -> adjust traffic

10.4 Input Data & Dataset Requirements
- DatasetVersion must include:
  - dataset_id, version tag or checksum, storage URI, schema, preprocessing version, split definitions, license.
- Minimum metadata for training run:
  - code commit SHA, training framework/version, hyperparameters, random seed, hardware profile, start/end timestamps.

11. API Specifications
11.1 REST Endpoints
- POST /models
  - Create a model {name, owner_team}
- GET /models?query=
  - Search models
- POST /models/{name}/versions
  - Register a version {semver, build_id, lineage, env_spec, hyperparams}; returns version_id and presigned URLs for artifacts
- POST /models/{name}/versions/{version_id}/artifacts
  - Request presigned URLs for artifact uploads {type, size_bytes, filename}
- GET /models/{name}/versions/{version_id}
  - Retrieve version metadata, metrics, artifacts
- POST /models/{name}/versions/{version_id}/metrics
  - Upsert metrics payload
- PUT /models/{name}/aliases/{alias}
  - Update alias target {version_id} with atomic switch
- GET /models/{name}/aliases/{alias}
  - Get alias target version
- POST /models/{name}/versions/{version_id}/promotions
  - Start promotion workflow to {stage: staging|production}, with optional {strategy, thresholds}
- POST /models/{name}/versions/{version_id}/approvals
  - Approve/reject with comment
- POST /models/{name}/versions/{version_id}/signatures
  - Upload input/output schemas
- GET /events/stream
  - SSE or WebSocket stream of events
- POST /policies/validate
  - Run ad-hoc validation on a version {policy_set}

11.2 Request/Response Examples
- Register Version
Request:
POST /models/sentiment-classifier/versions
{
  "semver": "1.3.0",
  "build_id": "build.2025-11-25+git.abc1234",
  "lineage": {
    "dataset_version_id": "dsv_72b1",
    "training_run_id": "run_9f22",
    "code_commit": "abc1234",
    "docker_image": "ghcr.io/org/sentiment:1.3.0@sha256:deadbeef",
    "seed": 42
  },
  "env_spec": {
    "requirements_txt": "numpy==2.1.2\ntorch==2.3.1\ntransformers==4.45.0",
    "python": "3.11.8",
    "cuda": "12.2"
  },
  "hyperparams": {"lr": 3e-5, "batch_size": 64, "epochs": 3}
}
Response:
{
  "version_id": "ver_01JABCDEF",
  "checksum": "sha256:4c1f...",
  "artifact_uploads": [
    {"type":"pytorch","upload_url":"https://s3...","artifact_id":"art_1"},
    {"type":"tokenizer","upload_url":"https://s3...","artifact_id":"art_2"}
  ]
}

- Update Alias
PUT /models/sentiment-classifier/aliases/production
{"version_id":"ver_01JABCDEF"}
Response: {"alias":"production","version_id":"ver_01JABCDEF","updated_at":"2025-11-25T10:10:10Z"}

- Promotion with Policy
POST /models/sentiment-classifier/versions/ver_01JABCDEF/promotions
{
  "stage":"production",
  "strategy":{"type":"canary","traffic_percent":10,"monitor_window_min":30},
  "thresholds":{"accuracy_delta_min":0.0,"p95_latency_ms_max":200,"parity_diff_max":0.1}
}
Response: {"promotion_id":"pr_123","status":"pending_checks"}

11.3 Authentication
- OIDC/OAuth2.1 Authorization Code + PKCE for UI; client credentials for automation.
- JWT bearer tokens with short TTL; refresh via OAuth; optional API keys scoped per project.
- Scopes: model:read, model:write, promote:write, policy:write, admin.
- Tenant isolation via org_id claim; row-level security in DB.

12. UI/UX Requirements
12.1 User Interface
- Key pages: Model Catalog, Model Detail, Version Comparison, Artifacts, Metrics & Evaluations, Aliases & Deployments, Approvals, Audit Log, Policies.
- Components: Search/filter, diff views (schema, metrics), lineage graph, promotion wizard, drift dashboard.

12.2 User Experience
- Intuitive flows: register -> validate -> compare -> request approval -> promote -> monitor -> rollback.
- Zero-downtime alias switch; contextual tooltips; keyboard navigation; dark/light mode.

12.3 Accessibility
- WCAG 2.1 AA compliance; semantic HTML; ARIA labels; high-contrast themes; screen-reader friendly tables and charts.

13. Security Requirements
13.1 Authentication
- OIDC federation; MFA optional; device trust; service accounts with least privilege.

13.2 Authorization
- RBAC: Roles (Admin, Owner, Contributor, Reviewer, Viewer). Fine-grained: per-model/project permissions. Row-level security and namespace isolation.

13.3 Data Protection
- TLS 1.3 in transit; SSE-KMS at rest (S3); transparent data encryption for Postgres; client-side hashing for integrity.
- Secrets managed via Kubernetes Secrets/External Secrets and cloud KMS.

13.4 Compliance
- Logging and audit trails for all changes; retention policies configurable.
- Support for data governance controls; PII redaction in logs; consent-aware dataset links.

14. Performance Requirements
14.1 Response Times
- p50/p95 for reads: 100/500 ms.
- p50/p95 for writes (metadata): 200/2000 ms.
- Alias update: < 1 s.

14.2 Throughput
- Sustain 300 RPS read, 50 RPS write per region; scale horizontally via autoscaling.

14.3 Resource Usage
- Registry service < 500 MiB RSS per pod under typical load; CPU target 70% per HPA; object store bandwidth per upload ≥ 100 MB/s (intra-region).

15. Scalability Requirements
15.1 Horizontal Scaling
- Stateless API pods; scale via HPA on CPU/RPS; Redis for caching; partitioned Kafka topics for events.

15.2 Vertical Scaling
- DB vertical headroom; read replicas for heavy read paths; connection pooling via pgbouncer.

15.3 Load Handling
- Burst handling with queue backpressure; rate limiting per tenant; circuit breakers and retries with jitter.

16. Testing Strategy
16.1 Unit Testing
- 80%+ coverage for registry logic, schema validation, policy evaluation, permission checks.

16.2 Integration Testing
- Test flows: register->upload->validate->promote; artifact store presigned URL lifecycle; RBAC; idempotent retries.

16.3 Performance Testing
- Load tests for read/write and alias switch; artifact upload throughput; chaos testing for DB and object store outages.

16.4 Security Testing
- SAST/DAST; dependency scanning; signature verification tests; authn/authz fuzzing; audit log tamper detection.

17. Deployment Strategy
17.1 Deployment Pipeline
- CI: build, test, SBOM, scan, sign (cosign) images; publish to registry.
- CD: Argo CD with GitOps manifests; progressive delivery for API service with canary.

17.2 Environments
- Dev -> Staging -> Production; isolated cloud accounts/projects; distinct object store buckets and DB instances.

17.3 Rollout Plan
- Phase 1: Internal tenants; enable read APIs.
- Phase 2: Write APIs and promotions gated.
- Phase 3: External integrations; feature parity and SLOs.

17.4 Rollback Procedures
- Helm rollback; Argo CD app rollback; database migrations with down scripts; revert aliases to previous versions atomically.

18. Monitoring & Observability
18.1 Metrics
- Service: request_rate, error_rate, latency_histogram, saturation, DB query timings.
- Business: versions_registered, promotions_completed, rollbacks, policy_failures.
- Model: accuracy, F1, AUROC, p95 latency, cost/inference, drift scores.

18.2 Logging
- Structured JSON logs; correlation IDs; sensitive data redaction; retention 30-180 days.

18.3 Alerting
- SLO burn alerts (availability, latency); error spikes; policy engine failures; promotion stuck; storage utilization.

18.4 Dashboards
- Service health; model catalog KPIs; promotion timelines; lineage graphs; cost/latency trends per version.

19. Risk Assessment
19.1 Technical Risks
- Complexity of multi-format artifact validation.
- Vendor lock-in with specific cloud services.
- Inconsistent lineage from external tools.

19.2 Business Risks
- Adoption resistance if workflows are rigid.
- Compliance requirements evolving.
- Cost overruns due to storage/compute.

19.3 Mitigation Strategies
- Pluggable adapters; cloud-agnostic interfaces (S3 API, OIDC).
- Policy engine with declarative configs; stakeholder training; budgets and lifecycle retention policies.

20. Timeline & Milestones
20.1 Phase breakdown
- Phase 0 (Week 0-2): Discovery, requirements finalization, architecture sign-off.
- Phase 1 (Week 3-8): Core registry service (models, versions, aliases), DB schema, REST read APIs, basic UI.
- Phase 2 (Week 9-14): Artifact uploads (presigned), lineage, signatures, metrics, policy engine MVP, approvals, audit logs.
- Phase 3 (Week 15-20): Promotion strategies (A/B, canary, shadow), serving integrations, rollback automation.
- Phase 4 (Week 21-24): Security (RBAC, OIDC, signing/attestation), observability, SLOs, performance tuning.
- Phase 5 (Week 25-28): Advanced features (model cards, vector index links, cross-region), UX polish, docs.
- Phase 6 (Week 29-32): Hardening, scale tests, DR drills, GA launch.

20.2 Key Milestones
- M1: Core metadata and alias API (Week 8)
- M2: Artifact store integration and lineage (Week 14)
- M3: Policy-gated promotion with UI (Week 20)
- M4: Security GA (RBAC + signatures) (Week 24)
- M5: GA Release with SLOs met (Week 32)

Budgetary estimate (12 months run-rate after GA)
- Cloud infra: $6k–$12k/month (region, storage volume dependent)
- Engineering: 4–6 FTE for build; 1–2 FTE ongoing ops
- Tooling/licenses (optional): $1k–$3k/month

21. Success Metrics & KPIs
21.1 Measurable targets
- Registry availability: ≥99.5% monthly.
- Read latency p95: ≤500 ms; write p95: ≤2 s.
- Promotion automation rate: ≥75%.
- Incident reduction: ≥60% decrease in model-related rollbacks due to failures within 6 months.
- Reproducibility: ≥95% of models with complete lineage and verified environment pinning.
- Model performance gates: ≥90% of production models meet or exceed baseline accuracy and fairness thresholds.
- Time-to-production: median reduced by ≥50%.
- Audit coverage: 100% of alias changes and promotions logged with approver identity and diffs.

22. Appendices & Glossary
22.1 Technical Background
- Immutable versions with mutable aliases mirror best practices in versioned vector indexes and alias cutovers; supports parallel builds, validation, and atomic switch.
- Content-addressed storage ensures integrity; semver with build metadata allows human-friendly releases with precise pinning.
- Lineage ties datasets, code, environment, and seeds for reproducibility; signatures provide schema guarantees for safe integration.
- Policy gates enforce performance, robustness, and fairness before promotions; rollout strategies minimize risk and enable automated rollback.

22.2 References
- SemVer 2.0.0 specification
- JSON Schema Draft 2020-12
- OpenTelemetry specification
- Sigstore/cosign documentation
- MLflow Model Registry (conceptual reference)
- ONNX and ONNX Runtime docs
- Kubernetes, Argo CD, KEDA docs
- Fairlearn documentation

22.3 Glossary
- Alias: Human-friendly pointer to an immutable model version (e.g., staging, production).
- Attestation: Cryptographically verifiable statement about build/provenance.
- Canary Deployment: Partial rollout to a subset of traffic to validate behavior.
- Lineage: Links between a model and its datasets, code, environment, and training run.
- Manifest: Machine-readable description of a model version and its constituent artifacts.
- Model Card: Documentation describing model intent, evaluation, and risks.
- Policy Gate: Automated rule enforcing thresholds (accuracy, fairness, latency).
- Provenance: Audit trail of how an artifact was produced.
- Schema/Signature: Input/output definitions (types, shapes) enforced at validation time.
- Version: Immutable, content-addressed model release with semver and build metadata.

Repository Structure (mono-repo)
- notebooks/
  - exploration.ipynb
  - evaluation_suite.ipynb
- src/
  - api/
    - main.py
    - routes/
      - models.py
      - versions.py
      - aliases.py
      - policies.py
  - services/
    - registry_service.py
    - policy_service.py
    - lineage_service.py
    - signing.py
  - clients/
    - python/
      - sdk.py
      - cli.py
  - validations/
    - schema_validator.py
    - metrics_checker.py
  - integrations/
    - serving/
      - kserve_adapter.py
    - storage/
      - s3.py
  - db/
    - models.sql
    - migrations/
- tests/
  - unit/
  - integration/
  - e2e/
- configs/
  - policy/
    - default_policy.yaml
  - app.yaml
  - logging.yaml
- data/
  - sample_manifests/
  - example_schemas/
- deploy/
  - helm/
  - k8s/
  - argo/
- docs/
  - api.md
  - architecture.md
  - model_card_template.md

Config Samples
- configs/policy/default_policy.yaml
policies:
  - name: baseline_performance
    checks:
      - type: metrics.threshold
        metric: accuracy
        op: ">="
        value: 0.90
      - type: metrics.delta_vs_alias
        alias: production
        metric: accuracy
        op: ">="
        value: 0.00
      - type: latency.threshold
        metric: p95_latency_ms
        op: "<="
        value: 200
      - type: fairness.parity_difference
        metric: demographic_parity
        op: "<="
        value: 0.10

- configs/app.yaml
server:
  port: 8080
  cors: ["*"]
auth:
  oidc_issuer: "https://org.okta.com/oauth2/default"
  audience: "aiml025"
storage:
  s3:
    bucket: "aiml025-artifacts"
    region: "us-east-1"
    sse: "aws:kms"
db:
  postgres:
    host: "postgres"
    port: 5432
    database: "registry"
    user: "registry"
    sslmode: "require"
observability:
  otel:
    exporter: "otlp"
    endpoint: "http://otel-collector:4317"

Python SDK Snippet
- src/clients/python/sdk.py (usage)
from aiml025 import RegistryClient

client = RegistryClient(base_url="https://registry.example.com", token="...")

ver = client.register_version(
    model_name="sentiment-classifier",
    semver="1.3.0",
    build_id="build.2025-11-25+git.abc1234",
    lineage={
        "dataset_version_id":"dsv_72b1",
        "training_run_id":"run_9f22",
        "code_commit":"abc1234",
        "docker_image":"ghcr.io/org/sentiment:1.3.0@sha256:deadbeef",
        "seed":42
    },
    env_spec={"python":"3.11.8","requirements_txt":"torch==2.3.1\n..."},
    hyperparams={"lr":3e-5}
)

client.upload_artifact(
    model_name="sentiment-classifier",
    version_id=ver["version_id"],
    artifact_type="pytorch",
    file_path="model.pt"
)

client.set_alias("sentiment-classifier", "staging", ver["version_id"])

Model Manifest Example (stored alongside artifacts)
{
  "model_name": "sentiment-classifier",
  "version_id": "ver_01JABCDEF",
  "semver": "1.3.0+build.2025-11-25.git.abc1234",
  "artifacts": [
    {"type":"pytorch","uri":"s3://.../model.pt","checksum":"sha256:..."},
    {"type":"tokenizer","uri":"s3://.../tokenizer.json","checksum":"sha256:..."}
  ],
  "signature": {
    "input_schema":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]},
    "output_schema":{"type":"object","properties":{"label":{"type":"string"},"score":{"type":"number"}},"required":["label","score"]}
  },
  "lineage": {
    "dataset_version_id":"dsv_72b1",
    "training_run_id":"run_9f22",
    "code_commit":"abc1234",
    "docker_image":"ghcr.io/org/sentiment:1.3.0@sha256:deadbeef",
    "seed":42
  }
}

Acceptance Criteria Summary Highlights
- Immutable version IDs with checksums; alias switches are atomic and logged.
- Policy gates enforce: accuracy ≥ 0.90, p95 latency ≤ 200 ms, fairness parity difference ≤ 0.10.
- Read p95 latency ≤ 500 ms; availability ≥ 99.5%.
- A/B, canary, shadow supported with automatic rollback.
- Full lineage present for ≥95% of registered models.