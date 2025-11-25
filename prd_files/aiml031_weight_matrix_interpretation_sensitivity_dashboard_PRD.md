# Product Requirements Document (PRD) / # `aiml031_weight_matrix_interpretation_sensitivity_dashboard`

Project ID: aiml031  
Category: AI/ML Developer Tools, Model Interpretability & Robustness  
Status: Draft for Review  
Version: v1.0  
Last Updated: 2025-11-25

## 1. Overview
### 1.1 Executive Summary
aiml031 delivers an interactive dashboard and API for interpreting weight matrices and analyzing sensitivity of machine learning models. It ingests checkpoints from common frameworks (PyTorch, TensorFlow, ONNX), computes layer-wise weight statistics (norms, spectra), sensitivity metrics (gradients, Fisher, curvature proxies), and links them with explainability artifacts (saliency, Integrated Gradients) and robustness evaluations (adversarial/stress testing). The system supports comparison across runs/checkpoints, pruning/regularization what-if analyses, representation similarity (CKA), and exports for reports. It is designed for researchers, ML engineers, and data scientists building and validating deep learning, NLP, and computer vision systems.

### 1.2 Document Purpose
This PRD specifies requirements for building aiml031: scope, user needs, features, architecture, data model, APIs, UI/UX, security, performance, testing, deployment, monitoring, risks, timeline, and success criteria to align product, engineering, and stakeholders.

### 1.3 Product Vision
Make weight and sensitivity analysis first-class in the ML workflow—fast, insightful, and actionable—so teams can understand, debug, and harden models with confidence. Provide a single pane of glass that correlates weight structure, sensitivity, explainability, and robustness, enabling evidence-based decisions on training, fine-tuning, and deployment.

## 2. Problem Statement
### 2.1 Current Challenges
- Interpreting model parameters is fragmented; tools exist for saliency or pruning but rarely connect weight spectra, curvature, and robustness in one place.
- Sensitivity analyses are compute-heavy and ad-hoc; reproducibility and caching are often missing.
- Comparing checkpoints, layers, and runs is time-consuming and error-prone.
- Lack of standardized APIs to programmatically extract weight/sensitivity metrics for pipelines.
- Limited visibility into how pruning/regularization affects generalization and stability.

### 2.2 Impact Analysis
- Slow iteration cycles and model regressions.
- Missed issues in sharpness or brittle regions that lead to instability or adversarial vulnerability.
- Inefficient resource usage due to repeated computations without caching/tracking.
- Difficulty communicating insights to stakeholders without cohesive visualizations.

### 2.3 Opportunity
- Provide a robust platform, with GPU acceleration and background workers, that computes and visualizes key weight/sensitivity metrics, linked to explainability and robustness.
- Enable comparative analytics and experiment tracking integrations (MLflow/W&B).
- Empower teams to optimize models via pruning/regularization while quantifying trade-offs.

## 3. Goals and Objectives
### 3.1 Primary Goals
- Build an interactive dashboard for weight-matrix interpretation and sensitivity analysis.
- Implement scalable backend computations for spectra, norms, Fisher/Hessian proxies, and influence.
- Support run/checkpoint/layer comparison, pruning what-if, and robustness stress tests.
- Provide REST APIs and SDK for automation and integration.
- Ensure reproducibility with caching, versioning, and experiment tracking hooks.

### 3.2 Business Objectives
- Reduce model debugging time by 50%.
- Improve robustness metrics (e.g., adversarial tolerance) by measurable deltas after interventions.
- Drive adoption across research and product teams; 100+ monthly active users within 6 months.
- Offer a foundation for premium features (auto-pruning recommendations, vector-based anomaly detection).

### 3.3 Success Metrics
- <500 ms latency for cached metric retrieval; <5 s for on-demand heavy computations for medium models.
- 99.5% monthly uptime.
- >90% agreement with reference computations (relative error <1e-6 on numerical benchmarks).
- >30% decrease in post-training regression incidents attributed to weight/sensitivity issues.
- Export usage: >200 exports/month within 3 months of launch.

## 4. Target Users/Audience
### 4.1 Primary Users
- ML Researchers exploring architectures and training dynamics.
- Machine Learning Engineers integrating interpretability and robustness checks into pipelines.
- Data Scientists validating model changes and communicating insights.

### 4.2 Secondary Users
- Product Managers needing interpretability evidence for go/no-go decisions.
- MLOps/SRE teams monitoring model health and compute efficiency.
- Security/Responsible AI reviewers auditing robustness and sensitivity.

### 4.3 User Personas
- Persona 1: Dr. Alice Nguyen (ML Researcher)
  - Background: PhD in Deep Learning; works on vision transformers and contrastive learning.
  - Pain points: Difficult to correlate weight spectra with generalization; manual scripts for SVD/Hessian approximations; inconsistent comparisons across runs.
  - Goals: Quickly compare checkpoints, understand flatness vs. accuracy trade-offs, publish robust figures.

- Persona 2: Marco Silva (ML Engineer)
  - Background: 6 years building NLP pipelines; owns CI/CD for model training and deployment.
  - Pain points: Lacks standardized API to fetch sensitivity metrics in pipelines; recomputation wastes GPU hours; difficult to track provenance.
  - Goals: Automate metric extraction, cache results, enforce robustness gates before deployment.

- Persona 3: Priya Desai (Data Scientist)
  - Background: Works on tabular + time-series models; responsible for explainability to stakeholders.
  - Pain points: Disconnected tools for feature attributions and parameter analysis; unclear impact of pruning on stability and accuracy.
  - Goals: Generate consistent, interpretable reports; quantify risk of interventions.

- Persona 4: Jamal Burton (MLOps/SRE)
  - Background: Observability and platform engineering for ML systems.
  - Pain points: No visibility into compute hotspots; unpredictable load from ad-hoc analysis.
  - Goals: Capacity plan for GPU workers; monitor job queues, set alerts, control costs.

## 5. User Stories
- US-001: As a researcher, I want to upload a checkpoint and see layer-wise weight heatmaps so that I can detect anomalies.
  - Acceptance: Upload succeeds for PyTorch/TF/ONNX; heatmaps render within 2 s for medium models; layers navigable.

- US-002: As an engineer, I want to compute per-layer spectral norm and effective rank so that I can correlate with validation accuracy.
  - Acceptance: API returns metrics with relative error <1e-6 vs. reference; completion <5 s for medium models.

- US-003: As a data scientist, I want sparsity histograms and before/after pruning comparisons so that I can quantify trade-offs.
  - Acceptance: Histograms and deltas rendered side-by-side; export CSV/PNG works.

- US-004: As a researcher, I want to compute Fisher diagonal approximations and Hutchinson-trace sharpness so that I can assess curvature.
  - Acceptance: Jobs queue; progress visible; results cached and retrievable.

- US-005: As an engineer, I want to run adversarial stress tests and view robustness vs. spectral statistics so that I can harden models.
  - Acceptance: Supports FGSM/PGD configurations; plots link to weight metrics; results exportable.

- US-006: As a user, I want to compare two checkpoints with synchronized cursors so that I can inspect metric drifts.
  - Acceptance: Side-by-side panels with linked brushing; diffs computed per layer.

- US-007: As a user, I want Integrated Gradients and saliency maps linked to parameter sensitivity so that I can connect inputs to parameters.
  - Acceptance: Selecting a layer highlights related attribution changes; tooltips show correlations.

- US-008: As an engineer, I want an API to retrieve layer metrics for CI so that I can gate deployments.
  - Acceptance: OAuth2-protected endpoint; stable schema; rate-limited; example SDK provided.

- US-009: As a researcher, I want loss-landscape cross-sections to evaluate flatness so that I can justify learning rate schedules.
  - Acceptance: 1D/2D planes computed with noise or basis vectors; interactive plots.

- US-010: As a user, I want representation similarity (CKA) across epochs so that I can study drift.
  - Acceptance: CKA matrices computed; cluster visualization; compare across runs.

- US-011: As an engineer, I want vector-based retrieval of similar runs/layers so that I can find anomalies or exemplars.
  - Acceptance: FAISS/Qdrant returns neighbors under 300 ms; metadata shown.

- US-012: As a PM, I want PDF/HTML report exports so that I can share insights with stakeholders.
  - Acceptance: Branded, timestamped, includes selected panels and metrics.

- US-013: As MLOps, I want to monitor worker GPU utilization and queue length so that I can scale resources.
  - Acceptance: Grafana dashboards; alerts on thresholds.

- US-014: As a user, I want run/layer filters by tags, dataset slices, and checkpoints so that I can focus analysis.
  - Acceptance: Multi-select filters; changes propagate to all linked panels.

- US-015: As a security reviewer, I want role-based access and audit logs so that I can enforce compliance.
  - Acceptance: RBAC with scopes; audit log entries on sensitive actions.

## 6. Functional Requirements
### 6.1 Core Features
- FR-001: Checkpoint ingestion (PyTorch .pt/.pth, TensorFlow .ckpt/.h5, ONNX .onnx).
- FR-002: Weight visualization: heatmaps, sparsity/density histograms, per-layer stats (L1/L2/Frobenius, spectral norm, condition number, effective rank).
- FR-003: Spectrum computation: SVD/EVD log-spectrum tracking; caching per layer/checkpoint.
- FR-004: Sensitivity metrics: gradients/Jacobians, Fisher diagonal/trace, Hessian proxies (Hutchinson), sharpness.
- FR-005: Explainability linkage: saliency, Integrated Gradients, SmoothGrad; map to layer sensitivity.
- FR-006: Robustness workbench: adversarial gradient norms, FGSM/PGD sweeps, noise injection perturbations; Lipschitz proxies.
- FR-007: Pruning/regularization playground: magnitude pruning, SNIP/GraSP, weight decay effects; before/after deltas.
- FR-008: Comparison tools: checkpoint/run diffing; synchronized cursors; small multiples; epoch timelines.
- FR-009: Representation similarity: CKA/linear CKA; cross-layer correlation matrices; PCA of weights/activations.
- FR-010: Vector search: embedding of layer statistics into vectorstore; neighbor retrieval and anomaly flagging.
- FR-011: Exports: CSV/JSON of metrics; PNG/SVG of figures; HTML/PDF reports.
- FR-012: Experiment tracking integration: MLflow/W&B run hooks; parameter and metric lineage.
- FR-013: Dataset slice management: register slices for sensitivity/robustness evaluations.
- FR-014: API + SDK: authenticated endpoints for metrics retrieval, job submission, and export.
- FR-015: Caching and recomputation control with cache keys (model hash, layer id, config).

### 6.2 Advanced Features
- FR-016: Loss landscape explorer: 1D/2D cross-sections along principal directions or random directions.
- FR-017: Auto-recommendations: highlight layers with anomalous spectra or sharpness; pruning candidate suggestions.
- FR-018: Background scheduling: priority queues, cancellation, retry policies.
- FR-019: Checkpoint comparator with structural diffs (parameter deltas, rank changes).
- FR-020: Webhooks for pipeline integration (e.g., notify when sensitivity job completes).
- FR-021: Role-based workspaces and shared dashboards with annotations.
- FR-022: Programmatic report templates with Jinja2.

## 7. Non-Functional Requirements
### 7.1 Performance
- Cached reads: p95 < 500 ms.
- Spectrum medium models (≤50M params): p95 < 5 s per layer group; batching allowed.
- Adversarial sweeps on 1k samples: p95 < 90 s on 1 x A10 GPU.
- Vector search latency: p95 < 300 ms for k=10 neighbors.

### 7.2 Reliability
- Uptime: ≥ 99.5% monthly.
- Job success rate: ≥ 99% excluding user cancellations.
- Retry policy: exponential backoff up to 3 retries for transient failures.

### 7.3 Usability
- Onboarding: first chart visible within 3 clicks after upload.
- Accessibility: WCAG 2.2 AA.
- Internationalization-ready; English baseline.

### 7.4 Maintainability
- Code coverage: ≥ 85%.
- Modular service boundaries; clear API versioning (v1).
- Linting/formatting enforced (ruff/black, eslint/prettier).

## 8. Technical Requirements
### 8.1 Technical Stack
- Backend: Python 3.11+, FastAPI 0.115+, Uvicorn 0.30+, Celery 5.4+ or Prefect 2.16+.
- ML: PyTorch 2.3+, TorchVision 0.18+; TensorFlow 2.16+; JAX optional; ONNX 1.16+.
- Data/Storage: PostgreSQL 15+, Redis 7+ (cache/queue), MinIO/S3 for artifacts.
- Vector store: FAISS 1.8+ or Qdrant 1.8+.
- Frontend: React 18+, TypeScript 5+, Vite 5+, Plotly.js 2.30+ or ECharts 5+, D3 7+.
- Container/Orchestration: Docker 24+, Kubernetes 1.29+, Helm 3.14+.
- Auth: Keycloak 24+ or Auth0; OAuth2/OIDC.
- Observability: OpenTelemetry 1.27+, Prometheus 2.53+, Grafana 11+.
- Experiment Tracking: MLflow 2.12+; WandB SDK 0.17+.
- CI/CD: GitHub Actions; ArgoCD or Flux optional.

Repository structure:
- notebooks/
- src/
  - api/
  - workers/
  - core/metrics/
  - core/visualization/
  - integrations/mlflow/
  - storage/
  - auth/
  - vectorstore/
- webapp/
- tests/
- configs/
- data/ (local dev only)
- infra/ (helm charts, k8s manifests)
- scripts/
- docs/

### 8.2 AI/ML Components
- Weight analytics: SVD (truncated), eigendecomposition, norms, effective rank (entropy-based), condition numbers.
- Sensitivity: per-sample/per-batch gradients, Jacobian norms, Fisher diagonal (empirical or using gradients), Hutchinson trace for Hessian, sharpness-aware estimates.
- Explainability: Saliency, Integrated Gradients, SmoothGrad; optional LRP plugin.
- Robustness: FGSM, PGD, noise injection; Lipschitz proxies via spectral norms.
- Pruning: magnitude, SNIP, GraSP; mask visualization and evaluation.
- Similarity: CKA/linear CKA; PCA; correlation matrices.
- Vector embeddings: concatenate standardized stats (e.g., log-spectra, norms) and reduce via PCA/UMAP for indexing.

## 9. System Architecture
### 9.1 High-Level Architecture (ASCII diagram)
+---------------------------+           +---------------------------+
|         Web App           |  HTTPS    |        REST API           |
|  React/TS, Plotly/D3      +-----------> FastAPI, OAuth2          |
|  Auth via OIDC            |           | Rate limit, validation    |
+------------+--------------+           +--------------+------------+
             |                                         |
             | WebSockets (progress)                   | Publishes Jobs
             |                                         v
             |                          +--------------+------------+
             |                          |    Task Queue (Redis)     |
             |                          +--------------+------------+
             |                                         |
             |                                         v
             |                          +--------------+------------+
             |                          |   Workers (GPU-enabled)   |
             |                          | PyTorch/TF, metrics calc  |
             |                          +--+-----------+-----------+
             |                             |           |           
             |       Reads metrics          |           | Writes artifacts
             v                             v           v
+------------+--------------+   +-----------+--+   +---+----------------+
|   PostgreSQL (metadata)   |   |  Object Store |   |  Vector Store      |
| projects, runs, metrics   |   |  (S3/MinIO)   |   |  (FAISS/Qdrant)    |
+---------------------------+   +-------------- +   +--------------------+
             ^
             | Telemetry
+------------+--------------+
| Observability Stack       |
| Prometheus/Grafana/OTel   |
+---------------------------+

### 9.2 Component Details
- Web App: Interactive dashboards, authentication, data fetching, WebSocket progress updates, export builder.
- REST API: AuthN/Z, input validation, resource endpoints, job submission, cache layer, streaming of large results.
- Workers: Background compute for spectra, gradients, adversarial sweeps; GPU-accelerated, batched; checkpoint loaders (PyTorch/TF/ONNX).
- Storage:
  - PostgreSQL: project/run metadata, metric indexes, configs, RBAC, audit logs.
  - Object Store: checkpoints, computed artifacts (plots, CSVs, NumPy arrays).
  - Redis: task queue, cache.
  - Vector Store: embeddings for similarity/anomaly queries.
- Integrations: MLflow/W&B hooks to ingest runs and parameters.
- Observability: Tracing, metrics, logs; dashboards and alerts.

### 9.3 Data Flow
1) User uploads checkpoint and selects dataset slice.  
2) API validates, stores metadata, uploads artifact to object store.  
3) User triggers computations (e.g., spectra, sensitivity).  
4) API enqueues tasks; workers consume, load model, run computations with caching.  
5) Workers write results to object store/Postgres; embed stats into vector store.  
6) API notifies Web App via WebSocket; UI fetches and renders results.  
7) Exports generated on-demand and stored for download.  

## 10. Data Model
### 10.1 Entity Relationships
- User (1..*) -> Project (1..*)
- Project (1..*) -> Run (1..*)
- Run (1..*) -> Checkpoint (1..*)
- Checkpoint (1..*) -> Layer (1..*)
- Layer (1..*) -> WeightStats (1..1 per config)
- Layer (1..*) -> SensitivityMetrics (1..1 per config)
- Run (1..*) -> RobustnessRun / PruningRun / SimilarityMetrics
- Run (1..*) -> VisualizationConfig
- DatasetSlice referenced by Run and analyses

### 10.2 Database Schema (selected tables)
- users: id, email, name, role, oidc_sub, created_at, updated_at
- teams: id, name, created_at
- team_members: id, team_id, user_id, role
- projects: id, team_id, name, tags (jsonb), created_at
- runs: id, project_id, name, framework, base_model, params (jsonb), tags (jsonb), created_at
- checkpoints: id, run_id, path, model_hash, step, epoch, created_at
- layers: id, checkpoint_id, name, type, shape, param_count, created_at
- weight_stats: id, layer_id, config_hash, l1, l2, fro, spectral_norm, cond_num, eff_rank, sparsity, spectrum_path, created_at
- sensitivity_metrics: id, layer_id, config_hash, grad_norm, jacobian_norm, fisher_diag_path, hutchinson_trace, sharpness, created_at
- robustness_runs: id, run_id, checkpoint_id, config (jsonb), results_path, metrics (jsonb), created_at
- pruning_runs: id, run_id, checkpoint_id, method, sparsity_target, mask_path, metrics_before (jsonb), metrics_after (jsonb), created_at
- similarity_metrics: id, run_id, ck_matrix_path, method, epochs, created_at
- dataset_slices: id, name, description, source_uri, spec (jsonb), created_at
- viz_configs: id, run_id, layout (jsonb), filters (jsonb), saved_by, created_at
- vector_index: id, object_type, object_id, embedding (vector), metadata (jsonb), created_at
- audit_logs: id, user_id, action, resource_type, resource_id, ip, user_agent, created_at
- api_tokens: id, user_id, hashed_token, scopes, expires_at, created_at

### 10.3 Data Flow Diagrams (ASCII)
[Upload Flow]
User -> API -> Object Store (checkpoint)
               -> Postgres (runs/checkpoints)
[Compute Flow]
API -> Redis Queue -> Worker -> Object Store (artifacts)
                               -> Postgres (metric rows)
                               -> Vector Store (embedding)

### 10.4 Input Data & Dataset Requirements
- Supported checkpoints: PyTorch (.pt/.pth with state_dict), TensorFlow (.ckpt/.h5), ONNX (.onnx).
- Optional sample datasets for sensitivity/robustness:
  - Vision: ImageNet-like subsets, CIFAR-10, custom image folders.
  - NLP: GLUE subsets, SST-2, custom text datasets via Hugging Face Datasets.
  - Tabular/time-series: CSV/Parquet with schema definitions and preprocessing functions.
- Dataset slices: JSON spec defining filter predicates, transforms, batch size, and device placement.
- Privacy: Avoid uploading sensitive or personal data; enable data masking/anonymization hooks.

## 11. API Specifications
### 11.1 REST Endpoints (v1)
- POST /v1/auth/login (if using password grant or token exchange)
- GET /v1/users/me
- GET/POST /v1/projects
- GET/POST /v1/projects/{project_id}/runs
- POST /v1/runs/{run_id}/checkpoints: multipart upload or signed URL
- GET /v1/runs/{run_id}/checkpoints
- GET /v1/checkpoints/{ckpt_id}/layers
- POST /v1/checkpoints/{ckpt_id}/compute/weights: body={config}
- POST /v1/checkpoints/{ckpt_id}/compute/sensitivity: body={config}
- POST /v1/checkpoints/{ckpt_id}/compute/robustness: body={config}
- POST /v1/checkpoints/{ckpt_id}/compute/similarity: body={config}
- GET /v1/layers/{layer_id}/weight-stats?config_hash=...
- GET /v1/layers/{layer_id}/sensitivity?config_hash=...
- GET /v1/runs/{run_id}/similarity
- POST /v1/checkpoints/{ckpt_id}/prune: body={method, sparsity_target}
- GET /v1/pruning/{prune_id}
- POST /v1/vector/search: body={object_type, query_embedding, top_k}
- GET /v1/exports/report?run_id=...&format=pdf
- GET /v1/metrics/diff?left_ckpt=...&right_ckpt=...
- POST /v1/dataset-slices
- GET /v1/dataset-slices
- POST /v1/webhooks
- GET /v1/jobs/{job_id}/status
- WS /v1/stream/progress

### 11.2 Request/Response Examples
Example: Compute spectral metrics
Request:
POST /v1/checkpoints/123/compute/weights
{
  "layers": ["encoder.layers.0.attn.q_proj.weight", "encoder.layers.0.mlp.fc1.weight"],
  "metrics": ["spectral_norm", "effective_rank", "spectrum"],
  "svd": {"k": 128, "solver": "randomized"},
  "cache": true
}
Response:
202 Accepted
{
  "job_id": "job_abc123",
  "estimated_seconds": 12
}

Example: Retrieve layer weight stats
GET /v1/layers/987/weight-stats?config_hash=8a1f...
Response:
{
  "layer_id": 987,
  "config_hash": "8a1f...",
  "l1": 1234.56,
  "l2": 78.90,
  "fro": 81.23,
  "spectral_norm": 2.345,
  "cond_num": 154.2,
  "eff_rank": 57.4,
  "sparsity": 0.32,
  "spectrum_uri": "s3://bucket/artifacts/spectra/987_8a1f.npy",
  "created_at": "2025-11-25T10:15:00Z"
}

Example: Vector search similar layers
POST /v1/vector/search
{
  "object_type": "layer",
  "query_embedding": [0.12, -0.05, ...],
  "top_k": 5
}
Response:
{
  "results": [
    {"object_id": 987, "score": 0.92, "metadata": {"run":"r1","layer":"enc.3.fc1"}},
    {"object_id": 654, "score": 0.89, "metadata": {"run":"r4","layer":"enc.2.fc2"}}
  ]
}

Python SDK snippet (FastAPI client):
import requests

token = "Bearer YOUR_TOKEN"
base = "https://aiml031.example.com/v1"
headers = {"Authorization": token}

# Submit sensitivity job
resp = requests.post(f"{base}/checkpoints/123/compute/sensitivity",
                     json={"layers":["layer1.weight"], "method":"fisher_diag", "batch_size":32, "dataset_slice_id": 42},
                     headers=headers)
job_id = resp.json()["job_id"]

# Poll status
status = requests.get(f"{base}/jobs/{job_id}/status", headers=headers).json()
print(status)

### 11.3 Authentication
- OAuth2/OIDC with PKCE for web.
- Bearer tokens for API; scopes: read:metrics, write:jobs, admin:*.
- Optional service accounts for CI with rotating tokens.
- CSRF protection for session-based flows; CORS configured for web app domain.

## 12. UI/UX Requirements
### 12.1 User Interface
- Pages:
  - Home/Projects: list and search projects/runs, tags.
  - Run Detail: high-level metrics, checkpoint timeline.
  - Layer Explorer: heatmaps, histograms, per-layer stats table; filter by type/size.
  - Sensitivity Panel: gradient norms, Fisher/Hessian proxies; configuration sidebar.
  - Explainability Panel: saliency/Integrated Gradients viewer with linked layer sensitivity overlays.
  - Robustness Workbench: adversarial sweeps, noise injection; plots of accuracy vs. epsilon; gradient norms.
  - Pruning Playground: method selection, sparsity sliders, before/after metrics and mask visualization.
  - Similarity Viewer: CKA matrices, PCA scatter; epoch evolution.
  - Compare: side-by-side diff with synchronized cursors and linked brushing.
  - Exports & Reports: build and download.
  - Settings: integrations (MLflow/W&B), tokens, dataset slices.
- Chart interactions: hover tooltips, zoom/pan, select, linked highlights across panels.

### 12.2 User Experience
- Guided wizards for first-time compute jobs with sensible defaults.
- Non-blocking operations: background tasks with progress bars and ETA.
- Keyboard shortcuts for navigation and toggle layers; command palette.
- Persistent filters and panel layouts per user/workspace.

### 12.3 Accessibility
- WCAG 2.2 AA contrast and keyboard navigation.
- Alt text for images; ARIA labels for graphs.
- Colorblind-safe palettes, adjustable font sizes.

## 13. Security Requirements
### 13.1 Authentication
- OAuth2/OIDC with MFA support via IdP.
- Refresh token rotation; device-bound sessions.

### 13.2 Authorization
- RBAC: roles (viewer, editor, admin) at team/project level.
- Fine-grained scopes per API token.
- Row-level access control for projects/runs.

### 13.3 Data Protection
- TLS 1.2+ in transit.
- Server-side encryption for object store; AES-256 at rest.
- Secrets management via Kubernetes Secrets/HashiCorp Vault.
- Signed URLs for artifact download with time-limited access.
- Audit logs for create/update/delete, export, and permission changes.

### 13.4 Compliance
- Best practices aligned with SOC 2 Type II, ISO 27001.
- GDPR readiness: data subject rights, data export/deletion; data residency options.
- HIPAA-ready deployment option (no PHI by default).

## 14. Performance Requirements
### 14.1 Response Times
- p95 latency:
  - GET metrics (cached): <500 ms
  - POST job submission: <300 ms
  - Websocket updates: <200 ms push delay
  - Export retrieval: <2 s for recent artifacts

### 14.2 Throughput
- API: ≥ 200 RPS sustained on 2 replicas.
- Queue: ≥ 5k jobs/day baseline; scalable to 50k/day.

### 14.3 Resource Usage
- GPU workers: configurable; target 70–85% utilization under load.
- Memory: cap per worker process; model offloading when possible.
- Storage: artifacts lifecycle policy; default retention 90 days.

## 15. Scalability Requirements
### 15.1 Horizontal Scaling
- Scale API and workers independently via HPA (CPU/GPU/queue length).
- Stateless API; Redis/Postgres pooled connections.

### 15.2 Vertical Scaling
- Worker node types from T4/A10/A100; auto-select kernels and batch sizes.

### 15.3 Load Handling
- Backpressure in queue; rate limiting per user.
- Priority queues for interactive vs. batch jobs.

## 16. Testing Strategy
### 16.1 Unit Testing
- Core metric computations validated with synthetic tensors and invariance tests.
- Numerical accuracy checks vs. NumPy/Scipy references.

### 16.2 Integration Testing
- End-to-end flows: upload -> compute -> fetch -> export.
- Multi-framework loaders (PyTorch/TF/ONNX).
- Auth and RBAC, dataset slice ingestion.

### 16.3 Performance Testing
- Load tests: API and job throughput.
- GPU benchmarking for spectra, Fisher, adversarial sweeps on standard models (ResNet50, BERT-base).

### 16.4 Security Testing
- Static analysis (Bandit, Semgrep).
- Dependency scanning (pip-audit).
- Pen tests on auth and artifact access; fuzz test APIs.

## 17. Deployment Strategy
### 17.1 Deployment Pipeline
- GitHub Actions: lint/test/build Docker images -> push to registry -> Helm deploy to dev/staging/prod.
- Infrastructure as Code (Helm charts under infra/).

### 17.2 Environments
- Dev: single-node, minimal GPU optional.
- Staging: mirrors prod; test data; chaos experiments.
- Prod: multi-AZ cluster; autoscaling; backups enabled.

### 17.3 Rollout Plan
- Phased rollout by team; feature flags for advanced features.
- Canary releases for API and workers.

### 17.4 Rollback Procedures
- Helm rollback to previous revision.
- Database migrations reversible; backup/restore runbooks.
- Artifact versioning retained for reprocessing.

## 18. Monitoring & Observability
### 18.1 Metrics
- API: requests/sec, p95 latency, error rate.
- Workers: job duration, GPU utilization, VRAM usage, batch size.
- Queue: length, wait time, retries.
- Business: #active users, #runs, #exports.

### 18.2 Logging
- Structured JSON logs with correlation IDs.
- PII scrubbing; retention per policy.

### 18.3 Alerting
- On-call alerts: API error rate >2%, queue wait >5 min, GPU utilization <30% for 30 min (underutilization), job failure rate >5%.
- PagerDuty/Slack integrations.

### 18.4 Dashboards
- Grafana: API SLOs, worker performance, storage growth, vector search latency.
- Kibana/OpenSearch optional for logs.

## 19. Risk Assessment
### 19.1 Technical Risks
- High compute costs for SVD/Hessian proxies on large models.
- Heterogeneous framework checkpoints causing loader failures.
- Numerical instability in spectra/effective rank.
- UI complexity leading to performance issues on large datasets.

### 19.2 Business Risks
- Low adoption if perceived as research-only.
- Data sensitivity concerns limit dataset uploads.
- Vendor lock-in concerns for vector store or auth.

### 19.3 Mitigation Strategies
- Offer truncated/randomized SVD with error bounds; batching strategies.
- Robust checkpoint parsing with schema introspection and clear errors.
- Numerical stabilization (epsilon, log scaling) and unit tests.
- Client-side virtualization; progressive rendering; caching layers.
- Support self-hosting and pluggable vector/auth providers.
- Local compute option: lightweight agent executing jobs within user VPC.

## 20. Timeline & Milestones
### 20.1 Phase breakdown
- Phase 0 (2 weeks): Requirements finalization, architecture/design docs.
- Phase 1 (6 weeks): Backend MVP (ingest, weight stats, spectra, caching), basic UI (Layer Explorer), MLflow integration.
- Phase 2 (6 weeks): Sensitivity metrics, explainability linkage, exports, API/SDK.
- Phase 3 (6 weeks): Robustness workbench, pruning playground, comparison tools, vector search.
- Phase 4 (4 weeks): Hardening, performance, security, observability, staging validation.
- Phase 5 (2 weeks): Production rollout, documentation, training.

Total: ~26 weeks.

### 20.2 Key Milestones
- M1: Ingestion + weight heatmaps live (end Phase 1).
- M2: Sensitivity panel with Fisher/Hutchinson (end Phase 2).
- M3: Robustness + pruning features (end Phase 3).
- M4: SLOs met; SOC2-aligned controls; 99.5% uptime in staging (end Phase 4).
- GA: Production release with onboarding kit (end Phase 5).

Estimated costs (6 months):
- Team: 1 PM, 1 Designer, 4 Backend/ML, 2 Frontend, 1 MLOps (~$1.6M fully loaded).
- Cloud/GPU: $8–15k/month depending on usage (A10/A100 mix).

## 21. Success Metrics & KPIs
### 21.1 Measurable targets
- Reliability: ≥99.5% uptime; API error rate <1%.
- Performance: cached requests p95 <500 ms; vector search p95 <300 ms; medium-model spectra <5 s per layer group.
- Adoption: ≥50 projects, ≥100 MAUs within 6 months.
- Engagement: median session length >10 min; ≥5 panels used per session.
- Impact: ≥30% reduction in debugging time (survey), ≥10% robustness improvement on benchmark sweeps post-intervention.
- Quality: numerical relative error vs. reference <1e-6; unit test coverage ≥85%.
- Exports: ≥200/month within 3 months.

## 22. Appendices & Glossary
### 22.1 Technical Background
- Weight metrics:
  - L1/L2/Frobenius norms; spectral norm via largest singular value.
  - Effective rank r_eff = exp(H(p)), where p are normalized singular values.
  - Condition number = σ_max / σ_min (regularized).
- Sensitivity metrics:
  - Grad/Jacobian norms; Fisher diagonal approximations via gradient squares.
  - Hutchinson estimator: tr(H) ≈ (1/T) Σ v_i^T H v_i with Rademacher/Gaussian v_i.
  - Sharpness proxies: loss increase under small perturbations.
- Explainability:
  - Saliency (∂y/∂x), Integrated Gradients path integral, SmoothGrad with noise.
- Robustness:
  - FGSM (single-step), PGD (iterative) attacks; robustness curves vs. ε.
  - Lipschitz proxy via product of spectral norms.
- Pruning:
  - Magnitude pruning (threshold on |w|), SNIP (saliency-based), GraSP (gradient signal preservation).
- Representation similarity:
  - CKA/linear CKA for comparing activations across layers/epochs.

### 22.2 References
- Li et al., Visualizing the Loss Landscape of Neural Nets.
- Ghorbani et al., Interpretation of Neural Networks Is Fragile.
- Neyshabur et al., Towards Understanding the Role of Over-Parametrization and Optimization in Generalization.
- Morcos et al., On the Importance of Single Directions for Generalization.
- Hooker et al., A Benchmark for Interpretability Methods.
- Singh et al., Evaluating Representation Similarity via CKA.
- Yu et al., Playing the Lottery with Pruning: The Lottery Ticket Hypothesis.
- Keskar et al., On Large-Batch Training: Sharp Minima Can Generalize Poorly.

### 22.3 Glossary
- Checkpoint: Serialized model parameters at a given training step/epoch.
- Layer: Logical module with parameters (e.g., Linear, Conv2d, Attention).
- Spectrum: Singular/eigenvalue distribution of a weight matrix.
- Effective Rank: Entropy-based measure of matrix rank.
- Fisher Information: Curvature approximation derived from gradients.
- Hessian: Second derivative matrix of the loss with respect to parameters.
- Sharpness: Sensitivity of loss to small parameter perturbations.
- CKA: Centered Kernel Alignment for comparing representations.
- Pruning: Removing parameters to increase sparsity.
- Saliency/Integrated Gradients/SmoothGrad: Input attribution methods.

Sample configuration (YAML):
compute:
  device: "cuda:0"
  precision: "float32"
  batch_size: 64
  svd:
    k: 128
    solver: "randomized"
  hutchinson:
    trials: 50
    noise: "rademacher"
robustness:
  attack: "pgd"
  steps: 10
  epsilon: 0.03
  step_size: 0.007

FastAPI route example:
from fastapi import APIRouter, Depends, HTTPException
from schemas import ComputeWeightsRequest
router = APIRouter()

@router.post("/v1/checkpoints/{ckpt_id}/compute/weights")
async def compute_weights(ckpt_id: str, req: ComputeWeightsRequest, user=Depends(auth)):
    job_id = enqueue_weights_job(ckpt_id, req, user.id)
    return {"job_id": job_id, "estimated_seconds": estimate_eta(req)}

Frontend snippet (React, fetch + WebSocket):
const ws = new WebSocket(`${WS_BASE}/v1/stream/progress?job=${jobId}`);
ws.onmessage = (evt) => setProgress(JSON.parse(evt.data));
const res = await fetch(`/v1/layers/${layerId}/weight-stats?config_hash=${hash}`, { headers: { Authorization: token }});
const data = await res.json();
renderSpectrum(data.spectrum_uri);

End of PRD.