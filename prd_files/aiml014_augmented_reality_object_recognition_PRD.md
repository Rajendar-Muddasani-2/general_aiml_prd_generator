# Product Requirements Document (PRD)
# `Aiml014_Augmented_Reality_Object_Recognition`

Project ID: aiml014  
Category: Computer Vision, AR, On-device ML  
Status: Draft for Review  
Version: 1.0.0  
Last Updated: 2025-11-25

## 1. Overview
### 1.1 Executive Summary
aiml014 delivers an on-device augmented reality (AR) object recognition and 6DoF pose estimation system enabling real-time identification, tracking, and overlay of digital content onto physical objects. It combines mobile AR frameworks (ARKit/ARCore), efficient deep learning models for detection/segmentation, visual-inertial odometry (VIO/SLAM), and robust 3D pose pipelines. The solution operates primarily on-device for privacy and low latency, with optional cloud services for catalog sync, analytics, and model updates.

### 1.2 Document Purpose
Define comprehensive product requirements spanning functionality, user experience, data/model pipelines, system architecture, APIs, security, performance, scalability, testing, deployment, monitoring, risks, and timelines to guide cross-functional teams (Product, Engineering, ML, QA, DevOps, Design).

### 1.3 Product Vision
Empower users to point a mobile device or AR headset at real-world objects and instantly see accurate, stable, and context-aware overlays: names, instructions, metadata, and guided steps. The system feels instantaneous and reliable across environments, supports offline usage, and improves continuously via privacy-preserving learning.

## 2. Problem Statement
### 2.1 Current Challenges
- Existing AR apps struggle with reliable recognition across lighting, occlusion, and textureless surfaces.
- Overlays jitter due to unstable pose estimates and tracking drift.
- Latency on commodity devices leads to poor UX and user drop-off.
- Solutions often require network connectivity, raising privacy and resilience concerns.
- Catalog management and continuous model updates are fragmented.

### 2.2 Impact Analysis
- Inaccurate or delayed overlays reduce trust and increase task time.
- High battery and thermal impact shortens sessions and user satisfaction.
- Lack of offline capability hinders field use cases.
- Inconsistent object identity across frames frustrates multi-step guidance.

### 2.3 Opportunity
- Deliver a robust, privacy-first AR recognition stack that runs in real-time on commodity devices, supports multimodal queries, and enables enterprise use cases (field assistance, training, guided assembly, retail) and consumer use cases (education, hobbyist tool identification).

## 3. Goals and Objectives
### 3.1 Primary Goals
- Real-time on-device object detection/recognition and 6DoF pose with stable overlays.
- Robustness across lighting, motion blur, occlusion, and texture variability.
- Offline-first operation with optional cloud augmentation and analytics.
- Easy catalog management and incremental updates.

### 3.2 Business Objectives
- Reduce task completion time in AR-guided workflows by ≥30%.
- Increase user retention by ≥20% vs baseline AR app without recognition.
- Enable enterprise subscriptions and SDK licensing.

### 3.3 Success Metrics
- Recognition mAP@0.5 ≥ 0.60 on target catalog; top-1 identification accuracy ≥ 90%.
- Pose accuracy: 80% objects with ADD-S ≤ 5 cm or 5% of object diameter; 2D reprojection error ≤ 3 px median.
- AR overlay stability: <1° rotational jitter RMS; <1 cm positional jitter RMS on supported devices.
- Latency: <120 ms per frame end-to-end on mid-tier device; cloud API P95 <300 ms.
- Uptime: ≥99.5% backend; Crash-free sessions ≥99.8%.
- Battery: ≤1.5 W average additional draw during recognition on supported devices.

## 4. Target Users/Audience
### 4.1 Primary Users
- Field technicians using AR-guided identification and steps.
- Retail associates and customers for product info overlays.
- Hobbyists/learners identifying tools/instruments.
- Developers integrating SDK into their AR apps.

### 4.2 Secondary Users
- Catalog managers and curators.
- ML Ops engineers managing models and indices.
- Product managers analyzing usage metrics.

### 4.3 User Personas
- Persona 1: Maya Chen, Field Technician
  - Background: 6 years maintaining HVAC systems; uses company-issued Android phone.
  - Pain points: Time lost searching manuals; poor connectivity in basements; gloves limit touch interaction.
  - Goals: Quick identification and step-by-step overlays; offline reliability; voice prompts.
- Persona 2: Luis Romero, Retail Associate
  - Background: Works in consumer electronics; assists customers on the floor.
  - Pain points: Keeping up with specs; mismatched products; crowded store lighting reflections.
  - Goals: Instant product info; “compare” overlays; durable battery during shifts.
- Persona 3: Priya Nair, AR App Developer
  - Background: Unity/AR Foundation developer building educational apps.
  - Pain points: Integrating robust recognition; balancing frame rate vs accuracy; cross-platform issues.
  - Goals: Easy SDK; sample scenes; configurable models; observability; CI/CD support.
- Persona 4: Omar Ali, Catalog Manager
  - Background: Manages 5k+ product SKUs with images and CAD models.
  - Pain points: Versioning; duplicates; inconsistent labels; cold starts for new items.
  - Goals: Bulk import; active learning suggestions; approval workflow; analytics.

## 5. User Stories
- US-001: As a field technician, I want the app to identify equipment and show overlays within 500 ms so that I can proceed without delays.
  - Acceptance: 95% of identifications under 500 ms; confidence ≥0.7 triggers overlay.
- US-002: As a user, I want stable overlays while I move so that the instructions remain readable.
  - Acceptance: Positional jitter <1.5 cm RMS; overlay drift <2% of object size over 10 s.
- US-003: As a developer, I want an SDK with simple APIs to start/stop recognition and get callbacks so that I can integrate quickly.
  - Acceptance: <50 LOC integration; sample app builds in <15 min.
- US-004: As a catalog manager, I want to upload new items and generate embeddings automatically so that they are discoverable in AR.
  - Acceptance: Upload to live index <10 minutes; validation reports generated.
- US-005: As a user, I want recognition to work offline so that I can use it in areas without connectivity.
  - Acceptance: Core features available with cached catalog; graceful degradation with notifications.
- US-006: As a user, I want to ask “find the 10mm wrench” so that the app highlights matching objects.
  - Acceptance: Text-to-vision query returns candidates with IoU ≥0.5 in top-3 85% of time.
- US-007: As an admin, I want analytics on recognition accuracy and latency per device so that I can improve models.
  - Acceptance: Dashboard with filters; daily aggregates; P90/P95 latency charts.

## 6. Functional Requirements
### 6.1 Core Features
- FR-001: Real-time 2D object detection on-device with bounding boxes and class labels.
- FR-002: 6DoF object pose estimation using 2D–3D correspondences and PnP with RANSAC; ICP refinement when depth is available.
- FR-003: Visual-inertial tracking and SLAM integration to maintain world/camera pose and persistent anchors.
- FR-004: Instance segmentation masks to improve occlusion and overlay placement.
- FR-005: On-device embedding search for catalog recognition using ANN (e.g., HNSW).
- FR-006: Temporal tracking across frames with optical flow and re-ID to stabilize identities.
- FR-007: Offline mode: local catalog, models, and indices; sync when online.
- FR-008: SDK APIs: start/stop, configure models, query results, subscribe to events, anchor management.
- FR-009: Admin dashboard for catalog management, model versions, and analytics.
- FR-010: Privacy controls, telemetry opt-in, and data export.

### 6.2 Advanced Features
- FR-011: Multimodal retrieval: text queries mapped to vision embeddings (CLIP-like).
- FR-012: Spatially aware retrieval prioritizing likely candidates by pose/FOV.
- FR-013: Active learning loop: flag low-confidence clusters for human labeling.
- FR-014: Domain adaptation and test-time adaptation to new environments.
- FR-015: Power-aware scheduling: dynamic FPS and model early-exit.

## 7. Non-Functional Requirements
### 7.1 Performance
- On-device pipeline latency: P90 ≤120 ms/frame on target devices; ≥24 FPS sustained.
- Backend API P95 latency ≤300 ms for recognition calls; ≤2 s for bulk catalog queries.

### 7.2 Reliability
- Backend uptime ≥99.5%; SDK crash-free sessions ≥99.8%.
- Anchor persistence across sessions: ≥95% re-localization success within 2 s.

### 7.3 Usability
- First-time experience under 2 minutes to setup offline pack.
- SDK documentation completeness score ≥90% in developer survey.

### 7.4 Maintainability
- >80% unit test coverage in core libraries; CI within 10 minutes.
- Modular model registry with semantic versioning; backward-compatible SDK for 12 months.

## 8. Technical Requirements
### 8.1 Technical Stack
- Mobile:
  - iOS: Swift 5.9+, iOS 16+, ARKit 6+, Core ML 3+, Metal.
  - Android: Kotlin 1.9+, Android 10+, ARCore 1.41+, TensorFlow Lite 2.16+, NNAPI.
  - Unity 2022.3 LTS with AR Foundation 5.1+ (optional SDK flavor).
- Backend:
  - Python 3.11+, FastAPI 0.115+, Uvicorn.
  - PyTorch 2.4+/TorchVision 0.19+ for training.
  - ONNX 1.16+, ONNX Runtime 1.19+; TensorRT 10.x (optional).
  - Faiss 1.8+ or HNSWlib 0.8+ for ANN.
  - PostgreSQL 15+, Redis 7+, object storage (S3-compatible).
  - Kubernetes 1.29+, Helm 3+, NGINX Ingress, Istio (optional).
- Web:
  - React 18+, TypeScript 5+, Vite.
  - TailwindCSS 3+; Chart.js 4+.
- MLOps:
  - MLflow 2.14+; DVC 3+; Weights & Biases (optional).
  - GitHub Actions; ArgoCD.

### 8.2 AI/ML Components
- Detection backbones: EfficientDet-Lite0 or YOLOv8n (quantized int8); lightweight FPN.
- Segmentation: Lightweight Mask head (e.g., YOLOv8n-seg) or MobileMask R-CNN variant.
- Feature extraction: MobileNetV3-Large/ViT-Tiny embeddings; SuperPoint+SuperGlue (optional) for keypoints.
- Pose estimation: PnP with EPnP + RANSAC; depth-assisted ICP; fallback: CAD-based pose nets for textureless items.
- Tracking: RAFT-lite optical flow or KLT; Kalman filter for pose smoothing; Siamese re-ID for multi-object.
- Depth: Device depth API/meshing; monocular depth (MiDaS-small) if no hardware depth.
- Retrieval: HNSW on-device; cosine similarity in normalized embedding space.
- Training: Self-supervised pretraining; knowledge distillation; quantization-aware training; pruning.
- Metrics: mAP/IoU; ADD(-S); 2D reprojection error; FPS/latency; battery and memory footprint.

## 9. System Architecture
### 9.1 High-Level Architecture (ASCII)
+--------------------+       +-------------------+       +--------------------+
|     Mobile AR App  |       |   API Gateway     |       |   Admin Dashboard  |
|  (iOS/Android/AR)  |<----->|  (FastAPI/HTTPS)  |<----->|   (React Web)      |
+---------+----------+       +---------+---------+       +---------+----------+
          |                            |                           |
          v                            v                           v
  +-------+--------+          +--------+--------+         +--------+---------+
  | On-Device ML   |          | Catalog Service |         | Analytics Service |
  | - Detector     |<-------->| - Items/Embeds  |<------->| - Metrics/Events  |
  | - Segmentation | (sync)   | - Versions      |         | - Dashboards      |
  | - Embedding NN |          +--------+--------+         +--------+---------+
  | - ANN (HNSW)   |                   |                            |
  | - Pose/Tracking|                   v                            v
  +-------+--------+          +--------+--------+         +--------+---------+
          |                   |  Model Registry |         |  Storage (S3)    |
          v                   +-----------------+         +------------------+
  +-------+--------+
  | AR Engine      |  (ARKit/ARCore: SLAM, IMU, depth, anchors)
  +----------------+

Data paths:
- On-device: Camera/IMU -> Detection/Seg -> Pose -> Anchors -> Render
- Sync: Catalog/models/index <-> Backend (when online)

### 9.2 Component Details
- Mobile AR App: Captures frames and IMU; invokes on-device ML; manages anchors; renders overlays.
- On-Device ML: Efficient models for detection, segmentation, embeddings; ANN index; pose/tracking module; energy-aware scheduler.
- AR Engine: Provides camera pose, world mapping, depth/meshes, anchors.
- API Gateway: AuthN/Z, rate limiting, routing to services.
- Catalog Service: CRUD for items, embeddings, versions; index builds; bulk ingest.
- Model Registry: Stores model artifacts, metadata, A/B rollout configs.
- Analytics Service: Collects anonymized telemetry; aggregates metrics.
- Storage: Assets (images, CAD, meshes), datasets, logs.

### 9.3 Data Flow
1) Capture: Camera frame + IMU -> AR engine (VIO) -> camera pose.  
2) Perception: Detector -> boxes/classes; segmentation -> masks; feature extractor -> embeddings.  
3) Retrieval: ANN search on device -> candidates; optional geometric verification with RANSAC.  
4) Pose: 2D–3D correspondences -> PnP; refine with depth ICP; Kalman smoothing.  
5) Tracking: Optical flow propagate; re-ID maintain identities; temporal smoothing.  
6) AR: Create/update anchors; render overlays with occlusion handling from masks/depth.  
7) Sync: When online, sync catalog updates, embeddings index, and optional telemetry.

## 10. Data Model
### 10.1 Entity Relationships
- User (1..*) Sessions
- Session (1..*) Detections, Poses, Anchors, Events
- CatalogItem (1..*) Embeddings; (0..1) CADModel; (1..*) Images
- Embedding belongs to CatalogItem (with version)
- ModelVersion (1..*) Artifacts; (1..*) Deployments
- Anchor belongs to Session; references CatalogItem (optional)

### 10.2 Database Schema
- users: id (uuid), email, role, created_at
- sessions: id, user_id, device_id, started_at, ended_at
- catalog_items: id, name, sku, category, metadata(jsonb), created_at, updated_at
- item_assets: id, item_id, type(enum: image,cad,mesh), uri, checksum, created_at
- embeddings: id, item_id, model_version, vector (faiss/hnsw store external), created_at
- model_versions: id, name, type(enum: detector,segmenter,embedder), semver, uri, checksum, metrics(jsonb), created_at
- deployments: id, model_version, environment(enum: dev,staging,prod), rollout(jsonb), created_at
- analytics_events: id, session_id, type, payload(jsonb), ts
- anchors: id, session_id, item_id, pose(jsonb), ts
- api_keys: id, user_id, key_hash, scopes, created_at, revoked_at

### 10.3 Data Flow Diagrams
[On-device]
Frame -> Detector -> Boxes
Boxes + Frame -> Segmenter -> Masks
Frame -> Embedder -> Vector -> ANN -> Candidates
Candidates + Keypoints -> RANSAC -> Inliers
Inliers -> PnP -> Pose -> Kalman -> Anchor

[Backend]
Admin Upload -> Catalog Service -> Storage
Storage -> Embedding Job -> Embeddings Store -> Index Build -> Publish

### 10.4 Input Data & Dataset Requirements
- Images: multi-view per item (≥30 views), varied backgrounds, scales, lighting, occlusions.
- CAD/3D models where available for textured/textureless handling.
- Synthetic generation: Blender pipeline with domain randomization (illumination, motion blur, noise, occluders).
- Public datasets for pretraining/benchmarking: COCO, LVIS, BOP Challenge (6DoF), T-LESS, YCB-Video.
- Annotations: bounding boxes, instance masks, keypoints/landmarks, 3D CAD alignment or 2D–3D correspondences.
- Splits: train/val/test; device-specific validation set.

## 11. API Specifications
### 11.1 REST Endpoints
- POST /v1/auth/token
- GET /v1/catalog/items
- POST /v1/catalog/items
- POST /v1/catalog/items/{id}/assets
- POST /v1/catalog/index/rebuild
- GET /v1/models
- POST /v1/models/deploy
- GET /v1/analytics/metrics
- POST /v1/telemetry/events
- GET /v1/offline/packs/latest
- GET /v1/offline/packs/{platform}/{version}

Mobile-local SDK interfaces (pseudo):
- startRecognition(config)
- stopRecognition()
- onResults(callback: DetectionResult[])
- queryByText(text): Result[]
- manageAnchors(create/update/delete/list)
- preloadOfflinePack(version)

### 11.2 Request/Response Examples
- Example: Get catalog items
Request:
GET /v1/catalog/items?limit=20&query=wrench
Authorization: Bearer <token>

Response:
200 {
  "items": [
    {"id":"it_123","name":"10mm Wrench","sku":"WR-10","category":"Tools","metadata":{"brand":"Acme"}}
  ],
  "next_cursor":"abc123"
}

- Example: Upload asset
POST /v1/catalog/items/it_123/assets
Body: {"type":"image","uri":"s3://bucket/path/wrench_01.jpg","checksum":"..."}
Response 201 { "id":"asset_456" }

- Example: Fetch offline pack
GET /v1/offline/packs/ios/1.2.0
Response 200 { "models":[...], "catalog":{"version":"2025.11.01","items":1234}, "index_uri":"..." }

- Example: Telemetry
POST /v1/telemetry/events
Body: {"session_id":"sess_1","type":"latency","payload":{"p90":95,"device":"iPhone 14"},"ts":"..."}

### 11.3 Authentication
- OAuth2 with JWT bearer tokens for users; API keys for service-to-service.
- Mobile SDK retrieves short-lived tokens via PKCE.
- Scopes: catalog:read/write, models:read/write, telemetry:write, analytics:read.

## 12. UI/UX Requirements
### 12.1 User Interface
- Mobile AR:
  - Minimal overlay HUD: recognition indicator, confidence, “lock” status.
  - Toggle: offline pack status, battery saver mode.
  - Object card: name, attributes, actions (instructions, compare, add note).
  - Visual “halo” highlighting recognized objects.
- Admin Dashboard:
  - Catalog table with search, filters, bulk actions.
  - Model registry view: versions, metrics, rollout status.
  - Analytics: latency, accuracy, session counts, device breakdown.

### 12.2 User Experience
- One-tap to start AR recognition; haptic feedback on lock.
- Progressive enhancement: coarse box -> refined mask -> pose lock.
- Graceful degradation: text prompts (“move closer,” “reduce glare”); offline banners.
- Voice prompts (optional) and large UI targets for gloved users.

### 12.3 Accessibility
- WCAG 2.1 AA compliance for dashboard.
- High-contrast overlays; adjustable font size in mobile app.
- VoiceOver/TalkBack labels for UI elements.
- Color-blind safe palettes.

## 13. Security Requirements
### 13.1 Authentication
- OAuth2/OIDC; MFA optional for admin roles; short-lived tokens; token revocation.

### 13.2 Authorization
- RBAC: roles (user, developer, admin, catalog_manager, analyst).
- Resource-level permissions on catalog items and models.

### 13.3 Data Protection
- TLS 1.3 in transit; AES-256 at rest.
- On-device data sandboxed; optional app-level encryption for offline packs.
- PII minimization; data retention policies; secure key management (KMS).

### 13.4 Compliance
- GDPR/CCPA readiness: consent management, data access/export, deletion.
- SOC 2 controls across change management, security monitoring, backup/restore.

## 14. Performance Requirements
### 14.1 Response Times
- On-device:
  - Detection+segmentation inference ≤60 ms/frame on target devices.
  - Retrieval and pose estimation ≤40 ms/frame average.
  - Total end-to-end ≤120 ms P90.
- Backend:
  - Catalog CRUD ≤200 ms P95.
  - Offline pack download initiates ≤1 s; throughput ≥10 MB/s on broadband.

### 14.2 Throughput
- Backend supports ≥500 RPS total across endpoints; autoscaling to 2,000 RPS.

### 14.3 Resource Usage
- Mobile memory budget: ≤350 MB peak additional; model bundle ≤150 MB.
- CPU/GPU utilization targets: <70% sustained on flagship; thermal throttling avoidance via duty-cycling.

## 15. Scalability Requirements
### 15.1 Horizontal Scaling
- Stateless API pods behind HPA; scale on CPU/RPS.
- Vector index service shards by catalog namespace.

### 15.2 Vertical Scaling
- Batch embedding jobs scale vertically for throughput; GPU-enabled nodes for training/inference services.

### 15.3 Load Handling
- Rate limiting per API key; circuit breakers; backpressure on telemetry ingestion.
- CDN for static packs and assets.

## 16. Testing Strategy
### 16.1 Unit Testing
- Coverage >80% for SDK core modules, pose pipeline, ANN retrieval wrappers.
- Deterministic seeds for model unit tests with golden outputs.

### 16.2 Integration Testing
- End-to-end mobile tests with recorded scenes; HIL tests on device farm.
- Backend integration: API-contract tests; data migrations; index build pipelines.

### 16.3 Performance Testing
- On-device benchmarks across representative hardware matrices.
- Load tests (k6/Locust) to 2k RPS; chaos testing for failure modes.

### 16.4 Security Testing
- Static analysis (SAST), dependency scanning.
- Dynamic app security tests; API fuzzing; pen-testing annually.

## 17. Deployment Strategy
### 17.1 Deployment Pipeline
- GitHub Actions:
  - Lint/test -> build -> containerize -> push to registry -> Helm deploy to dev/staging/prod.
  - Mobile SDK: build CI for iOS/Android; publish to internal repos.
  - Model artifacts: MLflow register -> smoke tests -> canary rollout.

### 17.2 Environments
- Dev: ephemeral namespaces.
- Staging: mirrors prod data schemas; limited traffic.
- Prod: multi-region active-active for APIs; object storage with cross-region replication.

### 17.3 Rollout Plan
- Feature flags for SDK capabilities.
- Phased model rollouts: 5% canary -> 25% -> 100% with rollback gates based on metrics (latency, accuracy).

### 17.4 Rollback Procedures
- Blue/green deployments; instant switch to prior version.
- SDK feature toggles remotely configurable.
- Model registry supports pinning previous versions; index rollback via snapshots.

## 18. Monitoring & Observability
### 18.1 Metrics
- On-device: FPS, per-stage latency, memory, thermal events; battery impact.
- Backend: RPS, latency (P50/P90/P95), error rates, saturation.
- ML: recognition accuracy by segment, false positive rate, pose error distributions.

### 18.2 Logging
- Structured JSON logs; correlation IDs; OpenTelemetry traces across services.

### 18.3 Alerting
- SLO-based alerts: uptime <99.5%, latency P95 > thresholds, error spikes.
- ML drift alerts: accuracy drop >5% week-over-week; embedding distribution shifts.

### 18.4 Dashboards
- Grafana dashboards: system health, API performance, ML metrics.
- Product analytics: adoption, retention, session durations.

## 19. Risk Assessment
### 19.1 Technical Risks
- Latency spikes on mid/low-tier devices.
- Lighting, reflections, or textureless surfaces reducing accuracy.
- Drift in SLAM causing anchor misalignment.
- Battery/thermal constraints.

### 19.2 Business Risks
- Catalog onboarding effort; lack of coverage for long-tail items.
- Privacy concerns restricting telemetry.
- Fragmentation across device capabilities.

### 19.3 Mitigation Strategies
- Quantization/pruning; ROI-cropping; early-exit heads; adaptive FPS.
- Synthetic data with domain randomization; CAD-based pose nets for textureless.
- Keyframe re-localization; pose smoothing; depth-based occlusion.
- Offline-first design; clear privacy controls; tiered device support.

## 20. Timeline & Milestones
### 20.1 Phase breakdown
- Phase 0 (2 weeks): Discovery, requirements, device matrix, data procurement plan.
- Phase 1 (6 weeks): Prototype on-device pipeline (detector+pose+tracking), basic SDK; initial datasets.
- Phase 2 (6 weeks): Segmentation integration, embedding retrieval, offline packs, admin MVP.
- Phase 3 (4 weeks): Robustness (augmentations, domain adaptation), analytics, dashboards.
- Phase 4 (4 weeks): Hardening, security, performance tuning, device farm tests.
- Phase 5 (2 weeks): Beta rollout, documentation, developer portal.
Total: 24 weeks.

Estimated costs (6 months):
- Team: 2 CV/ML engineers, 2 mobile engineers, 1 backend engineer, 1 DevOps, 1 designer, 1 PM, 0.5 QA: ~$1.2M loaded.
- Cloud/training/infrastructure: ~$80k.
- Device farm/testing: ~$20k.

### 20.2 Key Milestones
- M1 (Week 6): 2D detection at ≥20 FPS on target device.
- M2 (Week 12): 6DoF pose with ADD-S ≤ 7 cm for 70% cases; offline packs v1.
- M3 (Week 16): Segmentation-based occlusion; top-1 ≥90% on pilot catalog.
- M4 (Week 20): End-to-end latency ≤120 ms P90; admin dashboard GA.
- M5 (Week 24): Beta release; SDK v1.0; SLAs met.

## 21. Success Metrics & KPIs
### 21.1 Measurable targets
- Accuracy: mAP@0.5 ≥ 0.60; top-1 ≥ 90%; pose reprojection error ≤3 px median.
- Latency: On-device end-to-end ≤120 ms P90; backend uptime ≥99.5%.
- Stability: Positional jitter <1 cm RMS for ≥80% sessions.
- Adoption: ≥10 pilot customers; SDK integration time ≤1 day for 80% devs.
- Engagement: Session length ≥6 minutes average; repeat usage after 7 days ≥35%.

## 22. Appendices & Glossary
### 22.1 Technical Background
- End-to-end AR pipeline: camera capture → VIO/SLAM → world pose → detection/segmentation → embeddings/retrieval → 6DoF pose → anchors → rendering/occlusion.
- Markerless recognition via hybrid approach: deep detectors for coarse localization; local features and geometric verification; CAD-based pose nets for low-texture objects.
- 3D-aware perception: depth maps/meshing for occlusion, ICP for pose stability, normals for alignment.
- Temporal stability: optical flow propagation, multi-object tracking with re-ID, Kalman smoothing.
- Efficiency: compact backbones, quantization/pruning, knowledge distillation, ROI-cropping, early-exit heads.
- Robustness: synthetic data, heavy augmentation, test-time adaptation; temporal coherence with cache and majority voting.

### 22.2 References
- COCO, LVIS datasets for detection/segmentation.
- BOP Challenge, T-LESS, YCB-Video for 6DoF pose.
- MiDaS for monocular depth; RAFT for optical flow.
- PnP (EPnP), RANSAC, ICP algorithms.
- CLIP for multimodal embeddings; HNSW for ANN.

### 22.3 Glossary
- 6DoF: Six degrees of freedom (3D position + orientation).
- Anchor: Persistent reference in AR world space tied to real-world location.
- ANN: Approximate nearest neighbor search for fast similarity.
- ARKit/ARCore: Mobile AR frameworks providing tracking, depth, and anchors.
- CAD: 3D object model used for rendering and pose estimation.
- CLIP: Model mapping text and images into a shared embedding space.
- ICP: Iterative closest point algorithm for aligning 3D data.
- IoU: Intersection-over-Union, overlap metric for boxes/masks.
- Kalman filter: Algorithm to estimate system state from noisy measurements.
- mAP: Mean Average Precision, detection performance metric.
- PnP: Perspective-n-Point, computes pose from 2D-3D correspondences.
- RANSAC: Robust estimator for model fitting with outlier rejection.
- SLAM/VIO: Simultaneous localization and mapping / visual-inertial odometry.

Repository structure (mono-repo suggestion):
- README.md
- notebooks/
  - exploration/
  - training/
- src/
  - mobile/
    - ios/
    - android/
    - unity/
  - sdk/
    - core/
    - tracking/
    - pose/
    - segmentation/
    - embeddings/
  - backend/
    - api/
    - services/
      - catalog/
      - analytics/
      - model_registry/
  - ml/
    - models/
    - data/
    - pipelines/
  - web/
    - admin-dashboard/
- tests/
  - unit/
  - integration/
  - device/
- configs/
  - models/
  - deployment/
  - telemetry/
- data/
  - raw/
  - processed/
  - synthetic/
- scripts/
- deployments/
  - helm/
- docs/

Example configs:
- configs/models/detector.yaml
model: yolov8n
precision: int8
input_size: [640, 640]
confidence_threshold: 0.25
nms_iou: 0.5

- configs/models/embedder.yaml
backbone: mobilenetv3_large
embedding_dim: 256
quantized: true
index:
  type: hnsw
  ef_search: 64
  M: 32

Mobile SDK code snippet (Kotlin):
val config = RecognitionConfig(
  precision = Precision.INT8,
  enableSegmentation = true,
  powerMode = PowerMode.BALANCED
)
val engine = ARRecognitionEngine(context, config)
engine.onResults { results ->
  results.forEach { r ->
    overlay.render(r.anchorId, r.pose, r.mask, r.label, r.confidence)
  }
}
engine.start()

Backend FastAPI snippet (Python):
from fastapi import FastAPI, Depends
from pydantic import BaseModel

app = FastAPI()

class CatalogItem(BaseModel):
    name: str
    sku: str
    category: str
    metadata: dict = {}

@app.post("/v1/catalog/items")
def create_item(item: CatalogItem, user=Depends(authz("catalog:write"))):
    item_id = svc.create_item(item.dict())
    return {"id": item_id}

REST API auth example:
POST /v1/auth/token
Body: {"grant_type":"authorization_code","code":"...","code_verifier":"...","redirect_uri":"..."}

Evaluation script sketch:
# compute mAP/ADD-S and latency
for frame in dataset:
    t0 = time.time()
    dets = model.detect(frame)
    pose = estimate_pose(frame, dets)
    t = time.time() - t0
    latencies.append(t)
metrics = {
  "mAP50": compute_map(detections, ground_truth),
  "add_s": compute_adds(poses, gt_poses),
  "latency_p90_ms": np.percentile(latencies, 90)*1000
}

Service SLOs:
- Accuracy: ≥90% top-1 on pilot
- Latency: P95 <300 ms backend
- Uptime: ≥99.5%

This PRD establishes the blueprint for an on-device, privacy-first AR object recognition system with robust 6DoF pose estimation, multimodal retrieval, and an enterprise-ready backend for catalog and model lifecycle management.