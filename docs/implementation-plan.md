# MLOps platform implementation plan

Goal: a self-contained Compose project extending the existing sklearn/MLflow labs with automated Airflow training and alias-based FastAPI serving.

Architecture: Airflow LocalExecutor runs extract, validate/split, train, evaluate, register/promote, and serving verification. MLflow proxies artifacts to MinIO and stores tracking/registry metadata in PostgreSQL. Airflow uses a separate PostgreSQL database/user. One explicit bridge network connects all services; named volumes persist databases, objects and task logs.

Dataset: sklearn Breast Cancer Wisconsin, 569 rows, 30 features, target 0=malignant and 1=benign. Deterministic stratified 60/20/20 train/validation/test split. Fit preprocessing on training only. Use validation ROC AUC/accuracy for promotion; test metrics are reporting only.

- [x] Infrastructure: standalone Dockerfiles, pinned runtime dependencies, PostgreSQL/bucket initialization, healthchecks, Compose, environment example. Validated Compose and built public-base images with `pip check`; verified fresh PostgreSQL initialization in a disposable container.
- [x] Pipeline: tests for deterministic disjoint splits, invalid data rejection, threshold gate, retry-safe registration, preservation of champion on rejection and extraction failure before XCom. Implemented `mlops/data.py`, `mlops/pipeline.py`, `dags/training.py`. Snapshots, model/signature, metrics and provenance persist through MLflow. XCom carries only identifiers.
- [x] Serving: verified cold-start 503, input validation, successful prediction and failed refresh preservation. Implemented `mlops/api.py` with background alias refresh and atomic in-memory model swap, returning the resolved immutable version.
- [x] Operations: portability audit expanded coverage to 25 runtime tests and four Airflow DAG tests on ARM64 and emulated AMD64. Isolated smoke runs verify MinIO objects, PostgreSQL metadata, aliases, serving and one shared network. Supplied smoke checker, rollback CLI, Vietnamese runbook and `docs/verification.md`.

Scope: local operational lab with loopback-only published ports and configurable demo credentials. No production authentication/TLS platform, external dataset download, Kubernetes, manual fixed model version, or dependence on another tutorial's local Docker image.

Failure semantics: dependency startup gates; task retries; invalid data fails without retry; failed DAG marks MLflow run failed; rejected candidates remain registered with gate tags but do not replace champion. API stays live while unready and keeps its last loaded model on transient registry/storage failure. Registry operations are serialized by the DAG; external concurrent promotion is outside this lab's single-writer contract. MLflow registration is recovered by run ID after task retry; no distributed exactly-once guarantee is claimed.
