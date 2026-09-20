# Verification evidence

Verified locally on 2026-09-20 with Docker Desktop (Linux ARM64), Docker Compose v5.1.3.

- `docker compose config --quiet`: passed.
- Docker builds: passed; `pip check` passed in both application and Airflow images.
- Fresh PostgreSQL image initialization: created databases `mlflow` and `airflow`, owned by their respective users. The disposable verification container was stopped and removed automatically.
- Runtime tests in the built image: 18 passed, one Airflow-specific test skipped; 18 dependency warnings (primarily MLflow/Pydantic deprecations).
- Airflow-specific DAG test in the Airflow image: one passed.
- Scheduled DAG `scheduled__2026-09-19T00:00:00+00:00`: all six tasks successful after correcting the task argument collision and rerunning failed/downstream tasks.
- `docker compose run --rm tools python scripts/smoke.py`: passed two newly triggered DAG runs through the real scheduler.
- Six long-running services healthy; two initialization services exited with code 0.
- Actual Docker container network inspection: every project container, including tools and init containers, attached only to `sklearn-mlops_mlops`.

First smoke run:

```text
Airflow run: smoke_85e8dc1de5264dd6ac28f1cad738018e
MLflow run:  c61c3385795044e590ed1b94a0e0d3e2
Version:     2
MinIO files: 13
```

Second smoke run:

```text
Airflow run: smoke_25955f3f4523459ea9fa5c2dd44fd883
MLflow run:  e266b1978cff4aacb2a11e57dad02a4e
Version:     3
MinIO files: 13
```

Both runs produced validation accuracy 0.9912280701754386, validation ROC AUC 1.0, test accuracy 0.9912280701754386 and test ROC AUC 0.9970520799213888. These are deterministic educational dataset results, not estimates for a deployed clinical population.

The smoke checker read model versions and aliases directly from PostgreSQL, confirmed all six task states in the separate Airflow database, enumerated model/data/report objects in MinIO, and checked FastAPI returned the corresponding immutable version and MLflow run ID. A malformed prediction request returned HTTP 422.

Final aliases after verification: `champion=3`, `challenger=3`, `previous_champion=2`. FastAPI serves version 3. Named volumes retain the generated data. The stack is left running.

Known scope: local lab deployment, no TLS or external service authentication. Unit registry tests use temporary SQLite; the smoke tests exercise the actual PostgreSQL/MinIO deployment. No destructive volume reset was performed.
