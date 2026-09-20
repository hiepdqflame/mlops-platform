# Verification evidence

## Portability and workflow audit (2026-09-20 follow-up)

This follow-up supersedes the original workflow test counts below. It uses an isolated copy under a directory containing spaces, separate Compose project names and empty PostgreSQL/MinIO volumes. The PostgreSQL init script in the copy was deliberately converted to Windows CRLF before building. Database passwords contained URI-special characters (`@:/?#%`). No original lab database was reset.

Fixes verified:

- Registered runs no longer overwrite their data, model or metrics when earlier tasks are cleared.
- Replaying a completed registration does not move aliases back to an older version. Incomplete stale registration refuses to overwrite a newer champion.
- Model training no longer marks the MLflow run FINISHED early. Successful serving verification finalizes it and can recover a previously FAILED run.
- Serving verification has a configurable timeout; smoke polling tolerates transient connection/read timeouts while still reporting failed DAGs.
- Multi-platform base images are pinned by digest; Python transitive dependencies use version constraints captured in the two lock files.
- Init script line endings are normalized inside the image; all services use named volumes and service DNS, without developer-machine paths.
- Airflow webserver uses two workers and a 180-second master startup timeout to reduce lab resource pressure.

Verified builds and tests:

- Application, Airflow and PostgreSQL images built for both `linux/arm64` and `linux/amd64`; actual image architectures were inspected.
- ARM64: 25 runtime tests passed; four Airflow-specific tests passed.
- AMD64 under Docker Desktop emulation: 25 runtime tests passed; four Airflow-specific tests passed.
- Runtime test commands intentionally skip the Airflow module, which is tested separately inside the Airflow image. Dependency deprecation warnings remain.
- ARM64 isolated end-to-end smoke passed two runs: versions 3 and 4, run IDs `15af5b1ce6a04f69a2014b9d7944df1a` and `219ff96bb7244940b947abcb0fd83e74`. Each run had 13 MinIO artifacts and all six tasks successful.
- ARM64 API still returned HTTP 200 with version 4 after both MLflow and MinIO were stopped, proving continued inference from the loaded model.
- AMD64 isolated end-to-end smoke passed two runs: versions 2 and 3, run IDs `6f4a357ad0264555b5baf91a01c023f4` and `6793d6e3d24244b1923ccf7b78031b9c`. The automatically scheduled run also succeeded. The checker verified PostgreSQL metadata/aliases/task states, 13 MinIO artifacts per run, and FastAPI predictions from each new version.
- Actual AMD64 project network inspection confirmed every long-running container attached only to `mlops-portability-amd64_mlops`.
- The main `sklearn-mlops` stack was recreated from the final images and restored healthy. Versions 1 through 5 remain in its registry, including the original version 2/3 run IDs listed below. FastAPI reports ready and serves the registry's current champion, version 5 (`9b1039c7f81448dd81b57ccf8db6d655`), with no refresh error. Original volumes were preserved.
- Temporary ARM64/AMD64 test containers and networks were removed with `docker compose down`; their test volumes and image/build caches remain available for inspection. Other lab stacks were not stopped or changed.

Observed limits: one package download timed out and succeeded on rebuild. Running several stacks concurrently in an 8 GB Docker VM caused severe resource contention, an initial AMD64 Airflow startup timeout and one interrupted smoke poll; verification was changed to run sequentially. This is evidence for requiring sufficient Docker memory, not a promise of operation on arbitrary hardware. Native Windows/WSL2 and native Intel hosts were not available; AMD64 execution here uses emulation on Apple Silicon. Supported host guidance is in the README.

## Original implementation verification

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
