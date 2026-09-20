# Sklearn MLOps: Airflow + MLflow + FastAPI + MinIO + PostgreSQL

Project hoàn chỉnh kế thừa luồng từ các tutorial trong thư mục LAB. Build trực tiếp từ public images; không cần chạy hay build các tutorial cũ.

Kết quả kiểm chứng thực tế: xem [docs/verification.md](docs/verification.md).

## Chạy nhanh

Cần Docker Engine/Desktop và Docker Compose v2 trở lên. Nên cấp Docker khoảng 6 GB RAM và 4 CPU. Chạy trong thư mục `mlops-platform`:

```bash
cp .env.example .env
docker compose up -d --build --wait --wait-timeout 300
docker compose ps -a
```

Lần đầu cần internet để tải images và Python packages. Dữ liệu sklearn đã đi kèm package, không cần tải dataset riêng. Các giá trị mặc định trong Compose cho phép chạy cả khi chưa có `.env`.

- Airflow: http://localhost:18030 — `admin` / `admin_lab_password`.
- MLflow: http://localhost:15030.
- FastAPI Swagger: http://localhost:18031/docs.
- MinIO Console: http://localhost:19031 — `minioadmin` / `minio_lab_password`.
- MinIO S3 API: http://localhost:19030.

Đây là cấu hình lab cục bộ: ports chỉ bind `127.0.0.1`, PostgreSQL không publish ra host. Đổi credentials trong `.env` trước lần khởi tạo nếu cần. Đổi mật khẩu `.env` sau khi database đã tồn tại không tự đổi mật khẩu trong PostgreSQL/Airflow; cần thay qua công cụ quản trị tương ứng. Không công khai các giao diện này ra internet khi chưa bổ sung authentication/TLS.

`airflow-init` và `minio-init` kết thúc với mã `0` là bình thường. Sáu service thường trực gồm postgres, minio, mlflow, api, airflow-webserver và airflow-scheduler phải healthy. Healthcheck của API kiểm tra tiến trình sống; `/health/ready` trả `503` đến khi model đầu tiên đạt chất lượng và được tải xong.

## Luồng vận hành

```text
                    Docker network: sklearn-mlops_mlops

 Airflow scheduler/LocalExecutor ---> MLflow Tracking + Model Registry
         |                                 |                  |
         |                                 | metadata         | artifact proxy
         v                                 v                  v
 PostgreSQL database airflow     PostgreSQL database mlflow   MinIO bucket mlflow

 FastAPI -- tra alias champion --> MLflow -- tải artifact --> MinIO
```

Mỗi service, kể cả init và tools, chỉ gắn network `mlops` của project. Giao tiếp nội bộ dùng tên service: `postgres:5432`, `mlflow:5000`, `minio:9000`, `api:8000`; không dùng host gateway hoặc localhost để gọi service khác.

DAG `sklearn_training` được bật tự động, mặc định chạy `@daily`, `catchup=False`, tối đa một DAG run hoạt động. Scheduler tự tạo lần chạy đầu; không cần train bằng tay để bootstrap API. Có thể trigger ngay bằng UI hoặc:

```bash
docker compose exec airflow-scheduler airflow dags trigger sklearn_training
```

Sáu task nối tiếp:

1. `extract_dataset`: đọc `load_breast_cancer(as_frame=True)`, tạo MLflow run có liên kết Airflow run ID, lưu snapshot `raw.csv` và SHA-256 của dataset.
2. `validate_split`: kiểm tra schema, số dòng, giá trị hữu hạn, hai lớp và duplicate; chia stratified với seed 42 thành train 341, validation 114, test 114 dòng. Lưu snapshot từng tập và báo cáo validation.
3. `train_model`: fit `StandardScaler` + `LogisticRegression` trên train. Log params, model, signature, input example và môi trường model vào MLflow.
4. `evaluate_model`: tính accuracy, ROC AUC và confusion matrix trên validation và test. Test chỉ dùng báo cáo, không dùng để chọn model.
5. `register_promote`: tạo model version do MLflow tự tăng, gắn tags và alias `challenger`; chỉ chuyển `champion` nếu validation ROC AUC >= 0.95, accuracy >= 0.90 và ROC AUC không thấp hơn champion hiện tại. Lưu champion cũ ở alias `previous_champion`.
6. `verify_serving`: chờ FastAPI phục vụ đúng version của champion, rồi mới cho DAG thành công.

Dữ liệu có 569 dòng và 30 features; `0=malignant`, `1=benign`. Đây là dataset học tập, API minh họa không phải sản phẩm chẩn đoán y tế.

## Version, tags, alias và lưu trữ

Model mặc định: `breast-cancer-classifier`; experiment: `sklearn-breast-cancer`. Tên và ngưỡng có thể đổi trong `.env` rồi recreate các service bằng `docker compose up -d`.

Mỗi DAG run mới tạo một MLflow run và một model version mới. Retry cùng DAG run tái sử dụng run; task registry tìm version theo run ID trước khi đăng ký. DAG giới hạn một writer để tránh hai lần promotion chạy đua. Không chạy thêm writer bên ngoài đồng thời; đây không phải giao dịch exactly-once xuyên nhiều hệ thống.

Tags truy vết gồm dataset, dataset hash, algorithm, Airflow run ID, code version và phiên bản sklearn; version có thêm `quality_gate`, `gate.min_auc`, `gate.min_accuracy`. Run có `pipeline.status`, `model.version`, `promotion`. Trạng thái MLflow run có thể FINISHED sau task training; dùng `pipeline.status` và trạng thái Airflow DAG để xem toàn bộ workflow.

- `challenger`: ứng viên đăng ký gần nhất, kể cả bị từ chối.
- `champion`: version đang được phép phục vụ.
- `previous_champion`: version để rollback sau khi thay champion.

Ứng viên không đạt vẫn giữ artifacts, metrics và version với `quality_gate=rejected`; champion hiện tại không đổi. Nếu đây là model đầu và bị từ chối, DAG hoàn tất đánh giá nhưng API tiếp tục `503`; task cuối báo `no_champion`. Smoke test cố ý yêu cầu model mặc định phải được promotion.

PostgreSQL lưu hai database riêng, hai user riêng:

- `mlflow`: experiments, runs, params, metrics, tags, registered models, model versions, aliases và artifact URIs.
- `airflow`: DAG runs, task instances, XCom, lịch chạy và người dùng Airflow.

MinIO lưu các file lớn trong bucket `mlflow`, bật object versioning. MLflow chạy `--serve-artifacts --artifacts-destination s3://mlflow/artifacts`. Airflow và API upload/download qua MLflow HTTP; không cần S3 credentials. Artifact URI `mlflow-artifacts:/...` là đúng: MLflow proxy ánh xạ nó xuống MinIO.

XCom chỉ chứa run ID/kết quả đăng ký nhỏ. Snapshot và model được đọc lại từ artifact storage ở mỗi task; không truyền DataFrame/model qua XCom và không phụ thuộc file cục bộ của task trước.

## FastAPI và cập nhật model

API định kỳ tra alias `champion` (10 giây mặc định), resolve ra version cố định rồi mới tải model. Tải và kiểm tra schema thành công mới thay snapshot trong RAM. Request luôn trả đúng version/run ID đã dùng, kể cả alias thay đổi trong lúc dự đoán.

- `GET /health/live`: tiến trình còn sống.
- `GET /health/ready`: `200` nếu có model, `503` nếu chưa có; hiển thị version và lỗi refresh gần nhất.
- `GET /model/info`: danh sách 30 features theo đúng thứ tự và class mapping.
- `POST /predict`: nhận `{"features": [30 số hữu hạn]}`; từ chối sai độ dài, NaN/Infinity và trường thừa bằng `422`.

Nếu MLflow/MinIO tạm ngắt, API giữ model đã tải và thử refresh lại. Nếu API cũng restart trong lúc storage ngắt, nó phải chờ tải được model mới sẵn sàng.

Ví dụ dự đoán một dòng sklearn ngay trong network:

```bash
docker compose run --rm --no-deps tools python -c \
  'import requests; from sklearn.datasets import load_breast_cancer; r=requests.post("http://api:8000/predict",json={"features":load_breast_cancer().data[0].tolist()},timeout=10); print(r.status_code,r.json())'
```

## Kiểm thử

Unit/integration tests dùng database SQLite tạm để kiểm tra registry thật mà không làm thay đổi model trong PostgreSQL đang chạy:

```bash
docker compose run --rm --no-deps tools python -m pytest tests -q -p no:cacheprovider
docker compose run --rm --no-deps airflow-init python -m pytest /opt/airflow/project/tests/test_dag.py -q -p no:cacheprovider
```

Smoke end-to-end trigger **hai DAG runs thật qua Airflow REST API**, chờ scheduler thực thi, kiểm tra version tăng, aliases/tags, prediction, invalid input, file artifact trong MinIO, model/alias trong PostgreSQL và cả sáu task states trong Airflow database:

```bash
docker compose run --rm tools python scripts/smoke.py
```

Lệnh này tạo thêm hai phiên bản model theo đúng quy trình. Có thể dùng `--runs 1` để kiểm tra nhanh một lần. Muốn xem toàn bộ kết quả chạy, đọc JSON cuối output. Nếu thay ngưỡng để model bị từ chối, smoke sẽ báo lỗi do yêu cầu promotion không đạt.

Kiểm tra network:

```bash
docker network inspect sklearn-mlops_mlops
docker compose ps -a
```

Kiểm tra metadata trực tiếp:

```bash
docker compose exec postgres psql -U postgres -d mlflow -c \
  'SELECT name, version, run_id FROM model_versions ORDER BY version;'
docker compose exec postgres psql -U postgres -d mlflow -c \
  'SELECT name, alias, version FROM registered_model_aliases ORDER BY alias;'
docker compose exec postgres psql -U postgres -d airflow -c \
  'SELECT dag_id, run_id, state FROM dag_run ORDER BY execution_date DESC LIMIT 10;'
```

## Retry, rollback và sự cố

Lỗi tạm thời được retry hai lần, cách nhau 20 giây. Validation hỏng dừng ngay, không retry vô ích. Callback đánh dấu MLflow run failed khi task hết retry; khi MLflow ngắt hoàn toàn, callback cũng có thể không cập nhật được và Airflow logs là nguồn điều tra.

Rollback cần có ít nhất hai lần promotion. Tạm pause DAG để lịch tự động không lập tức thay đổi lại champion:

```bash
docker compose exec airflow-scheduler airflow dags pause sklearn_training
docker compose run --rm --no-deps tools python scripts/rollback.py
```

Script xác nhận version trước đã qua gate và tải được artifact, rồi đổi alias. API cập nhật trong chu kỳ refresh kế tiếp. Sau khi xử lý xong nguyên nhân:

```bash
docker compose exec airflow-scheduler airflow dags unpause sklearn_training
```

Nếu Airflow chưa thấy DAG, chờ lần parse đầu hoặc kiểm tra import errors. Nếu API chưa ready, kiểm tra run training trong UI trước khi restart:

```bash
docker compose exec airflow-scheduler airflow dags list-import-errors
docker compose logs --tail=100 airflow-scheduler mlflow api
```

Nếu thay code, chạy lại `docker compose up -d --build`; code được COPY vào image để mỗi lần triển khai có nội dung cố định. Không bind mount source code vào container đang vận hành.

## Dừng và bảo toàn dữ liệu

```bash
docker compose down
docker compose up -d --wait --wait-timeout 300
```

Named volumes giữ PostgreSQL, MinIO và task logs qua các lần dừng. `docker compose down -v` xóa toàn bộ dữ liệu của project, chỉ dùng khi chủ động reset lab. Database init script chỉ chạy với volume PostgreSQL trống. Nên backup PostgreSQL cùng object storage để giữ đúng liên kết metadata–artifact.

## Cấu trúc source

```text
Dockerfile                 Runtime MLflow / FastAPI / tools
docker/airflow.Dockerfile   Airflow và các thư viện ML cùng phiên bản
docker/postgres.Dockerfile  Đóng gói database init script, không phụ thuộc host mount
docker-compose.yml         Toàn bộ service, network, volumes, healthchecks
docker/init-databases.sh    Tạo database/user riêng cho MLflow và Airflow
dags/training.py           DAG sáu task và failure callback
mlops/data.py              Data validation và deterministic splitting
mlops/pipeline.py          Tracking, training, evaluation và promotion
mlops/api.py               Model cache, refresh, health và predict
scripts/                   Bootstrap, smoke test và rollback
tests/                     Data, quality gate, serving và registry tests
docs/implementation-plan.md Phạm vi và checklist triển khai
```
