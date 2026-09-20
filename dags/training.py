import os
from datetime import timedelta

import pendulum
from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException
from airflow.operators.python import get_current_context


def failure_callback(context):
    from mlops.pipeline import mark_failed
    run_id = context['ti'].xcom_pull(task_ids='extract_dataset')
    mark_failed(run_id, airflow_run_id=context['run_id'])


@dag(
    dag_id='sklearn_training', schedule=os.getenv('AIRFLOW_SCHEDULE', '@daily'),
    start_date=pendulum.datetime(2026, 1, 1, tz='UTC'), catchup=False,
    max_active_runs=1, dagrun_timeout=timedelta(minutes=30),
    default_args={'owner': 'mlops', 'retries': 2, 'retry_delay': timedelta(seconds=20),
                  'execution_timeout': timedelta(minutes=10), 'on_failure_callback': failure_callback},
    tags=['sklearn', 'mlflow', 'minio'],
    doc_md='Train sklearn data, log to MLflow/MinIO, gate promotion, and verify FastAPI serves the champion.',
)
def training_pipeline():
    @task
    def extract_dataset():
        from mlops.pipeline import extract
        return extract(get_current_context()['run_id'])

    @task
    def validate_split(mlflow_run_id):
        from mlops.pipeline import validate_and_split
        try:
            return validate_and_split(mlflow_run_id)
        except ValueError as exc:
            raise AirflowFailException(str(exc)) from exc

    @task
    def train_model(mlflow_run_id):
        from mlops.pipeline import train
        return train(mlflow_run_id)

    @task
    def evaluate_model(mlflow_run_id):
        from mlops.pipeline import evaluate
        return evaluate(mlflow_run_id)

    @task
    def register_promote(mlflow_run_id):
        from mlops.pipeline import register_and_promote
        return register_and_promote(mlflow_run_id)

    @task
    def verify_serving(result):
        import time
        import requests
        from mlops.pipeline import client, finalize_run, get_alias
        champion = get_alias(client(), result['model_name'], 'champion')
        if champion is None:
            finalize_run(result['run_id'])
            return {'status': 'no_champion', 'candidate_rejected': True}
        timeout = float(os.getenv('VERIFY_SERVING_TIMEOUT_SECONDS', '180'))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response = requests.get(os.environ['API_URL'] + '/health/ready', timeout=5)
            if response.ok and response.json()['model_version'] == str(champion.version):
                finalize_run(result['run_id'])
                return response.json()
            time.sleep(min(2, max(0, deadline - time.monotonic())))
        raise RuntimeError(f'FastAPI did not load the champion within {timeout:g} seconds')

    verify_serving(register_promote(evaluate_model(train_model(validate_split(extract_dataset())))))


training_pipeline()
