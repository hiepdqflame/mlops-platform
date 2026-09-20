"""Exercise real scheduled workers and verify all persistent stores end to end."""
import argparse
import json
import os
import time
import uuid

import boto3
import psycopg2
import requests
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_breast_cancer


def wait_for(description, check, seconds=600):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            value = check()
        except (requests.Timeout, requests.ConnectionError):
            value = None
        if value:
            return value
        time.sleep(3)
    raise TimeoutError(f'Timed out waiting for {description}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=2)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error('--runs must be positive')
    airflow = os.getenv('AIRFLOW_URL', 'http://airflow-webserver:8080') + '/api/v1'
    api = os.getenv('API_URL', 'http://api:8000')
    session = requests.Session()
    session.auth = (os.environ['AIRFLOW_ADMIN_USER'], os.environ['AIRFLOW_ADMIN_PASSWORD'])
    tracking = MlflowClient()
    name = os.environ['MODEL_NAME']
    s3 = boto3.client('s3', endpoint_url=os.environ['MLFLOW_S3_ENDPOINT_URL'])
    results = []

    def dag_ready():
        response = session.get(airflow + '/dags/sklearn_training', timeout=10)
        if response.status_code == 404:
            return False
        response.raise_for_status()
        return response.json().get('is_active')

    wait_for('DAG discovery', dag_ready, 180)
    response = session.patch(airflow + '/dags/sklearn_training', json={'is_paused': False}, timeout=10)
    response.raise_for_status()
    for _ in range(args.runs):
        dag_run_id = 'smoke_' + uuid.uuid4().hex
        url = airflow + '/dags/sklearn_training/dagRuns'
        response = session.post(url, json={'dag_run_id': dag_run_id}, timeout=10)
        response.raise_for_status()
        print(f'Triggered {dag_run_id}', flush=True)

        def finished():
            response = session.get(url + '/' + dag_run_id, timeout=10)
            response.raise_for_status()
            state = response.json()['state']
            if state == 'failed':
                raise RuntimeError(f'DAG failed: {dag_run_id}; inspect Airflow task logs')
            return state == 'success'

        wait_for('successful DAG run', finished)
        experiment = tracking.get_experiment_by_name(os.environ['MLFLOW_EXPERIMENT_NAME'])
        runs = tracking.search_runs([experiment.experiment_id], f"tags.`airflow.run_id` = '{dag_run_id}'")
        assert len(runs) == 1, 'A single DAG execution must map to one MLflow run'
        run = runs[0]
        version = tracking.get_model_version_by_alias(name, 'champion')
        assert version.run_id == run.info.run_id, 'Expected this deterministic candidate to pass promotion'
        assert run.info.status == 'FINISHED' and run.data.tags['pipeline.status'] == 'completed'
        assert version.tags['quality_gate'] == 'passed'
        assert tracking.get_model_version_by_alias(name, 'challenger').version == version.version
        ready = requests.get(api + '/health/ready', timeout=10)
        ready.raise_for_status()
        assert ready.json()['model_version'] == version.version
        prediction = requests.post(api + '/predict', json={'features': load_breast_cancer().data[0].tolist()}, timeout=10)
        prediction.raise_for_status()
        assert prediction.json()['run_id'] == run.info.run_id
        assert prediction.json()['prediction'] == 'malignant'
        assert 0 <= prediction.json()['probability_benign'] <= 1
        invalid = requests.post(api + '/predict', json={'features': [1.0]}, timeout=10)
        assert invalid.status_code == 422

        keys = [obj['Key'] for page in s3.get_paginator('list_objects_v2').paginate(Bucket='mlflow')
                for obj in page.get('Contents', []) if run.info.run_id in obj['Key']]
        for suffix in ('/model/MLmodel', '/model/model.pkl', '/data/raw.csv', '/data/train.csv',
                       '/data/validation.csv', '/data/test.csv', '/reports/evaluation.json'):
            assert any(key.endswith(suffix) for key in keys), f'Missing MinIO artifact: {suffix}'
        with psycopg2.connect(host='postgres', dbname='mlflow', user='mlflow', password=os.environ['MLFLOW_DB_PASSWORD']) as connection:
            with connection.cursor() as cursor:
                cursor.execute('SELECT version FROM model_versions WHERE name=%s AND run_id=%s', (name, run.info.run_id))
                assert [str(row[0]) for row in cursor.fetchall()] == [version.version]
                cursor.execute('SELECT version FROM registered_model_aliases WHERE name=%s AND alias=%s', (name, 'champion'))
                assert str(cursor.fetchone()[0]) == version.version
        with psycopg2.connect(host='postgres', dbname='airflow', user='airflow', password=os.environ['AIRFLOW_DB_PASSWORD']) as connection:
            with connection.cursor() as cursor:
                cursor.execute('SELECT task_id, state FROM task_instance WHERE dag_id=%s AND run_id=%s', ('sklearn_training', dag_run_id))
                tasks = cursor.fetchall()
                assert len(tasks) == 6 and all(state == 'success' for _, state in tasks)
        results.append({'dag_run_id': dag_run_id, 'mlflow_run_id': run.info.run_id,
                        'version': version.version, 'metrics': run.data.metrics,
                        'minio_artifacts': len(keys), 'prediction': prediction.json()})
        print(f'Verified version {version.version}: PostgreSQL, MinIO, registry and FastAPI', flush=True)
    if len(results) > 1:
        assert int(results[-1]['version']) > int(results[0]['version'])
        assert tracking.get_model_version_by_alias(name, 'previous_champion').version == results[-2]['version']
    print(json.dumps({'status': 'passed', 'runs': results}, indent=2))


if __name__ == '__main__':
    main()
