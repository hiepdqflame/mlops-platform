import pytest
from types import SimpleNamespace

pytest.importorskip('airflow')
from airflow.models import DagBag


def test_dag_task_arguments_do_not_collide_with_airflow_context():
    bag = DagBag('/opt/airflow/dags', include_examples=False)
    assert not bag.import_errors
    dag = bag.get_dag('sklearn_training')
    assert dag is not None
    for task in dag.tasks:
        task.determine_kwargs({'run_id': 'airflow-run-id', 'ti': object()})


@pytest.mark.parametrize('has_champion', [True, False])
def test_serving_success_finalizes_run(monkeypatch, has_champion):
    from mlops import pipeline
    import requests
    bag = DagBag('/opt/airflow/dags', include_examples=False)
    verify = bag.get_dag('sklearn_training').get_task('verify_serving').python_callable
    completed = []
    monkeypatch.setattr(pipeline, 'client', lambda: object())
    monkeypatch.setattr(pipeline, 'get_alias', lambda *args: SimpleNamespace(version='7') if has_champion else None)
    monkeypatch.setattr(pipeline, 'finalize_run', completed.append)
    monkeypatch.setattr(requests, 'get', lambda *args, **kwargs: SimpleNamespace(ok=True, json=lambda: {'model_version': '7'}))
    monkeypatch.setenv('API_URL', 'http://api:8000')
    verify({'run_id': 'recovered-run', 'model_name': 'classifier', 'promoted': has_champion})
    assert completed == ['recovered-run']


def test_serving_timeout_respects_configured_deadline(monkeypatch):
    from mlops import pipeline
    import requests
    import time
    bag = DagBag('/opt/airflow/dags', include_examples=False)
    verify = bag.get_dag('sklearn_training').get_task('verify_serving').python_callable
    monkeypatch.setattr(pipeline, 'client', lambda: object())
    monkeypatch.setattr(pipeline, 'get_alias', lambda *args: SimpleNamespace(version='7'))
    monkeypatch.setattr(requests, 'get', lambda *args, **kwargs: SimpleNamespace(ok=False))
    now = [0]
    monkeypatch.setattr(time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(time, 'sleep', lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setenv('API_URL', 'http://api:8000')
    monkeypatch.setenv('VERIFY_SERVING_TIMEOUT_SECONDS', '120')
    with pytest.raises(RuntimeError, match='120'):
        verify({'run_id': 'pending', 'model_name': 'classifier', 'promoted': True})
    assert now[0] == 120
