import pytest

pytest.importorskip('airflow')
from airflow.models import DagBag


def test_dag_task_arguments_do_not_collide_with_airflow_context():
    bag = DagBag('/opt/airflow/dags', include_examples=False)
    assert not bag.import_errors
    dag = bag.get_dag('sklearn_training')
    assert dag is not None
    for task in dag.tasks:
        task.determine_kwargs({'run_id': 'airflow-run-id', 'ti': object()})
