import mlflow
import pytest
from mlflow.tracking import MlflowClient

from mlops.pipeline import extract, validate_and_split, train, evaluate, register_and_promote, mark_failed


@pytest.fixture(autouse=True)
def isolated_artifact_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_real_registry_retries_and_rejected_candidate(tmp_path, monkeypatch):
    monkeypatch.setenv('MLFLOW_TRACKING_URI', f'sqlite:///{tmp_path}/tracking.db')
    monkeypatch.setenv('MLFLOW_EXPERIMENT_NAME', 'integration-test')
    monkeypatch.setenv('MODEL_NAME', 'integration-classifier')
    monkeypatch.setenv('MIN_VALIDATION_AUC', '0.95')
    monkeypatch.setenv('MIN_VALIDATION_ACCURACY', '0.90')
    mlflow.set_tracking_uri(f'sqlite:///{tmp_path}/tracking.db')
    mlflow.create_experiment('integration-test', artifact_location=str(tmp_path / 'artifacts'))
    run_id = extract('integration-first')
    assert extract('integration-first') == run_id
    validate_and_split(run_id)
    train(run_id)
    evaluate(run_id)
    first = register_and_promote(run_id)
    assert first['promoted'] is True
    assert register_and_promote(run_id)['version'] == first['version']
    client = MlflowClient()
    assert len(client.search_model_versions("name='integration-classifier'")) == 1
    model = mlflow.sklearn.load_model(f"models:/integration-classifier@champion")
    assert model.n_features_in_ == 30
    rejected = extract('integration-rejected')
    validate_and_split(rejected)
    train(rejected)
    evaluate(rejected)
    client.log_metric(rejected, 'validation_roc_auc', 0.5)
    result = register_and_promote(rejected)
    assert result['promoted'] is False
    assert result['version'] != first['version']
    assert str(client.get_model_version_by_alias('integration-classifier', 'champion').version) == first['version']
    assert str(client.get_model_version_by_alias('integration-classifier', 'challenger').version) == result['version']
    assert client.get_model_version('integration-classifier', result['version']).tags['quality_gate'] == 'rejected'
    assert client.get_run(rejected).data.tags['pipeline.status'] == 'completed'
    assert client.get_run(run_id).data.tags['dataset.sha256']


def test_extraction_upload_failure_is_tracked_without_xcom(tmp_path, monkeypatch):
    uri = f'sqlite:///{tmp_path}/failed-extract.db'
    monkeypatch.setenv('MLFLOW_TRACKING_URI', uri)
    monkeypatch.setenv('MLFLOW_EXPERIMENT_NAME', 'failed-extract')
    mlflow.set_tracking_uri(uri)
    experiment_id = mlflow.create_experiment('failed-extract', artifact_location=str(tmp_path / 'artifacts'))
    def fail_upload(*args, **kwargs):
        raise ConnectionError('artifact storage unavailable')
    monkeypatch.setattr(MlflowClient, 'log_artifact', fail_upload)
    with pytest.raises(ConnectionError):
        extract('failed-before-xcom')
    mark_failed(airflow_run_id='failed-before-xcom')
    run = MlflowClient().search_runs([experiment_id])[0]
    assert run.info.status == 'FAILED'
    assert run.data.tags['pipeline.status'] == 'failed'
