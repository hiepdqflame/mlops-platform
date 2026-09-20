import mlflow
import pytest
from pathlib import Path
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
    assert client.get_run(rejected).data.tags['pipeline.status'] == 'awaiting_serving'
    assert client.get_run(run_id).data.tags['dataset.sha256']


@pytest.fixture
def registered_run(tmp_path, monkeypatch):
    uri = f'sqlite:///{tmp_path}/recovery.db'
    monkeypatch.setenv('MLFLOW_TRACKING_URI', uri)
    monkeypatch.setenv('MLFLOW_EXPERIMENT_NAME', 'recovery')
    monkeypatch.setenv('MODEL_NAME', 'recovery-classifier')
    monkeypatch.setenv('MIN_VALIDATION_AUC', '0.95')
    monkeypatch.setenv('MIN_VALIDATION_ACCURACY', '0.90')
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment('recovery', artifact_location=str(tmp_path / 'artifacts'))
    run_id = evaluate(train(validate_and_split(extract('original'))))
    return run_id, register_and_promote(run_id)


def test_registered_run_replay_preserves_artifact_bytes_and_metrics(registered_run):
    run_id, result = registered_run
    tracking = MlflowClient()
    root = Path(tracking.get_run(run_id).info.artifact_uri)
    before = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob('*') if path.is_file()}
    # An audit correction must not be overwritten by clearing old Airflow tasks.
    tracking.log_metric(run_id, 'validation_roc_auc', 0.51)
    assert extract('original') == run_id
    assert validate_and_split(run_id) == run_id
    assert train(run_id) == run_id
    assert evaluate(run_id) == run_id
    after = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob('*') if path.is_file()}
    assert after == before
    assert tracking.get_run(run_id).data.metrics['validation_roc_auc'] == 0.51


def test_old_registration_replay_does_not_roll_back_new_champion(registered_run):
    run_id, original = registered_run
    newer_id = evaluate(train(validate_and_split(extract('newer'))))
    newer = register_and_promote(newer_id)
    assert register_and_promote(run_id) == original
    tracking = MlflowClient()
    assert str(tracking.get_model_version_by_alias('recovery-classifier', 'champion').version) == newer['version']
    assert str(tracking.get_model_version_by_alias('recovery-classifier', 'challenger').version) == newer['version']
    assert str(tracking.get_model_version_by_alias('recovery-classifier', 'previous_champion').version) == original['version']


def test_run_is_not_finished_until_serving_verification(registered_run):
    run_id, result = registered_run
    assert MlflowClient().get_run(run_id).info.status == 'RUNNING'
    assert MlflowClient().get_run(run_id).data.tags['pipeline.status'] != 'completed'


def test_serving_only_retry_restores_failed_run(registered_run):
    from mlops import pipeline
    run_id, result = registered_run
    mark_failed(run_id)
    assert MlflowClient().get_run(run_id).info.status == 'FAILED'
    pipeline.finalize_run(run_id)
    assert MlflowClient().get_run(run_id).info.status == 'FINISHED'
    assert MlflowClient().get_run(run_id).data.tags['pipeline.status'] == 'completed'


def test_incomplete_old_registration_cannot_replace_a_newer_champion(registered_run, monkeypatch):
    old_id, _ = registered_run
    tracking = MlflowClient()
    # Simulate loss of the final write after alias updates succeeded.
    tracking.delete_tag(old_id, 'pipeline.registration')
    newer_id = evaluate(train(validate_and_split(extract('newer-after-failure'))))
    newer = register_and_promote(newer_id)
    with pytest.raises(ValueError, match='newer champion'):
        register_and_promote(old_id)
    assert str(tracking.get_model_version_by_alias('recovery-classifier', 'champion').version) == newer['version']
    assert str(tracking.get_model_version_by_alias('recovery-classifier', 'challenger').version) == newer['version']


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
