"""Task boundaries persist artifacts via MLflow; XCom carries only the run ID."""
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlops.data import FEATURES, SEED, prepare_splits


def client() -> MlflowClient:
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    return MlflowClient()


def is_registered(tracking, run_id: str) -> bool:
    # Registry versions reference run artifacts; clearing old tasks must not
    # overwrite their contents. Start a new DAG run to retrain a registered model.
    return bool(tracking.search_model_versions(f"run_id='{run_id}'"))


def extract(airflow_run_id: str) -> str:
    tracking = client()
    experiment = mlflow.set_experiment(os.getenv('MLFLOW_EXPERIMENT_NAME', 'sklearn-breast-cancer'))
    key = hashlib.sha256(airflow_run_id.encode()).hexdigest()
    existing = tracking.search_runs([experiment.experiment_id], f"tags.`pipeline.run_key` = '{key}'")
    if existing:
        run_id = existing[0].info.run_id
        if is_registered(tracking, run_id):
            return run_id
    else:
        run_id = tracking.create_run(experiment.experiment_id, tags={
            'mlflow.runName': airflow_run_id,
            'pipeline.run_key': key, 'airflow.run_id': airflow_run_id,
            'dataset': 'sklearn.breast_cancer', 'algorithm': 'standard_scaler+logistic_regression',
            'pipeline.status': 'running', 'code.version': '1.0.0',
            'sklearn.version': sklearn.__version__,
        }).info.run_id
    raw = load_breast_cancer(as_frame=True).frame.to_csv(index=False)
    tracking.set_tag(run_id, 'dataset.sha256', hashlib.sha256(raw.encode()).hexdigest())
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'raw.csv'
        path.write_text(raw)
        tracking.log_artifact(run_id, str(path), 'data')
    return run_id


def read_frame(run_id: str, artifact: str) -> pd.DataFrame:
    tracking = client()
    with tempfile.TemporaryDirectory() as directory:
        path = tracking.download_artifacts(run_id, artifact, directory)
        return pd.read_csv(path)


def validate_and_split(run_id: str) -> str:
    tracking = client()
    if is_registered(tracking, run_id):
        return run_id
    splits = prepare_splits(read_frame(run_id, 'data/raw.csv'))
    with tempfile.TemporaryDirectory() as directory:
        for name, frame in splits.items():
            path = Path(directory) / f'{name}.csv'
            frame.to_csv(path, index=False)
            tracking.log_artifact(run_id, str(path), 'data')
            tracking.log_param(run_id, f'{name}_rows', len(frame))
        report = Path(directory) / 'validation.json'
        report.write_text(json.dumps({
            'valid': True, 'features': FEATURES, 'rows': sum(map(len, splits.values())),
            'split_seed': SEED, 'split_fractions': [0.6, 0.2, 0.2],
            'class_names': {'0': 'malignant', '1': 'benign'},
        }, indent=2))
        tracking.log_artifact(run_id, str(report), 'reports')
    return run_id


def train(run_id: str) -> str:
    tracking = client()
    if is_registered(tracking, run_id):
        return run_id
    frame = read_frame(run_id, 'data/train.csv')
    features, target = frame[FEATURES], frame.target
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=SEED))
    model.fit(features, target)
    for key, value in {'C': 1.0, 'max_iter': 2000, 'random_state': SEED, 'scaler': 'StandardScaler'}.items():
        tracking.log_param(run_id, key, value)
    # Client uploads keep the run open across Airflow tasks; a fluent run
    # context would mark it FINISHED before evaluation and serving verification.
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / 'model')
        mlflow.sklearn.save_model(
            model, path=path,
            mlflow_model=mlflow.models.Model(run_id=run_id, artifact_path='model'),
            signature=infer_signature(features, model.predict(features)),
            input_example=features.head(2),
            pip_requirements=[f'scikit-learn=={sklearn.__version__}', f'pandas=={pd.__version__}', 'numpy==1.26.4'],
        )
        tracking.log_artifacts(run_id, path, 'model')
    return run_id


def evaluate(run_id: str) -> str:
    tracking = client()
    if is_registered(tracking, run_id):
        return run_id
    model = mlflow.sklearn.load_model(f'runs:/{run_id}/model')
    metrics, matrices = {}, {}
    for name in ('validation', 'test'):
        frame = read_frame(run_id, f'data/{name}.csv')
        prediction = model.predict(frame[FEATURES])
        probability = model.predict_proba(frame[FEATURES])[:, 1]
        metrics[f'{name}_accuracy'] = float(accuracy_score(frame.target, prediction))
        metrics[f'{name}_roc_auc'] = float(roc_auc_score(frame.target, probability))
        matrices[name] = confusion_matrix(frame.target, prediction).tolist()
    for key, value in metrics.items():
        tracking.log_metric(run_id, key, value)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'evaluation.json'
        path.write_text(json.dumps({'metrics': metrics, 'confusion_matrices': matrices}, indent=2))
        tracking.log_artifact(run_id, str(path), 'reports')
    return run_id


def passes_gate(metrics, champion_auc=None, *, min_auc=0.95, min_accuracy=0.90) -> bool:
    auc, accuracy = metrics['validation_roc_auc'], metrics['validation_accuracy']
    return bool(
        math.isfinite(auc) and math.isfinite(accuracy)
        and auc >= min_auc and accuracy >= min_accuracy
        and (champion_auc is None or auc >= champion_auc)
    )


def get_alias(tracking, name, alias):
    try:
        registered = tracking.get_registered_model(name)
    except MlflowException as exc:
        if exc.error_code != 'RESOURCE_DOES_NOT_EXIST':
            raise
        return None
    if alias not in registered.aliases:
        return None
    return tracking.get_model_version(name, registered.aliases[alias])


def register_and_promote(run_id: str) -> dict:
    tracking = client()
    name = os.getenv('MODEL_NAME', 'breast-cancer-classifier')
    run = tracking.get_run(run_id)
    if 'pipeline.registration' in run.data.tags:
        # Replaying an old completed task must not undo a newer promotion or rollback.
        return json.loads(run.data.tags['pipeline.registration'])
    # One active DAG run serializes writers; recover an already registered retry.
    versions = tracking.search_model_versions(f"run_id='{run_id}'")
    matching = [v for v in versions if v.name == name]
    version = matching[0] if matching else mlflow.register_model(f'runs:/{run_id}/model', name)
    number = str(version.version)
    champion = get_alias(tracking, name, 'champion')
    if matching and champion is not None and int(champion.version) > int(number):
        raise ValueError('A newer champion exists; start a new DAG run instead of replaying incomplete registration')
    tracking.set_registered_model_tag(name, 'task', 'binary-classification')
    tracking.set_registered_model_tag(name, 'dataset', 'sklearn.breast_cancer')
    tracking.update_model_version(name, number, description='Sklearn pipeline with training-only scaling; validation-gated promotion.')
    for key in ('dataset', 'dataset.sha256', 'algorithm', 'airflow.run_id', 'code.version'):
        tracking.set_model_version_tag(name, number, key, run.data.tags[key])
    tracking.set_registered_model_alias(name, 'challenger', number)
    champion_auc = None
    if champion is not None:
        champion_run = tracking.get_run(champion.run_id)
        if champion_run.data.tags.get('dataset.sha256') != run.data.tags['dataset.sha256']:
            raise ValueError('Re-evaluate champion on the same dataset before comparing versions')
        champion_auc = champion_run.data.metrics['validation_roc_auc']
    min_auc = float(os.getenv('MIN_VALIDATION_AUC', '0.95'))
    min_accuracy = float(os.getenv('MIN_VALIDATION_ACCURACY', '0.90'))
    promoted = passes_gate(run.data.metrics, champion_auc, min_auc=min_auc, min_accuracy=min_accuracy)
    tracking.set_model_version_tag(name, number, 'quality_gate', 'passed' if promoted else 'rejected')
    tracking.set_model_version_tag(name, number, 'gate.min_auc', str(min_auc))
    tracking.set_model_version_tag(name, number, 'gate.min_accuracy', str(min_accuracy))
    if promoted:
        if champion is not None and str(champion.version) != number:
            tracking.set_registered_model_alias(name, 'previous_champion', champion.version)
        tracking.set_registered_model_alias(name, 'champion', number)
    result = {'run_id': run_id, 'model_name': name, 'version': number, 'promoted': promoted}
    tracking.set_tag(run_id, 'model.version', number)
    tracking.set_tag(run_id, 'promotion', 'promoted' if promoted else 'rejected')
    tracking.set_tag(run_id, 'pipeline.status', 'awaiting_serving')
    tracking.set_tag(run_id, 'pipeline.registration', json.dumps(result))
    return result


def finalize_run(run_id: str) -> None:
    tracking = client()
    tracking.set_tag(run_id, 'pipeline.status', 'completed')
    tracking.set_terminated(run_id, 'FINISHED')


def mark_failed(run_id: str | None = None, *, airflow_run_id: str | None = None) -> None:
    tracking = client()
    if run_id is None:
        experiment = tracking.get_experiment_by_name(os.getenv('MLFLOW_EXPERIMENT_NAME', 'sklearn-breast-cancer'))
        if experiment is None or airflow_run_id is None:
            return
        key = hashlib.sha256(airflow_run_id.encode()).hexdigest()
        runs = tracking.search_runs([experiment.experiment_id], f"tags.`pipeline.run_key` = '{key}'")
        if not runs:
            return
        run_id = runs[0].info.run_id
    tracking.set_tag(run_id, 'pipeline.status', 'failed')
    tracking.set_terminated(run_id, 'FAILED')
