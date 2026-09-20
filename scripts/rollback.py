"""Restore the previous validated champion without retraining or copying artifacts."""
import os

import mlflow.sklearn
from mlflow.tracking import MlflowClient

name = os.environ['MODEL_NAME']
client = MlflowClient()
previous = client.get_model_version_by_alias(name, 'previous_champion')
if previous.tags.get('quality_gate') != 'passed':
    raise RuntimeError('Rollback target has not passed the quality gate')
mlflow.sklearn.load_model(f'models:/{name}/{previous.version}')
client.set_registered_model_alias(name, 'champion', previous.version)
print(f'{name}@champion -> version {previous.version}; FastAPI will refresh automatically')
