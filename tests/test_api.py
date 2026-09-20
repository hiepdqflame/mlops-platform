from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlops.api import ModelStore, create_app


@pytest.fixture
def trained_model():
    data = load_breast_cancer(as_frame=True)
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(data.data, data.target)


def test_cold_start_is_live_but_not_ready():
    store = ModelStore(resolver=lambda: (_ for _ in ()).throw(RuntimeError('not registered')))
    with TestClient(create_app(store)) as client:
        assert client.get('/health/live').status_code == 200
        assert client.get('/health/ready').status_code == 503
        assert client.post('/predict', json={'features': [1.0] * 30}).status_code == 503


def test_failed_refresh_keeps_old_model_and_resolved_version(trained_model):
    version = SimpleNamespace(version='3', run_id='run-three')
    store = ModelStore(resolver=lambda: version, loader=lambda uri: trained_model)
    assert store.refresh()
    store.resolver = lambda: SimpleNamespace(version='4', run_id='run-four')
    store.loader = lambda uri: (_ for _ in ()).throw(RuntimeError('storage unavailable'))
    assert not store.refresh()
    with TestClient(create_app(store)) as client:
        row = load_breast_cancer().data[0].tolist()
        result = client.post('/predict', json={'features': row})
        assert result.status_code == 200
        assert result.json()['model_version'] == '3'
        assert result.json()['run_id'] == 'run-three'
        assert result.json()['prediction'] in ('malignant', 'benign')
        assert 0 <= result.json()['probability_benign'] <= 1
        assert client.get('/health/ready').json()['refresh_error'] is not None
        assert client.post('/predict', json={'features': [1.0] * 29}).status_code == 422
        assert client.post('/predict', json={'features': ['NaN'] * 30}).status_code == 422
        assert client.post('/predict', json={'features': [1.0] * 30, 'unexpected': 1}).status_code == 422


def test_refresh_loads_immutable_version_and_swaps_only_when_ready(trained_model):
    version = SimpleNamespace(version='7', run_id='seven')
    def load(uri):
        assert uri.endswith('/7')
        return trained_model
    store = ModelStore(resolver=lambda: version, loader=load)
    assert store.refresh()
    assert store.snapshot.version == '7'
