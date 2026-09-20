import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from threading import Lock

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from mlops.data import FEATURES

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Snapshot:
    model: object
    version: str
    run_id: str


class ModelStore:
    def __init__(self, resolver=None, loader=None):
        self.name = os.getenv('MODEL_NAME', 'breast-cancer-classifier')
        self.resolver = resolver or (lambda: MlflowClient().get_model_version_by_alias(self.name, 'champion'))
        self.loader = loader or mlflow.sklearn.load_model
        self.snapshot = None
        self.refresh_error = None
        self.lock = Lock()

    def refresh(self) -> bool:
        with self.lock:
            try:
                version = self.resolver()
                if self.snapshot is None or self.snapshot.version != str(version.version):
                    model = self.loader(f'models:/{self.name}/{version.version}')
                    if list(model.feature_names_in_) != FEATURES or list(model.classes_) != [0, 1]:
                        raise ValueError('Model schema/classes are incompatible with this API')
                    # Resolve the alias first, load an immutable version, then swap one object.
                    self.snapshot = Snapshot(model, str(version.version), version.run_id)
                self.refresh_error = None
                return True
            except Exception as exc:
                self.refresh_error = type(exc).__name__
                logger.warning('Model refresh failed: %s', exc)
                return False


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    features: list[FiniteFloat] = Field(min_length=30, max_length=30)


def create_app(store=None) -> FastAPI:
    store = store or ModelStore()

    @asynccontextmanager
    async def lifespan(app):
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
        stop = asyncio.Event()

        async def refresh_loop():
            while not stop.is_set():
                await asyncio.to_thread(store.refresh)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=max(1, float(os.getenv('MODEL_REFRESH_SECONDS', '10'))))
                except asyncio.TimeoutError:
                    pass

        task = asyncio.create_task(refresh_loop())
        yield
        stop.set()
        await task

    app = FastAPI(title='Sklearn MLOps Serving', version='1.0.0', lifespan=lifespan)
    app.state.model_store = store

    @app.get('/health/live')
    def live():
        return {'status': 'alive'}

    @app.get('/health/ready')
    def ready():
        snapshot = store.snapshot
        return JSONResponse(status_code=200 if snapshot else 503, content={
            'status': 'ready' if snapshot else 'waiting_for_champion',
            'model_name': store.name, 'alias': 'champion',
            'model_version': snapshot.version if snapshot else None,
            'run_id': snapshot.run_id if snapshot else None,
            'refresh_error': store.refresh_error,
        })

    @app.get('/model/info')
    def info():
        return {'model_name': store.name, 'alias': 'champion', 'features': FEATURES,
                'classes': {'0': 'malignant', '1': 'benign'}}

    @app.post('/predict')
    def predict(request: PredictRequest):
        snapshot = store.snapshot
        if snapshot is None:
            raise HTTPException(503, 'No champion model loaded yet')
        frame = pd.DataFrame([request.features], columns=FEATURES)
        probability = float(snapshot.model.predict_proba(frame)[0, 1])
        return {'prediction': 'benign' if probability >= 0.5 else 'malignant',
                'probability_benign': probability, 'model_name': store.name,
                'model_version': snapshot.version, 'run_id': snapshot.run_id}

    return app


app = create_app()
