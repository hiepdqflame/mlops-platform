import os

from db_uri import database_uri

os.execvp('mlflow', [
    'mlflow', 'server', '--backend-store-uri', database_uri('mlflow'),
    '--serve-artifacts', '--artifacts-destination', 's3://mlflow/artifacts',
    '--host', '0.0.0.0', '--port', '5000', '--workers', '2',
])
