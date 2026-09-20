import os
import subprocess
import sys
from pathlib import Path


def test_airflow_command_uri_has_no_newline_and_escapes_password():
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'db_uri.py'
    result = subprocess.run(
        [sys.executable, str(script), 'airflow'], capture_output=True, text=True, check=True,
        env={**os.environ, 'AIRFLOW_DB_PASSWORD': 'p@ss:/?#%'},
    )
    assert result.stdout == 'postgresql+psycopg2://airflow:p%40ss%3A%2F%3F%23%25@postgres:5432/airflow'
