import requests
import pytest

from scripts.smoke import wait_for


def test_polling_survives_a_transient_read_timeout(monkeypatch):
    attempts = iter([requests.ReadTimeout('busy worker'), {'state': 'success'}])
    monkeypatch.setattr('scripts.smoke.time.sleep', lambda _: None)
    def check():
        value = next(attempts)
        if isinstance(value, Exception):
            raise value
        return value
    assert wait_for('DAG', check) == {'state': 'success'}


def test_polling_does_not_hide_a_failed_dag():
    def check():
        raise RuntimeError('DAG failed')
    with pytest.raises(RuntimeError, match='DAG failed'):
        wait_for('DAG', check)
