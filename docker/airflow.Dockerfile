FROM apache/airflow:2.10.5-python3.11@sha256:b16e19f925ae0dcabaa2271b37212891ed68bf74e26c2db61f46dddd7d502937
COPY requirements.txt /tmp/requirements.txt
COPY requirements-airflow.lock /tmp/requirements-airflow.lock
RUN pip install --no-cache-dir "apache-airflow==2.10.5" -r /tmp/requirements.txt -c /tmp/requirements-airflow.lock && pip check
ENV PYTHONPATH=/opt/airflow/project
COPY --chown=airflow:root mlops /opt/airflow/project/mlops
COPY --chown=airflow:root dags /opt/airflow/dags
COPY --chown=airflow:root scripts /opt/airflow/project/scripts
COPY --chown=airflow:root tests /opt/airflow/project/tests
