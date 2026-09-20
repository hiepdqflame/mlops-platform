FROM apache/airflow:2.10.5-python3.11
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir "apache-airflow==2.10.5" -r /tmp/requirements.txt && pip check
ENV PYTHONPATH=/opt/airflow/project
COPY --chown=airflow:root mlops /opt/airflow/project/mlops
COPY --chown=airflow:root dags /opt/airflow/dags
COPY --chown=airflow:root scripts /opt/airflow/project/scripts
COPY --chown=airflow:root tests /opt/airflow/project/tests
