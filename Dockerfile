FROM python:3.11.11-slim-bookworm
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt && pip check
RUN useradd --create-home --uid 10001 app
ENV PYTHONPATH=/app
COPY --chown=app:app mlops ./mlops
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app tests ./tests
USER app
CMD ["uvicorn", "mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
