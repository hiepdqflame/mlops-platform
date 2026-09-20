.PHONY: up down test smoke logs status
up:
	docker compose up -d --build --wait --wait-timeout 300
down:
	docker compose down
test:
	docker compose run --rm --no-deps tools python -m pytest tests -q -p no:cacheprovider
	docker compose run --rm --no-deps airflow-init python -m pytest /opt/airflow/project/tests/test_dag.py -q -p no:cacheprovider
smoke:
	docker compose run --rm tools python scripts/smoke.py
logs:
	docker compose logs -f --tail=100
status:
	docker compose ps -a
