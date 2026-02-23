.PHONY: install fetch-data validate-data run-pipeline run-scenarios run-backtest test lint format

install:
	pip install -r requirements.txt
	pip install -e .

fetch-data:
	python scripts/fetch_data.py

validate-data:
	python -c "from src.data.storage.data_store import DataStore; DataStore().validate()"

run-pipeline:
	python scripts/run_pipeline.py

run-scenarios:
	python scripts/run_pipeline.py --scenarios-only

run-backtest:
	python scripts/run_pipeline.py --backtest-only

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	mypy src/

format:
	black src/ tests/ scripts/
	ruff check --fix src/ tests/
