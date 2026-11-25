.PHONY: setup run run-mock calibrate check-env test test-coverage lint clean

setup:
	bash scripts/install.sh

run:
	PYTHONPATH=src .venv/bin/python -m nutriscan.core.app

run-mock:
	PYTHONPATH=src python3 -m nutriscan.core.app --mock --once --no-web

calibrate:
	PYTHONPATH=src python3 scripts/calibrate_hx711.py

check-env:
	PYTHONPATH=src python3 scripts/check_env.py

test:
	PYTHONPATH=src python3 -m pytest tests/ -v

test-coverage:
	PYTHONPATH=src python3 -m pytest tests/ --cov=src/nutriscan --cov=scripts.calibrate_hx711 --cov-report=term-missing

lint:
	ruff check src/ scripts/ tests/ main.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov
