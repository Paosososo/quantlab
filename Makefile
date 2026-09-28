# Convenience targets.  Every one is a command you could type by hand; the
# Makefile exists so that "how do I run the tests" has one answer.

.DEFAULT_GOAL := help
PYTHON ?= python
PYTEST ?= $(PYTHON) -m pytest

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# --- environment ---------------------------------------------------------
.PHONY: install
install: ## Install the package with development extras
	$(PYTHON) -m pip install -e ".[api,dashboard,dev]"

.PHONY: install-all
install-all: ## Install everything, including Airflow and XGBoost
	$(PYTHON) -m pip install -e ".[api,dashboard,dev,airflow,boost]"

# --- quality -------------------------------------------------------------
.PHONY: lint
lint: ## Ruff check and format check
	$(PYTHON) -m ruff check src tests dags scripts
	$(PYTHON) -m ruff format --check src tests dags scripts

.PHONY: format
format: ## Apply Ruff formatting and autofixes
	$(PYTHON) -m ruff check --fix src tests dags scripts
	$(PYTHON) -m ruff format src tests dags scripts

.PHONY: typecheck
typecheck: ## Static type checking
	$(PYTHON) -m mypy

.PHONY: test
test: ## Full test suite
	$(PYTEST)

.PHONY: test-fast
test-fast: ## Unit and leakage tests only
	$(PYTEST) tests/unit tests/leakage -q

.PHONY: test-leakage
test-leakage: ## Only the look-ahead and leakage tests
	$(PYTEST) -m leakage -q

.PHONY: coverage
coverage: ## Test suite with a coverage report
	$(PYTEST) --cov=quantlab --cov-report=term-missing --cov-report=html

.PHONY: check
check: lint typecheck test ## Everything CI runs

# --- database ------------------------------------------------------------
.PHONY: migrate
migrate: ## Apply migrations
	alembic upgrade head

.PHONY: migration
migration: ## Autogenerate a migration: make migration m="add x"
	alembic revision --autogenerate -m "$(m)"

.PHONY: downgrade
downgrade: ## Roll back one migration
	alembic downgrade -1

# --- pipelines -----------------------------------------------------------
.PHONY: seed
seed: ## Load the deterministic synthetic demo dataset
	$(PYTHON) -m quantlab.demo_seed

.PHONY: ingest
ingest: ## Fetch real data from Stooq and FRED
	quantlab ingest-prices
	quantlab ingest-macro
	quantlab returns

.PHONY: features
features: ## Build and store features
	quantlab features

.PHONY: train
train: ## Walk-forward training of the model ladder
	quantlab train

.PHONY: backtest
backtest: ## Run the classical strategies
	quantlab backtest

.PHONY: research
research: ## Run the full research study and write the report
	$(PYTHON) scripts/run_research.py

.PHONY: research-offline
research-offline: ## Research study on synthetic data; no database, no network
	$(PYTHON) scripts/run_research.py --offline

# --- services ------------------------------------------------------------
.PHONY: api
api: ## Run the API with reload
	uvicorn quantlab.api.main:app --reload --host 0.0.0.0 --port 8000

.PHONY: dashboard
dashboard: ## Run the dashboard
	streamlit run src/quantlab/dashboard/app.py

# --- docker --------------------------------------------------------------
.PHONY: up
up: ## Start PostgreSQL, migrations, API and dashboard
	docker compose up -d --build

.PHONY: up-demo
up-demo: ## Start the stack with the synthetic demo dataset
	docker compose --profile demo up -d --build

.PHONY: up-airflow
up-airflow: ## Start the stack plus Airflow
	docker compose --profile airflow up -d --build

.PHONY: down
down: ## Stop the stack
	docker compose down

.PHONY: clean
clean: ## Stop the stack and delete its volumes
	docker compose --profile demo --profile airflow down -v

.PHONY: logs
logs: ## Tail service logs
	docker compose logs -f
