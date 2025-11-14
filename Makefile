.PHONY: help install test lint format clean run-backtest

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt
	pip install -e ".[dev]"

test: ## Run tests
	pytest tests/ -v --cov=libs --cov-report=html --cov-report=term

lint: ## Run linters
	ruff check .
	mypy libs/

format: ## Format code
	black .
	ruff check --fix .

clean: ## Clean build artifacts
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	rm -rf .pytest_cache .coverage htmlcov dist build

run-backtest: ## Run backtest locally
	./scripts/run_backtest_local.sh

sync-data: ## Sync data
	./scripts/sync_data.sh

