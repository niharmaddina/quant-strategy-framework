# Common developer tasks. Override PYTHON to point at a virtualenv interpreter, e.g.
#   make test PYTHON=.venv/bin/python
PYTHON ?= python

.PHONY: help install test test-all lint fix example clean

help:
	@echo "make install   - install runtime + dev requirements"
	@echo "make test      - fast test suite (no network, no slow tests); what CI runs"
	@echo "make test-all  - full suite via tests/run_all_tests.py --all"
	@echo "make lint      - ruff check (E, F, I rules; 120 columns)"
	@echo "make fix       - ruff check --fix (safe fixes only)"
	@echo "make example   - frozen-universe backtest -> results/ (network: yfinance, cached in data_cache/)"
	@echo "make clean     - remove local output/logs/caches (python -m utils.clean_local_state)"

install:
	$(PYTHON) -m pip install -r requirements-dev.txt

test:
	$(PYTHON) -m pytest tests -m "not network and not slow"

test-all:
	$(PYTHON) tests/run_all_tests.py --all

lint:
	$(PYTHON) -m ruff check .

fix:
	$(PYTHON) -m ruff check --fix .

example:
	$(PYTHON) examples/frozen_universe_backtest.py

clean:
	$(PYTHON) -m utils.clean_local_state
