# Convenient commands. Run `make help` for command info.
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@echo "\033[0;1mCommands\033[0m"
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[34;1m%-30s\033[0m %s\n", $$1, $$2}'

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts
.PHONY: clean

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +
.PHONY: clean-build

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
.PHONY: clean-pyc

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache
.PHONY: clean-test

lint: ## check style with pylint
	pylint ribs tests examples benchmarks
.PHONY: lint

test: ## run tests with the default Python
	pytest tests
.PHONY: test

test-core: ## only test the core of ribs
	pytest tests/core
.PHONY: test-core

test-extras: ## only test the extras of ribs
	pytest tests/extras
.PHONY: test-extras

test-coverage: ## get better test coverage by running without numba on
	NUMBA_DISABLE_JIT=1 pytest tests
.PHONY: test-coverage

test-all: ## run tests on every Python version with tox
	tox
.PHONY: test-all

benchmarks: ## run benchmarks (may take a while)
	pytest -c pytest_benchmark.ini tests
.PHONY: benchmarks

xtest: ## run tests with n workers (e.g. make xtest n=4)
	pytest -n $(n) tests
.PHONY: xtest

xbenchmarks: ## run benchmarks with n workers (e.g. make xbenchmarks n=4)
	pytest -n $(n) -c pytest_no_benchmark.ini tests
.PHONY: xbenchmarks

test-examples: ## test examples are working
	bash tests/examples.sh
.PHONY: test-examples

docs: ## generate Sphinx HTML documentation, including API docs
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html
.PHONY: docs

servedocs: ## compile the docs watching for changes
	DOCS_MODE=dev sphinx-autobuild \
		--open-browser \
		--watch ribs/ \
		--watch examples/ \
		docs/ \
		docs/_build/html
.PHONY: servedocs

release-test: dist ## package and upload a release to TestPyPI
	twine upload --repository testpypi dist/*
.PHONY: release-test

release: dist ## package and upload a release
	twine upload dist/*
.PHONY: release

dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist
	check-wheel-contents dist/*.whl
