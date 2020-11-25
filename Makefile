# Convenient commands. Run `make help` for command info.
.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint: ## check style with pylint
	pylint ribs tests examples benchmarks

test: ## run tests with the default Python
	pytest tests
test-core: ## only test the core of ribs
	pytest tests/core
test-extras: ## only test the extras of ribs
	pytest tests/extras
test-failed: ## run only tests that filed
	pytest --last-failed
test-only: ## run tests without benchmarks (which take a while)
	pytest -c pytest_no_benchmark.ini tests
test-coverage: ## get better test coverage by running without numba on
	NUMBA_DISABLE_JIT=1 pytest -c pytest_no_benchmark.ini tests
test-all: ## run tests on every Python version with tox
	tox

NUM_CPUS=4
xtest: ## run tests distributed with 4 workers
	pytest -n $(NUM_CPUS) tests
xtest-only: ## run tests without benchmarks distributed over 4 workers
	pytest -n $(NUM_CPUS) -c pytest_no_benchmark.ini tests

examples-test: ## test examples are working
	bash tests/examples.sh

docs: ## generate Sphinx HTML documentation, including API docs
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

servedocs: ## compile the docs watching for changes
	DOCS_MODE=dev sphinx-autobuild \
		--open-browser \
		--watch ribs/ \
		docs/ \
		docs/_build/html

servedocs-ignore-vim: ## compile the docs watching for changes, ignore vim .swp files
	DOCS_MODE=dev sphinx-autobuild \
		--open-browser \
		--watch ribs/ \
		--ignore *.swp \
		docs/ \
		docs/_build/html

release: dist ## package and upload a release
	twine upload dist/*

dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	python setup.py install
