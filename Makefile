VENV := .venv
PY := $(VENV)/bin/python
UV := $(shell command -v uv 2>/dev/null)

.DEFAULT_GOAL := help
.PHONY: help install format lint typecheck test check clean analyze generate

define require_jd
	if [ -z "$(JD)" ]; then \
		echo "Set JD to a job description file, e.g. make $@ JD=examples/sample_jd.txt"; \
		exit 1; \
	fi
endef

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create the virtualenv and install the package with dev dependencies
ifdef UV
	uv venv --allow-existing $(VENV)
	uv pip install --python $(PY) -e ".[dev]"
else
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"
endif

format: ## Apply ruff formatting and import fixes
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

lint: ## Check style and formatting without modifying files
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

typecheck: ## Run mypy in strict mode over the package
	$(PY) -m mypy src

test: ## Run the test suite with coverage
	$(PY) -m pytest

check: lint typecheck test ## Run all quality gates

analyze: ## Analyze a job description (JD=examples/sample_jd.txt)
	@$(call require_jd)
	$(VENV)/bin/resumelab analyze --jd $(JD)

generate: ## Generate a resume for a job description (JD=examples/sample_jd.txt)
	@$(call require_jd)
	$(VENV)/bin/resumelab generate --jd $(JD)

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
