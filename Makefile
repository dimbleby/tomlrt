UV ?= uv

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

.PHONY: test
test: ## run the test suite (excludes slow/fuzz)
	pytest -q

.PHONY: fuzz
fuzz: ## run the slow property-based suite
	pytest -q -m slow

.PHONY: coverage
coverage: ## tests + branch coverage
	pytest --cov

.PHONY: lint
lint:fmt ruff ty mypy ## formatting, typechecking, etc

.PHONY: fmt
fmt:
	ruff format --check .

.PHONY: ruff
ruff:
	ruff check .

.PHONY: ty
ty:
	ty check

.PHONY: mypy
mypy:
	mypy

.PHONY: docs
docs: ## build the MkDocs site (strict)
	$(UV) run --group docs zensical build

.PHONY: docs-serve
docs-serve: ## preview the docs locally
	$(UV) run --group docs zensical serve

.PHONY: bench
bench: ## run the parse-throughput benchmark
	benchmarks/bench_parse.py
	benchmarks/bench_mutate.py
	benchmarks/bench_pyproject.py

.PHONY: clean
clean: ## remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov site dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
