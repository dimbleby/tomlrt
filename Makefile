UV ?= uv

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Common targets:"
	@echo "  make test         # run the test suite (excludes slow/fuzz)"
	@echo "  make fuzz         # run the slow property-based suite"
	@echo "  make coverage     # tests + branch coverage"
	@echo "  make lint         # formatting, typechecking, etc"
	@echo "  make docs         # build the MkDocs site (strict)"
	@echo "  make docs-serve   # preview the docs locally"
	@echo "  make bench        # run the parse-throughput benchmark"
	@echo "  make clean        # remove caches and build artefacts"

.PHONY: test
test:
	pytest -q

.PHONY: fuzz
fuzz:
	pytest -q -m slow

.PHONY: coverage
coverage:
	pytest --cov

.PHONY: lint
lint: fmt ruff ty mypy

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
docs:
	$(UV) run --group docs zensical build

.PHONY: docs-serve
docs-serve:
	$(UV) run --group docs zensical serve

.PHONY: bench
bench:
	benchmarks/bench_parse.py
	benchmarks/bench_mutate.py
	benchmarks/bench_pyproject.py

.PHONY: clean
clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov site dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
