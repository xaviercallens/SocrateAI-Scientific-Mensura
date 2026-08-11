.PHONY: setup develop test test-rust lint typecheck fmt bench fetch-data docs clean

UV ?= uv

setup:
	$(UV) venv --allow-existing
	$(UV) pip install -e ".[dev,notebooks]"
	$(UV) run maturin develop --release -m rust/socrates_numerics/Cargo.toml

develop:
	$(UV) run maturin develop --release -m rust/socrates_numerics/Cargo.toml

test:
	$(UV) run pytest -m "not network" -q
	cd rust/socrates_numerics && cargo test --lib

test-all:
	$(UV) run pytest -q
	cd rust/socrates_numerics && cargo test --lib

lint:
	$(UV) run ruff check src tests
	cd rust/socrates_numerics && cargo clippy -- -D warnings

typecheck:
	$(UV) run mypy src/socrates

fmt:
	$(UV) run ruff format src tests
	cd rust/socrates_numerics && cargo fmt

bench:
	$(UV) run pytest tests/benchmarks -m "not network" --benchmark-only

fetch-data:
	$(UV) run python scripts/fetch_data.py

docs:
	$(UV) run mkdocs serve

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build site
	find . -type d -name __pycache__ -exec rm -rf {} +
