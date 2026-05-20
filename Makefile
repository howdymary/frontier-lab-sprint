.PHONY: setup check test data

setup:
	uv sync

check:
	uv run python scripts/check_env.py

test:
	uv run pytest

data:
	uv run python scripts/generate_addition_sample.py --n 10

