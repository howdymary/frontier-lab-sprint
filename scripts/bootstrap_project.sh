#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it with Homebrew first:"
  echo "  brew install uv"
  exit 1
fi

echo "Installing project dependencies..."
uv sync

echo "Checking environment..."
uv run python scripts/check_env.py

echo "Running tests..."
uv run pytest

echo "Generating sample addition data..."
uv run python scripts/generate_addition_sample.py --n 5

echo "Running spiking-neuron simulation..."
uv run python scripts/simulate_lif.py

echo "Project bootstrap complete."

