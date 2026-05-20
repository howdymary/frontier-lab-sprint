# Frontier Lab Sprint

Two-month frontier-lab preparation project based on Vlad Feinberg's "How to Land a Frontier Lab Job".

The goal is to produce a public evidence packet around a specific frontier-lab-relevant skill:

- Primary wedge: JAX-based transformer, scaling-law analysis, and kernel / inference-performance experiments.
- Backup wedge: rigorous controlled experiments on agentic loops.

Target finish line: 2026-07-20.

## Setup

This repo uses `uv` for Python and dependency management.

```bash
uv sync
uv run python scripts/check_env.py
uv run pytest
```

## First Milestone

Get a synthetic addition dataset running, then train a tiny decoder-only transformer on it.

```bash
uv run python scripts/generate_addition_sample.py --n 10
```

## Evidence Packet

By the end, this repo should contain:

- A JAX/Flax/Optax transformer trained on synthetic addition.
- Scaling-law experiments and writeup.
- Dense vs MoE notes or experiment.
- One Pallas or JAX performance benchmark.
- Paper notes on FlashAttention, quantization, kernels/DSLs, and agent experiments.
- A concise hiring memo and resume bullets.

