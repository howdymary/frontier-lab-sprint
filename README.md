# Frontier Lab Sprint

Two-month frontier-lab preparation project based on Vlad Feinberg's "How to Land a Frontier Lab Job".

New computer setup starts here: [START_HERE.md](START_HERE.md)

The goal is to produce a public evidence packet around a specific frontier-lab-relevant skill:

- Foundation track: neural networks, backpropagation, optimization, and spiking neural networks.
- Primary wedge: JAX-based transformer, scaling-law analysis, and kernel / inference-performance experiments.
- Backup wedge: rigorous controlled experiments on agentic loops.

Target finish line: 2026-07-20.

## Setup

This repo uses `uv` for Python and dependency management.

```bash
uv sync
uv run python scripts/check_env.py
uv run pytest
uv run python scripts/simulate_lif.py
```

Or run:

```bash
./scripts/bootstrap_project.sh
```

## First Milestone

Get a synthetic addition dataset running, then train a tiny decoder-only transformer on it.

```bash
uv run python scripts/generate_addition_sample.py --n 10
```

The neural-network and spiking-neural-network foundation track lives in [docs/neural-networks-and-snn.md](docs/neural-networks-and-snn.md).

## Evidence Packet

By the end, this repo should contain:

- A JAX/Flax/Optax transformer trained on synthetic addition.
- Foundation notes on neural networks, optimization, and spiking neural networks.
- Scaling-law experiments and writeup.
- Dense vs MoE notes or experiment.
- One Pallas or JAX performance benchmark.
- Paper notes on FlashAttention, quantization, kernels/DSLs, and agent experiments.
- A concise hiring memo and resume bullets.
