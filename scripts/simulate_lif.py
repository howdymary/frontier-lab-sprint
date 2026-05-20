"""Run a small leaky integrate-and-fire neuron simulation."""

from __future__ import annotations

import argparse

import jax.numpy as jnp
from rich.console import Console

from frontier_lab_sprint.snn import LIFParams, simulate_lif


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--current", type=float, default=1.6)
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--threshold", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()
    params = LIFParams(tau=args.tau, threshold=args.threshold)
    currents = jnp.full((args.steps,), args.current, dtype=jnp.float32)
    voltages, spikes = simulate_lif(currents, params)

    console.print("[bold]Leaky integrate-and-fire simulation[/bold]")
    console.print(f"steps={args.steps} current={args.current} tau={args.tau} threshold={args.threshold}")
    console.print(f"spike_count={int(spikes.sum())}")
    console.print("first_10_voltages=" + ", ".join(f"{float(v):.3f}" for v in voltages[:10]))
    console.print("spike_train=" + "".join("|" if float(s) else "." for s in spikes))


if __name__ == "__main__":
    main()

