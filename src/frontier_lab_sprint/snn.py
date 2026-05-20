"""Small spiking-neural-network utilities for foundation experiments."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class LIFParams:
    """Leaky integrate-and-fire neuron parameters."""

    tau: float = 20.0
    dt: float = 1.0
    threshold: float = 1.0
    reset: float = 0.0
    resistance: float = 1.0


def lif_step(voltage: jax.Array, input_current: jax.Array, params: LIFParams) -> tuple[jax.Array, jax.Array]:
    """Advance one leaky integrate-and-fire neuron step.

    The update is an Euler discretization of:

        tau * dv/dt = -v + resistance * input_current

    A spike is emitted when the updated voltage crosses the threshold, then the
    voltage is reset.
    """

    dv = (params.dt / params.tau) * (-voltage + params.resistance * input_current)
    updated = voltage + dv
    spike = updated >= params.threshold
    next_voltage = jnp.where(spike, params.reset, updated)
    return next_voltage, spike.astype(jnp.float32)


def simulate_lif(input_current: jax.Array, params: LIFParams | None = None) -> tuple[jax.Array, jax.Array]:
    """Simulate a single LIF neuron over a one-dimensional input-current trace."""

    params = params or LIFParams()

    def step(voltage: jax.Array, current: jax.Array) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        next_voltage, spike = lif_step(voltage, current, params)
        return next_voltage, (next_voltage, spike)

    _, (voltages, spikes) = jax.lax.scan(step, jnp.array(0.0, dtype=jnp.float32), input_current)
    return voltages, spikes

