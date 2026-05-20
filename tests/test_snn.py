import jax.numpy as jnp

from frontier_lab_sprint.snn import LIFParams, lif_step, simulate_lif


def test_lif_step_spikes_and_resets() -> None:
    next_voltage, spike = lif_step(
        jnp.array(0.95, dtype=jnp.float32),
        jnp.array(2.0, dtype=jnp.float32),
        LIFParams(tau=10.0, threshold=1.0, reset=0.0),
    )
    assert float(spike) == 1.0
    assert float(next_voltage) == 0.0


def test_simulate_lif_eventually_spikes() -> None:
    currents = jnp.full((40,), 1.6, dtype=jnp.float32)
    _, spikes = simulate_lif(currents, LIFParams(tau=10.0, threshold=1.0))
    assert float(spikes.sum()) > 0

