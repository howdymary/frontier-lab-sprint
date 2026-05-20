"""Check that the local sprint environment is usable."""

from __future__ import annotations

import importlib
import platform

from rich.console import Console


REQUIRED_MODULES = [
    "jax",
    "flax",
    "optax",
    "numpy",
    "matplotlib",
    "pandas",
    "tqdm",
]


def main() -> None:
    console = Console()
    console.print("[bold]Frontier Lab Sprint environment check[/bold]")
    console.print(f"Python: {platform.python_version()}")
    console.print(f"Machine: {platform.machine()}")

    for module_name in REQUIRED_MODULES:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        console.print(f"{module_name}: {version}")

    import jax

    console.print(f"JAX backend: {jax.default_backend()}")
    console.print(f"JAX devices: {jax.devices()}")
    console.print("[green]Environment looks ready.[/green]")


if __name__ == "__main__":
    main()

