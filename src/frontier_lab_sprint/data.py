"""Synthetic arithmetic data for the first transformer milestone."""

from __future__ import annotations

from dataclasses import dataclass
import random


VOCAB = tuple("0123456789 +=\n")
STOI = {token: index for index, token in enumerate(VOCAB)}
ITOS = {index: token for token, index in STOI.items()}


@dataclass(frozen=True)
class AdditionExample:
    """One fixed-format addition example."""

    a: int
    b: int

    @property
    def text(self) -> str:
        return f"{self.a} + {self.b} = {self.a + self.b}\n"


def generate_addition_examples(
    n: int,
    *,
    max_digits: int = 3,
    seed: int = 0,
) -> list[AdditionExample]:
    """Generate simple addition examples with operands up to ``max_digits`` digits."""

    if n < 0:
        raise ValueError("n must be non-negative")
    if max_digits < 1:
        raise ValueError("max_digits must be at least 1")

    rng = random.Random(seed)
    upper = 10**max_digits - 1
    return [
        AdditionExample(a=rng.randint(0, upper), b=rng.randint(0, upper))
        for _ in range(n)
    ]


def encode(text: str) -> list[int]:
    """Encode text into token ids."""

    try:
        return [STOI[token] for token in text]
    except KeyError as exc:
        raise ValueError(f"unsupported token: {exc.args[0]!r}") from exc


def decode(token_ids: list[int]) -> str:
    """Decode token ids back to text."""

    try:
        return "".join(ITOS[token_id] for token_id in token_ids)
    except KeyError as exc:
        raise ValueError(f"unsupported token id: {exc.args[0]!r}") from exc

