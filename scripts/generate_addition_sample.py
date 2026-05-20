"""Print a few synthetic addition examples."""

from __future__ import annotations

import argparse

from frontier_lab_sprint.data import encode, generate_addition_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Number of examples to print.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--max-digits", type=int, default=3, help="Maximum operand digits.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = generate_addition_examples(args.n, max_digits=args.max_digits, seed=args.seed)
    for example in examples:
        token_ids = encode(example.text)
        print(f"{example.text.rstrip()}  tokens={token_ids}")


if __name__ == "__main__":
    main()

