"""Top-level dispatcher for the main public entrypoints."""

from __future__ import annotations

import sys


HELP_TEXT = """usage: main.py [-h] {combined,ablation,harmfulness-eval} ...

Dynamic Target Attack project entrypoint

positional arguments:
  {combined,ablation,harmfulness-eval}
    combined            Run the combined DTA attacker pipeline.
    ablation            Run ablation experiments.
    harmfulness-eval    Run the unified harmfulness evaluation CLI.

options:
  -h, --help            show this help message and exit
"""


def main() -> int:
    if len(sys.argv) == 1 or sys.argv[1] in {"-h", "--help"}:
        print(HELP_TEXT)
        return 0

    command = sys.argv[1]
    remaining = sys.argv[2:]

    if command == "combined":
        from combined_attacker import main as combined_main

        sys.argv = ["combined_attacker.py", *remaining]
        combined_main()
        return 0

    if command == "ablation":
        from experiments_ablation import main as ablation_main

        sys.argv = ["experiments_ablation.py", *remaining]
        ablation_main()
        return 0

    if command == "harmfulness-eval":
        from eval.harmfulness.cli import main as harmfulness_main

        return harmfulness_main(remaining)

    print(f"Unsupported command: {command}\n")
    print(HELP_TEXT)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
