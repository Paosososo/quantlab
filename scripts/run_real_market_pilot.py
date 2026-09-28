"""Run a close-only S&P 500 forecast pilot from a downloaded FRED CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from quantlab.research.market_pilot import run_close_only_pilot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_csv", type=Path, help="FRED SP500 CSV snapshot")
    parser.add_argument("output_dir", type=Path, help="new output directory")
    args = parser.parse_args()
    result = run_close_only_pilot(args.source_csv, args.output_dir)
    print(result["report"])
    print(result["results"])


if __name__ == "__main__":
    main()
