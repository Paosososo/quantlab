#!/usr/bin/env python
"""Run the research study and write the report.

python scripts/run_research.py             # uses the database
python scripts/run_research.py --offline   # synthetic data, no database needed
"""

from __future__ import annotations

import argparse
import sys

from quantlab.logging import configure_logging
from quantlab.research_run import run_research_study


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbols", nargs="*", help="tickers; defaults to everything stored")
    parser.add_argument("--horizon", type=int, default=1, help="forecast horizon in days")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use the synthetic generator instead of the database",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--test-size", type=int, default=252)
    parser.add_argument("--min-train-size", type=int, default=756)
    args = parser.parse_args(argv)

    configure_logging()
    result = run_research_study(
        symbols=args.symbols or None,
        horizon=args.horizon,
        use_database=not args.offline,
        n_splits=args.n_splits,
        test_size=args.test_size,
        min_train_size=args.min_train_size,
    )
    print("\n" + "=" * 78)
    print(result["verdict"])
    print("=" * 78)
    print(f"\nReport:  {result['report']}")
    print(f"Results: {result['results']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
