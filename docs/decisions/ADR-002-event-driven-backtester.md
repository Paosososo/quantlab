# ADR-002: The backtester is an event loop, not vectorised

**Status.** Accepted.

## Context

The compact way to write a backtest is three lines:

```python
positions = signal.shift(1)
pnl = positions * returns
equity = (1 + pnl).cumprod()
```

It is fast, and it is how most tutorials do it.

## Decision

The engine walks bars one at a time, maintaining explicit orders, fills,
positions and cash.

## Rationale

The correctness of the vectorised version rests entirely on that single
`shift(1)`, and nobody can verify it by reading. Is the signal computed from the
close of day *t*? Does `returns` mean open-to-open or close-to-close? Does the
shift account for the gap between decision and execution? Every one of those is a
place where a plausible-looking implementation is off by a day, and being off by
a day in the favourable direction produces spectacular backtests.

In the loop, every price used is fetched from a named bar at an explicit
timestamp. A reader can follow one order from decision to fill. For a project
whose entire claim is temporal correctness, auditability beats speed.

The loop also makes things possible that the vectorised form cannot express
naturally: per-order commission with a minimum, participation caps against the
bar's actual volume, orders that stay pending when a symbol did not trade, and
the accounting invariant checked after every bar.

## Alternatives rejected

**Vectorised with careful documentation.** The documentation is not executable.

**An existing library (backtrader, vectorbt, zipline).** Reasonable in
production. Rejected here because the backtester is one of the components the
project is meant to demonstrate; using a library would move the interesting part
out of the repository.

## Consequences

*Good.* Explicit, testable, auditable. The temporal guard is a single assertion
in one function that every fill passes through.

*Costs.* Slower, and the first version was slow enough to matter: an `O(n²)`
point-in-time rebuild made a fifteen-year run take minutes. That was fixed with
prefix slicing and binary search (19.1s to 1.75s, identical results), but the
lesson is recorded: a safe path that is slow gets bypassed, so performance is a
correctness concern here, not just a comfort.

*Also costs.* Roughly 700 lines where three would do.
