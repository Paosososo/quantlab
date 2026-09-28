# ADR-003: Stooq, FRED and a synthetic generator

**Status.** Accepted.

## Context

The project must run after a clone, without a paid subscription, and its tests
must be deterministic and offline.

## Decision

Three providers behind one interface:

- **Stooq** for daily OHLCV. A plain CSV endpoint, no key, no registration, long
  history for US equities, ETFs and major indices.
- **FRED** for macro, via `fredgraph.csv`, which needs no key.
- **A seeded synthetic generator** for tests, CI and the offline demo.

## Alternatives rejected

**yfinance.** Scrapes an undocumented endpoint that changes without notice and
sits in a licensing grey area. A portfolio project should not depend on it.

**Alpha Vantage / Tiingo / Polygon.** Good APIs, but all require a key, which
breaks "clone and run" and means CI needs a secret.

**Committing a CSV snapshot.** Deterministic, but frozen, and it makes the
ingestion framework decorative.

## Why the synthetic provider matters more than it looks

It implements the same `PriceProvider` interface, returns a real `RawPayload` and
has a pure `parse`. A test that passes against it is testing the real loader, not
a mock of it. It also has a `autocorrelation` parameter, which is what makes the
two-sided validation possible: generate a series with known predictability and
check the pipeline finds it; generate a random walk and check it finds nothing. A
backtester that cannot tell those apart is broken, and you cannot run that test
without synthetic data.

## Consequences

*Good.* No keys, no secrets in CI, deterministic tests, an offline demo.

*Costs, stated plainly.* Stooq returns one close series. It is split-adjusted;
we do not assume dividend adjustment, so every return in this project is a price
return, understating total return for dividend payers. Delisted symbols are not
available, which caps how far survivorship bias can be addressed. The FRED CSV
endpoint returns the latest vintage, so macro values are revised ones. All three
are documented in `docs/methodology.md` rather than glossed over.
