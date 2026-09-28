# Architecture decision records

One short document per decision that a reader might otherwise question. Each
states the context, the decision, the alternatives that were rejected and why,
and the consequences we accepted.

| # | Decision |
|---|---|
| [001](ADR-001-availability-model.md) | Every fact row carries `available_at` alongside `ts` |
| [002](ADR-002-event-driven-backtester.md) | The backtester is an event loop, not vectorised |
| [003](ADR-003-free-data-providers.md) | Stooq, FRED and a synthetic generator; no keys, no scraping |
| [004](ADR-004-postgres-and-constraints.md) | PostgreSQL, with domain rules as CHECK constraints |
| [005](ADR-005-experiment-tracking.md) | Content hashes and JSON manifests instead of MLflow |
| [006](ADR-006-src-layout.md) | An installable `src/` package instead of top-level directories |
