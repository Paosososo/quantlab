# ADR-004: PostgreSQL, with domain rules as CHECK constraints

**Status.** Accepted.

## Context

The warehouse holds a few million rows of daily bars, features and backtest
output. The workload is append-mostly with analytical reads.

## Decision

PostgreSQL, with the project's invariants expressed as database constraints
rather than only as Python assertions.

## Rationale

On raw analytical speed, DuckDB or Parquet would win. The reason for PostgreSQL
is that **constraints are the point**:

```sql
CHECK (available_at >= ts)              -- on prices, returns, features, economic_data
CHECK (execution_ts > decision_ts)      -- on orders and trades
CHECK (high >= low AND high >= open AND high >= close)
CHECK (target_ts > ts)                  -- on predictions
```

Those four lines encode the project's central claims at a layer no application
bug can bypass. A leaky feature pipeline would have to violate a constraint to
write its results, and it cannot.

Postgres also gives concurrent writers (Airflow tasks run in parallel), real
transactions, `ON CONFLICT DO UPDATE` for idempotent loads, and JSONB for the
config and metrics blobs.

## Related decisions

**Enums are VARCHAR + CHECK, not native `CREATE TYPE`.** Native enums cannot have
a value removed without recreating the type, and do not exist on SQLite, which
the unit tests use. The CHECK constraint still enforces the domain.

**Money is `NUMERIC`, statistics are `DOUBLE PRECISION`.** The database is the
system of record for accounting quantities and exact decimal arithmetic is free
there. Returns, feature values and metrics are estimates, where float range
matters more than the last decimal place. The Decimal/float conversion happens in
exactly one module, `db/repository.py`, so no other code has to think about it.

**Features are stored long, not wide.** `(feature_set_id, asset_id, name, ts)`
instead of one column per feature. Adding a feature then costs an insert, not a
migration, which matters when the research loop is the point of the system. The
cost is a larger table and a pivot on read.

## Consequences

*Good.* Invariants enforced at the storage layer. CI has a job that fails if the
ORM and the migrations disagree, so the deployed schema always matches the code.

*Costs.* A service to run. Unit tests use SQLite, which required a `UTCDateTime`
column type to restore timezone awareness on read — worth it, because without it
the tests would exercise naive timestamps while production used aware ones.
