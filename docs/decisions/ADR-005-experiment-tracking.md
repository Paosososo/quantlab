# ADR-005: Content hashes and JSON manifests instead of MLflow

**Status.** Accepted.

## Context

Model runs need to be reproducible and comparable.

## Decision

Reproducibility rests on four content hashes — feature-set spec, model config,
data fingerprint, seed — written to a JSON manifest on disk and to a `model_runs`
row.

## Alternatives rejected

**MLflow.** A good tool. Rejected for three project-specific reasons: it is
another service in `docker compose` and another thing that must be running before
a result can be reproduced; the database already has `model_runs` and
`predictions` tables, which are the same tables a tracking server would keep, and
having predictions in SQL next to prices makes the API and dashboard joins
trivial; and a run ID handed out by a server is not the same guarantee as a
content hash.

**Weights & Biases.** Cloud-hosted, needs an account and a key. Same objection as
a paid data API.

**Nothing.** Rejected because "same config, different numbers" then has no
diagnosis.

## The property being bought

Two runs agreeing on all four hashes should produce the same numbers, and anyone
with the repository can check that. It does not depend on a server being up, an
account existing, or a version string somebody typed by hand being accurate.

The data fingerprint earns its place: without it, "same config, different
numbers" is a mystery. With it, it is diagnosed as "the data changed".

## Consequences

*Good.* No extra service. Results queryable in SQL alongside prices and trades.
The reproducibility claim is verifiable rather than asserted.

*Costs.* No experiment-comparison UI beyond `/model-runs/comparison` and the
dashboard table. No artefact store for fitted models — they are refitted per fold
by design, so there is nothing durable to store, but a production system that
served a fitted model would need one.
