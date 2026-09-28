# ADR-006: An installable `src/` package instead of top-level directories

**Status.** Accepted; a deliberate departure from the brief's suggested layout.

## Context

The brief suggested top-level `ingestion/`, `database/`, `models/`, `api/` and so
on. The same code must be importable from pytest, from Airflow, from a Docker
image, from a Streamlit process and from a notebook.

## Decision

`src/quantlab/{ingestion,db,features,models,backtesting,research,risk,api,dashboard}`,
installed with `pip install -e .`. The subdirectory names are the ones the brief
asked for; only their parent changed.

## Rationale

With top-level packages, the import path depends on the current working
directory. That works in a notebook opened at the repository root and breaks when
Airflow's scheduler imports a DAG from `/opt/airflow/dags`, or when a container
copies the code to `/app`, or when pytest's rootdir handling differs from what
the developer expected. Every one of those is debugged with `sys.path`
manipulation, and `sys.path.append("../..")` at the top of a file is a smell that
never goes away.

An installed package gives one import path that is identical everywhere:
`pip install -e .` in development, a wheel in the Docker image,
`PYTHONPATH=/opt/quantlab/src` in the Airflow image.

The `src` layout specifically (rather than a top-level `quantlab/`) prevents the
classic accident where tests import the local directory instead of the installed
package, and pass against code that was never actually packaged.

## Consequences

*Good.* One import path. `quantlab` is a console script. The Airflow image
installs the package with `--no-deps` so our requirements cannot re-resolve
Airflow's pinned tree.

*Costs.* One more directory level. A contributor must run `pip install -e .`
before tests work, which `make install` does.

*Kept from the brief.* `dags/`, `tests/`, `docs/` and `scripts/` stay at the top
level, because none of them is imported as a library.
