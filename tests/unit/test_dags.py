"""Structural tests for the Airflow DAGs.

Airflow is a heavy optional dependency, so the important checks here are static:
they parse the DAG files with ``ast`` and need nothing installed.  The dynamic
check (does Airflow actually import them?) runs only when Airflow is present.

What is being enforced is the rule from ``dags/_common.py``: a DAG file is a
schedule and a dependency graph, never a place where analysis happens.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

DAGS_DIR = Path(__file__).resolve().parents[2] / "dags"
DAG_FILES = sorted(p for p in DAGS_DIR.glob("*_dag.py"))

#: Libraries that indicate analysis is happening in the DAG file itself.
ANALYSIS_MODULES = {"pandas", "sklearn", "statsmodels", "scipy"}


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def module_level_imports(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:  # module level only, not inside functions
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


class TestDagFilesExist:
    def test_every_required_dag_is_present(self):
        expected = {
            "market_data_ingestion_dag.py",
            "data_validation_dag.py",
            "feature_generation_dag.py",
            "model_training_dag.py",
            "backtest_dag.py",
        }
        assert expected <= {p.name for p in DAG_FILES}

    def test_all_dag_files_parse(self):
        for path in DAG_FILES:
            parse(path)


class TestNoBusinessLogicInDags:
    @pytest.mark.parametrize("path", DAG_FILES, ids=lambda p: p.name)
    def test_no_analysis_libraries_at_module_level(self, path):
        """A DAG that imports pandas at module scope is doing work it should delegate.

        Deferred imports inside a task body are fine and are the pattern used
        throughout: they keep DAG parsing fast, which matters because the
        scheduler re-parses every file on a short interval.
        """
        imported = module_level_imports(parse(path))
        assert not (imported & ANALYSIS_MODULES), (
            f"{path.name} imports {sorted(imported & ANALYSIS_MODULES)} at module level"
        )

    @pytest.mark.parametrize("path", DAG_FILES, ids=lambda p: p.name)
    def test_quantlab_is_imported_lazily(self, path):
        """Project imports belong inside task bodies, not at module scope."""
        assert "quantlab" not in module_level_imports(parse(path))

    @pytest.mark.parametrize("path", DAG_FILES, ids=lambda p: p.name)
    def test_task_bodies_are_short(self, path):
        """Each task should orchestrate, not compute.

        The threshold is generous; it exists to catch a task that has quietly
        grown into a script, not to police formatting.
        """
        tree = parse(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            decorators = {
                d.id if isinstance(d, ast.Name) else getattr(d.func, "id", "")
                for d in node.decorator_list
                if isinstance(d, ast.Name | ast.Call)
            }
            if "task" not in decorators:
                continue
            statements = sum(1 for _ in ast.walk(node) if isinstance(_, ast.stmt))
            assert statements < 60, f"{path.name}:{node.name} has {statements} statements"

    @pytest.mark.parametrize("path", DAG_FILES, ids=lambda p: p.name)
    def test_every_dag_file_documents_itself(self, path):
        assert ast.get_docstring(parse(path)), f"{path.name} has no module docstring"


class TestSchedulingConventions:
    @pytest.mark.parametrize("path", DAG_FILES, ids=lambda p: p.name)
    def test_catchup_is_not_enabled(self, path):
        """These pipelines pull full histories; replaying every past date would
        repeat the same work hundreds of times."""
        source = path.read_text(encoding="utf-8")
        assert "catchup=True" not in source

    def test_shared_defaults_configure_retries(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("_dag_common", DAGS_DIR / "_common.py")
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.DEFAULT_ARGS["retries"] >= 1
        assert module.DEFAULT_ARGS["retry_exponential_backoff"] is True
        defaults = module.dag_defaults()
        assert defaults["catchup"] is False
        assert defaults["max_active_runs"] == 1

    def test_summarise_trims_oversized_payloads(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("_dag_common", DAGS_DIR / "_common.py")
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        big = {"values": ["x" * 100 for _ in range(200)]}
        trimmed = module.summarise(big, limit=500)
        assert trimmed["truncated"] is True
        assert module.summarise({"a": 1}) == {"a": 1}


class TestAirflowImports:
    def test_dagbag_loads_without_errors(self):
        """The real check, when Airflow is available."""
        pytest.importorskip("airflow", reason="Airflow is an optional extra")
        import os

        from airflow.models import DagBag

        os.environ.setdefault("AIRFLOW__CORE__LOAD_EXAMPLES", "False")
        bag = DagBag(dag_folder=str(DAGS_DIR), include_examples=False)
        assert not bag.import_errors, bag.import_errors
        assert {
            "market_data_ingestion",
            "data_validation",
            "feature_generation",
            "model_training",
            "backtesting",
        } <= set(bag.dags)
