"""Experiment tracking.

Why not MLflow / Weights & Biases
---------------------------------
Both are good tools.  They were rejected here for three reasons specific to this
project:

1.  A tracking server is another service in ``docker compose``, another thing to
    be running before a result can be reproduced, and another dependency for a
    reviewer to install.
2.  The database in this project already has ``model_runs`` and ``predictions``
    tables, which are the same tables a tracking server would keep, and having
    predictions in SQL next to prices is what makes the joins in the API and the
    dashboard trivial.
3.  Reproducibility here rests on **content hashes**, not on a run ID handed out
    by a server: the feature-set spec hash, the model config hash, the data
    fingerprint and the seed.  Two runs agreeing on those four values should
    produce the same numbers, and that property is checkable by anyone with the
    repository, with no server involved.

What is recorded
----------------
A JSON manifest on disk (``data/artifacts/runs/<run>.json``) and a row in
``model_runs``.  The manifest is the portable artefact; the row is the queryable
one.  Both carry the same four hashes plus the settings snapshot, so a result
can be traced back to the exact configuration that produced it.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from quantlab.config import get_settings
from quantlab.db import models as m
from quantlab.db import repository as repo
from quantlab.db.enums import RunStatus, SplitKind
from quantlab.logging import get_logger
from quantlab.models.walkforward import WalkForwardResult

log = get_logger(__name__)


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def frame_fingerprint(frame: pd.DataFrame) -> str:
    """Content hash of a dataframe's values, shape and column names.

    Uses pandas' row hashing, which is fast and stable across runs on the same
    pandas version.  Recorded so that "same config, different numbers" can be
    diagnosed as "the data changed" rather than remaining a mystery.
    """
    from pandas.util import hash_pandas_object

    digest = hashlib.sha256()
    digest.update(str(frame.shape).encode())
    digest.update(",".join(map(str, frame.columns)).encode())
    digest.update(hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def git_revision() -> str | None:
    """Short git SHA, or None outside a repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


@dataclass(slots=True)
class RunManifest:
    run_name: str
    model_type: str
    config: dict[str, Any]
    config_hash: str
    data_fingerprint: str | None
    feature_set_hash: str | None
    seed: int
    code_version: str | None
    started_at: dt.datetime
    finished_at: dt.datetime | None = None
    metrics: dict[str, Any] | None = None
    settings: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "model_type": self.model_type,
            "config": self.config,
            "config_hash": self.config_hash,
            "data_fingerprint": self.data_fingerprint,
            "feature_set_hash": self.feature_set_hash,
            "seed": self.seed,
            "code_version": self.code_version,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "metrics": self.metrics,
            "settings": self.settings,
        }


class ExperimentTracker:
    def __init__(self, artifacts_dir: Path | None = None) -> None:
        settings = get_settings()
        self.root = Path(artifacts_dir or settings.artifacts_dir) / "runs"
        self.root.mkdir(parents=True, exist_ok=True)

    def build_manifest(
        self,
        result: WalkForwardResult,
        *,
        run_name: str,
        data_fingerprint: str | None = None,
        feature_set_hash: str | None = None,
        started_at: dt.datetime | None = None,
        extra_config: dict[str, Any] | None = None,
    ) -> RunManifest:
        config = {
            "model": result.model_name,
            "params": result.model_params,
            "splitter": result.splitter,
            "target": result.target_name,
            "features": result.feature_names,
            **(extra_config or {}),
        }
        return RunManifest(
            run_name=run_name,
            model_type=result.model_name,
            config=config,
            config_hash=stable_hash(config),
            data_fingerprint=data_fingerprint,
            feature_set_hash=feature_set_hash,
            seed=result.seed,
            code_version=git_revision(),
            started_at=started_at or dt.datetime.now(tz=dt.timezone.utc),
            finished_at=dt.datetime.now(tz=dt.timezone.utc),
            metrics=result.summary(),
            settings=get_settings().redacted(),
        )

    def write_manifest(self, manifest: RunManifest) -> Path:
        path = self.root / f"{manifest.run_name}_{manifest.config_hash[:12]}.json"
        path.write_text(json.dumps(manifest.to_dict(), indent=2, default=str), encoding="utf-8")
        log.info("tracking.manifest_written", path=str(path), run=manifest.run_name)
        return path

    def persist_run(
        self,
        session: Session,
        result: WalkForwardResult,
        manifest: RunManifest,
        *,
        feature_set_id: int | None = None,
        store_predictions: bool = True,
        replace_existing: bool = True,
    ) -> m.ModelRun:
        """Write a run, replacing any identical earlier one.

        "Identical" means the same config hash, data fingerprint and seed -- the
        reproducibility triple.  Same configuration on the same data with the
        same seed *is* the same run, so a retried Airflow task must overwrite it
        rather than add a second copy.

        An earlier version inserted unconditionally.  A task that failed after
        writing predictions and then retried left two ``model_runs`` rows and two
        sets of predictions for one logical run, which quietly doubled the
        apparent trial count in the deflated-Sharpe calculation -- a bookkeeping
        bug that changes a reported statistic.
        """
        if replace_existing:
            existing = (
                session.query(m.ModelRun)
                .filter(
                    m.ModelRun.config_hash == manifest.config_hash,
                    m.ModelRun.data_fingerprint == manifest.data_fingerprint,
                    m.ModelRun.seed == manifest.seed,
                )
                .all()
            )
            for row in existing:
                log.info("tracking.replacing_identical_run", model_run_id=row.id)
                session.delete(row)  # predictions cascade
            if existing:
                session.flush()

        predictions = result.predictions
        run = m.ModelRun(
            name=manifest.run_name,
            model_type=manifest.model_type,
            feature_set_id=feature_set_id,
            target_name=result.target_name,
            config=manifest.config,
            config_hash=manifest.config_hash,
            data_fingerprint=manifest.data_fingerprint,
            code_version=manifest.code_version,
            seed=manifest.seed,
            validation_scheme=str(result.splitter.get("splitter", "unknown")),
            train_start=_first_fold_bound(result, "train_start"),
            train_end=_last_fold_bound(result, "train_end"),
            test_start=_first_fold_bound(result, "test_start"),
            test_end=_last_fold_bound(result, "test_end"),
            status=RunStatus.SUCCEEDED,
            started_at=manifest.started_at,
            finished_at=manifest.finished_at,
            metrics=result.overall.to_dict() if result.overall else None,
        )
        session.add(run)
        session.flush()

        if store_predictions and not predictions.empty:
            asset_ids = {
                symbol: asset.id
                for symbol in predictions["symbol"].unique()
                if (asset := repo.get_asset_by_symbol(session, str(symbol))) is not None
            }
            rows = []
            for record in predictions.to_dict("records"):
                asset_id = asset_ids.get(str(record["symbol"]))
                if asset_id is None:
                    continue
                ts = pd.Timestamp(record["ts"]).to_pydatetime()
                target_ts = record.get("target_ts")
                rows.append(
                    {
                        "model_run_id": run.id,
                        "asset_id": asset_id,
                        "ts": ts,
                        "target_ts": pd.Timestamp(target_ts).to_pydatetime()
                        if target_ts is not None and pd.notna(target_ts)
                        else ts + dt.timedelta(days=1),
                        "fold": int(record.get("fold", 0)),
                        "split": SplitKind.TEST,
                        "y_pred": float(record["y_pred"]),
                        "y_true": None
                        if pd.isna(record.get("y_true"))
                        else float(record["y_true"]),
                        "y_proba": None
                        if record.get("y_proba") is None or pd.isna(record.get("y_proba"))
                        else float(record["y_proba"]),
                    }
                )
            repo.bulk_insert(session, m.Prediction, rows)
        log.info("tracking.run_persisted", model_run_id=run.id, model=manifest.model_type)
        return run


def _first_fold_bound(result: WalkForwardResult, key: str) -> dt.datetime | None:
    if not result.folds:
        return None
    value = result.folds[0].split.get(key)
    return pd.Timestamp(value).to_pydatetime() if value else None


def _last_fold_bound(result: WalkForwardResult, key: str) -> dt.datetime | None:
    if not result.folds:
        return None
    value = result.folds[-1].split.get(key)
    return pd.Timestamp(value).to_pydatetime() if value else None


__all__ = [
    "ExperimentTracker",
    "RunManifest",
    "frame_fingerprint",
    "git_revision",
    "stable_hash",
]
