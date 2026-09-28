"""Feature computation and persistence.

Macro joins are the subtle part
-------------------------------
A macro series is joined onto price bars with ``merge_asof`` on
``available_at``, not on ``ts``.  Concretely: on 2024-02-15 a model may use the
January CPI print only if January CPI had actually been released by then.
Joining on observation date instead would hand the model the January number on
31 January, roughly two weeks before anyone had it.  That single join is the
difference between a macro feature and a time machine.

Labels are computed here too, in one place, and are never returned in the same
frame as the features.  ``build_training_frame`` returns ``(X, y)`` separately,
so a label cannot end up in a feature matrix by accident.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from quantlab.db import models as m
from quantlab.db import repository as repo
from quantlab.features.base import FeatureSetSpec
from quantlab.features.guards import assert_availability_ordering
from quantlab.features.transforms import forward_return
from quantlab.logging import get_logger

log = get_logger(__name__)

REQUIRED_PRICE_COLUMNS = (
    "symbol",
    "ts",
    "available_at",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
)


# ---------------------------------------------------------------------------
# Macro alignment
# ---------------------------------------------------------------------------
def attach_macro(
    prices: pd.DataFrame,
    macro: pd.DataFrame | None,
    *,
    codes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """As-of join macro observations onto one symbol's price frame.

    ``prices`` must be sorted by ``available_at``.  ``macro`` is long-form
    ``[code, ts, available_at, value]``.  For each price bar, each macro column
    takes the most recent value whose ``available_at`` is at or before the
    bar's own ``available_at``.
    """
    out = prices.sort_values("available_at").reset_index(drop=True)
    if macro is None or macro.empty:
        return out

    wanted = list(codes) if codes else sorted(macro["code"].unique())
    for code in wanted:
        series = (
            macro.loc[macro["code"] == code, ["available_at", "value"]]
            .dropna(subset=["available_at"])
            .sort_values("available_at")
            .reset_index(drop=True)
        )
        if series.empty:
            out[f"macro__{code}"] = np.nan
            continue
        series = series.rename(columns={"value": f"macro__{code}"})
        out = pd.merge_asof(
            out,
            series,
            on="available_at",
            direction="backward",
            allow_exact_matches=True,
        )
    return out


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------
def compute_symbol_features(
    frame: pd.DataFrame, feature_set: FeatureSetSpec, *, symbol: str | None = None
) -> pd.DataFrame:
    """Compute every feature for one symbol.  Row count and order are preserved.

    Preserving row count matters: the leakage harness compares the output
    row-for-row against a run on corrupted data, and a function that dropped
    NaN rows would make that comparison meaningless.
    """
    ordered = frame.sort_values("ts").reset_index(drop=True)
    values: dict[str, pd.Series] = {}
    for spec in feature_set.specs:
        try:
            series = spec.fn(ordered)
        except KeyError as exc:
            raise KeyError(
                f"feature {spec.name!r} needs column {exc.args[0]!r} which is not in the frame"
            ) from exc
        if len(series) != len(ordered):
            raise ValueError(
                f"feature {spec.name!r} returned {len(series)} values for {len(ordered)} rows"
            )
        values[spec.name] = pd.Series(
            np.asarray(series, dtype="float64"), index=ordered.index, name=spec.name
        )
    out = pd.DataFrame(values, index=ordered.index)
    if symbol is not None:
        out.insert(0, "symbol", symbol)
    return out


def build_feature_frame(
    prices: pd.DataFrame,
    feature_set: FeatureSetSpec,
    *,
    macro: pd.DataFrame | None = None,
    macro_codes: Sequence[str] | None = None,
    drop_warmup: bool = True,
) -> pd.DataFrame:
    """Compute features for every symbol.

    Returns a tidy frame indexed by ``[symbol, ts]`` with ``available_at`` and
    one column per feature.  A feature's ``available_at`` is the availability of
    the most recent input observation, which for a causal feature is the bar's
    own availability.
    """
    missing = [c for c in REQUIRED_PRICE_COLUMNS if c not in prices.columns]
    if missing:
        raise KeyError(f"price frame is missing columns: {missing}")
    assert_availability_ordering(prices, label="prices")

    pieces: list[pd.DataFrame] = []
    for symbol, group in prices.groupby("symbol", sort=True):
        enriched = (
            attach_macro(group, macro, codes=macro_codes).sort_values("ts").reset_index(drop=True)
        )
        features = compute_symbol_features(enriched, feature_set, symbol=str(symbol))
        features.insert(1, "ts", enriched["ts"].to_numpy())
        features.insert(2, "available_at", enriched["available_at"].to_numpy())
        pieces.append(features)

    if not pieces:
        return pd.DataFrame(columns=["symbol", "ts", "available_at", *feature_set.names])

    out = pd.concat(pieces, ignore_index=True)
    if drop_warmup:
        feature_columns = feature_set.names
        out = out.dropna(subset=feature_columns, how="all").reset_index(drop=True)
    return out


def build_labels(
    prices: pd.DataFrame,
    *,
    horizon: int = 1,
    price_field: str = "adj_close",
    label_name: str | None = None,
    classification: bool = False,
) -> pd.DataFrame:
    """Forward returns, with an honest availability timestamp.

    The label for bar ``t`` over horizon ``h`` is only knowable once bar ``t+h``
    has been published, so its ``available_at`` is bar ``t+h``'s availability.
    Carrying that column is what lets the splitters purge overlapping labels
    instead of assuming a one-day horizon.
    """
    name = label_name or (f"label_up_{horizon}d" if classification else f"fwd_ret_{horizon}d")
    pieces: list[pd.DataFrame] = []
    for symbol, group in prices.groupby("symbol", sort=True):
        ordered = group.sort_values("ts").reset_index(drop=True)
        values = forward_return(ordered[price_field].astype("float64"), horizon)
        label_available = ordered["available_at"].shift(-horizon)
        frame = pd.DataFrame(
            {
                "symbol": str(symbol),
                "ts": ordered["ts"],
                "available_at": ordered["available_at"],
                "label_available_at": label_available,
                name: (values > 0).astype("float64").where(values.notna())
                if classification
                else values,
            }
        )
        pieces.append(frame)
    out = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    return out.dropna(subset=[name]).reset_index(drop=True)


def build_training_frame(
    prices: pd.DataFrame,
    feature_set: FeatureSetSpec,
    *,
    macro: pd.DataFrame | None = None,
    horizon: int = 1,
    classification: bool = False,
    price_field: str = "adj_close",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return ``(X, y, meta)`` with features and labels kept in separate objects.

    ``meta`` carries ``symbol``, ``ts``, ``available_at`` (feature availability)
    and ``label_available_at`` (when the label became knowable).  Splitters need
    both to purge and embargo correctly.
    """
    features = build_feature_frame(prices, feature_set, macro=macro)
    labels = build_labels(
        prices, horizon=horizon, price_field=price_field, classification=classification
    )
    reserved = {"symbol", "ts", "available_at", "label_available_at"}
    label_name = next(c for c in labels.columns if c not in reserved)

    merged = features.merge(
        labels[["symbol", "ts", "label_available_at", label_name]],
        on=["symbol", "ts"],
        how="inner",
        validate="one_to_one",
    )
    feature_columns = feature_set.names
    merged = merged.dropna(subset=feature_columns).reset_index(drop=True)
    merged = merged.sort_values(["ts", "symbol"]).reset_index(drop=True)

    meta = merged[["symbol", "ts", "available_at", "label_available_at"]].copy()
    X = merged[feature_columns].copy()
    y = merged[label_name].astype("float64").copy()
    y.name = label_name
    log.info(
        "features.training_frame",
        rows=len(X),
        features=len(feature_columns),
        symbols=int(meta["symbol"].nunique()),
        label=label_name,
    )
    return X, y, meta


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def persist_feature_set(session: Session, feature_set: FeatureSetSpec) -> m.FeatureSet:
    spec_hash = feature_set.spec_hash()
    existing = (
        session.query(m.FeatureSet)
        .filter(m.FeatureSet.name == feature_set.name, m.FeatureSet.version == feature_set.version)
        .one_or_none()
    )
    if existing is not None:
        if existing.spec_hash != spec_hash:
            raise ValueError(
                f"feature set {feature_set.name} v{feature_set.version} already exists with a "
                f"different definition (stored {existing.spec_hash[:12]}, new {spec_hash[:12]}). "
                "Bump the version rather than mutating a published definition."
            )
        return existing
    row = m.FeatureSet(
        name=feature_set.name,
        version=feature_set.version,
        spec_hash=spec_hash,
        definition=feature_set.definition(),
        description=feature_set.description,
    )
    session.add(row)
    session.flush()
    log.info("features.set_persisted", name=row.name, version=row.version, hash=spec_hash[:12])
    return row


def persist_feature_values(
    session: Session,
    feature_set_row: m.FeatureSet,
    frame: pd.DataFrame,
    *,
    feature_names: Sequence[str] | None = None,
) -> int:
    """Write a wide feature frame into the long ``features`` table."""
    if frame.empty:
        return 0
    names = list(
        feature_names or [c for c in frame.columns if c not in {"symbol", "ts", "available_at"}]
    )
    asset_ids: dict[str, int] = {}
    for symbol in frame["symbol"].unique():
        asset = repo.get_asset_by_symbol(session, str(symbol))
        if asset is None:
            log.warning("features.unknown_symbol", symbol=symbol)
            continue
        asset_ids[str(symbol)] = asset.id

    rows: list[dict[str, Any]] = []
    for record in frame.to_dict("records"):
        asset_id = asset_ids.get(str(record["symbol"]))
        if asset_id is None:
            continue
        ts = pd.Timestamp(record["ts"]).to_pydatetime()
        available_at = pd.Timestamp(record["available_at"]).to_pydatetime()
        for name in names:
            value = record.get(name)
            rows.append(
                {
                    "feature_set_id": feature_set_row.id,
                    "asset_id": asset_id,
                    "name": name,
                    "ts": ts,
                    "available_at": available_at,
                    "value": None if value is None or pd.isna(value) else float(value),
                }
            )
    written = repo.upsert_feature_values(session, rows)
    log.info("features.values_persisted", rows=written, feature_set_id=feature_set_row.id)
    return written


def load_feature_matrix(
    session: Session,
    feature_set_id: int,
    symbols: Sequence[str] | None = None,
    *,
    as_of: dt.datetime | None = None,
) -> pd.DataFrame:
    """Read features back as a wide frame indexed by ``(symbol, ts)``."""
    long_frame = repo.load_features(session, feature_set_id, symbols, as_of=as_of)
    return repo.pivot_features(long_frame)


def summarise_coverage(frame: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    """Per-feature non-null coverage, for the validation DAG and the docs."""
    present = [c for c in feature_names if c in frame.columns]
    if frame.empty or not present:
        return pd.DataFrame(columns=["feature", "rows", "non_null", "coverage"])
    rows = [
        {
            "feature": name,
            "rows": int(len(frame)),
            "non_null": int(frame[name].notna().sum()),
            "coverage": float(frame[name].notna().mean()),
        }
        for name in present
    ]
    return pd.DataFrame(rows).sort_values("coverage").reset_index(drop=True)


__all__ = [
    "REQUIRED_PRICE_COLUMNS",
    "attach_macro",
    "build_feature_frame",
    "build_labels",
    "build_training_frame",
    "compute_symbol_features",
    "load_feature_matrix",
    "persist_feature_set",
    "persist_feature_values",
    "summarise_coverage",
]
