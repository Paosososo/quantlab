"""Data validation and quality checks.

Two layers, deliberately separated:

``validate_schema``
    A hard gate.  Wrong columns or unparseable types mean the pipeline stops.
    Loading a frame whose ``close`` column is actually a string is not a
    "warning", it is a bug that will silently produce nonsense returns.

``run_quality_checks``
    A soft gate that produces a report.  Real market data has gaps, zero-volume
    days and genuine 20% moves.  Failing the pipeline on every anomaly means the
    pipeline never runs; recording them means a human can look.

Each check returns a :class:`CheckResult` which is persisted to
``data_quality_checks``, so quality is a queryable time series rather than a
line in a log file.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantlab.db.enums import CheckStatus
from quantlab.exceptions import DuplicateObservationError, SchemaValidationError
from quantlab.logging import get_logger
from quantlab.timeutils import UTC

log = get_logger(__name__)


@dataclass(slots=True)
class CheckResult:
    name: str
    status: CheckStatus
    message: str
    observed: dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.status is CheckStatus.FAILED


@dataclass(slots=True)
class ValidationReport:
    dataset: str
    entity: str | None
    results: list[CheckResult] = field(default_factory=list)
    checked_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(tz=UTC))

    @property
    def ok(self) -> bool:
        return not any(r.failed for r in self.results)

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if r.failed]

    @property
    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if r.status is CheckStatus.WARNED]

    def to_rows(self, ingestion_run_id: int | None = None) -> list[dict[str, Any]]:
        return [
            {
                "dataset": self.dataset,
                "entity": self.entity,
                "check_name": r.name,
                "status": r.status,
                "observed": r.observed,
                "message": r.message,
                "checked_at": self.checked_at,
                "ingestion_run_id": ingestion_run_id,
            }
            for r in self.results
        ]


# ---------------------------------------------------------------------------
# Hard schema gate
# ---------------------------------------------------------------------------
def validate_schema(
    frame: pd.DataFrame,
    *,
    required: Sequence[str],
    numeric: Sequence[str] = (),
    datetime_utc: Sequence[str] = (),
    label: str = "frame",
) -> pd.DataFrame:
    """Check required columns exist and coerce declared types.  Raises on failure."""
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise SchemaValidationError(
            "missing required columns", label=label, missing=missing, got=list(frame.columns)
        )

    out = frame.copy()
    for col in datetime_utc:
        if col not in out.columns:
            continue
        try:
            out[col] = pd.to_datetime(out[col], utc=True, errors="raise")
        except (ValueError, TypeError) as exc:
            raise SchemaValidationError(
                "column is not parseable as a UTC timestamp", label=label, column=col
            ) from exc

    for col in numeric:
        if col not in out.columns:
            continue
        coerced = pd.to_numeric(out[col], errors="coerce")
        newly_null = int(coerced.isna().sum() - out[col].isna().sum())
        if newly_null > 0:
            raise SchemaValidationError(
                "column contains values that are not numeric",
                label=label,
                column=col,
                unparseable=newly_null,
            )
        out[col] = coerced.astype("float64")
    return out


def assert_unique_key(frame: pd.DataFrame, key: Sequence[str], *, label: str = "frame") -> None:
    dupes = frame.duplicated(subset=list(key), keep=False)
    if dupes.any():
        sample = frame.loc[dupes, list(key)].head(5).to_dict("records")
        raise DuplicateObservationError(
            "duplicate key values", label=label, key=list(key), examples=sample
        )


def deduplicate(
    frame: pd.DataFrame, key: Sequence[str], *, keep: str = "last", label: str = "frame"
) -> tuple[pd.DataFrame, int]:
    """Drop duplicate keys, returning the cleaned frame and how many were dropped.

    ``keep='last'`` because providers append corrections after the original
    observation; the later row is the corrected one.
    """
    before = len(frame)
    out = frame.drop_duplicates(subset=list(key), keep=keep).reset_index(drop=True)
    dropped = before - len(out)
    if dropped:
        log.warning("validation.duplicates_dropped", label=label, dropped=dropped, key=list(key))
    return out, dropped


# ---------------------------------------------------------------------------
# Soft quality checks
# ---------------------------------------------------------------------------
def check_no_duplicates(frame: pd.DataFrame, key: Sequence[str]) -> CheckResult:
    n = int(frame.duplicated(subset=list(key)).sum())
    return CheckResult(
        "no_duplicate_keys",
        CheckStatus.PASSED if n == 0 else CheckStatus.FAILED,
        f"{n} duplicate rows on {list(key)}",
        {"duplicates": n},
    )


def check_monotonic_time(frame: pd.DataFrame, column: str = "ts") -> CheckResult:
    ok = frame[column].is_monotonic_increasing
    return CheckResult(
        "timestamps_increasing",
        CheckStatus.PASSED if ok else CheckStatus.FAILED,
        "timestamps are sorted" if ok else "timestamps are out of order",
        {"monotonic": bool(ok)},
    )


def check_availability_not_before_ts(frame: pd.DataFrame) -> CheckResult:
    """The invariant that protects against look-ahead at the source."""
    if "available_at" not in frame.columns or "ts" not in frame.columns:
        return CheckResult(
            "availability_after_ts", CheckStatus.WARNED, "columns absent", {"checked": False}
        )
    bad = int((frame["available_at"] < frame["ts"]).sum())
    return CheckResult(
        "availability_after_ts",
        CheckStatus.PASSED if bad == 0 else CheckStatus.FAILED,
        f"{bad} rows claim to be available before the period they describe",
        {"violations": bad},
    )


def check_ohlc_consistency(frame: pd.DataFrame) -> CheckResult:
    needed = {"open", "high", "low", "close"}
    if not needed.issubset(frame.columns):
        return CheckResult("ohlc_consistency", CheckStatus.WARNED, "OHLC columns absent", {})
    high, low = frame["high"], frame["low"]
    violations = int(
        (
            (high < low)
            | (high < frame["open"])
            | (high < frame["close"])
            | (low > frame["open"])
            | (low > frame["close"])
        ).sum()
    )
    return CheckResult(
        "ohlc_consistency",
        CheckStatus.PASSED if violations == 0 else CheckStatus.FAILED,
        f"{violations} bars violate high >= max(o,c) >= min(o,c) >= low",
        {"violations": violations},
    )


def check_positive_prices(frame: pd.DataFrame, columns: Sequence[str] = ("close",)) -> CheckResult:
    bad = 0
    for col in columns:
        if col in frame.columns:
            bad += int((frame[col] <= 0).sum())
    return CheckResult(
        "positive_prices",
        CheckStatus.PASSED if bad == 0 else CheckStatus.FAILED,
        f"{bad} non-positive price observations",
        {"violations": bad},
    )


def check_missing_values(
    frame: pd.DataFrame, columns: Sequence[str], threshold: float = 0.02
) -> CheckResult:
    present = [c for c in columns if c in frame.columns]
    if not present or frame.empty:
        return CheckResult("missing_values", CheckStatus.WARNED, "nothing to check", {})
    ratios = {c: float(frame[c].isna().mean()) for c in present}
    worst = max(ratios, key=lambda k: ratios[k])
    status = CheckStatus.PASSED if ratios[worst] <= threshold else CheckStatus.WARNED
    return CheckResult(
        "missing_values",
        status,
        f"worst missing ratio {ratios[worst]:.4f} in '{worst}' (threshold {threshold})",
        {"ratios": ratios, "threshold": threshold},
    )


def check_extreme_returns(
    frame: pd.DataFrame, column: str = "adj_close", threshold: float = 0.5
) -> CheckResult:
    """Flag single-day moves above ``threshold``.

    Warn rather than fail: a 50% move is usually an unadjusted split or a bad
    print, but sometimes it is a real event.  The check exists to make the
    researcher look, not to decide for them.
    """
    if column not in frame.columns or len(frame) < 3:
        return CheckResult("extreme_returns", CheckStatus.WARNED, "insufficient data", {})
    series = pd.to_numeric(frame[column], errors="coerce")
    # ``fill_method=None``: forward-filling would hide a gap as a 0% move and
    # so suppress the very outlier this check exists to find.
    moves = series.pct_change(fill_method=None).abs()
    n = int((moves > threshold).sum())
    worst = float(np.nanmax(moves.to_numpy())) if len(moves.dropna()) else float("nan")
    return CheckResult(
        "extreme_returns",
        CheckStatus.PASSED if n == 0 else CheckStatus.WARNED,
        f"{n} moves above {threshold:.0%}; largest {worst:.4f}",
        {"count": n, "largest": None if np.isnan(worst) else worst, "threshold": threshold},
    )


def check_calendar_gaps(
    frame: pd.DataFrame, column: str = "ts", max_gap_days: int = 7
) -> CheckResult:
    """Detect suspicious holes in a daily series.

    Weekends make a 3-day gap normal; a run of holidays can make 4 or 5.  A gap
    beyond a week in a liquid daily series usually means a failed ingestion, not
    a market closure.
    """
    if len(frame) < 3:
        return CheckResult("calendar_gaps", CheckStatus.WARNED, "insufficient data", {})
    gaps = pd.Series(pd.to_datetime(frame[column], utc=True)).diff().dt.days.dropna()
    big = gaps[gaps > max_gap_days]
    return CheckResult(
        "calendar_gaps",
        CheckStatus.PASSED if big.empty else CheckStatus.WARNED,
        f"{len(big)} gaps longer than {max_gap_days} days (largest {int(gaps.max()) if len(gaps) else 0})",
        {"count": int(len(big)), "largest_days": int(gaps.max()) if len(gaps) else 0},
    )


def check_row_count(frame: pd.DataFrame, minimum: int = 1) -> CheckResult:
    n = len(frame)
    return CheckResult(
        "row_count",
        CheckStatus.PASSED if n >= minimum else CheckStatus.FAILED,
        f"{n} rows (minimum {minimum})",
        {"rows": n, "minimum": minimum},
    )


def run_price_quality_checks(
    frame: pd.DataFrame, *, entity: str | None = None, minimum_rows: int = 1
) -> ValidationReport:
    report = ValidationReport(dataset="prices", entity=entity)
    report.results = [
        check_row_count(frame, minimum_rows),
        check_no_duplicates(frame, ["symbol", "ts"] if "symbol" in frame.columns else ["ts"]),
        check_monotonic_time(frame),
        check_availability_not_before_ts(frame),
        check_ohlc_consistency(frame),
        check_positive_prices(frame, ("open", "high", "low", "close")),
        check_missing_values(frame, ("open", "high", "low", "close", "adj_close")),
        check_extreme_returns(frame),
        check_calendar_gaps(frame),
    ]
    if not report.ok:
        log.error(
            "validation.failed",
            entity=entity,
            failures=[r.name for r in report.failures],
        )
    return report


def run_economic_quality_checks(
    frame: pd.DataFrame, *, entity: str | None = None
) -> ValidationReport:
    report = ValidationReport(dataset="economic_data", entity=entity)
    report.results = [
        check_row_count(frame, 1),
        check_no_duplicates(frame, ["code", "ts"] if "code" in frame.columns else ["ts"]),
        check_monotonic_time(frame),
        check_availability_not_before_ts(frame),
        check_missing_values(frame, ("value",), threshold=0.20),
    ]
    return report


__all__ = [
    "CheckResult",
    "ValidationReport",
    "assert_unique_key",
    "check_availability_not_before_ts",
    "check_calendar_gaps",
    "check_extreme_returns",
    "check_missing_values",
    "check_monotonic_time",
    "check_no_duplicates",
    "check_ohlc_consistency",
    "check_positive_prices",
    "check_row_count",
    "deduplicate",
    "run_economic_quality_checks",
    "run_price_quality_checks",
    "validate_schema",
]
