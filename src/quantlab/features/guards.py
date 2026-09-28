"""Leakage detection.

Two independent controls, because either alone is escapable:

**Structural** (:mod:`quantlab.pit`)
    Code is handed a view that physically contains no future rows.

**Behavioural** (this module)
    Compute features on the full history.  Then corrupt everything after a cut
    point and recompute.  If any feature value at or before the cut moved, that
    feature read the future.

The behavioural test is the strong one.  It makes no assumption about *how* a
feature is implemented -- pandas, numpy, a fitted scaler, a call into another
library -- and it catches the leaks that a code review misses:

* ``rolling(window, center=True)``
* ``fillna(method="bfill")`` and time interpolation across gaps
* whole-sample standardisation or min-max scaling
* a target column accidentally left in the feature matrix
* a scaler fitted on train+test and applied to both

Corruption strategy
-------------------
Future rows are replaced with a random permutation of the observed values, sign
flipped and modestly rescaled, with a fraction set to NaN.  Two properties
matter:

*Same order of magnitude.*  An earlier version of this harness replaced the
future with values 1000x the series' scale.  That produced false positives in
rolling moments on the pandas version used then: a global numerical
re-centring step let a huge future tail affect rounding in the visible head.
The implementation detail varies by pandas version, so tests assert the
behavioural contract instead: causal rolling skew must pass the harness.
Keeping corruption near the series' own scale avoids the historical artefact.

*Still unmistakable if a leak exists.*  A feature that actually reads a future
value changes by an amount comparable to its own standard deviation -- a
relative difference of order 1.  The tolerance below is a relative one at 1e-6
of the feature's scale, so a genuine leak clears it by six orders of magnitude
while library round-off does not.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantlab.exceptions import LookAheadError
from quantlab.logging import get_logger

log = get_logger(__name__)

FeatureFn = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass(slots=True)
class LeakageFinding:
    feature: str
    cut_index: int
    n_changed: int
    max_absolute_difference: float
    first_changed_index: int

    def describe(self) -> str:
        return (
            f"{self.feature}: {self.n_changed} value(s) at or before index "
            f"{self.cut_index} changed when future data was corrupted "
            f"(largest change {self.max_absolute_difference:.6g}, "
            f"first at index {self.first_changed_index})"
        )


@dataclass(slots=True)
class LeakageReport:
    findings: list[LeakageFinding] = field(default_factory=list)
    checked_features: list[str] = field(default_factory=list)
    cut_points: list[int] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.findings

    def raise_if_leaky(self) -> None:
        if self.findings:
            raise LookAheadError(
                "feature computation depends on future data",
                findings=[f.describe() for f in self.findings][:10],
                n_findings=len(self.findings),
            )


def corrupt_future(
    frame: pd.DataFrame,
    cut_index: int,
    *,
    numeric_columns: Sequence[str] | None = None,
    rng: np.random.Generator | None = None,
    nan_fraction: float = 0.3,
    magnitude: float = 3.0,
) -> pd.DataFrame:
    """Return a copy of ``frame`` whose rows after ``cut_index`` are destroyed.

    The replacement is a shuffled, sign-flipped, ``magnitude``-scaled version of
    the column's own observed values, with ``nan_fraction`` of them blanked.
    Shuffling and sign flipping guarantee the tail no longer carries the
    original information; the NaNs catch backward-fill; staying near the
    column's own scale avoids the pandas re-centring artefact described in the
    module docstring.
    """
    rng = rng or np.random.default_rng(12345)
    out = frame.copy()
    if numeric_columns is None:
        numeric_columns = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    n_tail = len(out) - (cut_index + 1)
    if n_tail <= 0:
        return out
    tail_index = out.index[cut_index + 1 :]
    for column in numeric_columns:
        values = out[column].to_numpy(dtype="float64", na_value=np.nan)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        pool = rng.permutation(finite)
        replacement = np.resize(pool, n_tail).astype("float64")
        signs = rng.choice(np.array([-1.0, 1.0]), size=n_tail)
        centre = float(np.mean(finite))
        # Reflect around the column mean and rescale, so the tail keeps the
        # column's order of magnitude while carrying none of its sequence.
        corrupted = centre + signs * magnitude * (replacement - centre)
        corrupted[rng.random(n_tail) < nan_fraction] = np.nan
        out.loc[tail_index, column] = corrupted
    return out


def _compare(
    baseline: pd.DataFrame,
    perturbed: pd.DataFrame,
    cut_index: int,
    *,
    relative_tolerance: float,
    absolute_floor: float,
) -> list[tuple[str, int, float, int]]:
    """Columns whose values at or before ``cut_index`` differ materially.

    The threshold for each column is ``relative_tolerance * scale`` where
    ``scale`` is the column's own standard deviation over the visible region
    (floored at ``absolute_floor``).  Comparing on the feature's own scale is
    what makes one tolerance work for a return of order 0.01 and an RSI of order
    50 without hand-tuning either.
    """
    findings: list[tuple[str, int, float, int]] = []
    shared = [c for c in baseline.columns if c in perturbed.columns]
    for column in shared:
        left = baseline[column].iloc[: cut_index + 1]
        right = perturbed[column].iloc[: cut_index + 1]
        if not pd.api.types.is_numeric_dtype(left) or not pd.api.types.is_numeric_dtype(right):
            differs = (left.astype(str) != right.astype(str)).to_numpy()
            diff = np.where(differs, np.inf, 0.0)
        else:
            a = left.to_numpy(dtype="float64", na_value=np.nan)
            b = right.to_numpy(dtype="float64", na_value=np.nan)
            both_nan = np.isnan(a) & np.isnan(b)
            one_nan = np.isnan(a) ^ np.isnan(b)
            scale = float(np.nanstd(a)) if np.isfinite(a).any() else 0.0
            threshold = max(relative_tolerance * scale, absolute_floor)
            diff = np.abs(np.where(both_nan, 0.0, a - b))
            diff = np.where(one_nan, np.inf, diff)
            differs = diff > threshold
        n_changed = int(differs.sum())
        if n_changed:
            changed = diff[differs]
            largest = float(np.max(changed)) if changed.size else float("inf")
            findings.append((column, n_changed, largest, int(np.argmax(differs))))
    return findings


def punch_gap(
    frame: pd.DataFrame,
    cut_index: int,
    *,
    numeric_columns: Sequence[str] | None = None,
    width: int = 3,
) -> pd.DataFrame:
    """Blank the last ``width`` rows before ``cut_index`` in every numeric column.

    This exists to make backward-fill leaks detectable.  ``bfill`` on data with
    no gaps is a no-op, so a harness run on clean input would give it a pass.
    Punching an identical hole into both the clean and the corrupted frame means
    a causal feature produces the same values on both sides of the comparison,
    while anything that fills the hole from later rows produces different ones,
    because those later rows are exactly what the corruption changed.
    """
    out = frame.copy()
    if numeric_columns is None:
        numeric_columns = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    start = max(cut_index - width + 1, 0)
    if start > cut_index:
        return out
    hole = out.index[start : cut_index + 1]
    for column in numeric_columns:
        out.loc[hole, column] = np.nan
    return out


def assert_causal(
    compute: FeatureFn,
    frame: pd.DataFrame,
    *,
    cut_fractions: Sequence[float] = (0.4, 0.6, 0.85),
    relative_tolerance: float = 1e-6,
    absolute_floor: float = 1e-12,
    input_columns: Sequence[str] | None = None,
    gap_width: int = 3,
    seed: int = 12345,
    raise_on_finding: bool = True,
) -> LeakageReport:
    """Verify that ``compute`` uses no information after each cut point.

    ``compute`` must be a pure function from an input frame to an output frame
    with the same number of rows in the same order.  For each cut point it is
    called twice on frames that are **identical up to the cut** and differ only
    afterwards.  Any output difference at or before the cut is a leak.

    Set ``gap_width`` to 0 to skip the backward-fill probe.
    """
    if len(compute(frame)) != len(frame):
        raise ValueError(
            "assert_causal requires the feature function to preserve row count and order"
        )

    report = LeakageReport(checked_features=[])
    rng = np.random.default_rng(seed)
    for fraction in cut_fractions:
        cut_index = int(len(frame) * fraction)
        if cut_index < 2 or cut_index >= len(frame) - 1:
            continue
        report.cut_points.append(cut_index)

        # Two probes per cut point, because they catch different bugs and each
        # blinds the other.  The clean probe detects windows that reach forward
        # (``center=True``, a negative shift, a whole-sample statistic).  The
        # gapped probe detects fills that reach forward (``bfill``, time
        # interpolation) -- but its NaN hole also blanks the very rows where a
        # forward-reaching window would have shown up, so it cannot replace the
        # first.  Running both costs two extra evaluations and closes the gap.
        probes: list[pd.DataFrame] = [frame]
        if gap_width > 0:
            probes.append(
                punch_gap(frame, cut_index, numeric_columns=input_columns, width=gap_width)
            )

        for probe in probes:
            baseline = compute(probe)
            corrupted = corrupt_future(probe, cut_index, numeric_columns=input_columns, rng=rng)
            perturbed = compute(corrupted)
            if not report.checked_features:
                report.checked_features = list(baseline.columns)
            for column, n_changed, largest, first in _compare(
                baseline,
                perturbed,
                cut_index,
                relative_tolerance=relative_tolerance,
                absolute_floor=absolute_floor,
            ):
                report.findings.append(
                    LeakageFinding(
                        feature=column,
                        cut_index=cut_index,
                        n_changed=n_changed,
                        max_absolute_difference=largest,
                        first_changed_index=first,
                    )
                )

    if report.findings:
        log.error(
            "leakage.detected",
            n_findings=len(report.findings),
            features=sorted({f.feature for f in report.findings}),
        )
    if raise_on_finding:
        report.raise_if_leaky()
    return report


def assert_availability_ordering(frame: pd.DataFrame, *, label: str = "frame") -> None:
    """Every row must claim availability at or after the period it describes."""
    if "ts" not in frame.columns or "available_at" not in frame.columns:
        raise LookAheadError("frame lacks ts/available_at columns", label=label)
    ts = pd.to_datetime(frame["ts"], utc=True)
    available = pd.to_datetime(frame["available_at"], utc=True)
    violations = int((available < ts).sum())
    if violations:
        raise LookAheadError(
            "rows are marked available before their observation time",
            label=label,
            violations=violations,
        )


def assert_split_ordering(
    train_index: pd.DatetimeIndex,
    test_index: pd.DatetimeIndex,
    *,
    embargo: pd.Timedelta | None = None,
    label: str = "split",
) -> None:
    """Every training timestamp must precede every test timestamp, plus an embargo.

    The embargo matters when labels overlap.  With a 5-day forward return, the
    label for the last training day is not known until 5 days later, which is
    inside the test window.  Training on it means the model has seen data from
    the test period.  Purging the overlap and embargoing a gap after it is the
    standard fix (Lopez de Prado, *Advances in Financial Machine Learning*).
    """
    if len(train_index) == 0 or len(test_index) == 0:
        return
    train_end = pd.Timestamp(train_index.max())
    test_start = pd.Timestamp(test_index.min())
    required = train_end + (embargo or pd.Timedelta(0))
    if test_start <= required:
        raise LookAheadError(
            "test window starts before the training window plus embargo",
            label=label,
            train_end=str(train_end),
            test_start=str(test_start),
            embargo=str(embargo),
        )


__all__ = [
    "LeakageFinding",
    "LeakageReport",
    "assert_availability_ordering",
    "assert_causal",
    "assert_split_ordering",
    "corrupt_future",
    "punch_gap",
]
