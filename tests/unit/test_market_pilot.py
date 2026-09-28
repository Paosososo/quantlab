"""Checks for the close-only real-market pilot's temporal construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.research.market_pilot import build_close_only_frame


def _write_closes(path, closes: np.ndarray) -> None:
    dates = pd.date_range("2020-01-01", periods=len(closes), freq="B")
    pd.DataFrame({"observation_date": dates.date, "SP500": closes}).to_csv(path, index=False)


def test_features_use_only_current_and_earlier_closes(tmp_path):
    rng = np.random.default_rng(43)
    closes = 1000 * np.exp(np.cumsum(rng.normal(0, 0.01, 1200)))
    original = tmp_path / "original.csv"
    changed = tmp_path / "changed.csv"
    _write_closes(original, closes)
    changed_closes = closes.copy()
    changed_closes[800:] *= 2
    _write_closes(changed, changed_closes)

    X, y, meta, summary = build_close_only_frame(original)
    changed_X, _, _, _ = build_close_only_frame(changed)
    pd.testing.assert_frame_equal(X.iloc[:700], changed_X.iloc[:700])
    assert len(X) == summary["model_rows"]
    assert (meta["label_available_at"] > meta["ts"]).all()
    assert (meta["available_at"] == meta["ts"]).all()
    assert y.iloc[0] == pytest.approx(closes[22] / closes[21] - 1)


def test_duplicate_dates_are_rejected(tmp_path):
    path = tmp_path / "duplicate.csv"
    closes = np.linspace(100.0, 200.0, 1200)
    _write_closes(path, closes)
    frame = pd.read_csv(path)
    frame.loc[1, "observation_date"] = frame.loc[0, "observation_date"]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate market dates"):
        build_close_only_frame(path)
