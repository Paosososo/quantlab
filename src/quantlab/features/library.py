"""The default feature library.

Selection rationale
-------------------
The features here are the ones with a documented economic story and published
out-of-sample evidence, not everything that could be computed.  Throwing 200
technical indicators at a model is how you manufacture a false positive: with
enough features and enough tuning, something always looks predictive in-sample.

* **Momentum** at 21, 63, 126 and 252 days, plus the 12-1 variant that skips the
  most recent month.  Cross-sectional and time-series momentum are among the
  most replicated anomalies; the 12-1 skip exists because the most recent month
  tends to reverse.
* **Volatility**: realised and downside, at 21 and 63 days.  Volatility is
  strongly autocorrelated, which makes it one of the few genuinely predictable
  quantities in finance, and it is what risk-scaled strategies need.
* **Trend**: price relative to its own moving averages, and the fast/slow ratio.
  These are the features underlying the classical strategies we benchmark
  against, so including them lets the model beat those strategies on their own
  terms if it can.
* **Mean reversion**: short-horizon return z-scores and RSI.
* **Volume**: turnover z-score and dollar volume, which proxy for attention and
  for whether a signal is tradable at size.
* **Range**: normalised ATR, a volatility measure that uses the whole bar rather
  than closes alone.
* **Macro**: level, change and z-score of a small set of series, joined on
  publication availability rather than observation date.

Every one is computed with the causal primitives in
:mod:`quantlab.features.transforms` and is verified by the leakage harness in
the test suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.features import transforms as tf
from quantlab.features.base import FeatureSetSpec, FeatureSpec

PRICE = "adj_close"


def _close(frame: pd.DataFrame) -> pd.Series:
    return frame[PRICE].astype("float64")


# ---------------------------------------------------------------------------
# Feature constructors
# ---------------------------------------------------------------------------
def _trailing_return(window: int):
    def fn(frame: pd.DataFrame) -> pd.Series:
        return tf.pct_change(_close(frame), window)

    return fn


def _momentum_skip(lookback: int, skip: int):
    """Return from t-lookback to t-skip.  The classic 12-1 momentum shape."""

    def fn(frame: pd.DataFrame) -> pd.Series:
        close = _close(frame)
        return close.shift(skip) / close.shift(lookback) - 1.0

    return fn


def _realised_vol(window: int):
    def fn(frame: pd.DataFrame) -> pd.Series:
        return tf.realised_volatility(tf.log_return(_close(frame), 1), window)

    return fn


def _downside_vol(window: int):
    def fn(frame: pd.DataFrame) -> pd.Series:
        return tf.downside_volatility(tf.log_return(_close(frame), 1), window)

    return fn


def _price_to_ma(window: int):
    def fn(frame: pd.DataFrame) -> pd.Series:
        close = _close(frame)
        return close / tf.rolling_mean(close, window) - 1.0

    return fn


def _ma_ratio(fast: int, slow: int):
    def fn(frame: pd.DataFrame) -> pd.Series:
        close = _close(frame)
        return tf.rolling_mean(close, fast) / tf.rolling_mean(close, slow) - 1.0

    return fn


def _return_zscore(window: int):
    def fn(frame: pd.DataFrame) -> pd.Series:
        return tf.rolling_zscore(tf.pct_change(_close(frame), 1), window)

    return fn


def _rsi(window: int):
    def fn(frame: pd.DataFrame) -> pd.Series:
        return tf.rsi(_close(frame), window)

    return fn


def _volume_zscore(window: int):
    def fn(frame: pd.DataFrame) -> pd.Series:
        volume = frame["volume"].astype("float64")
        # A negative volume is bad data, not a small number; make it NaN rather
        # than letting log1p produce a warning and a silent NaN anyway.
        return tf.rolling_zscore(np.log1p(volume.where(volume >= 0.0)), window)

    return fn


def _dollar_volume_ratio(window: int):
    def fn(frame: pd.DataFrame) -> pd.Series:
        dollar = frame["volume"].astype("float64") * frame["close"].astype("float64")
        return dollar / tf.rolling_mean(dollar, window) - 1.0

    return fn


def _normalised_atr(window: int):
    def fn(frame: pd.DataFrame) -> pd.Series:
        atr = tf.average_true_range(
            frame["high"].astype("float64"),
            frame["low"].astype("float64"),
            frame["close"].astype("float64"),
            window,
        )
        return atr / frame["close"].astype("float64")

    return fn


def _rolling_skew(window: int):
    def fn(frame: pd.DataFrame) -> pd.Series:
        return tf.rolling_skew(tf.pct_change(_close(frame), 1), window)

    return fn


def _drawdown_from_high(window: int):
    def fn(frame: pd.DataFrame) -> pd.Series:
        close = _close(frame)
        return close / tf.rolling_max(close, window) - 1.0

    return fn


def macro_level(code: str):
    def fn(frame: pd.DataFrame) -> pd.Series:
        column = f"macro__{code}"
        if column not in frame.columns:
            return pd.Series(np.nan, index=frame.index, dtype="float64")
        return frame[column].astype("float64")

    return fn


def macro_change(code: str, window: int):
    def fn(frame: pd.DataFrame) -> pd.Series:
        column = f"macro__{code}"
        if column not in frame.columns:
            return pd.Series(np.nan, index=frame.index, dtype="float64")
        return frame[column].astype("float64").diff(window)

    return fn


def macro_zscore(code: str, window: int):
    def fn(frame: pd.DataFrame) -> pd.Series:
        column = f"macro__{code}"
        if column not in frame.columns:
            return pd.Series(np.nan, index=frame.index, dtype="float64")
        return tf.rolling_zscore(frame[column].astype("float64"), window)

    return fn


# ---------------------------------------------------------------------------
# The default set
# ---------------------------------------------------------------------------
def price_feature_specs() -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []
    for window in (1, 5, 21):
        specs.append(
            FeatureSpec(
                f"ret_{window}d",
                _trailing_return(window),
                lookback=window + 1,
                category="return",
                params={"window": window},
                description=f"Trailing {window}-day simple return",
            )
        )
    for window in (21, 63, 126, 252):
        specs.append(
            FeatureSpec(
                f"mom_{window}d",
                _trailing_return(window),
                lookback=window + 1,
                category="momentum",
                params={"window": window},
                description=f"Trailing {window}-day price momentum",
            )
        )
    specs.append(
        FeatureSpec(
            "mom_252d_skip21",
            _momentum_skip(252, 21),
            lookback=253,
            category="momentum",
            params={"lookback": 252, "skip": 21},
            description="12-1 momentum: return from t-252 to t-21, skipping the recent month",
        )
    )
    for window in (21, 63):
        specs.append(
            FeatureSpec(
                f"vol_{window}d",
                _realised_vol(window),
                lookback=window + 2,
                category="volatility",
                params={"window": window},
                description=f"Annualised realised volatility over {window} days",
            )
        )
        specs.append(
            FeatureSpec(
                f"downside_vol_{window}d",
                _downside_vol(window),
                lookback=window + 2,
                category="volatility",
                params={"window": window},
                description=f"Annualised downside deviation over {window} days",
            )
        )
    for window in (20, 50, 200):
        specs.append(
            FeatureSpec(
                f"price_to_ma_{window}",
                _price_to_ma(window),
                lookback=window,
                category="trend",
                params={"window": window},
                description=f"Close divided by its {window}-day moving average, minus one",
            )
        )
    specs.append(
        FeatureSpec(
            "ma_ratio_20_100",
            _ma_ratio(20, 100),
            lookback=100,
            category="trend",
            params={"fast": 20, "slow": 100},
            description="Fast over slow moving average, minus one",
        )
    )
    for window in (21, 63):
        specs.append(
            FeatureSpec(
                f"ret_zscore_{window}d",
                _return_zscore(window),
                lookback=window + 2,
                category="mean_reversion",
                params={"window": window},
                description=f"Daily return standardised by its trailing {window}-day distribution",
            )
        )
    specs.append(
        FeatureSpec(
            "rsi_14",
            _rsi(14),
            lookback=30,
            category="mean_reversion",
            params={"window": 14},
            description="Wilder's 14-day relative strength index",
        )
    )
    specs.append(
        FeatureSpec(
            "volume_zscore_21d",
            _volume_zscore(21),
            lookback=23,
            category="volume",
            params={"window": 21},
            description="Log volume standardised by its trailing 21-day distribution",
        )
    )
    specs.append(
        FeatureSpec(
            "dollar_volume_ratio_21d",
            _dollar_volume_ratio(21),
            lookback=22,
            category="volume",
            params={"window": 21},
            description="Dollar volume relative to its 21-day average",
        )
    )
    specs.append(
        FeatureSpec(
            "atr_14_normalised",
            _normalised_atr(14),
            lookback=30,
            category="volatility",
            params={"window": 14},
            description="Average true range divided by close",
        )
    )
    specs.append(
        FeatureSpec(
            "ret_skew_63d",
            _rolling_skew(63),
            lookback=65,
            category="distribution",
            params={"window": 63},
            description="Skewness of daily returns over 63 days",
        )
    )
    specs.append(
        FeatureSpec(
            "drawdown_252d",
            _drawdown_from_high(252),
            lookback=252,
            category="trend",
            params={"window": 252},
            description="Distance below the trailing 252-day high",
        )
    )
    return specs


def macro_feature_specs(codes: list[str] | None = None) -> list[FeatureSpec]:
    codes = codes or ["DGS10", "T10Y2Y", "VIXCLS", "BAMLH0A0HYM2", "UNRATE"]
    specs: list[FeatureSpec] = []
    for code in codes:
        key = code.lower()
        specs.append(
            FeatureSpec(
                f"macro_{key}_level",
                macro_level(code),
                lookback=1,
                category="macro",
                params={"code": code},
                description=f"Latest published value of {code}",
            )
        )
        specs.append(
            FeatureSpec(
                f"macro_{key}_chg_21d",
                macro_change(code, 21),
                lookback=22,
                category="macro",
                params={"code": code, "window": 21},
                description=f"21-day change in {code}",
            )
        )
        specs.append(
            FeatureSpec(
                f"macro_{key}_z_252d",
                macro_zscore(code, 252),
                lookback=252,
                category="macro",
                params={"code": code, "window": 252},
                description=f"{code} standardised by its trailing 252-day distribution",
            )
        )
    return specs


def default_feature_set(
    *, include_macro: bool = True, macro_codes: list[str] | None = None
) -> FeatureSetSpec:
    specs = price_feature_specs()
    if include_macro:
        specs = specs + macro_feature_specs(macro_codes)
    return FeatureSetSpec(
        name="core_v1" if include_macro else "price_only_v1",
        specs=specs,
        version=1,
        description=(
            "Momentum, volatility, trend, mean-reversion, volume and range features "
            "computed causally from daily bars"
            + (", plus macro series joined on publication availability." if include_macro else ".")
        ),
    )


__all__ = [
    "PRICE",
    "default_feature_set",
    "macro_change",
    "macro_feature_specs",
    "macro_level",
    "macro_zscore",
    "price_feature_specs",
]
