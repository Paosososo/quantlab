"""Deterministic synthetic market data.

Purpose
-------
Every test, every CI run and the offline demo path use this provider.  It exists
so that:

* CI needs no network and no secrets, and never flakes because a public endpoint
  is slow.
* Tests can assert exact numbers, because the generator is seeded and the
  process is known in closed form.
* The research harness can be *validated*: generate a series with a known,
  injected predictable component and check the pipeline finds it; generate a
  pure random walk and check the pipeline finds nothing.  A backtester that
  cannot tell those two apart is broken, and without synthetic data you cannot
  run that test.

Everything produced here is clearly labelled synthetic: symbols are prefixed
``SYN_`` by convention in the demo config, ``prices.source`` is ``"synthetic"``,
and the parameters are written into the raw manifest.  No synthetic number is
ever presented as a real market observation.

The process
-----------
Log returns are

    r_t = mu_d + phi * (r_{t-1} - mu_d) + sigma_t * eps_t

with ``eps_t`` iid standard normal and ``sigma_t`` optionally switching between
a calm and a stressed regime via a two-state Markov chain.  ``phi`` is the
injected autocorrelation: ``phi = 0`` gives a random walk with no exploitable
structure, ``phi != 0`` gives a series a linear model should be able to exploit
before costs.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from quantlab.ingestion.base import PRICE_COLUMNS, PriceProvider, RawPayload
from quantlab.timeutils import (
    US_EQUITY_SESSION,
    UTC,
    SessionSpec,
    price_available_at,
    session_date_to_ts,
)


@dataclass(frozen=True, slots=True)
class SyntheticSpec:
    """Parameters of the generating process.  Serialised into the raw manifest."""

    start: dt.date = dt.date(2010, 1, 4)
    end: dt.date = dt.date(2024, 12, 31)
    initial_price: float = 100.0
    annual_drift: float = 0.06
    annual_volatility: float = 0.18
    #: AR(1) coefficient on daily log returns.  0.0 => unpredictable.
    autocorrelation: float = 0.0
    #: Two-state volatility regime; set ``regime_switching=False`` for constant vol.
    regime_switching: bool = True
    stress_volatility_multiple: float = 2.5
    calm_to_stress_prob: float = 0.01
    stress_to_calm_prob: float = 0.06
    #: Standard deviation of the overnight gap as a fraction of daily vol.
    overnight_share: float = 0.35
    #: Mean daily volume and its log-normal dispersion.
    mean_volume: float = 5_000_000.0
    volume_dispersion: float = 0.35
    trading_days_per_year: int = 252

    def to_json(self) -> str:
        payload = asdict(self)
        payload["start"] = self.start.isoformat()
        payload["end"] = self.end.isoformat()
        return json.dumps(payload, sort_keys=True)


def _business_days(start: dt.date, end: dt.date) -> list[dt.date]:
    """Weekdays between ``start`` and ``end`` inclusive.

    No holiday calendar: synthetic data does not need to match a real exchange,
    and inventing holidays would only make the fixture harder to reason about.
    """
    index = pd.bdate_range(start=start, end=end, freq="C", weekmask="Mon Tue Wed Thu Fri")
    return [d.date() for d in index]


def generate_price_frame(
    symbol: str, spec: SyntheticSpec | None = None, *, seed: int = 0
) -> pd.DataFrame:
    """Generate a canonical price frame for ``symbol``.

    The seed is combined with a stable hash of the symbol so that different
    symbols get different, but still reproducible, paths.
    """
    spec = spec or SyntheticSpec()
    symbol_seed = (seed * 1_000_003 + sum(ord(c) * (i + 1) for i, c in enumerate(symbol))) % (2**32)
    rng = np.random.default_rng(symbol_seed)

    days = _business_days(spec.start, spec.end)
    n = len(days)
    if n < 2:
        raise ValueError("synthetic spec covers fewer than two business days")

    dt_years = 1.0 / spec.trading_days_per_year
    mu_d = (spec.annual_drift - 0.5 * spec.annual_volatility**2) * dt_years
    sigma_d = spec.annual_volatility * np.sqrt(dt_years)

    # Volatility regime path.
    if spec.regime_switching:
        state = np.zeros(n, dtype=int)
        for i in range(1, n):
            if state[i - 1] == 0:
                state[i] = 1 if rng.random() < spec.calm_to_stress_prob else 0
            else:
                state[i] = 0 if rng.random() < spec.stress_to_calm_prob else 1
        sigma_path = np.where(state == 1, sigma_d * spec.stress_volatility_multiple, sigma_d)
    else:
        sigma_path = np.full(n, sigma_d)

    # AR(1) log returns.
    eps = rng.standard_normal(n)
    log_ret = np.empty(n)
    log_ret[0] = mu_d + sigma_path[0] * eps[0]
    phi = float(spec.autocorrelation)
    for i in range(1, n):
        log_ret[i] = mu_d + phi * (log_ret[i - 1] - mu_d) + sigma_path[i] * eps[i]

    close = spec.initial_price * np.exp(np.cumsum(log_ret))

    # Split the daily move into an overnight gap and an intraday move so that
    # open != previous close, which is what makes next-open execution a
    # meaningful test rather than a relabelled close.
    prev_close = np.concatenate([[spec.initial_price], close[:-1]])
    overnight = rng.standard_normal(n) * sigma_path * spec.overnight_share
    open_ = prev_close * np.exp(overnight)

    body_high = np.maximum(open_, close)
    body_low = np.minimum(open_, close)
    wick = np.abs(rng.standard_normal(n)) * sigma_path * 0.5
    high = body_high * np.exp(wick)
    low = body_low * np.exp(-wick)

    volume = np.round(
        spec.mean_volume
        * np.exp(rng.standard_normal(n) * spec.volume_dispersion - 0.5 * spec.volume_dispersion**2)
    )

    session: SessionSpec = US_EQUITY_SESSION
    frame = pd.DataFrame(
        {
            "symbol": symbol.upper(),
            "ts": pd.to_datetime([session_date_to_ts(d) for d in days], utc=True),
            "available_at": pd.to_datetime(
                [price_available_at(d, session) for d in days], utc=True
            ),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": volume,
        }
    )
    return frame.loc[:, list(PRICE_COLUMNS)]


class SyntheticProvider(PriceProvider):
    """A :class:`PriceProvider` that fabricates data instead of fetching it.

    ``fetch`` still returns a :class:`RawPayload` and ``parse`` is still pure, so
    the synthetic path exercises exactly the same pipeline code as a real
    provider.  A test that passes here is testing the real loader, not a mock of
    it.
    """

    name = "synthetic"
    dataset = "daily_bars"

    def __init__(self, spec: SyntheticSpec | None = None, *, seed: int = 0) -> None:
        self.spec = spec or SyntheticSpec()
        self.seed = seed

    async def fetch(
        self, symbol: str, start: dt.date | None = None, end: dt.date | None = None
    ) -> RawPayload:
        spec = self.spec
        if start is not None or end is not None:
            spec = SyntheticSpec(
                **{
                    **asdict(self.spec),
                    "start": start or self.spec.start,
                    "end": end or self.spec.end,
                }
            )
        frame = generate_price_frame(symbol, spec, seed=self.seed)
        csv_bytes = frame.to_csv(index=False).encode("utf-8")
        return RawPayload(
            provider=self.name,
            dataset=self.dataset,
            entity=symbol.upper(),
            content=csv_bytes,
            content_type="text/csv",
            fetched_at=dt.datetime.now(tz=UTC),
            request={"generator": "synthetic", "seed": self.seed, "spec": spec.to_json()},
        )

    def parse(self, payload: RawPayload) -> pd.DataFrame:
        import io

        frame = pd.read_csv(io.StringIO(payload.content.decode("utf-8")))
        frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
        frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
        return frame.loc[:, list(PRICE_COLUMNS)]


__all__ = ["SyntheticProvider", "SyntheticSpec", "generate_price_frame"]
