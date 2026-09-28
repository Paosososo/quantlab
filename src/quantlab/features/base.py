"""Feature specifications.

A feature is a named, pure function from one symbol's OHLCV frame to a Series
of the same length, plus the metadata needed to reason about it:

``lookback``
    How many prior bars the feature needs.  The pipeline leaves NaN for the
    warm-up period rather than producing a value from a partial window.

``category``
    Grouping for reporting ("momentum", "volatility", "volume", "macro").

``params``
    Serialised into ``feature_sets.definition`` so a stored feature set fully
    describes how its values were produced.

The spec hash over all of the above is what makes a model run reproducible: two
runs with the same hash used the same feature logic, no version string
required.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

FeatureFunction = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    name: str
    fn: FeatureFunction
    lookback: int
    category: str = "other"
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "lookback": self.lookback,
            "category": self.category,
            "params": self.params,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class FeatureSetSpec:
    """A named, versioned collection of features."""

    name: str
    specs: Sequence[FeatureSpec]
    version: int = 1
    description: str = ""

    def __post_init__(self) -> None:
        names = [s.name for s in self.specs]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f"duplicate feature names: {sorted(duplicates)}")

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.specs]

    @property
    def max_lookback(self) -> int:
        return max((s.lookback for s in self.specs), default=0)

    def definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "features": [s.definition() for s in self.specs],
        }

    def spec_hash(self) -> str:
        payload = json.dumps(self.definition(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def subset(self, names: Iterable[str]) -> FeatureSetSpec:
        wanted = set(names)
        return FeatureSetSpec(
            name=f"{self.name}_subset",
            specs=[s for s in self.specs if s.name in wanted],
            version=self.version,
            description=f"Subset of {self.name}",
        )


__all__ = ["FeatureFunction", "FeatureSetSpec", "FeatureSpec"]
