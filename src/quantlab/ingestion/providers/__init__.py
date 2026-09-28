"""Concrete data providers."""

from quantlab.ingestion.providers.fred import FredProvider
from quantlab.ingestion.providers.stooq import StooqProvider
from quantlab.ingestion.providers.synthetic import (
    SyntheticProvider,
    SyntheticSpec,
    generate_price_frame,
)

__all__ = [
    "FredProvider",
    "StooqProvider",
    "SyntheticProvider",
    "SyntheticSpec",
    "generate_price_frame",
]
