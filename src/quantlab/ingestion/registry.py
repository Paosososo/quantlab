"""Provider registry.

A tiny name-to-factory map.  It exists so that Airflow DAGs, the CLI and the
config file can all say ``provider="stooq"`` without importing concrete classes,
which is what keeps the orchestration layer free of business logic.
"""

from __future__ import annotations

from collections.abc import Callable

from quantlab.exceptions import ProviderError
from quantlab.ingestion.base import EconomicProvider, PriceProvider
from quantlab.ingestion.providers.fred import FredProvider
from quantlab.ingestion.providers.stooq import StooqProvider
from quantlab.ingestion.providers.synthetic import SyntheticProvider

PriceProviderFactory = Callable[..., PriceProvider]
EconomicProviderFactory = Callable[..., EconomicProvider]

_PRICE_PROVIDERS: dict[str, PriceProviderFactory] = {
    "stooq": StooqProvider,
    "synthetic": SyntheticProvider,
}

_ECONOMIC_PROVIDERS: dict[str, EconomicProviderFactory] = {
    "fred": FredProvider,
}


def register_price_provider(name: str, factory: PriceProviderFactory) -> None:
    _PRICE_PROVIDERS[name] = factory


def register_economic_provider(name: str, factory: EconomicProviderFactory) -> None:
    _ECONOMIC_PROVIDERS[name] = factory


def get_price_provider(name: str, **kwargs: object) -> PriceProvider:
    try:
        factory = _PRICE_PROVIDERS[name]
    except KeyError as exc:
        raise ProviderError(
            "unknown price provider", name=name, known=sorted(_PRICE_PROVIDERS)
        ) from exc
    return factory(**kwargs)


def get_economic_provider(name: str, **kwargs: object) -> EconomicProvider:
    try:
        factory = _ECONOMIC_PROVIDERS[name]
    except KeyError as exc:
        raise ProviderError(
            "unknown economic provider", name=name, known=sorted(_ECONOMIC_PROVIDERS)
        ) from exc
    return factory(**kwargs)


def available_price_providers() -> list[str]:
    return sorted(_PRICE_PROVIDERS)


def available_economic_providers() -> list[str]:
    return sorted(_ECONOMIC_PROVIDERS)


__all__ = [
    "available_economic_providers",
    "available_price_providers",
    "get_economic_provider",
    "get_price_provider",
    "register_economic_provider",
    "register_price_provider",
]
