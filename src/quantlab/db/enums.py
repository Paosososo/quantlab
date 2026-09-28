"""String enumerations shared by the ORM, the API schemas and the engine."""

from __future__ import annotations

import sys
from enum import Enum

if sys.version_info >= (3, 11):  # pragma: no cover - version branch
    from enum import StrEnum
else:  # pragma: no cover - version branch

    class StrEnum(str, Enum):
        """Minimal ``enum.StrEnum`` backport for Python 3.10.

        Only the behaviour this project relies on: members compare equal to
        their string values and ``str()`` yields the value rather than
        ``ClassName.MEMBER``.
        """

        def __str__(self) -> str:
            return str(self.value)


def enum_values(enum_cls: type[Enum]) -> list[str]:
    """Values of an enum, for ``sqlalchemy.Enum(values_callable=...)``."""
    return [str(member.value) for member in enum_cls]


class AssetClass(StrEnum):
    EQUITY = "equity"
    ETF = "etf"
    INDEX = "index"
    FX = "fx"
    COMMODITY = "commodity"
    RATE = "rate"
    CRYPTO = "crypto"


class AssetStatus(StrEnum):
    ACTIVE = "active"
    DELISTED = "delisted"
    SUSPENDED = "suspended"


class ReturnMethod(StrEnum):
    SIMPLE = "simple"
    LOG = "log"


class ReturnDirection(StrEnum):
    #: Computed from past prices; knowable at ``ts``.  Safe as a feature.
    TRAILING = "trailing"
    #: Computed from future prices; knowable only after the horizon closes.
    #: Legitimate as a *label*, never as a feature.
    FORWARD = "forward"


class Frequency(StrEnum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class RunStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class CheckStatus(StrEnum):
    PASSED = "passed"
    WARNED = "warned"
    FAILED = "failed"


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(StrEnum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class ExecutionTiming(StrEnum):
    """Which bar's price fills an order decided at the close of bar ``t``.

    ``NEXT_OPEN`` is the default and the most realistic for a daily strategy
    that computes signals after the close.  ``NEXT_CLOSE`` models an execution
    algorithm that works the order through the following session.  There is
    deliberately no ``SAME_CLOSE`` option: filling at the price that generated
    the signal is the classic look-ahead bug, so the engine makes it
    unexpressible rather than merely discouraged.
    """

    NEXT_OPEN = "next_open"
    NEXT_CLOSE = "next_close"


class SplitKind(StrEnum):
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"


__all__ = [
    "AssetClass",
    "AssetStatus",
    "CheckStatus",
    "ExecutionTiming",
    "Frequency",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "ReturnDirection",
    "ReturnMethod",
    "RunStatus",
    "SplitKind",
    "StrEnum",
    "enum_values",
]
