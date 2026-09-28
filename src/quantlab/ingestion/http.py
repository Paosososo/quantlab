"""Async HTTP client with rate limiting, retries and structured logging.

Retry policy
------------
Only *transient* failures are retried: connection errors, timeouts, 5xx, and
429.  A 404 is never retried, because hammering a provider with a request that
cannot succeed is both rude and useless.  ``Retry-After`` is honoured when the
provider sends it; otherwise the wait is exponential with full jitter.

Full jitter (``uniform(0, base * 2**attempt)``) rather than plain exponential
backoff, because several symbols are fetched concurrently and unjittered
backoff makes them retry in lockstep, reproducing the burst that triggered the
rate limit in the first place.
"""

from __future__ import annotations

import asyncio
import random
import time
from types import TracebackType
from typing import Any

import httpx

from quantlab.config import get_settings
from quantlab.exceptions import PermanentHTTPError, RateLimitError, TransientHTTPError
from quantlab.logging import get_logger

log = get_logger(__name__)

RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class TokenBucketRateLimiter:
    """Classic token bucket.

    Chosen over a fixed sleep between requests because it permits a short burst
    (up to ``capacity``) and then settles to the sustained rate, which is how
    most public endpoints actually police usage.
    """

    def __init__(self, rate_per_second: float, capacity: float | None = None) -> None:
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be positive")
        self.rate = float(rate_per_second)
        self.capacity = float(capacity if capacity is not None else max(1.0, rate_per_second))
        self._tokens = self.capacity
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self._updated
                self._updated = now
                self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                deficit = tokens - self._tokens
                await asyncio.sleep(deficit / self.rate)


class AsyncHttpClient:
    """Thin wrapper over httpx that turns HTTP failures into typed exceptions."""

    def __init__(
        self,
        *,
        rate_limit_per_second: float | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
        headers: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        settings = get_settings()
        self.max_retries = settings.http_max_retries if max_retries is None else max_retries
        self.backoff_base = settings.http_backoff_base_seconds
        self.backoff_max = settings.http_backoff_max_seconds
        self.limiter = TokenBucketRateLimiter(
            rate_limit_per_second or settings.default_rate_limit_per_second
        )
        merged_headers = {"User-Agent": settings.http_user_agent, **(headers or {})}
        self._client = httpx.AsyncClient(
            timeout=timeout_seconds or settings.http_timeout_seconds,
            headers=merged_headers,
            follow_redirects=True,
            transport=transport,
        )

    async def __aenter__(self) -> AsyncHttpClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    def _backoff_seconds(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, self.backoff_max)
        ceiling = min(self.backoff_max, self.backoff_base * (2**attempt))
        return random.uniform(0.0, ceiling)

    async def get_bytes(self, url: str, params: dict[str, Any] | None = None) -> bytes:
        """GET ``url``, retrying transient failures.  Raises on permanent ones."""
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            await self.limiter.acquire()
            try:
                response = await self._client.get(url, params=params)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = TransientHTTPError("transport failure", url=url, error=str(exc))
                wait = self._backoff_seconds(attempt, None)
                log.warning(
                    "http.transient", url=url, attempt=attempt, wait=round(wait, 3), error=str(exc)
                )
                await asyncio.sleep(wait)
                continue

            status = response.status_code
            if status == 429:
                retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                last_error = RateLimitError("rate limited", retry_after=retry_after, url=url)
                wait = self._backoff_seconds(attempt, retry_after)
                log.warning("http.rate_limited", url=url, attempt=attempt, wait=round(wait, 3))
                await asyncio.sleep(wait)
                continue
            if status in RETRYABLE_STATUS:
                last_error = TransientHTTPError("retryable status", url=url, status=status)
                wait = self._backoff_seconds(attempt, None)
                log.warning("http.retryable_status", url=url, status=status, attempt=attempt)
                await asyncio.sleep(wait)
                continue
            if status >= 400:
                raise PermanentHTTPError(
                    "non-retryable HTTP status",
                    url=url,
                    status=status,
                    body=response.text[:200],
                )

            log.debug("http.ok", url=url, status=status, bytes=len(response.content))
            return response.content

        assert last_error is not None
        raise last_error


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        # HTTP-date form.  We do not parse it: falling back to jittered backoff
        # is safe and avoids a date-parsing dependency for a rare header shape.
        return None


__all__ = ["RETRYABLE_STATUS", "AsyncHttpClient", "TokenBucketRateLimiter"]
