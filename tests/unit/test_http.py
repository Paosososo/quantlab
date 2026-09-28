"""HTTP client: rate limiting, retry policy, error classification.

Uses ``httpx.MockTransport`` so the tests exercise the real client code path
without touching the network.
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from quantlab.exceptions import PermanentHTTPError, RateLimitError, TransientHTTPError
from quantlab.ingestion.http import AsyncHttpClient, TokenBucketRateLimiter

pytestmark = pytest.mark.unit


class TestTokenBucket:
    def test_a_burst_up_to_capacity_is_immediate(self):
        async def run() -> float:
            limiter = TokenBucketRateLimiter(rate_per_second=10.0, capacity=5.0)
            start = time.monotonic()
            for _ in range(5):
                await limiter.acquire()
            return time.monotonic() - start

        assert asyncio.run(run()) < 0.05

    def test_beyond_capacity_the_sustained_rate_applies(self):
        async def run() -> float:
            limiter = TokenBucketRateLimiter(rate_per_second=50.0, capacity=1.0)
            start = time.monotonic()
            for _ in range(5):
                await limiter.acquire()
            return time.monotonic() - start

        elapsed = asyncio.run(run())
        assert elapsed >= 4 / 50.0 * 0.8  # four refills at 50/s, with slack

    def test_a_non_positive_rate_is_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            TokenBucketRateLimiter(0.0)


def transport_returning(*responses: httpx.Response) -> tuple[httpx.MockTransport, list[int]]:
    calls: list[int] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return httpx.MockTransport(handler), calls


class TestRetryPolicy:
    def test_a_successful_response_is_returned(self):
        transport, calls = transport_returning(httpx.Response(200, content=b"hello"))

        async def run() -> bytes:
            async with AsyncHttpClient(transport=transport, rate_limit_per_second=1000) as client:
                return await client.get_bytes("https://example.test/data")

        assert asyncio.run(run()) == b"hello"
        assert len(calls) == 1

    def test_a_500_is_retried_then_succeeds(self):
        transport, calls = transport_returning(
            httpx.Response(500), httpx.Response(200, content=b"ok")
        )

        async def run() -> bytes:
            async with AsyncHttpClient(
                transport=transport, rate_limit_per_second=1000, max_retries=3
            ) as client:
                return await client.get_bytes("https://example.test/data")

        assert asyncio.run(run()) == b"ok"
        assert len(calls) == 2

    def test_a_429_is_retried_and_honours_retry_after(self):
        transport, calls = transport_returning(
            httpx.Response(429, headers={"Retry-After": "0.01"}),
            httpx.Response(200, content=b"ok"),
        )

        async def run() -> bytes:
            async with AsyncHttpClient(
                transport=transport, rate_limit_per_second=1000, max_retries=3
            ) as client:
                return await client.get_bytes("https://example.test/data")

        assert asyncio.run(run()) == b"ok"
        assert len(calls) == 2

    def test_a_404_is_not_retried(self):
        """Hammering an endpoint that cannot succeed is rude and pointless."""
        transport, calls = transport_returning(httpx.Response(404, text="nope"))

        async def run() -> bytes:
            async with AsyncHttpClient(
                transport=transport, rate_limit_per_second=1000, max_retries=5
            ) as client:
                return await client.get_bytes("https://example.test/data")

        with pytest.raises(PermanentHTTPError):
            asyncio.run(run())
        assert len(calls) == 1

    def test_retries_are_exhausted_and_the_last_error_is_raised(self):
        transport, calls = transport_returning(httpx.Response(503))

        async def run() -> bytes:
            async with AsyncHttpClient(
                transport=transport, rate_limit_per_second=1000, max_retries=2
            ) as client:
                return await client.get_bytes("https://example.test/data")

        with pytest.raises(TransientHTTPError):
            asyncio.run(run())
        assert len(calls) == 3  # the initial attempt plus two retries

    def test_persistent_rate_limiting_raises_rate_limit_error(self):
        transport, _ = transport_returning(httpx.Response(429))

        async def run() -> bytes:
            async with AsyncHttpClient(
                transport=transport, rate_limit_per_second=1000, max_retries=1
            ) as client:
                return await client.get_bytes("https://example.test/data")

        with pytest.raises(RateLimitError):
            asyncio.run(run())

    def test_a_transport_failure_is_retried(self):
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) == 1:
                raise httpx.ConnectError("boom", request=request)
            return httpx.Response(200, content=b"recovered")

        async def run() -> bytes:
            async with AsyncHttpClient(
                transport=httpx.MockTransport(handler),
                rate_limit_per_second=1000,
                max_retries=3,
            ) as client:
                return await client.get_bytes("https://example.test/data")

        assert asyncio.run(run()) == b"recovered"
        assert len(calls) == 2

    def test_a_user_agent_is_always_sent(self):
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.headers.get("user-agent", ""))
            return httpx.Response(200, content=b"")

        async def run() -> None:
            async with AsyncHttpClient(
                transport=httpx.MockTransport(handler), rate_limit_per_second=1000
            ) as client:
                await client.get_bytes("https://example.test/data")

        asyncio.run(run())
        assert "quantlab" in seen[0]
