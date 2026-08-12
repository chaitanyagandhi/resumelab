"""Fixtures for the unit suite.

The HTTP client factory here is what keeps posting-retrieval tests off the network.
Every test drives a real :class:`httpx.Client` whose transport is a function, so the
code under test takes the same code path it takes in production — including redirect
handling, streaming, and header negotiation — while the responses stay scripted.
"""

from collections.abc import Callable

import httpx
import pytest


@pytest.fixture
def make_http_client() -> Callable[[Callable[[httpx.Request], httpx.Response]], httpx.Client]:
    """Build a client that answers from ``handler`` instead of the network."""

    def factory(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)

    return factory


@pytest.fixture
def routed_client(
    make_http_client: Callable[[Callable[[httpx.Request], httpx.Response]], httpx.Client],
) -> Callable[..., httpx.Client]:
    """Build a client that answers a fixed URL-to-response map.

    Requests to a URL that was not scripted fail the test loudly rather than
    returning a 404 that a fallback path might quietly swallow.
    """

    def factory(
        routes: dict[str, httpx.Response], *, record: list[str] | None = None
    ) -> httpx.Client:
        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if record is not None:
                record.append(url)
            if url not in routes:
                raise AssertionError(f"unexpected request to {url}; scripted: {sorted(routes)}")
            return routes[url]

        return make_http_client(handler)

    return factory
