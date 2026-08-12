"""The one place ResumeLab makes an outbound request to somewhere it does not control.

Job postings come from arbitrary hosts, so every request is bounded on all four axes
that can otherwise hang or exhaust a process: time, redirects, response size, and
content type. A posting that violates any of them fails with a sentence a researcher
can act on rather than a stack trace.

The user agent identifies the tool rather than imitating a browser. Some sites behind
bot protection refuse it; that is reported as what it is, with the suggestion to paste
the posting instead. Working around bot detection is out of scope.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlsplit

import httpx

from resumelab import __version__
from resumelab.exceptions import JDFetchError

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS: Final = 20.0
"""Generous for a single page, short enough that a hung host does not stall a run."""

MAX_REDIRECTS: Final = 5
MAX_RESPONSE_BYTES: Final = 5_000_000
"""A job posting is kilobytes. Anything at this size is a page we cannot use anyway."""

USER_AGENT: Final = f"Mozilla/5.0 (compatible; ResumeLab/{__version__}; research prototype)"

_ALLOWED_SCHEMES: Final = frozenset({"http", "https"})

_READABLE_CONTENT_TYPES: Final = (
    "text/",
    "application/json",
    "application/ld+json",
    "application/xhtml+xml",
    "application/xml",
)


@dataclass(frozen=True)
class FetchedDocument:
    """A retrieved document, decoded to text."""

    url: str
    """Where the body was finally read from, after any redirects."""

    content_type: str
    text: str


def validate_url(url: str) -> str:
    """Return ``url`` if it is a fetchable absolute HTTP(S) URL.

    Raises:
        JDFetchError: If the scheme is missing, unsupported, or the host is absent.
            ``file://`` and friends are rejected here rather than at request time, so
            a mistyped argument cannot be turned into a local file read.
    """
    parts = urlsplit(url.strip())
    if parts.scheme not in _ALLOWED_SCHEMES:
        raise JDFetchError(
            f"Job posting URLs must start with http:// or https:// — got {url!r}."
            if parts.scheme
            else f"Not a valid job posting URL: {url!r}. Include https://."
        )
    if not parts.netloc:
        raise JDFetchError(f"Job posting URL has no host: {url!r}.")
    return parts.geturl()


def fetch_document(
    url: str,
    *,
    client: httpx.Client | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    accept: str | None = None,
) -> FetchedDocument:
    """GET ``url`` and return its body as text.

    Args:
        url: An absolute HTTP(S) URL.
        client: An existing client to send through. Supplying one is how tests reach
            a mock transport, and how several requests share a connection pool.
        timeout: Per-request timeout in seconds.
        accept: Value for the ``Accept`` header, when an adapter wants JSON.

    Returns:
        The decoded document and the URL it was finally read from.

    Raises:
        JDFetchError: On an invalid URL, a network failure, a non-2xx status, a
            response that is too large, or a body that is not text.
    """
    target = validate_url(url)
    headers = {"User-Agent": USER_AGENT, "Accept": accept} if accept else {"User-Agent": USER_AGENT}

    if client is not None:
        return _get(client, target, headers=headers)
    with httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        max_redirects=MAX_REDIRECTS,
    ) as owned:
        return _get(owned, target, headers=headers)


def _get(client: httpx.Client, url: str, *, headers: dict[str, str]) -> FetchedDocument:
    """Send the request and turn every failure mode into a readable message."""
    try:
        with client.stream("GET", url, headers=headers, follow_redirects=True) as response:
            _check_status(response, url)
            content_type = response.headers.get("content-type", "")
            _check_content_type(content_type, url)
            body = _read_bounded(response, url)
    except httpx.TooManyRedirects as exc:
        raise JDFetchError(f"Job posting URL redirected too many times: {url}") from exc
    except httpx.TimeoutException as exc:
        raise JDFetchError(f"Timed out fetching the job posting: {url}") from exc
    except httpx.HTTPError as exc:
        raise JDFetchError(f"Could not fetch the job posting: {url} ({exc})") from exc

    logger.debug("fetched url=%s status=ok bytes=%d type=%s", url, len(body), content_type)
    return FetchedDocument(url=str(response.url), content_type=content_type, text=body)


def _check_status(response: httpx.Response, url: str) -> None:
    """Report a refusal in terms of what the researcher should do about it."""
    if response.status_code < 400:
        return
    if response.status_code in (401, 403):
        raise JDFetchError(
            f"The site refused an automated request for {url} "
            f"(HTTP {response.status_code}). Some job boards allow only real browsers. "
            "Copy the posting text and use --jd-text, or save it to a file and use --jd."
        )
    if response.status_code == 404:
        raise JDFetchError(f"No job posting at {url} (HTTP 404). It may have been taken down.")
    raise JDFetchError(f"The site returned HTTP {response.status_code} for {url}.")


def _check_content_type(content_type: str, url: str) -> None:
    """Reject bodies that cannot be read as text before downloading them."""
    if not content_type:
        return
    normalized = content_type.split(";", 1)[0].strip().lower()
    if not normalized.startswith(_READABLE_CONTENT_TYPES):
        raise JDFetchError(
            f"{url} returned {normalized}, which is not a readable job posting. "
            "Expected an HTML page or a JSON response."
        )


def _read_bounded(response: httpx.Response, url: str) -> str:
    """Read the body, refusing to buffer more than :data:`MAX_RESPONSE_BYTES`.

    Streamed rather than read whole, because a ``Content-Length`` header is optional
    and a hostile or merely broken host can otherwise stream indefinitely.
    """
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_bytes():
        total += len(chunk)
        if total > MAX_RESPONSE_BYTES:
            raise JDFetchError(
                f"The page at {url} is larger than {MAX_RESPONSE_BYTES:,} bytes. "
                "That is a site index rather than a single posting."
            )
        chunks.append(chunk)
    return b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")
