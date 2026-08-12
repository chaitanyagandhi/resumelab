"""Retrieval of a job posting from a URL.

One entry point, :func:`fetch_posting`. It routes a URL to the applicant tracking
system that published it and reads the posting out of that system's JSON, falling
back to reading the page when the URL belongs to no board it knows.

Nothing here reaches a language model. This layer's only job is to turn a link into
the same kind of text a researcher would otherwise have pasted, so that everything
downstream cannot tell the difference.
"""

from __future__ import annotations

import logging

import httpx

from resumelab.fetching.boards import fetch_from_board, select_board
from resumelab.fetching.generic import fetch_generic
from resumelab.fetching.http import validate_url
from resumelab.fetching.models import FetchedPosting, PostingBoard

logger = logging.getLogger(__name__)

__all__ = ["FetchedPosting", "PostingBoard", "fetch_posting"]


def fetch_posting(url: str, *, client: httpx.Client | None = None) -> FetchedPosting:
    """Retrieve the job posting published at ``url``.

    Args:
        url: The link to the posting, as pasted from a browser.
        client: An HTTP client to send through. Tests supply one wired to a mock
            transport; production leaves it unset and one request gets its own client.

    Returns:
        The posting as text, with whatever title, company, and location the source
        made available.

    Raises:
        JDFetchError: If the URL is not fetchable, the site refuses the request, or
            no posting can be read from what came back.
    """
    target = validate_url(url)
    routed = select_board(target)

    if routed is None:
        logger.info("fetching posting url=%s board=generic", target)
        return _log_result(fetch_generic(target, client=client))

    adapter, request = routed
    logger.info("fetching posting url=%s board=%s", target, adapter.board.value)
    return _log_result(fetch_from_board(adapter, request, requested_url=target, client=client))


def _log_result(posting: FetchedPosting) -> FetchedPosting:
    """Record what was read, without putting the posting itself in the log."""
    logger.info(
        "fetched posting board=%s characters=%d title=%r company=%r",
        posting.board.value,
        len(posting.text),
        posting.title,
        posting.company,
    )
    return posting
