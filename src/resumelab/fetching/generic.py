"""Reading a posting from a page nobody wrote an adapter for.

Two strategies, in order of how much they can be trusted:

1. A schema.org ``JobPosting`` block, which the site published for search engines and
   which therefore contains the posting and nothing else.
2. The page's own text, preferring a ``<main>`` or ``<article>`` region.

The second is a fallback in the honest sense: it will sometimes carry a cookie banner
or a "related jobs" list into the posting. That shows up in ``jd.txt``, which is why
the run directory keeps it — a strange analysis is explained by reading what was
actually fetched.
"""

from __future__ import annotations

import logging
from typing import Final

import httpx

from resumelab.exceptions import JDFetchError
from resumelab.fetching.html_text import html_to_text
from resumelab.fetching.http import fetch_document
from resumelab.fetching.jsonld import find_job_posting
from resumelab.fetching.models import FetchedPosting, PostingBoard

logger = logging.getLogger(__name__)

_HTML_ACCEPT: Final = "text/html,application/xhtml+xml"

_CONTENT_REGIONS: Final = frozenset({"main", "article"})

MIN_USABLE_CHARACTERS: Final = 200
"""Below this a page yielded navigation, not a posting.

Set well under the 50-character floor a job description is held to, because the
failure this catches is different: a JavaScript-rendered page returns a valid HTML
document whose body is an empty div, and the useful thing to say about it is that
the posting never arrived, not that it was short.
"""


def fetch_generic(url: str, *, client: httpx.Client | None = None) -> FetchedPosting:
    """Fetch ``url`` as a web page and extract whatever posting it holds.

    Raises:
        JDFetchError: If the page cannot be fetched, or holds too little text to be a
            posting — most often because the site renders its content with JavaScript.
    """
    document = fetch_document(url, client=client, accept=_HTML_ACCEPT)

    posting = find_job_posting(document.text)
    if posting is not None and len(posting.description_html) >= MIN_USABLE_CHARACTERS:
        logger.debug("read posting from a schema.org JobPosting block url=%s", url)
        return FetchedPosting(
            text=_assemble(posting.title, posting.location, html_to_text(posting.description_html)),
            board=PostingBoard.GENERIC,
            requested_url=url,
            final_url=document.url,
            title=posting.title,
            company=posting.company,
            location=posting.location,
        )

    text = html_to_text(document.text, prefer=_CONTENT_REGIONS)
    if len(text) < MIN_USABLE_CHARACTERS:
        raise JDFetchError(
            f"No job posting could be read from {url}. The page returned "
            f"{len(text)} characters of text, which usually means it renders its "
            "content with JavaScript. Copy the posting and use --jd-text, or save it "
            "to a file and use --jd."
        )
    logger.debug("read posting from page text characters=%d url=%s", len(text), url)
    return FetchedPosting(
        text=text,
        board=PostingBoard.GENERIC,
        requested_url=url,
        final_url=document.url,
        title=posting.title if posting else None,
        company=posting.company if posting else None,
        location=posting.location if posting else None,
    )


def _assemble(*sections: str | None) -> str:
    """Join the parts of a posting, dropping the ones the page did not supply."""
    return "\n\n".join(section for section in sections if section)
