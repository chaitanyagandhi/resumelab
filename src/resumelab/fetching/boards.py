"""Adapters for the applicant tracking systems that publish postings as JSON.

Greenhouse, Lever, Ashby, and Workday all serve the posting behind the page as
structured data. Reading that instead of the rendered HTML is worth a dedicated
adapter each: the text arrives already free of navigation and boilerplate, the title
and company are separate fields rather than a guess, and it works on Workday, whose
visible page is rendered by JavaScript and has no readable body at all.

Every adapter is written against a real captured response. Their quirks are not
uniform and are commented where they bite:

* Greenhouse escapes the HTML in ``content`` a second time.
* Lever splits a posting across ``descriptionPlain``, ``lists``, and ``additionalPlain``.
* Ashby publishes no per-job endpoint, so the board is fetched and filtered.
* Workday's JSON lives at a ``/wday/cxs/`` path derived from the page URL.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final
from urllib.parse import SplitResult, unquote_plus, urlsplit

import httpx

from resumelab.exceptions import JDFetchError
from resumelab.fetching.html_text import html_to_text, tidy
from resumelab.fetching.http import fetch_document
from resumelab.fetching.models import FetchedPosting, PostingBoard

_JSON_ACCEPT: Final = "application/json"
_LOCALE_SEGMENT: Final = re.compile(r"^[a-z]{2}-[A-Za-z]{2}$")


@dataclass(frozen=True)
class BoardRequest:
    """Where a board's structured copy of a posting lives."""

    api_url: str
    job_id: str
    org: str


Matcher = Callable[[SplitResult], BoardRequest | None]
Reader = Callable[[object, BoardRequest, str, str], FetchedPosting]


@dataclass(frozen=True)
class BoardAdapter:
    """A board ResumeLab can read structurally."""

    board: PostingBoard
    match: Matcher
    read: Reader


def fetch_from_board(
    adapter: BoardAdapter,
    request: BoardRequest,
    *,
    requested_url: str,
    client: httpx.Client | None = None,
) -> FetchedPosting:
    """Fetch and parse a posting through ``adapter``.

    Raises:
        JDFetchError: If the board's response is not JSON, or does not hold the
            posting the URL named.
    """
    document = fetch_document(request.api_url, client=client, accept=_JSON_ACCEPT)
    try:
        payload = json.loads(document.text)
    except json.JSONDecodeError as exc:
        raise JDFetchError(
            f"{adapter.board.value.title()} returned a response that was not JSON "
            f"for {requested_url}."
        ) from exc
    return adapter.read(payload, request, requested_url, document.url)


# ---------------------------------------------------------------------------
# Greenhouse
# ---------------------------------------------------------------------------

_GREENHOUSE_HOSTS: Final = frozenset(
    {
        "boards.greenhouse.io",
        "job-boards.greenhouse.io",
        "boards.eu.greenhouse.io",
        "job-boards.eu.greenhouse.io",
    }
)


def match_greenhouse(parts: SplitResult) -> BoardRequest | None:
    """Recognize ``/<board>/jobs/<id>`` and the ``embed/job_app`` variant."""
    host = parts.hostname or ""
    if host not in _GREENHOUSE_HOSTS:
        return None
    api_host = "boards-api.eu.greenhouse.io" if ".eu." in host else "boards-api.greenhouse.io"

    segments = _segments(parts)
    if len(segments) >= 3 and segments[1] == "jobs":
        board, job_id = segments[0], segments[2]
    else:
        # The embed form carries both in the query string instead of the path.
        query = _query(parts)
        board, job_id = query.get("for", ""), query.get("token", "")
        if not (board and job_id):
            return None
    return BoardRequest(
        api_url=f"https://{api_host}/v1/boards/{board}/jobs/{job_id}",
        job_id=job_id,
        org=board,
    )


def read_greenhouse(
    payload: object, request: BoardRequest, requested_url: str, final_url: str
) -> FetchedPosting:
    """Read Greenhouse's job payload.

    ``content`` is HTML that was entity-escaped on its way into the JSON, so it
    arrives as ``&lt;p&gt;``; :func:`html_to_text` detects and undoes that.
    """
    node = _require_object(payload, request, requested_url)
    location_node = node.get("location")
    location = _string(location_node.get("name")) if isinstance(location_node, dict) else None
    title = _string(node.get("title"))
    return FetchedPosting(
        text=_assemble(title, location, html_to_text(_string(node.get("content")) or "")),
        board=PostingBoard.GREENHOUSE,
        requested_url=requested_url,
        final_url=final_url,
        title=title,
        company=_string(node.get("company_name")) or request.org,
        location=location,
    )


# ---------------------------------------------------------------------------
# Lever
# ---------------------------------------------------------------------------

_LEVER_HOSTS: Final = frozenset({"jobs.lever.co", "jobs.eu.lever.co"})


def match_lever(parts: SplitResult) -> BoardRequest | None:
    """Recognize ``jobs.lever.co/<org>/<posting-id>``."""
    if (parts.hostname or "") not in _LEVER_HOSTS:
        return None
    segments = _segments(parts)
    if len(segments) < 2:
        return None
    org, job_id = segments[0], segments[1]
    return BoardRequest(
        api_url=f"https://api.lever.co/v0/postings/{org}/{job_id}?mode=json",
        job_id=job_id,
        org=org,
    )


def read_lever(
    payload: object, request: BoardRequest, requested_url: str, final_url: str
) -> FetchedPosting:
    """Reassemble a Lever posting from the four fields it is split across.

    ``lists`` holds the requirements and responsibilities as separate blocks, each
    with its own heading — the part a naive read of ``descriptionPlain`` alone misses
    entirely, since that field stops before them.
    """
    node = _require_object(payload, request, requested_url)
    categories = node.get("categories")
    location = _string(categories.get("location")) if isinstance(categories, dict) else None

    sections = [_string(node.get("text")), location, _string(node.get("descriptionPlain"))]
    for block in node.get("lists") or []:
        if isinstance(block, dict):
            sections.append(_string(block.get("text")))
            sections.append(html_to_text(_string(block.get("content")) or ""))
    sections.append(_string(node.get("additionalPlain")))

    return FetchedPosting(
        text=_assemble(*sections),
        board=PostingBoard.LEVER,
        requested_url=requested_url,
        final_url=final_url,
        title=_string(node.get("text")),
        company=request.org,
        location=location,
    )


# ---------------------------------------------------------------------------
# Ashby
# ---------------------------------------------------------------------------

_ASHBY_HOST: Final = "jobs.ashbyhq.com"


def match_ashby(parts: SplitResult) -> BoardRequest | None:
    """Recognize ``jobs.ashbyhq.com/<org>/<posting-uuid>``."""
    if (parts.hostname or "") != _ASHBY_HOST:
        return None
    segments = _segments(parts)
    if len(segments) < 2:
        return None
    org, job_id = segments[0], segments[1]
    return BoardRequest(
        api_url=f"https://api.ashbyhq.com/posting-api/job-board/{org}",
        job_id=job_id,
        org=org,
    )


def read_ashby(
    payload: object, request: BoardRequest, requested_url: str, final_url: str
) -> FetchedPosting:
    """Pick one posting out of the whole board.

    Ashby's public API publishes a board rather than a job, so the URL's identifier is
    matched here. It hands back ``descriptionPlain``, so no HTML is involved at all.
    """
    node = _require_object(payload, request, requested_url)
    jobs = node.get("jobs")
    listings = [job for job in jobs if isinstance(job, dict)] if isinstance(jobs, list) else []
    for job in listings:
        if _string(job.get("id")) == request.job_id:
            return FetchedPosting(
                text=_assemble(
                    _string(job.get("title")),
                    _string(job.get("location")),
                    tidy(_string(job.get("descriptionPlain")) or ""),
                ),
                board=PostingBoard.ASHBY,
                requested_url=requested_url,
                final_url=final_url,
                title=_string(job.get("title")),
                company=request.org,
                location=_string(job.get("location")),
            )
    raise JDFetchError(
        f"Ashby's board for {request.org!r} does not list a posting with id "
        f"{request.job_id!r}. It may have been closed or unlisted: {requested_url}"
    )


# ---------------------------------------------------------------------------
# Workday
# ---------------------------------------------------------------------------

_WORKDAY_HOST_SUFFIX: Final = ".myworkdayjobs.com"


def match_workday(parts: SplitResult) -> BoardRequest | None:
    """Derive the ``/wday/cxs/`` JSON path from a Workday careers URL.

    A posting URL is ``<tenant>.wdN.myworkdayjobs.com/[<locale>/]<site>/job/<path>``.
    The tenant is the first host label and the site is the path segment before
    ``job``; the optional locale segment between them is skipped.
    """
    host = parts.hostname or ""
    if not host.endswith(_WORKDAY_HOST_SUFFIX):
        return None
    tenant = host.split(".", 1)[0]

    segments = _segments(parts)
    if segments and _LOCALE_SEGMENT.match(segments[0]):
        segments = segments[1:]
    if len(segments) < 3 or segments[1] != "job":
        return None

    site, rest = segments[0], "/".join(segments[2:])
    return BoardRequest(
        api_url=f"https://{host}/wday/cxs/{tenant}/{site}/job/{rest}",
        job_id=rest,
        org=tenant,
    )


def read_workday(
    payload: object, request: BoardRequest, requested_url: str, final_url: str
) -> FetchedPosting:
    """Read the posting out of Workday's CXS response."""
    node = _require_object(payload, request, requested_url)
    info = node.get("jobPostingInfo")
    if not isinstance(info, dict):
        raise JDFetchError(
            f"Workday returned no job posting for {requested_url}. "
            "The posting may have closed, or the URL may point at a search page."
        )
    organization = node.get("hiringOrganization")
    company = _string(organization.get("name")) if isinstance(organization, dict) else None

    return FetchedPosting(
        text=_assemble(
            _string(info.get("title")),
            _string(info.get("location")),
            html_to_text(_string(info.get("jobDescription")) or ""),
        ),
        board=PostingBoard.WORKDAY,
        requested_url=requested_url,
        final_url=final_url,
        title=_string(info.get("title")),
        company=company or request.org,
        location=_string(info.get("location")),
    )


BOARD_ADAPTERS: Final = (
    BoardAdapter(PostingBoard.GREENHOUSE, match_greenhouse, read_greenhouse),
    BoardAdapter(PostingBoard.LEVER, match_lever, read_lever),
    BoardAdapter(PostingBoard.ASHBY, match_ashby, read_ashby),
    BoardAdapter(PostingBoard.WORKDAY, match_workday, read_workday),
)


def select_board(url: str) -> tuple[BoardAdapter, BoardRequest] | None:
    """Return the adapter that recognizes ``url``, if any."""
    parts = urlsplit(url)
    for adapter in BOARD_ADAPTERS:
        request = adapter.match(parts)
        if request is not None:
            return adapter, request
    return None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _segments(parts: SplitResult) -> list[str]:
    """Path segments with empties removed, so a trailing slash changes nothing."""
    return [segment for segment in parts.path.split("/") if segment]


def _query(parts: SplitResult) -> dict[str, str]:
    """Parse the query string, keeping the first value for each name."""
    pairs = (pair.split("=", 1) for pair in parts.query.split("&") if "=" in pair)
    result: dict[str, str] = {}
    for name, value in pairs:
        result.setdefault(unquote_plus(name), unquote_plus(value))
    return result


def _require_object(payload: object, request: BoardRequest, requested_url: str) -> dict[str, Any]:
    """Insist the board returned an object, naming the URL when it did not."""
    if not isinstance(payload, dict):
        raise JDFetchError(
            f"Unexpected response for {requested_url}: expected a job posting object "
            f"from {request.api_url}."
        )
    return payload


def _string(value: object) -> str | None:
    """Accept a non-empty string; treat anything else as absent."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _assemble(*sections: str | None) -> str:
    """Join the parts of a posting, dropping the ones the board did not supply."""
    return "\n\n".join(section for section in sections if section)
