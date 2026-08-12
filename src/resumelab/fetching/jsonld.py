"""schema.org ``JobPosting`` extraction.

This is the single highest-yield way to read a posting off a page nobody wrote an
adapter for. Job boards embed a ``JobPosting`` block for Google Jobs, so the posting
arrives already separated from the site's navigation, cookie banner, and footer —
which is exactly the separation a text-scraping heuristic gets wrong.

It is also frequently present on pages whose visible content is rendered by
JavaScript, because the block is emitted server-side for crawlers. That is why this
is tried before falling back to reading the body.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Final

logger = logging.getLogger(__name__)

_LD_JSON_TYPE: Final = "application/ld+json"
_JOB_POSTING: Final = "JobPosting"


@dataclass(frozen=True)
class JsonLdPosting:
    """The fields of a ``JobPosting`` block that a resume pipeline can use."""

    title: str | None
    company: str | None
    location: str | None
    description_html: str


class _LdJsonCollector(HTMLParser):
    """Collect the body of every ``application/ld+json`` script on the page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self._capturing = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "script":
            values = {name.lower(): (value or "") for name, value in attrs}
            self._capturing = values.get("type", "").strip().lower() == _LD_JSON_TYPE

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._capturing = False

    def handle_data(self, data: str) -> None:
        # HTMLParser puts script content in CDATA mode, so the JSON arrives with its
        # entities and escapes intact rather than partially decoded.
        if self._capturing:
            self.blocks.append(data)


def find_job_posting(html: str) -> JsonLdPosting | None:
    """Return the first ``JobPosting`` described on the page, if there is one."""
    collector = _LdJsonCollector()
    collector.feed(html)
    collector.close()

    for block in collector.blocks:
        try:
            document = json.loads(block)
        except json.JSONDecodeError:
            # A malformed block is not a reason to give up: pages often carry several,
            # and the fallback path is still available if none of them parse.
            logger.debug("skipping unparseable ld+json block")
            continue
        posting = _find_posting_node(document)
        if posting is not None:
            return _read_posting(posting)
    return None


def _find_posting_node(node: object) -> dict[str, Any] | None:
    """Search a decoded JSON-LD document for a ``JobPosting``.

    Recursive because the block may be the posting, a list of things one of which is
    the posting, or an ``@graph`` wrapper containing it.
    """
    if isinstance(node, list):
        for item in node:
            found = _find_posting_node(item)
            if found is not None:
                return found
        return None
    if not isinstance(node, dict):
        return None
    if _JOB_POSTING in _types_of(node):
        return node
    return _find_posting_node(list(node.values()))


def _types_of(node: dict[str, Any]) -> list[str]:
    """Normalize ``@type``, which is a string on most pages and a list on some."""
    declared = node.get("@type")
    if isinstance(declared, str):
        return [declared]
    if isinstance(declared, list):
        return [entry for entry in declared if isinstance(entry, str)]
    return []


def _read_posting(node: dict[str, Any]) -> JsonLdPosting:
    return JsonLdPosting(
        title=_text(node.get("title")),
        company=_organization_name(node.get("hiringOrganization")),
        location=_location_name(node.get("jobLocation")),
        description_html=_text(node.get("description")) or "",
    )


def _text(value: object) -> str | None:
    """Accept a plain string; ignore anything else rather than stringifying it."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _organization_name(value: object) -> str | None:
    """``hiringOrganization`` is an Organization object, or sometimes just a name."""
    if isinstance(value, dict):
        return _text(value.get("name"))
    return _text(value)


def _location_name(value: object) -> str | None:
    """Reduce a ``Place`` to the "City, Region" a resume would print.

    A posting may list several locations; the first is enough for provenance, and the
    full text of the posting still carries the rest.
    """
    if isinstance(value, list):
        for item in value:
            found = _location_name(item)
            if found is not None:
                return found
        return None
    if not isinstance(value, dict):
        return _text(value)

    address = value.get("address")
    if not isinstance(address, dict):
        return _text(value.get("name"))
    parts = [
        _text(address.get(field))
        for field in ("addressLocality", "addressRegion", "addressCountry")
    ]
    present = [part for part in parts if part]
    return ", ".join(present) if present else _text(value.get("name"))
