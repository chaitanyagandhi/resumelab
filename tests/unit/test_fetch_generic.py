"""Tests for reading a posting from a page that belongs to no known board."""

import json

import httpx
import pytest

from resumelab.exceptions import JDFetchError
from resumelab.fetching.generic import fetch_generic
from resumelab.fetching.models import PostingBoard

URL = "https://northlake.example.com/careers/storage-engineer"

BODY = "Build distributed storage services in Go and Java on Linux, close to NVMe devices. "
LONG_TEXT = BODY * 4
DESCRIPTION_HTML = f"<p>{LONG_TEXT}</p><ul><li>Go and Java</li></ul>"

POSTING_BLOCK = {
    "@context": "https://schema.org",
    "@type": "JobPosting",
    "title": "Software Engineer, Cloud Storage",
    "hiringOrganization": {"name": "Northlake Systems"},
    "jobLocation": {"address": {"addressLocality": "Sunnyvale", "addressRegion": "CA"}},
    "description": DESCRIPTION_HTML,
}


def html_response(body: str) -> httpx.Response:
    return httpx.Response(200, text=body, headers={"content-type": "text/html"})


def page_with_block(block, body: str = "<p>rendered by javascript</p>") -> httpx.Response:
    script = f'<script type="application/ld+json">{json.dumps(block)}</script>'
    return html_response(f"<html><head>{script}</head><body>{body}</body></html>")


# --- the structured path --------------------------------------------------


def test_a_posting_is_read_from_a_job_posting_block(routed_client):
    posting = fetch_generic(URL, client=routed_client({URL: page_with_block(POSTING_BLOCK)}))

    assert posting.board is PostingBoard.GENERIC
    assert posting.title == "Software Engineer, Cloud Storage"
    assert posting.company == "Northlake Systems"
    assert posting.location == "Sunnyvale, CA"
    assert "- Go and Java" in posting.text


def test_the_structured_block_is_preferred_over_the_page_body(routed_client):
    """The block is the posting; the body is the posting plus the whole website."""
    body = "<body><p>Cookie banner. Related jobs. " + LONG_TEXT + "</p></body>"
    client = routed_client({URL: page_with_block(POSTING_BLOCK, body)})

    posting = fetch_generic(URL, client=client)

    assert "Cookie banner" not in posting.text


def test_a_block_with_too_thin_a_description_falls_back_to_the_page(routed_client):
    """Some sites emit a stub block with a one-line description; the page has more."""
    stub = POSTING_BLOCK | {"description": "<p>Apply now.</p>"}
    body = f"<body><main><p>{LONG_TEXT}</p></main></body>"

    posting = fetch_generic(URL, client=routed_client({URL: page_with_block(stub, body)}))

    assert "Build distributed storage services" in posting.text
    assert posting.title == "Software Engineer, Cloud Storage"


# --- the page-text fallback -----------------------------------------------


def test_a_posting_is_read_from_the_page_when_there_is_no_block(routed_client):
    body = f"<html><body><main><p>{LONG_TEXT}</p></main></body></html>"

    posting = fetch_generic(URL, client=routed_client({URL: html_response(body)}))

    assert "Build distributed storage services" in posting.text
    assert posting.title is None


def test_site_chrome_outside_the_content_region_is_left_out(routed_client):
    body = (
        "<html><body><header>Careers at Northlake</header>"
        f"<main><p>{LONG_TEXT}</p></main>"
        "<footer>Privacy policy</footer></body></html>"
    )

    posting = fetch_generic(URL, client=routed_client({URL: html_response(body)}))

    assert "Careers at Northlake" not in posting.text
    assert "Privacy policy" not in posting.text


# --- pages that hold no posting -------------------------------------------


def test_a_javascript_rendered_page_is_reported_as_such(routed_client):
    """The common real failure: a valid page whose body is an empty container."""
    body = '<html><body><div id="root"></div></body></html>'

    with pytest.raises(JDFetchError, match="renders its content with JavaScript"):
        fetch_generic(URL, client=routed_client({URL: html_response(body)}))


def test_the_failure_message_points_at_the_manual_alternatives(routed_client):
    body = "<html><body><p>Loading...</p></body></html>"

    with pytest.raises(JDFetchError, match="--jd-text") as error:
        fetch_generic(URL, client=routed_client({URL: html_response(body)}))

    assert "--jd" in str(error.value)


def test_provenance_records_where_the_posting_was_actually_read(routed_client):
    posting = fetch_generic(URL, client=routed_client({URL: page_with_block(POSTING_BLOCK)}))

    assert posting.requested_url == URL
    assert posting.final_url == URL
