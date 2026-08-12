"""Tests for the single entry point that turns a pasted link into posting text."""

import json

import httpx
import pytest

from resumelab.exceptions import JDFetchError
from resumelab.fetching import fetch_posting
from resumelab.fetching.models import FetchedPosting, PostingBoard

GREENHOUSE_URL = "https://job-boards.greenhouse.io/northlake/jobs/8077887"
GREENHOUSE_API = "https://boards-api.greenhouse.io/v1/boards/northlake/jobs/8077887"
GENERIC_URL = "https://northlake.example.com/careers/storage-engineer"

BODY = "Build distributed storage services in Go and Java on Linux, close to NVMe devices. "


def json_response(payload) -> httpx.Response:
    return httpx.Response(
        200, text=json.dumps(payload), headers={"content-type": "application/json"}
    )


# --- routing --------------------------------------------------------------


def test_a_board_url_is_read_through_its_adapter(routed_client):
    payload = {"title": "Storage Engineer", "content": "&lt;p&gt;Build storage.&lt;/p&gt;"}
    requested: list[str] = []

    posting = fetch_posting(
        GREENHOUSE_URL,
        client=routed_client({GREENHOUSE_API: json_response(payload)}, record=requested),
    )

    assert posting.board is PostingBoard.GREENHOUSE
    assert requested == [GREENHOUSE_API]


def test_an_unrecognized_url_falls_back_to_reading_the_page(routed_client):
    body = f"<html><body><main><p>{BODY * 4}</p></main></body></html>"
    client = routed_client({GENERIC_URL: httpx.Response(200, text=body)})

    posting = fetch_posting(GENERIC_URL, client=client)

    assert posting.board is PostingBoard.GENERIC
    assert "Build distributed storage services" in posting.text


def test_a_pasted_url_with_stray_whitespace_still_works(routed_client):
    body = f"<html><body><main><p>{BODY * 4}</p></main></body></html>"
    client = routed_client({GENERIC_URL: httpx.Response(200, text=body)})

    assert fetch_posting(f"  {GENERIC_URL}\n", client=client).text


def test_an_unusable_url_fails_before_any_request(routed_client):
    with pytest.raises(JDFetchError):
        fetch_posting("not-a-url", client=routed_client({}))


# --- what comes back ------------------------------------------------------


def test_the_posting_text_is_what_a_researcher_would_have_pasted(routed_client):
    """Nothing downstream should be able to tell a fetched posting from a pasted one."""
    payload = {
        "title": "Storage Engineer",
        "company_name": "Northlake Systems",
        "location": {"name": "Sunnyvale, CA"},
        "content": f"&lt;p&gt;{BODY}&lt;/p&gt;",
    }

    posting = fetch_posting(
        GREENHOUSE_URL, client=routed_client({GREENHOUSE_API: json_response(payload)})
    )

    assert isinstance(posting.text, str)
    assert "<" not in posting.text
    assert "&lt;" not in posting.text
    assert posting.text.strip() == posting.text


def test_a_posting_is_frozen():
    """The text handed downstream is the text that was fetched."""
    posting = FetchedPosting(
        text="Storage engineer",
        board=PostingBoard.GENERIC,
        requested_url=GENERIC_URL,
        final_url=GENERIC_URL,
    )

    with pytest.raises(ValueError, match="frozen"):
        posting.text = "something else"


# --- labels for run directories -------------------------------------------


def test_a_posting_is_labelled_by_its_company_and_title():
    posting = FetchedPosting(
        text="...",
        board=PostingBoard.GREENHOUSE,
        requested_url=GREENHOUSE_URL,
        final_url=GREENHOUSE_URL,
        title="Storage Engineer",
        company="Northlake Systems",
    )

    assert posting.label == "Northlake Systems Storage Engineer"


def test_a_posting_with_only_a_title_is_labelled_by_it():
    posting = FetchedPosting(
        text="...",
        board=PostingBoard.GENERIC,
        requested_url=GENERIC_URL,
        final_url=GENERIC_URL,
        title="Storage Engineer",
    )

    assert posting.label == "Storage Engineer"


def test_a_posting_that_named_nothing_still_has_a_label():
    """A run directory always needs a name, even from a page that gave us none."""
    posting = FetchedPosting(
        text="...",
        board=PostingBoard.GENERIC,
        requested_url=GENERIC_URL,
        final_url=GENERIC_URL,
    )

    assert posting.label == "generic"


# --- logging --------------------------------------------------------------


def test_the_posting_text_is_not_written_to_the_log(routed_client, caplog):
    """Logs record that a posting was read and how long it was, never its content."""
    payload = {"title": "Storage Engineer", "content": f"&lt;p&gt;{BODY}&lt;/p&gt;"}
    client = routed_client({GREENHOUSE_API: json_response(payload)})

    with caplog.at_level("INFO", logger="resumelab.fetching"):
        fetch_posting(GREENHOUSE_URL, client=client)

    assert "distributed storage services" not in caplog.text
    assert "board=greenhouse" in caplog.text
