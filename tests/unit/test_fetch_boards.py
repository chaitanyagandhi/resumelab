"""Tests for the applicant-tracking-system adapters.

The payloads here mirror the shape of real responses captured from each board,
including the parts that are awkward: Greenhouse's twice-escaped HTML, Lever's
posting split across four fields, and Ashby publishing a board rather than a job.
"""

import json

import httpx
import pytest

from resumelab.exceptions import JDFetchError
from resumelab.fetching.boards import fetch_from_board, select_board
from resumelab.fetching.models import PostingBoard

GREENHOUSE_URL = "https://job-boards.greenhouse.io/northlake/jobs/8077887"
GREENHOUSE_API = "https://boards-api.greenhouse.io/v1/boards/northlake/jobs/8077887"
LEVER_URL = "https://jobs.lever.co/northlake/33538a2f-d27d-4a96-8f05-fa4b0e4d940e"
LEVER_API = (
    "https://api.lever.co/v0/postings/northlake/33538a2f-d27d-4a96-8f05-fa4b0e4d940e?mode=json"
)
ASHBY_URL = "https://jobs.ashbyhq.com/northlake/34413f8d-26bf-4bbc-8ade-eb309a0e2245"
ASHBY_API = "https://api.ashbyhq.com/posting-api/job-board/northlake"
WORKDAY_URL = (
    "https://northlake.wd5.myworkdayjobs.com/en-US/NorthlakeCareers"
    "/job/US-CA-Sunnyvale/Storage-Engineer_JR2017740"
)
WORKDAY_API = (
    "https://northlake.wd5.myworkdayjobs.com/wday/cxs/northlake/NorthlakeCareers"
    "/job/US-CA-Sunnyvale/Storage-Engineer_JR2017740"
)


def json_response(payload) -> httpx.Response:
    return httpx.Response(
        200, text=json.dumps(payload), headers={"content-type": "application/json"}
    )


def fetch(url, routes, routed_client, record=None):
    routed = select_board(url)
    assert routed is not None, f"no adapter matched {url}"
    adapter, request = routed
    return fetch_from_board(
        adapter, request, requested_url=url, client=routed_client(routes, record=record)
    )


# --- routing --------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "board", "api_url"),
    [
        (GREENHOUSE_URL, PostingBoard.GREENHOUSE, GREENHOUSE_API),
        (
            "https://boards.greenhouse.io/northlake/jobs/8077887",
            PostingBoard.GREENHOUSE,
            GREENHOUSE_API,
        ),
        (
            "https://job-boards.eu.greenhouse.io/northlake/jobs/8077887",
            PostingBoard.GREENHOUSE,
            "https://boards-api.eu.greenhouse.io/v1/boards/northlake/jobs/8077887",
        ),
        (
            "https://boards.greenhouse.io/embed/job_app?token=8077887&for=northlake",
            PostingBoard.GREENHOUSE,
            GREENHOUSE_API,
        ),
        (LEVER_URL, PostingBoard.LEVER, LEVER_API),
        (ASHBY_URL, PostingBoard.ASHBY, ASHBY_API),
        (WORKDAY_URL, PostingBoard.WORKDAY, WORKDAY_API),
        (
            "https://northlake.wd5.myworkdayjobs.com/NorthlakeCareers"
            "/job/US-CA-Sunnyvale/Storage-Engineer_JR2017740",
            PostingBoard.WORKDAY,
            WORKDAY_API,
        ),
    ],
)
def test_a_posting_url_routes_to_its_board_api(url, board, api_url):
    routed = select_board(url)

    assert routed is not None
    adapter, request = routed
    assert adapter.board is board
    assert request.api_url == api_url


@pytest.mark.parametrize(
    "url",
    [
        "https://northlake.example.com/careers/storage-engineer",
        "https://www.indeed.com/viewjob?jk=abc123",
        "https://jobs.lever.co/northlake",
        "https://jobs.ashbyhq.com/northlake",
        "https://job-boards.greenhouse.io/northlake",
        "https://boards.greenhouse.io/embed/job_app?token=8077887",
        "https://northlake.wd5.myworkdayjobs.com/en-US/NorthlakeCareers",
    ],
)
def test_urls_without_a_recognizable_posting_are_not_routed(url):
    """A board URL that names no single posting must fall through, not guess."""
    assert select_board(url) is None


def test_a_trailing_slash_does_not_change_routing():
    routed = select_board(LEVER_URL + "/")

    assert routed is not None
    assert routed[1].api_url == LEVER_API


# --- Greenhouse -----------------------------------------------------------

GREENHOUSE_PAYLOAD = {
    "id": 8077887,
    "title": "Software Engineer, Cloud Storage Infrastructure",
    "company_name": "Northlake Systems",
    "location": {"name": "Sunnyvale, CA"},
    # Greenhouse escapes the HTML a second time on its way into this field.
    "content": (
        "&lt;h2&gt;What you will do&lt;/h2&gt;"
        "&lt;ul&gt;&lt;li&gt;Design services in Go and Java&lt;/li&gt;"
        "&lt;li&gt;Work with NVMe devices&lt;/li&gt;&lt;/ul&gt;"
    ),
}


def test_a_greenhouse_posting_is_read_from_the_board_api(routed_client):
    posting = fetch(
        GREENHOUSE_URL, {GREENHOUSE_API: json_response(GREENHOUSE_PAYLOAD)}, routed_client
    )

    assert posting.board is PostingBoard.GREENHOUSE
    assert posting.title == "Software Engineer, Cloud Storage Infrastructure"
    assert posting.company == "Northlake Systems"
    assert posting.location == "Sunnyvale, CA"
    assert posting.requested_url == GREENHOUSE_URL


def test_greenhouse_double_escaped_html_becomes_readable_text(routed_client):
    """Read naively, the whole posting arrives as literal `&lt;li&gt;` markup."""
    posting = fetch(
        GREENHOUSE_URL, {GREENHOUSE_API: json_response(GREENHOUSE_PAYLOAD)}, routed_client
    )

    assert "&lt;" not in posting.text
    assert "<li>" not in posting.text
    assert "- Design services in Go and Java" in posting.text
    assert "- Work with NVMe devices" in posting.text


def test_greenhouse_falls_back_to_the_board_name_when_the_company_is_absent(routed_client):
    payload = GREENHOUSE_PAYLOAD | {"company_name": None}

    posting = fetch(GREENHOUSE_URL, {GREENHOUSE_API: json_response(payload)}, routed_client)

    assert posting.company == "northlake"


# --- Lever ----------------------------------------------------------------

LEVER_PAYLOAD = {
    "text": "Software Engineer, Cloud Storage",
    "categories": {"location": "Sunnyvale, CA", "team": "Infrastructure"},
    "descriptionPlain": "Northlake builds the distributed storage platform.",
    "lists": [
        {"text": "What you will do", "content": "<ul><li>Design storage services</li></ul>"},
        {"text": "What we look for", "content": "<ul><li>Go or Java</li></ul>"},
    ],
    "additionalPlain": "Northlake is an equal opportunity employer.",
    "hostedUrl": LEVER_URL,
}


def test_a_lever_posting_is_reassembled_from_all_of_its_fields(routed_client):
    """Lever splits a posting across four fields; `descriptionPlain` alone stops
    before the requirements, which is most of what the analysis needs."""
    posting = fetch(LEVER_URL, {LEVER_API: json_response(LEVER_PAYLOAD)}, routed_client)

    assert posting.board is PostingBoard.LEVER
    assert posting.title == "Software Engineer, Cloud Storage"
    assert posting.location == "Sunnyvale, CA"
    assert "Northlake builds the distributed storage platform." in posting.text
    assert "What you will do" in posting.text
    assert "- Design storage services" in posting.text
    assert "What we look for" in posting.text
    assert "- Go or Java" in posting.text
    assert "equal opportunity" in posting.text


def test_a_malformed_entry_in_lever_lists_is_skipped(routed_client):
    payload = LEVER_PAYLOAD | {"lists": ["not an object", LEVER_PAYLOAD["lists"][0]]}

    posting = fetch(LEVER_URL, {LEVER_API: json_response(payload)}, routed_client)

    assert "- Design storage services" in posting.text


def test_a_lever_posting_without_lists_still_reads(routed_client):
    payload = LEVER_PAYLOAD | {"lists": None, "categories": None}

    posting = fetch(LEVER_URL, {LEVER_API: json_response(payload)}, routed_client)

    assert posting.location is None
    assert "Northlake builds the distributed storage platform." in posting.text


# --- Ashby ----------------------------------------------------------------

ASHBY_JOB_ID = "34413f8d-26bf-4bbc-8ade-eb309a0e2245"
ASHBY_PAYLOAD = {
    "jobs": [
        {
            "id": "00000000-0000-0000-0000-000000000000",
            "title": "Account Executive",
            "location": "New York, NY",
            "descriptionPlain": "Sell things.",
        },
        {
            "id": ASHBY_JOB_ID,
            "title": "Software Engineer, Storage",
            "location": "Sunnyvale, CA",
            "descriptionPlain": "Build   distributed storage\n\n\n\nservices in Go.",
        },
    ]
}


def test_an_ashby_posting_is_selected_out_of_the_whole_board(routed_client):
    """Ashby publishes no per-job endpoint, so the right posting must be picked here."""
    posting = fetch(ASHBY_URL, {ASHBY_API: json_response(ASHBY_PAYLOAD)}, routed_client)

    assert posting.board is PostingBoard.ASHBY
    assert posting.title == "Software Engineer, Storage"
    assert "Sell things." not in posting.text


def test_ashby_plain_text_is_tidied(routed_client):
    posting = fetch(ASHBY_URL, {ASHBY_API: json_response(ASHBY_PAYLOAD)}, routed_client)

    assert "Build distributed storage\n\nservices in Go." in posting.text


def test_a_posting_missing_from_the_ashby_board_is_reported(routed_client):
    posting_gone = {"jobs": [job for job in ASHBY_PAYLOAD["jobs"] if job["id"] != ASHBY_JOB_ID]}

    with pytest.raises(JDFetchError, match="does not list a posting"):
        fetch(ASHBY_URL, {ASHBY_API: json_response(posting_gone)}, routed_client)


# --- Workday --------------------------------------------------------------

WORKDAY_PAYLOAD = {
    "jobPostingInfo": {
        "title": "Software Engineer, Cloud Storage",
        "jobDescription": "<p>Build storage services.</p><ul><li>Go and Java</li></ul>",
        "location": "Sunnyvale, CA",
        "jobReqId": "JR2017740",
    },
    "hiringOrganization": {"name": "Northlake Systems"},
}


def test_a_workday_posting_is_read_from_the_cxs_endpoint(routed_client):
    """Workday's visible page is rendered by JavaScript and has no readable body,
    so this JSON path is the only way to read one at all."""
    posting = fetch(WORKDAY_URL, {WORKDAY_API: json_response(WORKDAY_PAYLOAD)}, routed_client)

    assert posting.board is PostingBoard.WORKDAY
    assert posting.title == "Software Engineer, Cloud Storage"
    assert posting.company == "Northlake Systems"
    assert posting.location == "Sunnyvale, CA"
    assert "Build storage services." in posting.text
    assert "- Go and Java" in posting.text


def test_a_workday_response_without_a_posting_is_reported(routed_client):
    routes = {WORKDAY_API: json_response({"userAuthenticated": False})}

    with pytest.raises(JDFetchError, match="no job posting"):
        fetch(WORKDAY_URL, routes, routed_client)


def test_workday_falls_back_to_the_tenant_when_no_organization_is_named(routed_client):
    payload = {"jobPostingInfo": WORKDAY_PAYLOAD["jobPostingInfo"]}

    posting = fetch(WORKDAY_URL, {WORKDAY_API: json_response(payload)}, routed_client)

    assert posting.company == "northlake"


# --- shared failure handling ----------------------------------------------


def test_a_non_json_board_response_is_reported(routed_client):
    routes = {GREENHOUSE_API: httpx.Response(200, text="<html>maintenance</html>")}

    with pytest.raises(JDFetchError, match="not JSON"):
        fetch(GREENHOUSE_URL, routes, routed_client)


def test_a_board_response_that_is_not_an_object_is_reported(routed_client):
    with pytest.raises(JDFetchError, match="Unexpected response"):
        fetch(GREENHOUSE_URL, {GREENHOUSE_API: json_response([1, 2, 3])}, routed_client)


def test_the_board_api_is_requested_rather_than_the_page(routed_client):
    """The whole point of an adapter is that the rendered page is never fetched."""
    requested: list[str] = []

    fetch(
        GREENHOUSE_URL,
        {GREENHOUSE_API: json_response(GREENHOUSE_PAYLOAD)},
        routed_client,
        record=requested,
    )

    assert requested == [GREENHOUSE_API]
