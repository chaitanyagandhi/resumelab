"""Tests for the bounded HTTP fetch that job posting retrieval is built on."""

import httpx
import pytest

from resumelab.exceptions import JDFetchError, ResumeLabError
from resumelab.fetching import http as fetch_http
from resumelab.fetching.http import USER_AGENT, fetch_document, validate_url

URL = "https://northlake.example.com/careers/storage-engineer"


def responder(response: httpx.Response):
    return lambda request: response


# --- URL validation -------------------------------------------------------


@pytest.mark.parametrize("url", ["https://example.com/job", "http://example.com/job"])
def test_http_and_https_urls_are_accepted(url):
    assert validate_url(url) == url


def test_surrounding_whitespace_in_a_pasted_url_is_ignored():
    assert validate_url("  https://example.com/job  ") == "https://example.com/job"


@pytest.mark.parametrize(
    "url",
    ["file:///etc/passwd", "ftp://example.com/job", "data:text/html,hi"],
)
def test_non_http_schemes_are_rejected(url):
    """A mistyped argument must not be turned into a local file read."""
    with pytest.raises(JDFetchError, match="http"):
        validate_url(url)


def test_a_url_without_a_scheme_is_rejected():
    with pytest.raises(JDFetchError, match="Include https://"):
        validate_url("northlake.example.com/careers")


def test_a_url_without_a_host_is_rejected():
    with pytest.raises(JDFetchError, match="no host"):
        validate_url("https:///careers")


def test_fetch_errors_are_resumelab_errors():
    with pytest.raises(ResumeLabError):
        validate_url("ftp://example.com")


# --- successful retrieval -------------------------------------------------


def test_a_document_is_returned_with_its_content_type(make_http_client):
    client = make_http_client(
        responder(httpx.Response(200, text="<p>Posting</p>", headers={"content-type": "text/html"}))
    )

    document = fetch_document(URL, client=client)

    assert document.text == "<p>Posting</p>"
    assert document.content_type == "text/html"
    assert document.url == URL


def test_the_tool_identifies_itself(make_http_client):
    """The user agent names ResumeLab rather than imitating a browser."""
    seen: dict[str, str] = {}

    def handler(request):
        seen.update(request.headers)
        return httpx.Response(200, text="ok")

    fetch_document(URL, client=make_http_client(handler))

    assert seen["user-agent"] == USER_AGENT
    assert "ResumeLab" in USER_AGENT


def test_an_accept_header_is_sent_when_an_adapter_asks_for_json(make_http_client):
    seen: dict[str, str] = {}

    def handler(request):
        seen.update(request.headers)
        return httpx.Response(200, text="{}")

    fetch_document(URL, client=make_http_client(handler), accept="application/json")

    assert seen["accept"] == "application/json"


def test_the_final_url_after_a_redirect_is_reported(make_http_client):
    """Provenance has to record where the posting was read, not where we aimed."""
    destination = "https://northlake.example.com/jobs/42"

    def handler(request):
        if str(request.url) == URL:
            return httpx.Response(301, headers={"location": destination})
        return httpx.Response(200, text="Posting")

    document = fetch_document(URL, client=make_http_client(handler))

    assert document.url == destination


# --- refusals and failures ------------------------------------------------


@pytest.mark.parametrize("status", [401, 403])
def test_a_refused_request_suggests_pasting_the_posting_instead(make_http_client, status):
    """Sites behind bot protection are a known limit, not a bug to work around."""
    client = make_http_client(responder(httpx.Response(status)))

    with pytest.raises(JDFetchError, match="--jd-text") as error:
        fetch_document(URL, client=client)

    assert str(status) in str(error.value)


def test_a_missing_posting_is_reported_as_taken_down(make_http_client):
    client = make_http_client(responder(httpx.Response(404)))

    with pytest.raises(JDFetchError, match="taken down"):
        fetch_document(URL, client=client)


def test_a_server_error_is_reported_with_its_status(make_http_client):
    client = make_http_client(responder(httpx.Response(503)))

    with pytest.raises(JDFetchError, match="HTTP 503"):
        fetch_document(URL, client=client)


def test_a_timeout_is_reported_as_a_timeout(make_http_client):
    def handler(request):
        raise httpx.ConnectTimeout("too slow", request=request)

    with pytest.raises(JDFetchError, match="Timed out"):
        fetch_document(URL, client=make_http_client(handler))


def test_a_connection_failure_is_reported(make_http_client):
    def handler(request):
        raise httpx.ConnectError("no route to host", request=request)

    with pytest.raises(JDFetchError, match="Could not fetch"):
        fetch_document(URL, client=make_http_client(handler))


def test_too_many_redirects_is_reported(make_http_client):
    def handler(request):
        return httpx.Response(302, headers={"location": f"{URL}?n={request.url}"})

    with pytest.raises(JDFetchError, match="redirected too many times"):
        fetch_document(URL, client=make_http_client(handler))


# --- bounds ---------------------------------------------------------------


@pytest.mark.parametrize("content_type", ["application/pdf", "image/png", "video/mp4"])
def test_a_body_that_is_not_text_is_refused(make_http_client, content_type):
    client = make_http_client(
        responder(httpx.Response(200, content=b"binary", headers={"content-type": content_type}))
    )

    with pytest.raises(JDFetchError, match="not a readable job posting"):
        fetch_document(URL, client=client)


@pytest.mark.parametrize(
    "content_type",
    ["text/html; charset=utf-8", "application/json", "TEXT/PLAIN", "application/xhtml+xml"],
)
def test_readable_content_types_are_accepted(make_http_client, content_type):
    client = make_http_client(
        responder(httpx.Response(200, text="ok", headers={"content-type": content_type}))
    )

    assert fetch_document(URL, client=client).text == "ok"


def test_a_response_without_a_content_type_is_still_read(make_http_client):
    """Absence is not evidence of a binary body, and some boards omit the header."""
    client = make_http_client(
        responder(httpx.Response(200, text="Posting", headers={"content-type": ""}))
    )

    assert fetch_document(URL, client=client).text == "Posting"


# --- the client this module owns ------------------------------------------


def test_a_request_without_a_supplied_client_is_bounded(monkeypatch):
    """The production path builds its own client; its limits are the ones that matter."""
    built: dict[str, object] = {}
    original = httpx.Client

    def record(**kwargs):
        built.update(kwargs)
        return original(transport=httpx.MockTransport(responder(httpx.Response(200, text="ok"))))

    monkeypatch.setattr(fetch_http.httpx, "Client", record)

    assert fetch_document(URL, timeout=5.0).text == "ok"
    assert built["timeout"] == 5.0
    assert built["follow_redirects"] is True
    assert built["max_redirects"] == fetch_http.MAX_REDIRECTS


def test_an_oversized_response_is_refused(make_http_client, monkeypatch):
    """Streamed rather than buffered, because Content-Length is optional."""
    monkeypatch.setattr(fetch_http, "MAX_RESPONSE_BYTES", 64)
    client = make_http_client(responder(httpx.Response(200, text="x" * 5000)))

    with pytest.raises(JDFetchError, match="larger than"):
        fetch_document(URL, client=client)


def test_a_response_at_the_size_limit_is_read(make_http_client, monkeypatch):
    monkeypatch.setattr(fetch_http, "MAX_RESPONSE_BYTES", 64)
    client = make_http_client(responder(httpx.Response(200, text="x" * 64)))

    assert len(fetch_document(URL, client=client).text) == 64


def test_an_invalid_url_is_rejected_before_any_request(make_http_client):
    def handler(request):
        raise AssertionError("no request should be made")

    with pytest.raises(JDFetchError):
        fetch_document("file:///etc/passwd", client=make_http_client(handler))
