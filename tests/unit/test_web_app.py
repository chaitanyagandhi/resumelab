"""Tests for the review UI's HTTP layer.

The app is built per test through the factory, so nothing here shares state with
anything else, and no server is ever started: Starlette's test client drives the
application directly. The pipeline is replaced at the app's own boundary — what is
under test here is routing, validation, and what leaves the process, not generation.
"""

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from resumelab import __version__
from resumelab.config import LLMProvider, Settings
from resumelab.exceptions import ConfigurationError, JDAnalysisError, PDFRenderingError
from resumelab.experiment.recorder import PDF_FILE, RESUME_FILE
from resumelab.web import create_app
from resumelab.web.app import INDEX_FILE, STATIC_DIRECTORY, _artifact
from resumelab.web.jobs import JobState

TIMEOUT = 5.0
RUN_ID = "2026-03-04T120000_acme-storage-engineer"


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None,
        openai_api_key="sk-test-not-a-real-key",
        output_dir=tmp_path / "output",
    )


@pytest.fixture
def app(settings, monkeypatch):
    """An app whose LLM client is never really built and never really called."""
    monkeypatch.setattr("resumelab.web.app.create_llm_client", lambda *_a, **_k: object())
    return create_app(settings)


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def completed(monkeypatch):
    """Make generation succeed instantly, reporting the run it wrote."""

    def fake(*_args, **_kwargs):
        return type("Result", (), {"run": type("Run", (), {"run_id": RUN_ID})()})()

    monkeypatch.setattr("resumelab.web.app.generate_resume", fake)
    monkeypatch.setattr("resumelab.web.app.load_job_description", lambda **_k: object())


@pytest.fixture
def run_directory(settings, generated_resume):
    """A finished run on disk, as the pipeline would have left it."""
    directory = settings.runs_dir / RUN_ID
    directory.mkdir(parents=True)
    (directory / RESUME_FILE).write_text(generated_resume.model_dump_json(), encoding="utf-8")
    (directory / PDF_FILE).write_bytes(b"%PDF-1.4 not really a pdf")
    return directory


def finish(client, response):
    """Poll a started job until it settles, the way the browser will."""
    job_id = response.json()["id"]
    for _ in range(int(TIMEOUT * 200)):
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["state"] != JobState.RUNNING:
            return body
    raise AssertionError(f"job {job_id} never finished")


# --- the application ------------------------------------------------------


def test_the_factory_builds_a_new_application_each_time(settings):
    """A module-level singleton would leak state between tests and between runs."""
    assert create_app(settings) is not create_app(settings)


def test_the_application_is_versioned_with_the_package(settings):
    assert create_app(settings).version == __version__


def test_the_factory_returns_a_fastapi_application(settings):
    assert isinstance(create_app(settings), FastAPI)


def test_settings_are_loaded_from_the_environment_when_none_are_given(monkeypatch, tmp_path):
    """How the CLI builds it: no settings to hand, so the process environment wins."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")

    assert create_app().state.settings.openai_api_key is not None


# --- health ---------------------------------------------------------------


def test_health_reports_the_running_version(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


# --- starting a run -------------------------------------------------------


def test_a_posting_url_starts_a_run(client, completed):
    response = client.post("/api/generate", json={"url": "https://example.test/jobs/1"})

    assert response.status_code == 202
    assert response.json()["state"] == JobState.RUNNING


def test_a_pasted_posting_starts_a_run(client, completed):
    response = client.post("/api/generate", json={"text": "Senior Go engineer, storage."})

    assert response.status_code == 202


def test_a_started_run_reports_the_run_it_produced(client, completed):
    started = client.post("/api/generate", json={"url": "https://example.test/jobs/1"})

    assert finish(client, started)["run_id"] == RUN_ID


def test_a_run_that_fails_reports_why(client, monkeypatch):
    def refuse(**_kwargs):
        raise JDAnalysisError("That posting could not be fetched.")

    monkeypatch.setattr("resumelab.web.app.load_job_description", refuse)

    started = client.post("/api/generate", json={"url": "https://example.test/jobs/1"})
    finished = finish(client, started)

    assert finished["state"] == JobState.FAILED
    assert finished["error"] == "That posting could not be fetched."


def test_supplying_neither_a_url_nor_text_is_rejected(client):
    assert client.post("/api/generate", json={}).status_code == 422


def test_supplying_both_a_url_and_text_is_rejected(client):
    """One posting per run: two sources is a mistake, not a merge."""
    response = client.post("/api/generate", json={"url": "https://a.test", "text": "b"})

    assert response.status_code == 422


def test_an_unknown_field_is_rejected_rather_than_ignored(client):
    response = client.post("/api/generate", json={"url": "https://a.test", "jd": "typo"})

    assert response.status_code == 422


def test_a_missing_api_key_is_reported_before_a_run_starts(settings, monkeypatch):
    """Immediate, because a job that fails a moment later reads like a real failure."""

    def refuse(*_args, **_kwargs):
        raise ConfigurationError("No API key is configured for openai.")

    monkeypatch.setattr("resumelab.web.app.create_llm_client", refuse)

    with TestClient(create_app(settings)) as client:
        response = client.post("/api/generate", json={"url": "https://a.test"})

    assert response.status_code == 400
    assert "No API key" in response.json()["detail"]


def test_the_requested_provider_is_the_one_used(settings, monkeypatch, completed):
    chosen = []
    monkeypatch.setattr(
        "resumelab.web.app.create_llm_client",
        lambda _settings, provider: chosen.append(provider) or object(),
    )

    with TestClient(create_app(settings)) as client:
        client.post("/api/generate", json={"url": "https://a.test", "provider": "anthropic"})

    assert chosen == [LLMProvider.ANTHROPIC]


# --- polling a run --------------------------------------------------------


def test_an_unknown_job_is_not_found(client):
    assert client.get("/api/jobs/nope").status_code == 404


# --- reading what a run produced ------------------------------------------


def test_the_generated_resume_is_served_as_json(client, run_directory, generated_resume):
    response = client.get(f"/api/runs/{RUN_ID}/resume")

    assert response.status_code == 200
    assert response.json()["summary"] == generated_resume.summary


def test_the_pdf_is_served(client, run_directory):
    response = client.get(f"/api/runs/{RUN_ID}/resume.pdf")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"


def test_the_pdf_is_shown_rather_than_downloaded(client, run_directory):
    """The browser's own viewer is a better preview than anything worth building."""
    assert "inline" in client.get(f"/api/runs/{RUN_ID}/resume.pdf").headers["content-disposition"]


def test_an_unknown_run_has_no_pdf(client):
    assert client.get("/api/runs/2026-01-01T000000_nope/resume.pdf").status_code == 404


def test_an_unknown_run_has_no_resume(client):
    assert client.get("/api/runs/2026-01-01T000000_nope/resume").status_code == 404


def test_a_run_whose_resume_is_unreadable_is_reported(client, settings):
    directory = settings.runs_dir / RUN_ID
    directory.mkdir(parents=True)
    (directory / RESUME_FILE).write_text("{ not json", encoding="utf-8")

    assert client.get(f"/api/runs/{RUN_ID}/resume").status_code == 500


# --- editing what a run produced ------------------------------------------


def test_an_edit_is_saved_and_rendered(client, run_directory, generated_resume):
    edited = generated_resume.model_copy(update={"summary": "A hand-written summary."})

    response = client.put(
        f"/api/runs/{RUN_ID}/edit", json={"resume": edited.model_dump(mode="json")}
    )

    assert response.status_code == 200
    assert response.json() == {"page_count": 1, "scale": 1.0, "fits_on_one_page": True}


def test_an_edit_leaves_what_the_model_wrote_untouched(client, run_directory, generated_resume):
    """The run is a research record; the edit is a separate document beside it."""
    original = (run_directory / RESUME_FILE).read_text(encoding="utf-8")
    edited = generated_resume.model_copy(update={"summary": "A hand-written summary."})

    client.put(f"/api/runs/{RUN_ID}/edit", json={"resume": edited.model_dump(mode="json")})

    assert (run_directory / RESUME_FILE).read_text(encoding="utf-8") == original
    assert client.get(f"/api/runs/{RUN_ID}/resume").json()["summary"] == generated_resume.summary


def test_an_edit_can_be_read_back(client, run_directory, generated_resume):
    edited = generated_resume.model_copy(update={"summary": "A hand-written summary."})

    client.put(f"/api/runs/{RUN_ID}/edit", json={"resume": edited.model_dump(mode="json")})

    assert client.get(f"/api/runs/{RUN_ID}/edit").json()["summary"] == "A hand-written summary."


def test_the_edited_pdf_is_served(client, run_directory, generated_resume):
    client.put(
        f"/api/runs/{RUN_ID}/edit", json={"resume": generated_resume.model_dump(mode="json")}
    )

    response = client.get(f"/api/runs/{RUN_ID}/edit.pdf")

    assert response.status_code == 200
    assert response.content.startswith(b"%PDF-")


def test_render_options_travel_with_the_edit(client, run_directory, generated_resume):
    response = client.put(
        f"/api/runs/{RUN_ID}/edit",
        json={
            "resume": generated_resume.model_dump(mode="json"),
            "options": {"include_summary": False, "include_gpa": False},
        },
    )

    assert response.status_code == 200


def test_an_invalid_section_order_is_rejected(client, run_directory, generated_resume):
    response = client.put(
        f"/api/runs/{RUN_ID}/edit",
        json={
            "resume": generated_resume.model_dump(mode="json"),
            "options": {"section_order": ["skills", "skills", "projects", "education"]},
        },
    )

    assert response.status_code == 422


def test_an_unrenderable_edit_is_reported(client, run_directory, generated_resume, monkeypatch):
    def refuse(*_args, **_kwargs):
        raise PDFRenderingError("the page could not be laid out")

    monkeypatch.setattr("resumelab.web.edits.render_resume", refuse)

    response = client.put(
        f"/api/runs/{RUN_ID}/edit", json={"resume": generated_resume.model_dump(mode="json")}
    )

    assert response.status_code == 400
    assert "could not be laid out" in response.json()["detail"]


def test_an_edit_to_an_unknown_run_is_not_found(client, generated_resume):
    response = client.put(
        "/api/runs/2026-01-01T000000_nope/edit",
        json={"resume": generated_resume.model_dump(mode="json")},
    )

    assert response.status_code == 404


def test_a_run_that_was_never_edited_has_no_edit(client, run_directory):
    assert client.get(f"/api/runs/{RUN_ID}/edit").status_code == 404
    assert client.get(f"/api/runs/{RUN_ID}/edit.pdf").status_code == 404


def test_an_edit_without_a_resume_is_rejected(client, run_directory):
    assert client.put(f"/api/runs/{RUN_ID}/edit", json={}).status_code == 422


# --- run identifiers come from the browser --------------------------------


@pytest.mark.parametrize("run_id", ["..", "../..", "../../etc"])
def test_a_run_id_cannot_escape_the_runs_directory(settings, run_id):
    """A traversal attempt and a typo get the same answer, so this is no oracle."""
    with pytest.raises(HTTPException) as raised:
        _artifact(settings, run_id, PDF_FILE)

    assert raised.value.status_code == 404


def test_an_absolute_run_id_is_refused(settings):
    with pytest.raises(HTTPException) as raised:
        _artifact(settings, "/etc/passwd", PDF_FILE)

    assert raised.value.status_code == 404


def test_a_real_run_resolves(settings, run_directory):
    assert _artifact(settings, RUN_ID, PDF_FILE).is_file()


# --- the page -------------------------------------------------------------


def test_the_root_serves_the_page(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "ResumeLab" in response.text


def test_the_page_carries_the_research_disclaimer(client):
    """The UI makes generated output look finished; the caveat has to travel with it."""
    assert "not in your source profile" in client.get("/").text


def test_the_stylesheet_is_served(client):
    response = client.get("/static/app.css")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/css")


def test_the_script_is_served(client):
    response = client.get("/static/app.js")

    assert response.status_code == 200
    assert "reportHealth" in response.text


def test_an_unknown_path_is_not_found(client):
    assert client.get("/nothing-here").status_code == 404


def test_a_static_file_cannot_shadow_the_api(client):
    """The mount is registered last, so /api stays the API whatever lands in static/."""
    assert client.get("/api/health").json()["status"] == "ok"


# --- what ships -----------------------------------------------------------


def test_the_static_directory_ships_with_the_package():
    """Served from the installed package, so these are files a wheel has to carry."""
    assert STATIC_DIRECTORY.is_dir()
    assert INDEX_FILE.is_file()


def test_the_page_loads_only_local_assets():
    """A local research tool that phoned out for a font or a CDN would be worse."""
    markup = INDEX_FILE.read_text(encoding="utf-8")

    assert "http://" not in markup
    assert "https://" not in markup
