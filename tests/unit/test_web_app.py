"""Tests for the review UI's HTTP layer.

The app is built per test through the factory, so nothing here shares state with
anything else, and no server is ever started: Starlette's test client drives the
application directly.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from resumelab import __version__
from resumelab.web import create_app
from resumelab.web.app import INDEX_FILE, STATIC_DIRECTORY


@pytest.fixture
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


# --- the application ------------------------------------------------------


def test_the_factory_builds_a_new_application_each_time():
    """A module-level singleton would leak state between tests and between runs."""
    assert create_app() is not create_app()


def test_the_application_is_versioned_with_the_package():
    assert create_app().version == __version__


def test_the_factory_returns_a_fastapi_application():
    assert isinstance(create_app(), FastAPI)


# --- health ---------------------------------------------------------------


def test_health_reports_the_running_version(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


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
