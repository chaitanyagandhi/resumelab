"""Tests for the ui command.

The server is never started: ``uvicorn.run`` blocks forever, so it is replaced and
inspected. What matters here is what the command hands it.
"""

import pytest
from fastapi import FastAPI
from typer.testing import CliRunner

from resumelab.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_working_directory(monkeypatch, tmp_path):
    """Run from an empty directory so a developer's real .env is never read."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")


@pytest.fixture
def served(monkeypatch):
    """Capture the arguments the command would have started a server with."""
    calls: list[dict] = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: calls.append({"app": app, **kwargs}))
    return calls


def test_the_server_is_started(served):
    result = runner.invoke(app, ["ui"])

    assert result.exit_code == 0
    assert len(served) == 1


def test_the_application_served_is_the_review_ui(served):
    runner.invoke(app, ["ui"])

    assert isinstance(served[0]["app"], FastAPI)
    assert served[0]["app"].title == "ResumeLab"


def test_it_binds_to_loopback_by_default(served):
    """The UI spends a real API budget and reads a real profile; the network is not
    invited by accident."""
    runner.invoke(app, ["ui"])

    assert served[0]["host"] == "127.0.0.1"
    assert served[0]["port"] == 8000


def test_the_host_and_port_can_be_overridden(served):
    runner.invoke(app, ["ui", "--host", "0.0.0.0", "--port", "9123"])

    assert served[0]["host"] == "0.0.0.0"
    assert served[0]["port"] == 9123


def test_the_url_to_open_is_printed(served):
    result = runner.invoke(app, ["ui", "--port", "9123"])

    assert "http://127.0.0.1:9123" in result.stdout


def test_the_configured_log_level_reaches_the_server(monkeypatch, served):
    """uvicorn wants the level in lower case; ours is stored upper."""
    monkeypatch.setenv("LOG_LEVEL", "WARNING")

    runner.invoke(app, ["ui"])

    assert served[0]["log_level"] == "warning"


def test_a_failure_to_start_is_reported_as_one_line(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise OSError("address already in use")

    monkeypatch.setattr("uvicorn.run", refuse)

    result = runner.invoke(app, ["ui"])

    assert result.exit_code == 1
    assert "address already in use" in result.stderr


def test_a_failure_to_start_can_be_debugged(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise OSError("address already in use")

    monkeypatch.setattr("uvicorn.run", refuse)

    result = runner.invoke(app, ["ui", "--debug"])

    assert isinstance(result.exception, OSError)
