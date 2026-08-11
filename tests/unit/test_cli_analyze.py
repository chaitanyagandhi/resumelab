"""Tests for the analyze command.

The LLM is replaced with a recording fake, so nothing here reaches a network.
"""

import json
import logging

import pytest
from typer.testing import CliRunner

from resumelab import __version__
from resumelab.cli import app
from resumelab.config import LLMProvider
from resumelab.exceptions import JDAnalysisError, UnsafePathError
from resumelab.models.analysis import JobAnalysis

OPENAI_KEY = "sk-test-not-a-real-key"
ANTHROPIC_KEY = "sk-ant-test-not-a-real-key"

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_working_directory(monkeypatch, tmp_path):
    """Run from an empty directory so a developer's real .env is never read."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)


@pytest.fixture
def fake_llm(monkeypatch, make_llm_client, job_analysis):
    """Install a recording client and hand it back for inspection."""
    client = make_llm_client([job_analysis])
    monkeypatch.setattr("resumelab.cli.create_llm_client", lambda *_a, **_k: client)
    return client


@pytest.fixture
def jd_file(tmp_path):
    path = tmp_path / "job.txt"
    path.write_text(
        "Storage Infrastructure Engineer. Build distributed storage services in Go "
        "and Java on Linux, working with NVMe devices and network storage protocols.",
        encoding="utf-8",
    )
    return path


def invoke(*args):
    return runner.invoke(app, list(args))


# --- the happy path -------------------------------------------------------


def test_a_job_description_file_is_analyzed(jd_file, fake_llm):
    result = invoke("analyze", "--jd", str(jd_file))

    assert result.exit_code == 0
    assert len(fake_llm.calls) == 1


def test_inline_text_is_analyzed(fake_llm):
    result = invoke(
        "analyze",
        "--jd-text",
        "Go engineer for distributed storage systems work on Linux with NVMe devices.",
    )

    assert result.exit_code == 0
    assert len(fake_llm.calls) == 1


def test_the_analysis_is_printed(jd_file, fake_llm, job_analysis):
    result = invoke("analyze", "--jd", str(jd_file))

    assert job_analysis.role_title in result.stdout
    assert job_analysis.company in result.stdout


def test_the_output_leads_with_what_drives_the_pipeline(jd_file, fake_llm, job_analysis):
    """technical_identity is what every later stage aims at, so it is not buried."""
    result = invoke("analyze", "--jd", str(jd_file))

    assert "TECHNICAL IDENTITY" in result.stdout
    assert job_analysis.technical_identity.split(".")[0] in " ".join(result.stdout.split())


def test_extracted_terms_are_shown(jd_file, fake_llm):
    result = invoke("analyze", "--jd", str(jd_file))

    assert "NVMe-oF" in result.stdout
    assert "HIGH VALUE KEYWORDS" in result.stdout


def test_empty_term_sections_are_omitted(jd_file, fake_llm):
    """The fixture analysis names no frameworks; an empty heading is noise."""
    result = invoke("analyze", "--jd", str(jd_file))

    assert "FRAMEWORKS" not in result.stdout


def test_no_resume_is_generated(jd_file, fake_llm):
    """analyze exists precisely so a researcher can read the plan input cheaply."""
    invoke("analyze", "--jd", str(jd_file))

    assert [call.purpose for call in fake_llm.calls] == ["jd_analysis"]


# --- saving the analysis --------------------------------------------------


def test_the_analysis_can_be_written_as_json(tmp_path, jd_file, fake_llm, job_analysis):
    target = tmp_path / "out" / "jd_analysis.json"

    result = invoke("analyze", "--jd", str(jd_file), "--output", str(target))

    assert result.exit_code == 0
    recorded = json.loads(target.read_text(encoding="utf-8"))
    assert recorded["technical_identity"] == job_analysis.technical_identity


def test_writing_json_reports_where_it_went(tmp_path, jd_file, fake_llm):
    target = tmp_path / "analysis.json"

    result = invoke("analyze", "--jd", str(jd_file), "-o", str(target))

    assert str(target) in result.stdout


def test_an_unwritable_output_path_is_reported_readably(tmp_path, jd_file, fake_llm):
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    blocked.chmod(0o500)
    try:
        result = invoke("analyze", "--jd", str(jd_file), "-o", str(blocked / "x" / "a.json"))

        assert result.exit_code == 1
        assert "analysis output path" in result.stderr
        assert "Traceback" not in result.stderr
    finally:
        blocked.chmod(0o700)


# --- input validation -----------------------------------------------------


def test_supplying_both_inputs_is_rejected(jd_file, fake_llm):
    result = invoke("analyze", "--jd", str(jd_file), "--jd-text", "Some posting text here.")

    assert result.exit_code == 1
    assert "not both" in result.stderr


def test_supplying_neither_input_is_rejected(fake_llm):
    result = invoke("analyze")

    assert result.exit_code == 1
    assert "required" in result.stderr


def test_a_missing_file_is_reported_without_a_traceback(tmp_path, fake_llm):
    result = invoke("analyze", "--jd", str(tmp_path / "absent.txt"))

    assert result.exit_code == 1
    assert "not found" in result.stderr
    assert "Traceback" not in result.stderr


# --- provider selection ---------------------------------------------------


def test_the_provider_can_be_overridden(monkeypatch, jd_file, make_llm_client, job_analysis):
    chosen: list[LLMProvider | None] = []
    client = make_llm_client([job_analysis])

    def record(_settings, *, provider=None):
        chosen.append(provider)
        return client

    monkeypatch.setattr("resumelab.cli.create_llm_client", record)

    assert invoke("analyze", "--jd", str(jd_file), "--provider", "anthropic").exit_code == 0
    assert chosen == [LLMProvider.ANTHROPIC]


def test_without_the_flag_the_configured_provider_is_used(
    monkeypatch, jd_file, make_llm_client, job_analysis
):
    chosen: list[LLMProvider | None] = []
    client = make_llm_client([job_analysis])
    monkeypatch.setattr(
        "resumelab.cli.create_llm_client",
        lambda _s, *, provider=None: (chosen.append(provider), client)[1],
    )

    invoke("analyze", "--jd", str(jd_file))

    assert chosen == [None]


def test_an_unknown_provider_is_rejected_by_the_parser(jd_file, fake_llm):
    result = invoke("analyze", "--jd", str(jd_file), "--provider", "gemini")

    assert result.exit_code != 0


# --- configuration and failures -------------------------------------------


def test_a_missing_api_key_is_reported_readably(monkeypatch, jd_file):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = invoke("analyze", "--jd", str(jd_file))

    assert result.exit_code == 1
    assert "OPENAI_API_KEY" in result.stderr
    assert "Traceback" not in result.stderr


def test_a_generation_failure_is_reported_readably(monkeypatch, jd_file, make_llm_client):
    failing = make_llm_client([JDAnalysisError("the provider was unreachable")])
    monkeypatch.setattr("resumelab.cli.create_llm_client", lambda *_a, **_k: failing)

    result = invoke("analyze", "--jd", str(jd_file))

    assert result.exit_code == 1
    assert "the provider was unreachable" in result.stderr


def test_debug_re_raises_so_the_traceback_is_available(monkeypatch, jd_file, make_llm_client):
    failing = make_llm_client([JDAnalysisError("the provider was unreachable")])
    monkeypatch.setattr("resumelab.cli.create_llm_client", lambda *_a, **_k: failing)

    result = invoke("analyze", "--jd", str(jd_file), "--debug")

    assert result.exit_code == 1
    assert isinstance(result.exception, JDAnalysisError)


def test_debug_raises_the_error_class_for_configuration_problems(monkeypatch, jd_file):
    from resumelab.exceptions import ConfigurationError

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = invoke("analyze", "--jd", str(jd_file), "--debug")

    assert isinstance(result.exception, ConfigurationError)


# --- logging --------------------------------------------------------------


def test_progress_is_logged_to_stderr_not_stdout(jd_file, fake_llm):
    """stdout carries the command's output, so the tool stays usable in a pipeline."""
    result = invoke("analyze", "--jd", str(jd_file))

    assert "analyzing job description" in result.stderr
    assert "analyzing job description" not in result.stdout


def test_debug_logging_names_the_stage(jd_file, fake_llm):
    result = invoke("analyze", "--jd", str(jd_file), "--debug")

    assert "resumelab.pipeline.jd_analyzer" in result.stderr


def test_logging_is_reconfigured_rather_than_duplicated(jd_file, fake_llm):
    """Two invocations in one process must not print every line twice."""
    invoke("analyze", "--jd", str(jd_file))
    handlers = len(logging.getLogger("resumelab").handlers)

    invoke("analyze", "--jd", str(jd_file))

    assert len(logging.getLogger("resumelab").handlers) == handlers


# --- the command surface --------------------------------------------------


def test_the_version_command_reports_the_installed_version():
    result = invoke("version")

    assert result.stdout.strip() == __version__


def test_bare_invocation_shows_help():
    result = invoke()

    assert "analyze" in result.stdout


def test_help_carries_the_research_disclaimer():
    result = invoke("--help")

    assert "not present in the source profile" in " ".join(result.stdout.split())


def test_the_analysis_stage_is_asked_for_the_documented_model(jd_file, fake_llm):
    invoke("analyze", "--jd", str(jd_file))

    assert fake_llm.last_call.response_model is JobAnalysis


def test_debug_re_raises_a_write_failure_too(tmp_path, jd_file, fake_llm):
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    blocked.chmod(0o500)
    try:
        result = invoke(
            "analyze", "--jd", str(jd_file), "-o", str(blocked / "x" / "a.json"), "--debug"
        )

        assert isinstance(result.exception, UnsafePathError)
    finally:
        blocked.chmod(0o700)


def test_an_output_path_that_names_a_directory_is_rejected(tmp_path, jd_file, fake_llm):
    """Writing over a directory is a mistake worth catching before the API call."""
    result = invoke("analyze", "--jd", str(jd_file), "-o", str(tmp_path))

    assert result.exit_code == 1
    assert "is a directory, not a file" in result.stderr


# --- unexpected failures --------------------------------------------------


def test_an_unexpected_error_is_not_a_wall_of_traceback(monkeypatch, jd_file):
    """A bug should still leave the user one readable line and a way to get more."""

    def explode(*_args, **_kwargs):
        raise ValueError("something the code did not anticipate")

    monkeypatch.setattr("resumelab.cli.create_llm_client", explode)

    result = invoke("analyze", "--jd", str(jd_file))

    assert result.exit_code == 1
    assert "ResumeLab failed unexpectedly: ValueError" in result.stderr
    assert "--debug" in result.stderr
    assert "Traceback" not in result.stderr


def test_debug_re_raises_an_unexpected_error(monkeypatch, jd_file):
    def explode(*_args, **_kwargs):
        raise ValueError("something the code did not anticipate")

    monkeypatch.setattr("resumelab.cli.create_llm_client", explode)

    result = invoke("analyze", "--jd", str(jd_file), "--debug")

    assert isinstance(result.exception, ValueError)


def test_an_interrupt_exits_cleanly(monkeypatch, jd_file):
    """Ctrl-C is a decision, not a crash."""

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("resumelab.cli.create_llm_client", interrupt)

    result = invoke("analyze", "--jd", str(jd_file))

    assert result.exit_code == 130
    assert "Interrupted." in result.stderr


def test_a_write_that_fails_after_validation_is_still_reported(
    monkeypatch, tmp_path, jd_file, fake_llm
):
    """The directory was writable when checked; the write can still fail."""

    def refuse(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("pathlib.Path.write_text", refuse)

    result = invoke("analyze", "--jd", str(jd_file), "-o", str(tmp_path / "out.json"))

    assert result.exit_code == 1
    assert "Could not write the analysis" in result.stderr
