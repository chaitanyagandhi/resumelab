"""Tests for resolving a job description from a file or inline text."""

import logging
from pathlib import Path

import pytest

from resumelab.exceptions import JDAnalysisError, ResumeLabError
from resumelab.loaders import load_job_description
from resumelab.models import JobDescriptionSource

JD_TEXT = (
    "Storage Infrastructure Engineer. Build distributed storage services in Go "
    "and Java on Linux, working with NVMe devices and network storage protocols."
)


@pytest.fixture
def jd_file(tmp_path):
    path = tmp_path / "job.txt"
    path.write_text(JD_TEXT, encoding="utf-8")
    return path


# --- input selection ------------------------------------------------------


def test_a_job_description_is_loaded_from_a_file(jd_file):
    jd = load_job_description(path=jd_file)

    assert jd.text == JD_TEXT
    assert jd.source is JobDescriptionSource.FILE
    assert jd.source_path == jd_file


def test_a_job_description_is_loaded_from_inline_text():
    jd = load_job_description(text=JD_TEXT)

    assert jd.text == JD_TEXT
    assert jd.source is JobDescriptionSource.TEXT
    assert jd.source_path is None


def test_supplying_both_inputs_is_rejected(jd_file):
    with pytest.raises(JDAnalysisError, match="not both"):
        load_job_description(path=jd_file, text=JD_TEXT)


def test_supplying_neither_input_is_rejected():
    with pytest.raises(JDAnalysisError, match="required"):
        load_job_description()


def test_loader_errors_are_resumelab_errors():
    with pytest.raises(ResumeLabError):
        load_job_description()


# --- file failures --------------------------------------------------------


def test_a_missing_file_names_the_path(tmp_path):
    missing = tmp_path / "absent.txt"

    with pytest.raises(JDAnalysisError) as exc_info:
        load_job_description(path=missing)

    assert str(missing) in str(exc_info.value)


def test_a_directory_is_rejected(tmp_path):
    with pytest.raises(JDAnalysisError, match="directory"):
        load_job_description(path=tmp_path)


def test_an_unreadable_file_is_rejected(jd_file):
    jd_file.chmod(0o000)
    try:
        with pytest.raises(JDAnalysisError, match="not readable"):
            load_job_description(path=jd_file)
    finally:
        jd_file.chmod(0o644)


def test_non_utf8_bytes_are_rejected(tmp_path):
    path = tmp_path / "latin1.txt"
    path.write_bytes(JD_TEXT.encode("utf-8") + "café".encode("latin-1"))

    with pytest.raises(JDAnalysisError, match="UTF-8"):
        load_job_description(path=path)


def test_unexpected_filesystem_errors_are_wrapped(monkeypatch, jd_file):
    def raise_io_error(*_args, **_kwargs):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(Path, "read_text", raise_io_error)

    with pytest.raises(JDAnalysisError, match="Could not read"):
        load_job_description(path=jd_file)


# --- content validation ---------------------------------------------------


def test_an_empty_file_names_the_path(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")

    with pytest.raises(JDAnalysisError) as exc_info:
        load_job_description(path=path)

    assert "empty" in str(exc_info.value)
    assert str(path) in str(exc_info.value)


def test_whitespace_only_inline_text_is_rejected():
    with pytest.raises(JDAnalysisError, match="empty"):
        load_job_description(text="   \n\t ")


def test_text_that_is_too_short_reports_the_reason():
    with pytest.raises(JDAnalysisError) as exc_info:
        load_job_description(text="Backend engineer.")

    message = str(exc_info.value)
    assert "Invalid job description" in message
    assert "inline text" in message
    assert "at least" in message


def test_a_short_file_names_the_path(tmp_path):
    path = tmp_path / "short.txt"
    path.write_text("Backend engineer.", encoding="utf-8")

    with pytest.raises(JDAnalysisError) as exc_info:
        load_job_description(path=path)

    assert str(path) in str(exc_info.value)


def test_file_contents_are_normalized(tmp_path):
    path = tmp_path / "crlf.txt"
    path.write_bytes(f"  {JD_TEXT}  ".replace(" ", "\r\n", 1).encode("utf-8"))

    jd = load_job_description(path=path)

    assert "\r" not in jd.text
    assert jd.text.endswith("protocols.")


# --- logging and the shipped example --------------------------------------


def test_loading_logs_counts_rather_than_content(jd_file, caplog):
    with caplog.at_level(logging.INFO, logger="resumelab.loaders.jd_loader"):
        load_job_description(path=jd_file)

    assert "loaded job description" in caplog.text
    assert "characters=" in caplog.text
    assert JD_TEXT not in caplog.text


def test_the_shipped_example_job_description_loads(repo_root):
    jd = load_job_description(path=repo_root / "examples" / "sample_jd.txt")

    assert jd.source is JobDescriptionSource.FILE
    assert "NVMe" in jd.text
