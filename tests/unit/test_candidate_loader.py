"""Tests for reading the source candidate profile from disk."""

import hashlib
import logging
from pathlib import Path

import pytest
import yaml

from resumelab.exceptions import CandidateProfileError, ResumeLabError
from resumelab.loaders import load_candidate_profile
from resumelab.models import CandidateProfile


@pytest.fixture
def profile_path(tmp_path, profile_data):
    """A valid profile written to a temporary YAML file."""
    path = tmp_path / "candidate_profile.yaml"
    path.write_text(yaml.safe_dump(profile_data, sort_keys=False), encoding="utf-8")
    return path


def write(tmp_path, content, name="candidate_profile.yaml"):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


# --- valid profile --------------------------------------------------------


def test_a_valid_profile_is_loaded(profile_path):
    profile = load_candidate_profile(profile_path)

    assert isinstance(profile, CandidateProfile)
    assert profile.personal.name == "Ada Lovelace"
    assert len(profile.projects) == 3
    assert profile.experiences[0].company == "Analytical Engines Inc."


def test_loading_does_not_modify_the_source_file(profile_path):
    """The profile is the experimental control and must survive every run untouched."""
    before = hashlib.sha256(profile_path.read_bytes()).hexdigest()

    load_candidate_profile(profile_path)

    assert hashlib.sha256(profile_path.read_bytes()).hexdigest() == before


def test_non_ascii_content_round_trips(tmp_path, profile_data):
    profile_data["personal"]["name"] = "José Ramírez"
    path = tmp_path / "profile.yaml"
    path.write_text(
        yaml.safe_dump(profile_data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    assert load_candidate_profile(path).personal.name == "José Ramírez"


def test_loading_is_logged(profile_path, caplog):
    with caplog.at_level(logging.INFO, logger="resumelab.loaders.candidate_loader"):
        load_candidate_profile(profile_path)

    assert "loading candidate profile" in caplog.text


# --- missing / unreadable files -------------------------------------------


def test_a_missing_file_reports_the_path_and_how_to_create_it(tmp_path):
    missing = tmp_path / "absent.yaml"

    with pytest.raises(CandidateProfileError) as exc_info:
        load_candidate_profile(missing)

    message = str(exc_info.value)
    assert str(missing) in message
    assert "candidate_profile.example.yaml" in message


def test_a_directory_is_rejected(tmp_path):
    with pytest.raises(CandidateProfileError, match="directory"):
        load_candidate_profile(tmp_path)


def test_an_unreadable_file_is_rejected(profile_path):
    profile_path.chmod(0o000)
    try:
        with pytest.raises(CandidateProfileError, match="not readable"):
            load_candidate_profile(profile_path)
    finally:
        profile_path.chmod(0o644)


def test_unexpected_filesystem_errors_are_wrapped(monkeypatch, profile_path):
    """An unforeseen OSError must still surface as a domain error, not a raw traceback."""

    def raise_io_error(*_args, **_kwargs):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(Path, "read_text", raise_io_error)

    with pytest.raises(CandidateProfileError, match="Could not read"):
        load_candidate_profile(profile_path)


def test_non_utf8_bytes_are_rejected(tmp_path):
    path = tmp_path / "latin1.yaml"
    path.write_bytes("personal:\n  name: José\n".encode("latin-1"))

    with pytest.raises(CandidateProfileError, match="UTF-8"):
        load_candidate_profile(path)


# --- malformed documents --------------------------------------------------


def test_malformed_yaml_is_rejected(tmp_path):
    path = write(tmp_path, "personal:\n  name: 'unterminated\nexperiences: [\n")

    with pytest.raises(CandidateProfileError, match="not valid YAML"):
        load_candidate_profile(path)


def test_an_empty_file_is_rejected(tmp_path):
    with pytest.raises(CandidateProfileError, match="empty"):
        load_candidate_profile(write(tmp_path, ""))


def test_a_comment_only_file_is_rejected(tmp_path):
    with pytest.raises(CandidateProfileError, match="empty"):
        load_candidate_profile(write(tmp_path, "# nothing here yet\n"))


@pytest.mark.parametrize(
    ("content", "described_as"),
    [("- one\n- two\n", "list"), ("just a string\n", "str"), ("42\n", "int")],
)
def test_a_document_that_is_not_a_mapping_is_rejected(tmp_path, content, described_as):
    with pytest.raises(CandidateProfileError) as exc_info:
        load_candidate_profile(write(tmp_path, content))

    assert described_as in str(exc_info.value)


def test_yaml_tags_cannot_construct_python_objects(tmp_path):
    """The profile is untrusted data; safe_load must refuse object construction."""
    path = write(tmp_path, "personal: !!python/object/apply:os.system ['echo pwned']\n")

    with pytest.raises(CandidateProfileError, match="not valid YAML"):
        load_candidate_profile(path)


# --- schema failures ------------------------------------------------------


def test_schema_errors_name_every_offending_location(tmp_path, profile_data):
    del profile_data["personal"]["name"]
    profile_data["projects"] = profile_data["projects"][:2]
    path = write(tmp_path, yaml.safe_dump(profile_data, sort_keys=False))

    with pytest.raises(CandidateProfileError) as exc_info:
        load_candidate_profile(path)

    message = str(exc_info.value)
    assert "personal.name" in message
    assert "projects" in message
    assert str(path) in message


def test_schema_errors_use_dotted_paths_into_nested_collections(tmp_path, profile_data):
    profile_data["projects"][1]["bullets"] = ["only one"]
    path = write(tmp_path, yaml.safe_dump(profile_data, sort_keys=False))

    with pytest.raises(CandidateProfileError, match=r"projects\.1\.bullets"):
        load_candidate_profile(path)


def test_schema_errors_do_not_echo_the_rejected_value(tmp_path, profile_data):
    profile_data["personal"]["email"] = ""
    path = write(tmp_path, yaml.safe_dump(profile_data, sort_keys=False))

    with pytest.raises(CandidateProfileError) as exc_info:
        load_candidate_profile(path)

    assert "personal.email" in str(exc_info.value)


def test_an_unknown_section_is_rejected(tmp_path, profile_data):
    profile_data["hobbies"] = ["chess"]
    path = write(tmp_path, yaml.safe_dump(profile_data, sort_keys=False))

    with pytest.raises(CandidateProfileError, match="hobbies"):
        load_candidate_profile(path)


def test_the_unpopulated_template_reports_useful_errors(profile_template_path):
    with pytest.raises(CandidateProfileError) as exc_info:
        load_candidate_profile(profile_template_path)

    assert "personal.name" in str(exc_info.value)


# --- error type -----------------------------------------------------------


def test_loader_errors_are_resumelab_errors(tmp_path):
    with pytest.raises(ResumeLabError):
        load_candidate_profile(tmp_path / "absent.yaml")
