"""Tests for per-run artifact directories and the metadata record."""

import json
import logging
from datetime import UTC, datetime, timedelta

import pytest

from resumelab import __version__
from resumelab.config import LLMProvider, Settings
from resumelab.experiment import build_metadata, create_run
from resumelab.experiment.recorder import (
    ANALYSIS_FILE,
    METADATA_FILE,
    PDF_FILE,
    RESUME_FILE,
    STRATEGY_FILE,
    ExperimentRun,
)
from resumelab.llm.client import LLMCallStats, TokenUsage
from resumelab.llm.prompts import JD_ANALYSIS_PROMPT_VERSION, TRANSFORMATION_PROMPT_VERSION
from resumelab.utils.text import slugify

OPENAI_KEY = "sk-test-not-a-real-key"
ANTHROPIC_KEY = "sk-ant-test-not-a-real-key"
STARTED_AT = datetime(2026, 8, 10, 15, 30, 0, tzinfo=UTC)


@pytest.fixture
def runs_dir(tmp_path):
    return tmp_path / "output" / "runs"


@pytest.fixture
def run(runs_dir):
    return create_run(runs_dir, label="crusoe", now=STARTED_AT)


# --- the run directory ----------------------------------------------------


def test_the_directory_is_named_from_the_time_and_label(run):
    assert run.directory.name == "2026-08-10T153000_crusoe"
    assert run.run_id == "2026-08-10T153000_crusoe"


def test_the_directory_is_created(run):
    assert run.directory.is_dir()


def test_parent_directories_are_created(runs_dir):
    run = create_run(runs_dir, label="x", now=STARTED_AT)

    assert run.directory.parent == runs_dir
    assert runs_dir.is_dir()


def test_run_directories_sort_chronologically(runs_dir):
    earlier = create_run(runs_dir, label="b", now=STARTED_AT)
    later = create_run(runs_dir, label="a", now=STARTED_AT + timedelta(hours=1))

    assert sorted([later.run_id, earlier.run_id])[0] == earlier.run_id


def test_two_runs_in_the_same_second_get_separate_directories(runs_dir):
    """Interleaving two runs' artifacts would silently corrupt both."""
    first = create_run(runs_dir, label="crusoe", now=STARTED_AT)
    second = create_run(runs_dir, label="crusoe", now=STARTED_AT)

    assert first.directory != second.directory
    assert second.directory.name == "2026-08-10T153000_crusoe-2"


def test_giving_up_on_a_unique_directory_fails_loudly(runs_dir, monkeypatch):
    """Better to stop than to write a second run's artifacts over a first one's."""
    monkeypatch.setattr("resumelab.experiment.recorder.MAX_COLLISION_ATTEMPTS", 1)
    create_run(runs_dir, label="crusoe", now=STARTED_AT)

    with pytest.raises(OSError, match="unique run directory"):
        create_run(runs_dir, label="crusoe", now=STARTED_AT)


def test_creating_a_run_is_logged(runs_dir, caplog):
    with caplog.at_level(logging.INFO, logger="resumelab.experiment.recorder"):
        create_run(runs_dir, label="crusoe", now=STARTED_AT)

    assert "recording run id=2026-08-10T153000_crusoe" in caplog.text


# --- labels cannot escape the output directory ----------------------------


@pytest.mark.parametrize(
    "label",
    ["../../etc", "/etc/passwd", "..", ".hidden", "a/b/c", "Crusoe Energy!", ""],
)
def test_a_hostile_label_stays_inside_the_runs_directory(runs_dir, label):
    run = create_run(runs_dir, label=label, now=STARTED_AT)

    assert run.directory.parent.resolve() == runs_dir.resolve()
    assert "/" not in run.run_id
    assert ".." not in run.run_id


def test_a_label_is_slugified_readably(runs_dir):
    run = create_run(runs_dir, label="Crusoe Energy — Storage!", now=STARTED_AT)

    assert run.directory.name == "2026-08-10T153000_crusoe-energy-storage"


def test_a_label_that_reduces_to_nothing_still_names_a_directory(runs_dir):
    assert create_run(runs_dir, label="!!!", now=STARTED_AT).directory.name.endswith("_run")


def test_a_long_label_is_truncated(runs_dir):
    run = create_run(runs_dir, label="x" * 200, now=STARTED_AT)

    assert len(slugify("x" * 200)) == 40
    assert run.directory.name.endswith("_" + "x" * 40)


# --- artifacts ------------------------------------------------------------


def test_the_job_description_is_recorded_as_analyzed(run, job_description):
    """The normalized text, not the file it came from."""
    path = run.record_job_description(job_description)

    assert path.name == "jd.txt"
    assert path.read_text(encoding="utf-8").strip() == job_description.text


@pytest.mark.parametrize(
    ("method", "filename"),
    [
        ("record_analysis", ANALYSIS_FILE),
        ("record_strategy", STRATEGY_FILE),
        ("record_resume", RESUME_FILE),
    ],
)
def test_each_structure_is_recorded_as_readable_json(
    run, job_analysis, transformation_strategy, generated_resume, method, filename
):
    payload = {
        "record_analysis": job_analysis,
        "record_strategy": transformation_strategy,
        "record_resume": generated_resume,
    }[method]

    path = getattr(run, method)(payload)

    assert path.name == filename
    assert json.loads(path.read_text(encoding="utf-8"))
    assert "\n  " in path.read_text(encoding="utf-8")


def test_the_recorded_analysis_round_trips(run, job_analysis):
    path = run.record_analysis(job_analysis)

    assert json.loads(path.read_text(encoding="utf-8"))["technical_identity"] == (
        job_analysis.technical_identity
    )


def test_the_recorded_resume_holds_the_generated_content(run, generated_resume):
    path = run.record_resume(generated_resume)

    recorded = json.loads(path.read_text(encoding="utf-8"))
    assert recorded["summary"] == generated_resume.summary
    assert len(recorded["projects"]) == 3


def test_artifacts_end_with_a_newline(run, job_analysis):
    """So the files behave in a terminal and in diffs."""
    assert run.record_analysis(job_analysis).read_text(encoding="utf-8").endswith("\n")


def test_artifacts_are_written_as_utf8(run, generated_resume):
    accented = generated_resume.model_copy(update={"summary": "José builds Go systems at scale."})

    path = run.record_resume(accented)

    assert "José" in path.read_text(encoding="utf-8")


def test_the_pdf_path_is_inside_the_run_directory(run):
    assert run.pdf_path == run.directory / PDF_FILE


def test_elapsed_time_is_measured_from_the_start(run):
    assert run.elapsed_seconds(now=STARTED_AT + timedelta(seconds=42)) == pytest.approx(42.0)


# --- metadata -------------------------------------------------------------


def openai_settings(tmp_path):
    profile = tmp_path / "candidate_profile.yaml"
    profile.write_text("personal: {}\n", encoding="utf-8")
    return Settings(
        _env_file=None,
        openai_api_key=OPENAI_KEY,
        candidate_profile_path=profile,
    )


def anthropic_settings(tmp_path):
    profile = tmp_path / "candidate_profile.yaml"
    profile.write_text("personal: {}\n", encoding="utf-8")
    return Settings(
        _env_file=None,
        anthropic_api_key=ANTHROPIC_KEY,
        candidate_profile_path=profile,
    )


def metadata_for(run, settings, provider, job_description, stats=None):
    return build_metadata(
        run,
        settings=settings,
        provider=provider,
        model=settings.model_for(provider),
        job_description=job_description,
        stats=stats or LLMCallStats(call_count=7, usage=TokenUsage(1000, 500, 1500)),
    )


def test_metadata_records_how_the_run_was_produced(tmp_path, run, job_description):
    metadata = metadata_for(run, openai_settings(tmp_path), LLMProvider.OPENAI, job_description)

    assert metadata.run_id == run.run_id
    assert metadata.timestamp == STARTED_AT
    assert metadata.provider == "openai"
    assert metadata.model == "gpt-4o"
    assert metadata.resumelab_version == __version__


def test_metadata_records_the_prompt_versions(tmp_path, run, job_description):
    metadata = metadata_for(run, openai_settings(tmp_path), LLMProvider.OPENAI, job_description)

    assert metadata.jd_analysis_prompt_version == JD_ANALYSIS_PROMPT_VERSION
    assert metadata.transformation_prompt_version == TRANSFORMATION_PROMPT_VERSION


def test_metadata_records_cost(tmp_path, run, job_description):
    metadata = metadata_for(run, openai_settings(tmp_path), LLMProvider.OPENAI, job_description)

    assert metadata.llm_calls == 7
    assert metadata.token_usage.total_tokens == 1500
    assert metadata.duration_seconds >= 0


def test_the_profile_is_hashed_so_runs_can_be_compared(tmp_path, run, job_description):
    settings = openai_settings(tmp_path)

    first = metadata_for(run, settings, LLMProvider.OPENAI, job_description)
    settings.candidate_profile_path.write_text("personal: {name: changed}\n", encoding="utf-8")
    second = metadata_for(run, settings, LLMProvider.OPENAI, job_description)

    assert len(first.candidate_profile_hash) == 64
    assert first.candidate_profile_hash != second.candidate_profile_hash


def test_the_same_profile_hashes_identically(tmp_path, run, job_description):
    settings = openai_settings(tmp_path)

    first = metadata_for(run, settings, LLMProvider.OPENAI, job_description)
    second = metadata_for(run, settings, LLMProvider.OPENAI, job_description)

    assert first.candidate_profile_hash == second.candidate_profile_hash


def test_openai_runs_record_temperature_and_no_effort(tmp_path, run, job_description):
    metadata = metadata_for(run, openai_settings(tmp_path), LLMProvider.OPENAI, job_description)

    assert metadata.temperature == pytest.approx(0.2)
    assert metadata.effort is None


def test_anthropic_runs_record_effort_and_no_temperature(tmp_path, run, job_description):
    """Current Claude models reject temperature, so recording one would be a fiction."""
    metadata = metadata_for(
        run, anthropic_settings(tmp_path), LLMProvider.ANTHROPIC, job_description
    )

    assert metadata.temperature is None
    assert metadata.effort == "high"
    assert metadata.model == "claude-opus-5"


def test_metadata_records_the_job_description_provenance(tmp_path, run, job_description):
    metadata = metadata_for(run, openai_settings(tmp_path), LLMProvider.OPENAI, job_description)

    assert metadata.job_description_source == "text"
    assert metadata.job_description_characters == job_description.character_count


# --- secrets never reach the artifacts ------------------------------------


def test_no_api_key_reaches_the_metadata_file(tmp_path, run, job_description):
    """A run directory is what a researcher shares when they share a result."""
    settings = Settings(
        _env_file=None,
        openai_api_key=OPENAI_KEY,
        anthropic_api_key=ANTHROPIC_KEY,
        candidate_profile_path=openai_settings(tmp_path).candidate_profile_path,
    )

    path = run.record_metadata(metadata_for(run, settings, LLMProvider.OPENAI, job_description))

    written = path.read_text(encoding="utf-8")
    assert OPENAI_KEY not in written
    assert ANTHROPIC_KEY not in written
    # Checked against the field names rather than the raw text, which also holds
    # filesystem paths that are outside our control.
    assert [field for field in json.loads(written) if "key" in field.lower()] == []


def test_the_metadata_file_is_valid_json_with_the_expected_shape(tmp_path, run, job_description):
    path = run.record_metadata(
        metadata_for(run, openai_settings(tmp_path), LLMProvider.OPENAI, job_description)
    )

    recorded = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == METADATA_FILE
    assert recorded["run_id"] == run.run_id
    assert recorded["candidate_profile_hash"]
    assert recorded["token_usage"]["total_tokens"] == 1500


def test_building_metadata_is_logged(tmp_path, run, job_description, caplog):
    with caplog.at_level(logging.INFO, logger="resumelab.experiment.metadata_builder"):
        metadata_for(run, openai_settings(tmp_path), LLMProvider.OPENAI, job_description)

    assert "run metadata provider=openai" in caplog.text


# --- the run object -------------------------------------------------------


def test_a_run_can_be_constructed_directly(tmp_path):
    """Used by tests and by anything replaying an existing directory."""
    run = ExperimentRun(directory=tmp_path, run_id="fixed", started_at=STARTED_AT)

    assert run.run_id == "fixed"
    assert run.pdf_path.parent == tmp_path
