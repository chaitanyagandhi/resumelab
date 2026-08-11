"""End-to-end pipeline runs against a deterministic fake LLM.

These exercise the couplings unit tests cannot: that the strategy names entries the
transformers can find, that what one stage writes is usable by the next, and that a
run leaves behind a complete, readable set of artifacts.

No network call is made. The one test that would make one is marked ``e2e`` and is
deselected by default.
"""

from __future__ import annotations

import json
import os

import pytest
from pypdf import PdfReader

from resumelab.config import LLMProvider
from resumelab.experiment.recorder import (
    ANALYSIS_FILE,
    METADATA_FILE,
    PDF_FILE,
    RESUME_FILE,
    STRATEGY_FILE,
)
from resumelab.loaders import load_job_description
from resumelab.models.analysis import JobAnalysis
from resumelab.models.metadata import RunMetadata
from resumelab.models.resume import GeneratedResume
from resumelab.models.strategy import TransformationStrategy
from resumelab.pipeline import generate_resume


def run_pipeline(jd_path, settings, client):
    return generate_resume(
        load_job_description(path=jd_path),
        settings=settings,
        provider=LLMProvider.OPENAI,
        client=client,
    )


@pytest.fixture
def result(storage_jd, settings, fake_llm):
    return run_pipeline(storage_jd, settings, fake_llm)


# --- the run completes ----------------------------------------------------


def test_the_pipeline_completes(result):
    assert result.resume.summary
    assert result.render.path.exists()


def test_every_stage_was_exercised(result, fake_llm):
    assert [purpose for purpose, _ in fake_llm.calls] == [
        "jd_analysis",
        "transformation_strategy",
        "summary",
        "experience",
        "project",
        "project",
        "project",
        "skills",
    ]


def test_the_resume_has_the_shape_the_research_design_requires(result):
    assert len(result.resume.projects) == 3
    for project in result.resume.projects:
        assert len(project.bullets) == 3
    assert len(result.resume.experiences) == 1
    assert result.resume.skills


# --- the artifacts --------------------------------------------------------


def test_every_artifact_is_written(result):
    written = {path.name for path in result.run.directory.iterdir()}

    assert written == {
        "jd.txt",
        ANALYSIS_FILE,
        STRATEGY_FILE,
        RESUME_FILE,
        METADATA_FILE,
        PDF_FILE,
    }


@pytest.mark.parametrize(
    ("filename", "model"),
    [
        (ANALYSIS_FILE, JobAnalysis),
        (STRATEGY_FILE, TransformationStrategy),
        (RESUME_FILE, GeneratedResume),
        (METADATA_FILE, RunMetadata),
    ],
)
def test_each_json_artifact_reloads_into_its_model(result, filename, model):
    """An artifact that cannot be read back is not a research record."""
    payload = json.loads((result.run.directory / filename).read_text(encoding="utf-8"))

    assert model.model_validate(payload)


def test_the_recorded_job_description_is_what_was_analyzed(result, storage_jd):
    recorded = (result.run.directory / "jd.txt").read_text(encoding="utf-8").strip()

    assert recorded == storage_jd.read_text(encoding="utf-8").strip()


def test_the_pdf_is_a_pdf(result):
    assert result.render.path.read_bytes().startswith(b"%PDF-")


def test_the_pdf_holds_the_generated_resume(result):
    text = " ".join(
        " ".join(page.extract_text().split()) for page in PdfReader(result.render.path).pages
    )

    assert result.resume.summary in text
    for bullet in result.resume.all_bullets:
        assert bullet in text


def test_the_pdf_fits_one_page(result):
    assert result.render.fits_on_one_page


def test_metadata_records_the_run(result):
    metadata = result.metadata

    assert metadata.run_id == result.run.run_id
    assert metadata.llm_calls == 8
    assert metadata.token_usage.total_tokens == 8 * 180
    assert metadata.candidate_profile_hash
    assert metadata.page_count == 1


# --- the stages actually agree with each other ----------------------------


def test_the_plan_covers_every_entry_it_has_to(result, fake_llm, candidate_profile):
    """The coupling a queue-based fake cannot exercise."""
    strategy = TransformationStrategy.model_validate(
        json.loads((result.run.directory / STRATEGY_FILE).read_text(encoding="utf-8"))
    )

    for experience in candidate_profile.experiences:
        assert strategy.direction_for_experience(experience.company) is not None
    for project in candidate_profile.projects:
        assert strategy.direction_for_project(project.name) is not None


def test_each_project_keeps_its_source_name(result, candidate_profile):
    """The anchor that lets a generated project be compared to its source."""
    assert [project.name for project in result.resume.projects] == [
        project.name for project in candidate_profile.projects
    ]


def test_later_stages_saw_what_earlier_ones_wrote(result, fake_llm):
    prompts = dict(fake_llm.calls[::-1])

    assert "BULLETS ALREADY WRITTEN" in prompts["skills"]
    assert "BULLETS ALREADY WRITTEN" in prompts["project"]


# --- the transformation under study ---------------------------------------


def test_the_resume_is_not_the_source_profile(result, candidate_profile):
    """If the output matched the input, there would be nothing to study."""
    source_bullets = {
        bullet for experience in candidate_profile.experiences for bullet in experience.bullets
    } | {bullet for project in candidate_profile.projects for bullet in project.bullets}

    assert not source_bullets & set(result.resume.all_bullets)


def test_project_framing_is_replaced(result, candidate_profile):
    for generated, source in zip(result.resume.projects, candidate_profile.projects, strict=True):
        assert generated.subtitle != source.subtitle
        assert generated.technologies != source.technologies


def test_two_different_postings_produce_two_different_resumes(
    storage_jd, genai_jd, settings, fake_llm
):
    """The research claim: the posting reshapes the candidate, not just the wording."""
    storage = run_pipeline(storage_jd, settings, fake_llm)
    genai = run_pipeline(genai_jd, settings, type(fake_llm)())

    assert storage.resume.summary != genai.resume.summary
    assert storage.resume.projects[0].subtitle != genai.resume.projects[0].subtitle
    assert storage.resume.all_bullets != genai.resume.all_bullets


def test_each_posting_pulls_in_its_own_vocabulary(storage_jd, genai_jd, settings, fake_llm):
    storage = run_pipeline(storage_jd, settings, fake_llm)
    genai = run_pipeline(genai_jd, settings, type(fake_llm)())

    storage_text = json.dumps(storage.resume.model_dump())
    genai_text = json.dumps(genai.resume.model_dump())
    assert "NVMe" in storage_text and "NVMe" not in genai_text
    assert "GenAI" in genai_text and "GenAI" not in storage_text


def test_the_source_profile_is_never_modified(result, profile_path, profile_data):
    import yaml

    assert yaml.safe_load(profile_path.read_text(encoding="utf-8")) == profile_data


# --- reproducibility ------------------------------------------------------


def test_the_same_inputs_produce_the_same_resume(storage_jd, settings, fake_llm):
    """Without this, no two runs of an experiment could be compared."""
    first = run_pipeline(storage_jd, settings, fake_llm)
    second = run_pipeline(storage_jd, settings, type(fake_llm)())

    assert first.resume.model_dump() == second.resume.model_dump()


def test_each_run_gets_its_own_directory(storage_jd, settings, fake_llm):
    first = run_pipeline(storage_jd, settings, fake_llm)
    second = run_pipeline(storage_jd, settings, type(fake_llm)())

    assert first.run.directory != second.run.directory


def test_both_runs_record_the_same_profile_hash(storage_jd, settings, fake_llm):
    """How two runs are shown to have shared an input."""
    first = run_pipeline(storage_jd, settings, fake_llm)
    second = run_pipeline(storage_jd, settings, type(fake_llm)())

    assert first.metadata.candidate_profile_hash == second.metadata.candidate_profile_hash


# --- against the real provider, on request only ---------------------------


@pytest.mark.e2e
def test_a_real_provider_produces_a_resume(storage_jd, tmp_path, profile_path):
    """Deselected by default. Run with: pytest -m e2e

    Requires a real API key and spends real money, so it is never part of an
    ordinary test run.
    """
    from resumelab.config import load_settings
    from resumelab.llm import create_llm_client

    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")):
        pytest.skip("no provider credentials configured")

    live_settings = load_settings().model_copy(
        update={"candidate_profile_path": profile_path, "output_dir": tmp_path / "output"}
    )
    client = create_llm_client(live_settings)

    outcome = generate_resume(
        load_job_description(path=storage_jd),
        settings=live_settings,
        provider=live_settings.resolved_provider,
        client=client,
    )

    assert outcome.render.path.read_bytes().startswith(b"%PDF-")
    assert outcome.metadata.llm_calls >= 8
