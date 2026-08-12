"""Tests for the deterministic checks run before a resume is rendered."""

import logging

import pytest

from resumelab.exceptions import ResumeLabError, ResumeValidationError
from resumelab.models.candidate import PersonalDetails
from resumelab.models.resume import (
    MAX_BULLET_CHARACTERS,
    MIN_BULLET_CHARACTERS,
    ResumeLimits,
)
from resumelab.validation import validate_resume


def replace(resume, **updates):
    return resume.model_copy(update=updates)


def failures(resume) -> str:
    with pytest.raises(ResumeValidationError) as exc_info:
        validate_resume(resume)
    return str(exc_info.value)


# --- the happy path -------------------------------------------------------


def test_a_complete_resume_passes(generated_resume):
    validate_resume(generated_resume)


def test_validation_is_logged(generated_resume, caplog):
    with caplog.at_level(logging.INFO, logger="resumelab.validation.resume_validator"):
        validate_resume(generated_resume)

    assert "validating generated resume" in caplog.text


def test_errors_are_resumelab_errors(generated_resume):
    with pytest.raises(ResumeLabError):
        validate_resume(replace(generated_resume, summary="  "))


# --- identity and contact -------------------------------------------------


def test_a_resume_without_a_name_is_rejected(generated_resume):
    anonymous = generated_resume.personal.model_copy(update={"name": "  "})

    assert "no name" in failures(replace(generated_resume, personal=anonymous))


def test_a_resume_with_no_way_to_reach_the_candidate_is_rejected(generated_resume):
    unreachable = PersonalDetails.model_construct(
        name="Ada Lovelace", email=None, phone=None, linkedin=None, github=None, location=None
    )

    assert "no contact information" in failures(replace(generated_resume, personal=unreachable))


def test_a_phone_alone_is_enough_contact(generated_resume):
    by_phone = PersonalDetails.model_construct(
        name="Ada Lovelace",
        email=None,
        phone="+1 555 0100",
        linkedin=None,
        github=None,
        location=None,
    )

    validate_resume(replace(generated_resume, personal=by_phone))


# --- required sections ----------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("summary", "   ", "summary is empty"),
        ("education", (), "no education entries"),
        ("experiences", (), "no experience entries"),
        ("skills", (), "no skills"),
    ],
)
def test_missing_sections_are_rejected(generated_resume, field, value, expected):
    assert expected in failures(replace(generated_resume, **{field: value}))


# --- the research design --------------------------------------------------


@pytest.mark.parametrize("count", [0, 1, 2])
def test_too_few_projects_are_rejected(generated_resume, count):
    """Project counts are the research design; a drift here invalidates the run."""
    resume = replace(generated_resume, projects=generated_resume.projects[:count])

    assert "expected exactly 3 projects" in failures(resume)


def test_too_many_projects_are_rejected(generated_resume):
    extra = generated_resume.projects[0].model_copy(update={"name": "Project 4"})

    resume = replace(generated_resume, projects=(*generated_resume.projects, extra))

    assert "expected exactly 3 projects, found 4" in failures(resume)


@pytest.mark.parametrize("count", [0, 1, 2])
def test_a_project_missing_bullets_is_rejected(generated_resume, count):
    first, *rest = generated_resume.projects
    short = first.model_copy(update={"bullets": first.bullets[:count]})

    message = failures(replace(generated_resume, projects=(short, *rest)))
    assert f"has {count} bullets, expected exactly 3" in message


def test_a_project_without_a_subtitle_is_rejected(generated_resume):
    first, *rest = generated_resume.projects
    bare = first.model_copy(update={"subtitle": "  "})

    assert "has no subtitle" in failures(replace(generated_resume, projects=(bare, *rest)))


def test_a_project_naming_no_technologies_is_rejected(generated_resume):
    first, *rest = generated_resume.projects
    bare = first.model_copy(update={"technologies": ()})

    assert "names no technologies" in failures(replace(generated_resume, projects=(bare, *rest)))


# --- bullets --------------------------------------------------------------


def test_an_experience_without_bullets_is_rejected(generated_resume):
    empty = generated_resume.experiences[0].model_copy(update={"bullets": ()})

    assert "has no bullets" in failures(replace(generated_resume, experiences=(empty,)))


@pytest.mark.parametrize("blank", ["", "   "])
def test_an_empty_bullet_is_rejected(generated_resume, blank):
    experience = generated_resume.experiences[0]
    broken = experience.model_copy(update={"bullets": (blank, *experience.bullets[1:])})

    assert "bullet 1 is empty" in failures(replace(generated_resume, experiences=(broken,)))


@pytest.mark.parametrize("length", [MIN_BULLET_CHARACTERS - 1, MAX_BULLET_CHARACTERS + 1])
def test_an_unreasonable_bullet_length_is_rejected(generated_resume, length):
    experience = generated_resume.experiences[0]
    broken = experience.model_copy(update={"bullets": ("x" * length, *experience.bullets[1:])})

    assert "expected between 40 and 220" in failures(
        replace(generated_resume, experiences=(broken,))
    )


def test_project_bullets_are_checked_too(generated_resume):
    first, *rest = generated_resume.projects
    broken = first.model_copy(update={"bullets": ("  ", *first.bullets[1:])})

    assert "bullet 1 is empty" in failures(replace(generated_resume, projects=(broken, *rest)))


# --- text hygiene ---------------------------------------------------------


@pytest.mark.parametrize("character", ["\x00", "\x07", "\x1b", "\n", "\t"])
def test_control_characters_in_a_bullet_are_rejected(generated_resume, character):
    """They survive into the PDF, break text extraction, and are invisible."""
    experience = generated_resume.experiences[0]
    tainted = experience.bullets[0].replace(" ", character, 1)
    broken = experience.model_copy(update={"bullets": (tainted, *experience.bullets[1:])})

    assert "control characters" in failures(replace(generated_resume, experiences=(broken,)))


def test_control_characters_in_the_summary_are_rejected(generated_resume):
    tainted = generated_resume.summary.replace(" ", "\x00", 1)

    assert "the summary contains control characters" in failures(
        replace(generated_resume, summary=tainted)
    )


def test_control_characters_in_a_skill_are_rejected(generated_resume):
    tainted = ("Go\x00", *generated_resume.skills[1:])

    assert "control characters" in failures(replace(generated_resume, skills=tainted))


def test_control_characters_in_a_project_subtitle_are_rejected(generated_resume):
    first, *rest = generated_resume.projects
    tainted = first.model_copy(update={"subtitle": "Storage\x1bEngine"})

    assert "subtitle contains control characters" in failures(
        replace(generated_resume, projects=(tainted, *rest))
    )


def test_ordinary_unicode_is_not_flagged(generated_resume):
    """Accents, dashes, and symbols are normal resume content."""
    validate_resume(replace(generated_resume, summary="José — builds ≥99.9% durable Go systems."))


# --- reporting ------------------------------------------------------------


def test_every_problem_is_reported_at_once(generated_resume):
    """Fixing one failure only to be shown the next is a poor way to debug a run."""
    message = failures(replace(generated_resume, summary="  ", education=(), projects=()))

    assert "summary is empty" in message
    assert "no education entries" in message
    assert "expected exactly 3 projects" in message


def test_the_message_names_the_offending_entry(generated_resume):
    first, *rest = generated_resume.projects
    short = first.model_copy(update={"bullets": first.bullets[:1]})

    assert "'Project 1'" in failures(replace(generated_resume, projects=(short, *rest)))


def test_all_bullets_collects_the_whole_document(generated_resume):
    assert len(generated_resume.all_bullets) == 3 + 3 * 3


# --- the configurable length budget ---------------------------------------


def test_the_default_budget_matches_what_generation_already_enforces(generated_resume):
    """Out of the box, nothing the stages produced can fail this check."""
    validate_resume(generated_resume, ResumeLimits())


def test_a_tighter_summary_budget_is_enforced(generated_resume):
    limits = ResumeLimits(summary_max_characters=40)

    with pytest.raises(ResumeValidationError, match="over the 40 allowed"):
        validate_resume(generated_resume, limits)


def test_a_tighter_bullet_budget_is_enforced(generated_resume):
    limits = ResumeLimits(bullet_max_characters=60)

    with pytest.raises(ResumeValidationError, match="expected between 40 and 60"):
        validate_resume(generated_resume, limits)


def test_a_configured_experience_bullet_count_is_enforced(generated_resume):
    """Configuration that cannot be honoured fails loudly rather than being ignored."""
    limits = ResumeLimits(experience_bullet_count=4)

    with pytest.raises(ResumeValidationError, match="expected exactly 4"):
        validate_resume(generated_resume, limits)


def test_a_configured_project_bullet_count_is_enforced(generated_resume):
    limits = ResumeLimits(project_bullet_count=2)

    with pytest.raises(ResumeValidationError, match="expected exactly 2"):
        validate_resume(generated_resume, limits)


def test_settings_supply_the_budget():
    from resumelab.config import Settings

    settings = Settings(
        _env_file=None,
        openai_api_key="sk-test-not-a-real-key",
        summary_max_characters=200,
        bullet_max_characters=150,
    )

    assert settings.resume_limits == ResumeLimits(
        summary_max_characters=200, bullet_max_characters=150
    )
