"""Tests for combining the generated sections into a validated resume."""

import logging

import pytest

from resumelab.exceptions import ResumeValidationError
from resumelab.pipeline import assemble_resume


@pytest.fixture
def parts(generated_resume):
    """The pieces the assembler is given, taken from a known-good resume."""
    return {
        "summary": generated_resume.summary,
        "experiences": generated_resume.experiences,
        "projects": generated_resume.projects,
        "skills": generated_resume.skills,
    }


# --- assembly -------------------------------------------------------------


def test_the_generated_sections_are_used(candidate_profile, parts):
    resume = assemble_resume(candidate_profile, **parts)

    assert resume.summary == parts["summary"]
    assert resume.experiences == parts["experiences"]
    assert resume.projects == parts["projects"]
    assert resume.skills == parts["skills"]


def test_identity_reaches_the_resume_here(candidate_profile, parts):
    """Personal details are withheld from every prompt and arrive at assembly."""
    resume = assemble_resume(candidate_profile, **parts)

    assert resume.personal == candidate_profile.personal
    assert resume.personal.email == "ada@example.edu"


def test_untransformed_sections_are_carried_through_verbatim(candidate_profile, parts):
    resume = assemble_resume(candidate_profile, **parts)

    assert resume.education == candidate_profile.education
    assert resume.achievements == candidate_profile.achievements


def test_sequences_are_frozen_into_tuples(candidate_profile, parts):
    """The resume is recorded as a run artifact and must not be mutable afterwards."""
    resume = assemble_resume(candidate_profile, **{**parts, "skills": list(parts["skills"])})

    assert isinstance(resume.skills, tuple)
    assert isinstance(resume.experiences, tuple)
    assert isinstance(resume.projects, tuple)


def test_the_assembled_resume_is_frozen(candidate_profile, parts):
    resume = assemble_resume(candidate_profile, **parts)

    with pytest.raises(Exception, match="frozen"):
        resume.summary = "something else"


def test_assembly_does_not_call_a_model(candidate_profile, parts):
    """Assembly is deterministic; it takes no client at all."""
    resume = assemble_resume(candidate_profile, **parts)

    assert resume.summary == parts["summary"]


# --- validation is not optional -------------------------------------------


def test_an_invalid_resume_cannot_be_assembled(candidate_profile, parts):
    with pytest.raises(ResumeValidationError, match="summary is empty"):
        assemble_resume(candidate_profile, **{**parts, "summary": "   "})


def test_the_project_count_is_enforced_at_assembly(candidate_profile, parts):
    with pytest.raises(ResumeValidationError, match="exactly 3 projects"):
        assemble_resume(candidate_profile, **{**parts, "projects": parts["projects"][:2]})


# --- logging --------------------------------------------------------------


def test_assembly_is_logged(candidate_profile, parts, caplog):
    with caplog.at_level(logging.INFO, logger="resumelab.pipeline.assembler"):
        assemble_resume(candidate_profile, **parts)

    assert "assembling resume" in caplog.text
    assert "experiences=1 projects=3 skills=10" in caplog.text
