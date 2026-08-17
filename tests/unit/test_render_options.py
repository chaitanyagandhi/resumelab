"""Tests for the presentation options a rendered resume is drawn under.

These arrive from outside the pipeline, so the tests are mostly about what is
rejected. An order that drew a section twice, or quietly dropped one, would produce a
plausible-looking PDF that is wrong in a way nobody would notice until they read it.
"""

import pytest
from pydantic import ValidationError

from resumelab.rendering import (
    DEFAULT_RENDER_OPTIONS,
    DEFAULT_SECTION_ORDER,
    RenderOptions,
    ResumeSection,
)

# --- defaults -------------------------------------------------------------


def test_everything_is_shown_by_default():
    options = RenderOptions()

    assert options.include_summary is True
    assert options.include_gpa is True
    assert options.section_order == DEFAULT_SECTION_ORDER


def test_the_shared_default_matches_a_freshly_built_one():
    assert RenderOptions() == DEFAULT_RENDER_OPTIONS


def test_the_default_order_names_every_section():
    assert set(DEFAULT_SECTION_ORDER) == set(ResumeSection)


def test_options_are_frozen_so_the_shared_default_cannot_be_edited():
    with pytest.raises(ValidationError):
        DEFAULT_RENDER_OPTIONS.include_summary = False


def test_an_unknown_option_is_rejected_rather_than_ignored():
    """A misspelled toggle must fail loudly, not silently leave the default in place."""
    with pytest.raises(ValidationError):
        RenderOptions(include_summry=False)


# --- the section order ----------------------------------------------------


def test_a_reordering_is_accepted():
    order = (
        ResumeSection.EXPERIENCE,
        ResumeSection.PROJECTS,
        ResumeSection.SKILLS,
        ResumeSection.EDUCATION,
    )

    assert RenderOptions(section_order=order).section_order == order


def test_sections_may_be_named_as_plain_strings():
    """The order crosses a JSON boundary, so it has to survive arriving as strings."""
    options = RenderOptions(section_order=("skills", "projects", "experience", "education"))

    assert options.section_order[0] is ResumeSection.SKILLS


def test_naming_a_section_twice_is_rejected():
    duplicated = (
        ResumeSection.SKILLS,
        ResumeSection.SKILLS,
        ResumeSection.EDUCATION,
        ResumeSection.EXPERIENCE,
    )

    with pytest.raises(ValidationError, match="more than once"):
        RenderOptions(section_order=duplicated)


def test_omitting_a_section_is_rejected():
    """Sections are reordered, never hidden; a short order is a mistake, not a request."""
    with pytest.raises(ValidationError, match="missing: skills"):
        RenderOptions(
            section_order=(
                ResumeSection.EDUCATION,
                ResumeSection.EXPERIENCE,
                ResumeSection.PROJECTS,
            )
        )


def test_the_rejection_names_every_missing_section():
    with pytest.raises(ValidationError, match="missing: projects, skills"):
        RenderOptions(section_order=(ResumeSection.EDUCATION, ResumeSection.EXPERIENCE))


def test_an_unknown_section_name_is_rejected():
    with pytest.raises(ValidationError):
        RenderOptions(section_order=("education", "experience", "projects", "hobbies"))
