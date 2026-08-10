"""Tests for the immutable source candidate profile schema."""

import copy

import pytest
import yaml
from pydantic import ValidationError

from resumelab.models import (
    REQUIRED_PROJECT_BULLET_COUNT,
    REQUIRED_PROJECT_COUNT,
    CandidateProfile,
    PersonalDetails,
    Project,
    Skills,
)


def make_project(bullet_count=REQUIRED_PROJECT_BULLET_COUNT, **overrides):
    project = {
        "name": "LoanFlow",
        "bullets": [f"Bullet {index}." for index in range(bullet_count)],
    }
    return {**project, **overrides}


# --- happy path -----------------------------------------------------------


def test_a_complete_profile_validates(profile_data):
    profile = CandidateProfile.model_validate(profile_data)

    assert profile.personal.name == "Ada Lovelace"
    assert len(profile.projects) == REQUIRED_PROJECT_COUNT
    assert profile.education[0].coursework == ("Distributed Systems", "Machine Learning")
    assert profile.skills.programming_languages == ("Python", "Go")
    assert profile.achievements == ("Dean's List",)


def test_skills_default_to_empty_when_the_section_is_omitted(profile_data):
    del profile_data["skills"]

    assert CandidateProfile.model_validate(profile_data).skills == Skills()


def test_optional_sections_may_be_omitted(profile_data):
    del profile_data["achievements"]

    assert CandidateProfile.model_validate(profile_data).achievements == ()


# --- immutability ---------------------------------------------------------


def test_the_profile_cannot_be_reassigned(profile_data):
    profile = CandidateProfile.model_validate(profile_data)

    with pytest.raises(ValidationError):
        profile.personal = PersonalDetails(name="Someone Else", email="x@example.com")


def test_nested_models_cannot_be_reassigned(profile_data):
    profile = CandidateProfile.model_validate(profile_data)

    with pytest.raises(ValidationError):
        profile.projects[0].name = "Renamed"


def test_collections_are_tuples_so_they_cannot_be_appended_to(profile_data):
    """A list would let a pipeline stage mutate the experimental control in place."""
    profile = CandidateProfile.model_validate(profile_data)

    assert isinstance(profile.projects, tuple)
    assert isinstance(profile.experiences, tuple)
    assert isinstance(profile.projects[0].bullets, tuple)
    assert isinstance(profile.skills.programming_languages, tuple)
    assert not hasattr(profile.projects, "append")


def test_validation_does_not_mutate_the_input_data(profile_data):
    original = copy.deepcopy(profile_data)

    CandidateProfile.model_validate(profile_data)

    assert profile_data == original


# --- text normalization ---------------------------------------------------


def test_surrounding_whitespace_is_stripped(profile_data):
    profile_data["personal"]["name"] = "  Ada Lovelace  "

    assert CandidateProfile.model_validate(profile_data).personal.name == "Ada Lovelace"


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_optional_fields_become_none(profile_data, blank):
    profile_data["personal"]["github"] = blank

    assert CandidateProfile.model_validate(profile_data).personal.github is None


@pytest.mark.parametrize("field_name", ["name", "email"])
def test_blank_required_contact_fields_are_rejected(profile_data, field_name):
    profile_data["personal"][field_name] = "   "

    with pytest.raises(ValidationError) as exc_info:
        CandidateProfile.model_validate(profile_data)

    assert field_name in str(exc_info.value)


# --- structural rules -----------------------------------------------------


@pytest.mark.parametrize("project_count", [0, 1, 2, 4])
def test_the_profile_requires_exactly_three_projects(profile_data, project_count):
    profile_data["projects"] = [make_project() for _ in range(project_count)]

    with pytest.raises(ValidationError) as exc_info:
        CandidateProfile.model_validate(profile_data)

    assert "projects" in str(exc_info.value)


@pytest.mark.parametrize("bullet_count", [0, 1, 2, 4])
def test_each_project_requires_exactly_three_bullets(bullet_count):
    with pytest.raises(ValidationError):
        Project.model_validate(make_project(bullet_count=bullet_count))


def test_a_project_with_three_bullets_is_accepted():
    project = Project.model_validate(make_project())

    assert len(project.bullets) == REQUIRED_PROJECT_BULLET_COUNT
    assert project.subtitle is None
    assert project.technologies == ()


def test_empty_project_bullets_are_rejected():
    with pytest.raises(ValidationError):
        Project.model_validate(make_project(bullets=["Real bullet.", "  ", "Another."]))


def test_experiences_require_at_least_one_bullet(profile_data):
    profile_data["experiences"][0]["bullets"] = []

    with pytest.raises(ValidationError):
        CandidateProfile.model_validate(profile_data)


def test_experience_bullet_counts_are_not_capped(profile_data):
    """Only projects are fixed at three; experiences carry all available material."""
    profile_data["experiences"][0]["bullets"] = [f"Bullet {index}." for index in range(6)]

    profile = CandidateProfile.model_validate(profile_data)

    assert len(profile.experiences[0].bullets) == 6


@pytest.mark.parametrize("section", ["education", "experiences"])
def test_required_sections_cannot_be_empty(profile_data, section):
    profile_data[section] = []

    with pytest.raises(ValidationError) as exc_info:
        CandidateProfile.model_validate(profile_data)

    assert section in str(exc_info.value)


@pytest.mark.parametrize("section", ["personal", "education", "experiences", "projects"])
def test_required_sections_cannot_be_missing(profile_data, section):
    del profile_data[section]

    with pytest.raises(ValidationError) as exc_info:
        CandidateProfile.model_validate(profile_data)

    assert section in str(exc_info.value)


def test_a_mistyped_key_is_rejected_rather_than_silently_dropped(profile_data):
    """`bullet:` instead of `bullets:` must fail loudly in a hand-maintained file."""
    experience = profile_data["experiences"][0]
    experience["bullet"] = experience.pop("bullets")

    with pytest.raises(ValidationError) as exc_info:
        CandidateProfile.model_validate(profile_data)

    assert "bullet" in str(exc_info.value)


# --- the shipped template -------------------------------------------------


def test_the_template_matches_the_schema_sections(profile_template_path):
    """Guards against the template drifting away from the model."""
    template = yaml.safe_load(profile_template_path.read_text(encoding="utf-8"))

    assert set(template) == set(CandidateProfile.model_fields)


def test_the_template_ships_the_required_project_scaffolding(profile_template_path):
    template = yaml.safe_load(profile_template_path.read_text(encoding="utf-8"))

    assert len(template["projects"]) == REQUIRED_PROJECT_COUNT
    for project in template["projects"]:
        assert set(project) == set(Project.model_fields)
        assert len(project["bullets"]) == REQUIRED_PROJECT_BULLET_COUNT


def test_the_unpopulated_template_does_not_validate(profile_template_path):
    """The template is a skeleton: it must fail until a researcher fills it in."""
    template = yaml.safe_load(profile_template_path.read_text(encoding="utf-8"))

    with pytest.raises(ValidationError):
        CandidateProfile.model_validate(template)
