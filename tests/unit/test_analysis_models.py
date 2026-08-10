"""Tests for the structured job description analysis model."""

import pytest
from openai.lib._pydantic import to_strict_json_schema
from pydantic import ValidationError

from resumelab.models.analysis import MAX_LIST_ITEMS, JobAnalysis

TEXT_FIELDS = ("role_title", "role_archetype", "technical_identity", "ideal_candidate_profile")
LIST_FIELDS = (
    "core_languages",
    "frameworks",
    "infrastructure",
    "databases",
    "ai_ml_concepts",
    "domain_concepts",
    "engineering_concepts",
    "responsibilities",
    "high_priority_requirements",
    "bonus_requirements",
    "soft_traits",
    "high_value_keywords",
)


def build(**overrides) -> JobAnalysis:
    fields = {
        "company": "Northlake Systems",
        "role_title": "Storage Engineer",
        "role_archetype": "storage infrastructure engineer",
        "seniority": "early-career",
        "technical_identity": "Early-career storage infrastructure engineer.",
        "ideal_candidate_profile": "Someone who has worked close to the OS.",
        **dict.fromkeys(LIST_FIELDS, ()),
    }
    return JobAnalysis.model_validate(fields | overrides)


# --- happy path -----------------------------------------------------------


def test_a_complete_analysis_validates(job_analysis):
    assert job_analysis.role_archetype == "storage infrastructure engineer"
    assert "NVMe-oF" in job_analysis.domain_concepts
    assert job_analysis.frameworks == ()


def test_the_analysis_is_frozen():
    analysis = build()

    with pytest.raises(ValidationError):
        analysis.role_title = "Something Else"


# --- structured-output compatibility --------------------------------------


def test_every_field_is_required():
    """Strict structured-output modes reject schemas with optional fields."""
    schema = JobAnalysis.model_json_schema()

    assert set(schema["required"]) == set(schema["properties"])


def test_extra_keys_are_forbidden():
    schema = JobAnalysis.model_json_schema()

    assert schema["additionalProperties"] is False
    with pytest.raises(ValidationError):
        build(unexpected_field="x")


def test_the_schema_avoids_unsupported_constraint_keywords():
    """String and array constraints are not supported by strict structured outputs."""
    unsupported = {"minLength", "maxLength", "minItems", "maxItems", "pattern", "default"}
    schema = JobAnalysis.model_json_schema()

    for name, definition in schema["properties"].items():
        assert not unsupported & set(definition), f"{name} carries an unsupported keyword"


def test_the_schema_survives_strict_mode_conversion():
    """Guards against a future field shape the provider would reject at request time."""
    strict = to_strict_json_schema(JobAnalysis)

    assert strict["additionalProperties"] is False
    assert set(strict["required"]) == set(strict["properties"])


# --- validation and hygiene -----------------------------------------------


@pytest.mark.parametrize("field", TEXT_FIELDS)
@pytest.mark.parametrize("value", ["", "   "])
def test_fields_the_pipeline_depends_on_cannot_be_blank(field, value):
    """A blank value here is worth a repair attempt, not a degraded run."""
    with pytest.raises(ValidationError, match=field):
        build(**{field: value})


def test_an_unnamed_company_is_allowed():
    """Some postings genuinely never name the employer."""
    assert build(company="").company == ""


@pytest.mark.parametrize("field", LIST_FIELDS)
def test_list_fields_may_legitimately_be_empty(field):
    assert getattr(build(**{field: ()}), field) == ()


def test_blank_list_entries_are_dropped():
    assert build(core_languages=("Go", "", "  ", "Java")).core_languages == ("Go", "Java")


def test_duplicate_terms_are_removed_case_insensitively():
    analysis = build(high_value_keywords=("NVMe", "nvme", "NVME", "Go"))

    assert analysis.high_value_keywords == ("NVMe", "Go")


def test_the_first_spelling_of_a_duplicate_is_kept():
    assert build(core_languages=("Golang", "golang")).core_languages == ("Golang",)


def test_extraction_order_is_preserved():
    assert build(core_languages=("Go", "Java", "C")).core_languages == ("Go", "Java", "C")


def test_list_entries_are_stripped():
    assert build(frameworks=("  React  ",)).frameworks == ("React",)


def test_over_long_lists_are_capped_rather_than_rejected():
    """Extraction noise should not cost an API call to repair."""
    analysis = build(high_value_keywords=tuple(f"term-{index}" for index in range(100)))

    assert len(analysis.high_value_keywords) == MAX_LIST_ITEMS
    assert analysis.high_value_keywords[0] == "term-0"


def test_text_fields_are_stripped():
    assert build(role_title="  Storage Engineer  ").role_title == "Storage Engineer"


@pytest.mark.parametrize("field", (*TEXT_FIELDS, *LIST_FIELDS, "company", "seniority"))
def test_a_missing_field_is_rejected(field):
    fields = build().model_dump()
    del fields[field]

    with pytest.raises(ValidationError, match=field):
        JobAnalysis.model_validate(fields)
