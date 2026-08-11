"""Tests for the transformation strategy schema."""

import pytest
from openai.lib._pydantic import to_strict_json_schema
from pydantic import ValidationError

from resumelab.models.candidate import REQUIRED_PROJECT_COUNT
from resumelab.models.strategy import (
    ExperienceDirection,
    ProjectDirection,
    TransformationStrategy,
    match_key,
)


def experience_direction(company="Analytical Engines Inc.") -> ExperienceDirection:
    return ExperienceDirection(
        experience=company,
        target_framing="Reframe as storage data path work.",
        concepts_to_emphasize=("replication",),
        jd_terms_to_incorporate=("Go",),
    )


def project_direction(name="LoanFlow") -> ProjectDirection:
    return ProjectDirection(
        project=name,
        new_positioning="An NVMe-oF backed event processing engine.",
        possible_title_direction="LoanFlow — Storage Engine",
        concepts_to_incorporate=("erasure coding",),
    )


def build(**overrides) -> TransformationStrategy:
    fields = {
        "target_identity": "Storage infrastructure engineer.",
        "summary_direction": "Lead with distributed storage.",
        "experience_directions": (experience_direction(),),
        "project_directions": tuple(
            project_direction(f"Project {index}") for index in range(REQUIRED_PROJECT_COUNT)
        ),
        "skills_priority": ("Go", "Linux"),
        "tone": "Direct and systems-oriented.",
        "overall_strategy": "Present the candidate as a storage engineer.",
    }
    return TransformationStrategy.model_validate(fields | overrides)


# --- happy path -----------------------------------------------------------


def test_a_complete_strategy_validates(transformation_strategy):
    assert transformation_strategy.target_identity.startswith("Early-career storage")
    assert len(transformation_strategy.project_directions) == REQUIRED_PROJECT_COUNT


def test_the_strategy_is_frozen():
    """A run's recorded plan must not change after later stages have read it."""
    with pytest.raises(ValidationError):
        build().tone = "Playful"


# --- structured-output compatibility --------------------------------------


def test_the_nested_schema_survives_strict_mode_conversion():
    """Nested direction objects must also be strict-clean, not just the root."""
    strict = to_strict_json_schema(TransformationStrategy)

    assert strict["additionalProperties"] is False
    assert set(strict["required"]) == set(strict["properties"])
    for definition in strict["$defs"].values():
        assert definition["additionalProperties"] is False
        assert set(definition["required"]) == set(definition["properties"])


# --- shape the transformers depend on -------------------------------------


@pytest.mark.parametrize("count", [0, 1, 2, 4])
def test_project_directions_must_match_the_required_project_count(count):
    with pytest.raises(ValidationError, match="exactly 3"):
        build(project_directions=tuple(project_direction(f"P{i}") for i in range(count)))


def test_at_least_one_experience_direction_is_required():
    with pytest.raises(ValidationError, match="at least one role"):
        build(experience_directions=())


def test_duplicate_experience_directions_are_rejected():
    """Two directions for one role would make the lookup ambiguous."""
    with pytest.raises(ValidationError, match="more than one direction"):
        build(experience_directions=(experience_direction("Acme"), experience_direction("acme")))


def test_duplicate_project_directions_are_rejected():
    with pytest.raises(ValidationError, match="more than one direction"):
        build(
            project_directions=(
                project_direction("LoanFlow"),
                project_direction("loanflow"),
                project_direction("Other"),
            )
        )


@pytest.mark.parametrize(
    "field",
    ["target_identity", "summary_direction", "tone", "overall_strategy"],
)
def test_blank_planning_fields_are_rejected(field):
    with pytest.raises(ValidationError, match=field):
        build(**{field: "   "})


def test_a_blank_direction_name_is_rejected():
    """An unnamed direction could never be matched to a profile entry."""
    with pytest.raises(ValidationError):
        build(
            project_directions=(
                project_direction(""),
                project_direction("B"),
                project_direction("C"),
            )
        )


# --- lookup ---------------------------------------------------------------


def test_directions_are_found_by_name(transformation_strategy):
    direction = transformation_strategy.direction_for_project("Project 2")

    assert direction is not None
    assert direction.possible_title_direction == "Project 2 — Storage Engine"


@pytest.mark.parametrize(
    "echoed",
    ["Project 1", "project 1", "  PROJECT 1  ", "Project  1"],
)
def test_lookup_tolerates_how_a_model_echoes_a_name(transformation_strategy, echoed):
    """Models rarely echo a name back byte-identical."""
    assert transformation_strategy.direction_for_project(echoed) is not None


def test_experience_lookup_tolerates_the_same_variation(transformation_strategy):
    assert transformation_strategy.direction_for_experience("analytical engines inc.") is not None


def test_an_unknown_name_returns_none(transformation_strategy):
    assert transformation_strategy.direction_for_project("Nonexistent") is None
    assert transformation_strategy.direction_for_experience("Nonexistent") is None


@pytest.mark.parametrize(
    ("left", "right"),
    [("LoanFlow", "loanflow"), ("A  B", "a b"), ("  X  ", "x")],
)
def test_match_keys_normalize_case_and_whitespace(left, right):
    assert match_key(left) == match_key(right)
