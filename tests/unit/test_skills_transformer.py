"""Tests for skills transformation and its schema.

The LLM is a recording fake; no network call is made.
"""

import logging

import pytest
from openai.lib._pydantic import to_strict_json_schema
from pydantic import ValidationError

from resumelab.exceptions import LLMGenerationError
from resumelab.llm.prompts import SKILLS_PROMPT
from resumelab.models.resume import (
    MAX_SKILL_GROUPS,
    MAX_SKILLS_PER_GROUP,
    MAX_TOTAL_SKILLS,
    MIN_SKILL_GROUPS,
    GeneratedSkills,
    SkillGroup,
)
from resumelab.pipeline import transform_skills

GROUPS = (
    SkillGroup(label="Languages", skills=("Go", "Java", "C", "Python")),
    SkillGroup(label="Storage & Systems", skills=("Linux", "NVMe", "NVMe-oF", "NFS", "SMB")),
    SkillGroup(label="Infrastructure", skills=("Kubernetes", "Terraform", "AWS")),
)


def skills(groups=GROUPS) -> GeneratedSkills:
    return GeneratedSkills(groups=groups)


def group(label: str, *items: str) -> SkillGroup:
    """Build a group; pass no skills to test the empty case."""
    return SkillGroup(label=label, skills=items)


@pytest.fixture
def client(make_llm_client):
    return make_llm_client([skills()])


def run(profile, analysis, strategy, client, **kwargs):
    return transform_skills(profile, analysis, strategy, client=client, **kwargs)


# --- the schema -----------------------------------------------------------


def test_the_nested_schema_survives_strict_mode_conversion():
    strict = to_strict_json_schema(GeneratedSkills)

    assert strict["additionalProperties"] is False
    for definition in strict["$defs"].values():
        assert definition["additionalProperties"] is False
        assert set(definition["required"]) == set(definition["properties"])


def test_a_complete_skills_section_validates():
    assert len(skills().groups) == 3
    assert skills().skill_count == 12


@pytest.mark.parametrize("count", [0, 1, MAX_SKILL_GROUPS + 1])
def test_the_group_count_is_bounded(count):
    groups = tuple(group(f"Group {index}", "Go") for index in range(count))

    with pytest.raises(ValidationError, match="between 2 and 6 groups"):
        skills(groups)


def test_the_minimum_number_of_groups_is_accepted():
    assert len(skills((group("A", "Go"), group("B", "Java"))).groups) == MIN_SKILL_GROUPS


def test_a_group_must_list_a_skill():
    with pytest.raises(ValidationError, match="at least one skill"):
        skills((group("Languages"), group("Empty")))


def test_an_overlong_group_is_rejected():
    too_many = tuple(f"Skill{index}" for index in range(MAX_SKILLS_PER_GROUP + 1))

    with pytest.raises(ValidationError, match="at most 12 skills"):
        skills((group("Languages", *too_many), group("Other", "Go")))


def test_the_whole_section_is_capped():
    """The total is where keyword stuffing shows up."""
    groups = tuple(
        group(f"Group {index}", *(f"Skill{index}-{n}" for n in range(MAX_SKILLS_PER_GROUP)))
        for index in range(4)
    )

    with pytest.raises(ValidationError, match=f"at most {MAX_TOTAL_SKILLS} skills in total"):
        skills(groups)


def test_duplicate_labels_are_rejected():
    with pytest.raises(ValidationError, match="labels must be distinct"):
        skills((group("Languages", "Go"), group("languages", "Java")))


def test_the_same_skill_in_two_groups_is_rejected():
    """It reads as carelessness, and it is worth a repair attempt."""
    with pytest.raises(ValidationError, match="more than one group"):
        skills((group("Languages", "Go"), group("Backend", "go")))


def test_duplicates_inside_a_group_are_cleaned_rather_than_rejected():
    cleaned = skills((group("Languages", "Go", "go", "  Java  "), group("Other", "Linux")))

    assert cleaned.groups[0].skills == ("Go", "Java")


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_label_is_rejected(blank):
    with pytest.raises(ValidationError):
        skills((group(blank, "Go"), group("Other", "Java")))


# --- what the stage asks for ----------------------------------------------


def test_the_groups_are_returned_in_order(
    candidate_profile, job_analysis, transformation_strategy, client
):
    returned = run(candidate_profile, job_analysis, transformation_strategy, client)

    assert [entry.label for entry in returned] == [
        "Languages",
        "Storage & Systems",
        "Infrastructure",
    ]


def test_one_call_builds_the_section(
    candidate_profile, job_analysis, transformation_strategy, client
):
    run(candidate_profile, job_analysis, transformation_strategy, client)

    assert len(client.calls) == 1


def test_the_versioned_prompt_is_used_verbatim(
    candidate_profile, job_analysis, transformation_strategy, client
):
    run(candidate_profile, job_analysis, transformation_strategy, client)

    assert client.last_call.system_prompt == SKILLS_PROMPT.system
    assert client.last_call.purpose == "skills"
    assert client.last_call.response_model is GeneratedSkills


def test_the_source_skills_are_the_starting_point(
    candidate_profile, job_analysis, transformation_strategy, client
):
    run(candidate_profile, job_analysis, transformation_strategy, client)

    user_prompt = client.last_call.user_prompt
    assert "PyTorch" in user_prompt
    assert "FastAPI" in user_prompt


def test_the_priority_from_the_plan_reaches_the_model(
    candidate_profile, job_analysis, transformation_strategy, client
):
    run(candidate_profile, job_analysis, transformation_strategy, client)

    assert "TRANSFORMATION STRATEGY:" in client.last_call.user_prompt
    for skill in transformation_strategy.skills_priority:
        assert skill in client.last_call.user_prompt


@pytest.mark.parametrize("secret", ["ada@example.edu", "+1 555 0100", "Ada Lovelace"])
def test_personal_details_never_reach_the_model(
    candidate_profile, job_analysis, transformation_strategy, client, secret
):
    run(candidate_profile, job_analysis, transformation_strategy, client)

    assert secret not in client.last_call.user_prompt


# --- consistency with what was already written ----------------------------


def test_the_generated_bullets_are_supplied_so_the_section_matches_them(
    candidate_profile, job_analysis, transformation_strategy, client
):
    """A technology in a bullet but missing from skills is noticed immediately."""
    written = ("Built an erasure-coded tier in Go across NVMe-oF targets.",)

    run(candidate_profile, job_analysis, transformation_strategy, client, already_written=written)

    assert "BULLETS ALREADY WRITTEN" in client.last_call.user_prompt
    assert written[0] in client.last_call.user_prompt


def test_with_nothing_written_yet_the_section_is_omitted(
    candidate_profile, job_analysis, transformation_strategy, client
):
    run(candidate_profile, job_analysis, transformation_strategy, client)

    assert "BULLETS ALREADY WRITTEN" not in client.last_call.user_prompt


def test_the_prompt_requires_consistency_and_forbids_keyword_dumping(
    candidate_profile, job_analysis, transformation_strategy, client
):
    run(candidate_profile, job_analysis, transformation_strategy, client)

    system_prompt = client.last_call.system_prompt
    assert "Every technology named anywhere else on this resume must appear here" in system_prompt
    assert "Do not dump the job description's keyword list" in system_prompt


# --- failure handling -----------------------------------------------------


def test_a_generation_failure_propagates(
    candidate_profile, job_analysis, transformation_strategy, make_llm_client
):
    failing = make_llm_client([LLMGenerationError("schema repair exhausted")])

    with pytest.raises(LLMGenerationError, match="schema repair exhausted"):
        run(candidate_profile, job_analysis, transformation_strategy, failing)


# --- logging --------------------------------------------------------------


def test_the_stage_logs_what_it_produced(
    candidate_profile, job_analysis, transformation_strategy, client, caplog
):
    with caplog.at_level(logging.DEBUG, logger="resumelab.pipeline.skills_transformer"):
        run(candidate_profile, job_analysis, transformation_strategy, client)

    assert "transforming skills" in caplog.text
    assert "groups=3 skills=12" in caplog.text
    assert "Storage & Systems" in caplog.text
