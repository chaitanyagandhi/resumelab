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
    MAX_SKILL_COUNT,
    MIN_SKILL_COUNT,
    GeneratedSkills,
)
from resumelab.pipeline import transform_skills

SKILLS = (
    "Go",
    "Java",
    "C",
    "Linux",
    "NVMe",
    "NVMe-oF",
    "NFS",
    "SMB",
    "Kubernetes",
    "Terraform",
    "Distributed Systems",
    "Replication",
)


def skills(items=SKILLS) -> GeneratedSkills:
    return GeneratedSkills(skills=items)


def listing(count: int) -> tuple[str, ...]:
    """A distinct skill list of exactly ``count`` entries, with no fallback."""
    return tuple(f"Skill{index}" for index in range(count))


@pytest.fixture
def client(make_llm_client):
    return make_llm_client([skills()])


def run(profile, analysis, strategy, client, **kwargs):
    return transform_skills(profile, analysis, strategy, client=client, **kwargs)


# --- the schema -----------------------------------------------------------


def test_the_schema_survives_strict_mode_conversion():
    """Adding a JSON-Schema constraint here would pass locally and fail at request time."""
    strict = to_strict_json_schema(GeneratedSkills)

    assert strict["additionalProperties"] is False
    assert set(strict["required"]) == set(strict["properties"])


def test_the_section_is_a_flat_list():
    """No labels, no nesting: the section is rendered as one line."""
    assert to_strict_json_schema(GeneratedSkills)["properties"]["skills"]["type"] == "array"
    assert skills().skills[0] == "Go"


def test_a_complete_skills_section_validates():
    assert skills().skill_count == 12


# --- the count is the whole point -----------------------------------------


@pytest.mark.parametrize("count", [0, 1, MIN_SKILL_COUNT - 1, MAX_SKILL_COUNT + 1])
def test_the_skill_count_is_bounded(count):
    """A section that lists everything says nothing about who is being presented."""
    with pytest.raises(ValidationError, match="between 10 and 20 skills"):
        skills(listing(count))


@pytest.mark.parametrize("count", [MIN_SKILL_COUNT, MAX_SKILL_COUNT])
def test_the_bounds_themselves_are_accepted(count):
    assert skills(listing(count)).skill_count == count


def test_duplicates_are_cleaned_before_the_count_is_judged():
    """A repeated term is a formatting slip, not worth an API call to repair."""
    cleaned = skills(("Go", "go", "  GO  ", *listing(9)))

    assert cleaned.skills[0] == "Go"
    assert cleaned.skill_count == MIN_SKILL_COUNT


def test_duplicates_that_drop_the_count_below_the_floor_are_rejected():
    """Cleaning happens first, so the rejection describes what is actually wrong."""
    with pytest.raises(ValidationError, match="got 2"):
        skills(("Go", "go", "Java", "java", "GO"))


def test_blank_entries_are_dropped():
    cleaned = skills(("Go", "", "   ", *listing(9)))

    assert "" not in cleaned.skills
    assert cleaned.skill_count == MIN_SKILL_COUNT


def test_the_chosen_order_is_preserved():
    """Order is the only emphasis the section has."""
    assert skills().skills == SKILLS


# --- what the stage asks for ----------------------------------------------


def test_the_skills_are_returned_in_order(
    candidate_profile, job_analysis, transformation_strategy, client
):
    returned = run(candidate_profile, job_analysis, transformation_strategy, client)

    assert returned == SKILLS


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


def test_the_source_skills_are_available_as_a_fallback(
    candidate_profile, job_analysis, transformation_strategy, client
):
    """When the posting names too few skills, the profile is what is drawn from."""
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


def test_the_prompt_states_the_selection_rule(
    candidate_profile, job_analysis, transformation_strategy, client
):
    """The posting first, the profile's related skills only to make up the number."""
    system_prompt = client.last_call.system_prompt if client.calls else SKILLS_PROMPT.system

    assert "Between 10 and 20 skills" in system_prompt
    assert "No categories" in system_prompt
    assert "The job description decides what belongs here" in system_prompt
    assert "Only if the posting names fewer than 10" in system_prompt
    assert "raw material, not a checklist" in system_prompt
    # Reversed deliberately: the section is read by keyword matching before a person
    # sees it, and a near-synonym does not match.
    assert "Take the posting's terms verbatim" in system_prompt
    assert "do not transcribe" not in system_prompt


def test_the_prompt_admits_entries_that_are_not_technologies(
    candidate_profile, job_analysis, transformation_strategy, client
):
    """A phrase the posting leans on counts as a skill for this section's purposes."""
    system_prompt = client.last_call.system_prompt if client.calls else SKILLS_PROMPT.system

    assert "Entries do not have to be technologies" in system_prompt
    assert "Omit proficiency ratings and years of experience" in system_prompt


def test_the_prompt_asks_for_consistent_casing(
    candidate_profile, job_analysis, transformation_strategy, client
):
    """Lifting the posting's words should not mean lifting its capitalisation.

    A list mixing "GPU Nodes" with "throughput optimization" reads as pasted, which
    is the one way this section can look worse than the selection behind it.
    """
    system_prompt = client.last_call.system_prompt if client.calls else SKILLS_PROMPT.system

    assert "in the case a resume would use" in system_prompt


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
    assert "count=12" in caplog.text
    assert "NVMe-oF" in caplog.text
