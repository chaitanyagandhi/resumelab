"""Tests for the transformation strategy stage.

The LLM is a recording fake; no network call is made.
"""

import json
import logging

import pytest

from resumelab.exceptions import LLMGenerationError
from resumelab.llm.prompts import FENCE_MARKER, STRATEGY_PROMPT
from resumelab.models.strategy import TransformationStrategy
from resumelab.pipeline import build_transformation_strategy


@pytest.fixture
def client(make_llm_client, transformation_strategy):
    return make_llm_client([transformation_strategy])


def build(profile, analysis, client):
    return build_transformation_strategy(profile, analysis, client=client)


# --- happy path -----------------------------------------------------------


def test_the_strategy_is_returned(candidate_profile, job_analysis, client, transformation_strategy):
    assert build(candidate_profile, job_analysis, client) is transformation_strategy


def test_a_single_call_produces_the_whole_plan(candidate_profile, job_analysis, client):
    """One plan, one call: per-section planning is what makes a resume incoherent."""
    build(candidate_profile, job_analysis, client)

    assert len(client.calls) == 1


# --- what the stage asks for ----------------------------------------------


def test_the_response_is_constrained_to_the_strategy_schema(
    candidate_profile, job_analysis, client
):
    build(candidate_profile, job_analysis, client)

    assert client.last_call.response_model is TransformationStrategy


def test_the_versioned_prompt_is_used_verbatim(candidate_profile, job_analysis, client):
    build(candidate_profile, job_analysis, client)

    assert client.last_call.system_prompt == STRATEGY_PROMPT.system
    assert client.last_call.purpose == "transformation_strategy"


def test_both_the_profile_and_the_analysis_are_supplied(candidate_profile, job_analysis, client):
    build(candidate_profile, job_analysis, client)

    user_prompt = client.last_call.user_prompt
    assert "CANDIDATE PROFILE:" in user_prompt
    assert "JOB ANALYSIS:" in user_prompt


def test_the_profile_content_the_plan_needs_is_present(candidate_profile, job_analysis, client):
    build(candidate_profile, job_analysis, client)

    user_prompt = client.last_call.user_prompt
    assert "Analytical Engines Inc." in user_prompt
    assert "Project 1" in user_prompt
    assert "Cut p99 latency by 40%" in user_prompt


def test_the_target_identity_reaches_the_planner(candidate_profile, job_analysis, client):
    build(candidate_profile, job_analysis, client)

    assert job_analysis.technical_identity in client.last_call.user_prompt


def test_internal_state_is_not_fenced_as_untrusted(candidate_profile, job_analysis, client):
    """The profile and our own analysis are trusted; fencing them would invite distrust.

    The untrusted job description text does not reach this stage at all — only the
    structured analysis derived from it.
    """
    build(candidate_profile, job_analysis, client)

    assert FENCE_MARKER not in client.last_call.user_prompt


# --- personal details are withheld ----------------------------------------


@pytest.mark.parametrize(
    "secret",
    ["ada@example.edu", "+1 555 0100", "linkedin.com/in/ada", "Ada Lovelace"],
)
def test_personal_details_never_reach_the_model(candidate_profile, job_analysis, client, secret):
    """Contact details play no part in planning, so they are not sent to a provider."""
    build(candidate_profile, job_analysis, client)

    assert secret not in client.last_call.user_prompt


def test_the_profile_payload_omits_the_personal_section(candidate_profile, job_analysis, client):
    build(candidate_profile, job_analysis, client)

    payload = client.last_call.user_prompt.split("CANDIDATE PROFILE:\n", 1)[1]
    payload = payload.split("\n\nJOB ANALYSIS:", 1)[0]
    assert "personal" not in json.loads(payload)


# --- coverage of the profile ----------------------------------------------


def test_a_plan_missing_a_project_fails_the_run(
    candidate_profile, job_analysis, make_llm_client, transformation_strategy
):
    """A missing direction would leave a section untransformed but still look like a result."""
    first, second, third = transformation_strategy.project_directions
    directions = (first, second, third.model_copy(update={"project": "Ghost"}))
    client = make_llm_client(
        [transformation_strategy.model_copy(update={"project_directions": directions})]
    )

    with pytest.raises(LLMGenerationError) as exc_info:
        build(candidate_profile, job_analysis, client)

    assert "Project 3" in str(exc_info.value)


def test_a_plan_missing_an_experience_fails_the_run(
    candidate_profile, job_analysis, make_llm_client, transformation_strategy
):
    renamed = transformation_strategy.experience_directions[0].model_copy(
        update={"experience": "Some Other Company"}
    )
    client = make_llm_client(
        [transformation_strategy.model_copy(update={"experience_directions": (renamed,)})]
    )

    with pytest.raises(LLMGenerationError) as exc_info:
        build(candidate_profile, job_analysis, client)

    assert "Analytical Engines Inc." in str(exc_info.value)


def test_a_plan_that_echoes_names_loosely_is_accepted(
    candidate_profile, job_analysis, make_llm_client, transformation_strategy
):
    """Case and spacing drift is normal model behaviour, not a failure."""
    loose = transformation_strategy.model_copy(
        update={
            "project_directions": tuple(
                direction.model_copy(update={"project": direction.project.upper()})
                for direction in transformation_strategy.project_directions
            )
        }
    )
    client = make_llm_client([loose])

    assert build(candidate_profile, job_analysis, client) is loose


def test_a_direction_for_an_unknown_entry_is_warned_about(
    candidate_profile, job_analysis, make_llm_client, transformation_strategy, caplog
):
    """It cannot be applied, but it is not worth failing an otherwise complete plan."""
    real = transformation_strategy.experience_directions[0]
    extra = real.model_copy(update={"experience": "Imaginary Corp"})
    client = make_llm_client(
        [transformation_strategy.model_copy(update={"experience_directions": (real, extra)})]
    )

    with caplog.at_level(logging.WARNING, logger="resumelab.pipeline.strategist"):
        build(candidate_profile, job_analysis, client)

    assert "Imaginary Corp" in caplog.text


def test_a_stray_project_direction_is_reported_as_missing_coverage(
    candidate_profile, job_analysis, make_llm_client, transformation_strategy
):
    """With the count pinned, a direction for an unknown project displaces a real one."""
    first, second, third = transformation_strategy.project_directions
    directions = (first, second, third.model_copy(update={"project": "Phantom Project"}))
    client = make_llm_client(
        [transformation_strategy.model_copy(update={"project_directions": directions})]
    )

    with pytest.raises(LLMGenerationError, match="Project 3"):
        build(candidate_profile, job_analysis, client)


# --- failure handling -----------------------------------------------------


def test_a_generation_failure_propagates(candidate_profile, job_analysis, make_llm_client):
    client = make_llm_client([LLMGenerationError("model was unreachable")])

    with pytest.raises(LLMGenerationError, match="model was unreachable"):
        build(candidate_profile, job_analysis, client)


# --- logging --------------------------------------------------------------


def test_the_stage_logs_its_start_and_target(candidate_profile, job_analysis, client, caplog):
    with caplog.at_level(logging.INFO, logger="resumelab.pipeline.strategist"):
        build(candidate_profile, job_analysis, client)

    assert "building transformation strategy" in caplog.text
    assert "Early-career storage infrastructure engineer." in caplog.text
