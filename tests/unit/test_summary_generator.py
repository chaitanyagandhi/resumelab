"""Tests for the summary generation stage and its length bounds.

The LLM is a recording fake; no network call is made.
"""

import logging

import pytest
from openai.lib._pydantic import to_strict_json_schema
from pydantic import ValidationError

from resumelab.exceptions import LLMGenerationError
from resumelab.llm.prompts import SUMMARY_PROMPT
from resumelab.models.resume import (
    MAX_SUMMARY_CHARACTERS,
    MIN_SUMMARY_CHARACTERS,
    GeneratedSummary,
)
from resumelab.pipeline import generate_summary

SUMMARY = (
    "Storage infrastructure engineer who builds distributed data-path services in Go "
    "and Java on Linux, working close to NVMe devices and network storage protocols."
)


@pytest.fixture
def client(make_llm_client):
    return make_llm_client([GeneratedSummary(summary=SUMMARY)])


def run(profile, analysis, strategy, client):
    return generate_summary(profile, analysis, strategy, client=client)


# --- the model ------------------------------------------------------------


def test_the_summary_schema_survives_strict_mode_conversion():
    strict = to_strict_json_schema(GeneratedSummary)

    assert strict["additionalProperties"] is False
    assert set(strict["required"]) == set(strict["properties"])


def test_a_valid_summary_is_kept_verbatim():
    assert GeneratedSummary(summary=SUMMARY).summary == SUMMARY


def test_stray_line_breaks_are_collapsed_rather_than_rejected():
    """A newline is a formatting slip, not worth spending an API call to repair."""
    wrapped = SUMMARY.replace(" ", "\n  ", 1)

    assert GeneratedSummary(summary=wrapped).summary == SUMMARY


def test_an_over_long_summary_is_kept_rather_than_rejected():
    """A summary rejected at 308 characters against a limit of 300 ended a run once.

    Length is a layout matter: a long summary takes another line and the renderer
    tightens. Spending the retry budget on it, and losing everything generated before
    it, is the trade the house rule forbids.
    """
    long_summary = "x" * (MAX_SUMMARY_CHARACTERS + 8)

    assert GeneratedSummary(summary=long_summary).summary == long_summary


def test_a_summary_of_several_sentences_is_shortened_cleanly():
    """Whole sentences go, because cutting at a character count leaves a fragment."""
    first = "Backend engineer who builds ad serving systems on Kafka and Postgres."
    filler = " Also writes tooling for internal teams across the company every day."
    summary = first + filler * 4

    shortened = GeneratedSummary(summary=summary).summary

    assert shortened.startswith(first)
    assert len(shortened) <= MAX_SUMMARY_CHARACTERS
    assert shortened.endswith(".")


def test_a_summary_at_the_ceiling_is_accepted():
    assert len(GeneratedSummary(summary="x" * MAX_SUMMARY_CHARACTERS).summary) == (
        MAX_SUMMARY_CHARACTERS
    )


def test_a_summary_too_short_to_establish_an_identity_is_rejected():
    with pytest.raises(ValidationError, match="at least 60 characters"):
        GeneratedSummary(summary="Software engineer.")


@pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
def test_a_blank_summary_is_rejected(blank):
    with pytest.raises(ValidationError):
        GeneratedSummary(summary=blank)


def test_the_length_check_applies_after_whitespace_is_collapsed():
    padded = "  " + "x" * MAX_SUMMARY_CHARACTERS + "  \n"

    assert len(GeneratedSummary(summary=padded).summary) == MAX_SUMMARY_CHARACTERS


def test_the_bounds_leave_room_for_a_real_summary():
    assert MIN_SUMMARY_CHARACTERS < len(SUMMARY) < MAX_SUMMARY_CHARACTERS


# --- the stage ------------------------------------------------------------


def test_the_summary_text_is_returned(
    candidate_profile, job_analysis, transformation_strategy, client
):
    assert run(candidate_profile, job_analysis, transformation_strategy, client) == SUMMARY


def test_one_call_is_made(candidate_profile, job_analysis, transformation_strategy, client):
    run(candidate_profile, job_analysis, transformation_strategy, client)

    assert len(client.calls) == 1


def test_the_versioned_prompt_is_used_verbatim(
    candidate_profile, job_analysis, transformation_strategy, client
):
    run(candidate_profile, job_analysis, transformation_strategy, client)

    assert client.last_call.system_prompt == SUMMARY_PROMPT.system
    assert client.last_call.purpose == "summary"
    assert client.last_call.response_model is GeneratedSummary


def test_the_stage_works_from_the_plan(
    candidate_profile, job_analysis, transformation_strategy, client
):
    """The summary executes summary_direction; it does not decide the framing itself."""
    run(candidate_profile, job_analysis, transformation_strategy, client)

    user_prompt = client.last_call.user_prompt
    assert "TRANSFORMATION STRATEGY:" in user_prompt
    assert transformation_strategy.summary_direction in user_prompt
    assert transformation_strategy.target_identity in user_prompt
    assert transformation_strategy.tone in user_prompt


def test_the_source_material_and_analysis_are_supplied(
    candidate_profile, job_analysis, transformation_strategy, client
):
    run(candidate_profile, job_analysis, transformation_strategy, client)

    user_prompt = client.last_call.user_prompt
    assert "CANDIDATE PROFILE:" in user_prompt
    assert "JOB ANALYSIS:" in user_prompt


@pytest.mark.parametrize("secret", ["ada@example.edu", "+1 555 0100", "Ada Lovelace"])
def test_personal_details_never_reach_the_model(
    candidate_profile, job_analysis, transformation_strategy, client, secret
):
    run(candidate_profile, job_analysis, transformation_strategy, client)

    assert secret not in client.last_call.user_prompt


def test_the_prompt_rules_out_the_generic_openers(
    candidate_profile, job_analysis, transformation_strategy, client
):
    """Generic filler is the documented failure mode for this stage."""
    run(candidate_profile, job_analysis, transformation_strategy, client)

    system_prompt = client.last_call.system_prompt
    assert "Passionate software engineer seeking opportunities" in system_prompt
    assert "Results-driven professional" in system_prompt


# --- failure handling -----------------------------------------------------


def test_a_generation_failure_propagates(
    candidate_profile, job_analysis, transformation_strategy, make_llm_client
):
    failing = make_llm_client([LLMGenerationError("could not shorten the summary")])

    with pytest.raises(LLMGenerationError, match="could not shorten"):
        run(candidate_profile, job_analysis, transformation_strategy, failing)


# --- logging --------------------------------------------------------------


def test_the_stage_logs_the_identity_it_is_aiming_at(
    candidate_profile, job_analysis, transformation_strategy, client, caplog
):
    with caplog.at_level(logging.INFO, logger="resumelab.pipeline.summary_generator"):
        run(candidate_profile, job_analysis, transformation_strategy, client)

    assert "generating summary" in caplog.text
    assert transformation_strategy.target_identity in caplog.text
