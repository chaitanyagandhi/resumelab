"""Tests for shortening a resume that does not fit.

The LLM is a recording fake; no network call is made.
"""

import logging

import pytest

from resumelab.exceptions import LLMGenerationError
from resumelab.llm.prompts import CONDENSE_PROMPT
from resumelab.models.resume import (
    MAX_BULLET_CHARACTERS,
    CondensedContent,
    ResumeLimits,
)
from resumelab.pipeline import condense_resume

SHORT_SUMMARY = "Storage infrastructure engineer building distributed data-path services in Go."


def shortened(resume, *, summary=SHORT_SUMMARY, bullets=None):
    """A plausible condensation response for ``resume``."""
    return CondensedContent(
        summary=summary,
        bullets=bullets
        if bullets is not None
        else tuple(
            f"Short bullet number {index} about Go and NVMe."
            for index in range(len(resume.all_bullets))
        ),
    )


@pytest.fixture
def client(make_llm_client, generated_resume):
    return make_llm_client([shortened(generated_resume)])


# --- the shortened resume -------------------------------------------------


def test_the_resume_gets_shorter(generated_resume, client):
    result = condense_resume(generated_resume, client=client)

    assert len(result.summary) < len(generated_resume.summary)
    assert sum(map(len, result.all_bullets)) < sum(map(len, generated_resume.all_bullets))


def test_the_structure_is_preserved(generated_resume, client):
    """Condensation shortens prose; it must not drop a section or an entry."""
    result = condense_resume(generated_resume, client=client)

    assert len(result.experiences) == len(generated_resume.experiences)
    assert len(result.projects) == len(generated_resume.projects)
    assert len(result.all_bullets) == len(generated_resume.all_bullets)


def test_bullets_are_returned_to_the_entries_they_came_from(generated_resume, client):
    """Position is how the flat response is matched back to sections."""
    result = condense_resume(generated_resume, client=client)

    assert len(result.experiences[0].bullets) == len(generated_resume.experiences[0].bullets)
    for original, rewritten in zip(result.projects, generated_resume.projects, strict=True):
        assert len(original.bullets) == len(rewritten.bullets)


def test_everything_not_being_shortened_is_left_alone(generated_resume, client):
    result = condense_resume(generated_resume, client=client)

    assert result.personal == generated_resume.personal
    assert result.education == generated_resume.education
    assert result.skills == generated_resume.skills
    assert result.achievements == generated_resume.achievements
    assert result.projects[0].subtitle == generated_resume.projects[0].subtitle
    assert result.experiences[0].company == generated_resume.experiences[0].company


# --- what the stage asks for ----------------------------------------------


def test_the_versioned_prompt_is_used_verbatim(generated_resume, client):
    condense_resume(generated_resume, client=client)

    assert client.last_call.system_prompt == CONDENSE_PROMPT.system
    assert client.last_call.purpose == "condense"
    assert client.last_call.response_model is CondensedContent


def test_one_call_shortens_the_whole_document(generated_resume, client):
    """Shortening bullets independently would take the same words out of each."""
    condense_resume(generated_resume, client=client)

    assert len(client.calls) == 1


def test_every_bullet_is_sent_numbered_for_matching(generated_resume, client):
    condense_resume(generated_resume, client=client)

    user_prompt = client.last_call.user_prompt
    for index, bullet in enumerate(generated_resume.all_bullets, start=1):
        assert f"{index}. {bullet}" in user_prompt


def test_the_budget_is_stated(generated_resume, client):
    condense_resume(generated_resume, client=client)

    user_prompt = client.last_call.user_prompt
    assert f"Return exactly {len(generated_resume.all_bullets)} bullets." in user_prompt
    assert "at most 300 characters" in user_prompt
    assert f"at most {MAX_BULLET_CHARACTERS} characters" in user_prompt


def test_a_tighter_budget_is_passed_through(generated_resume, client):
    condense_resume(
        generated_resume,
        client=client,
        limits=ResumeLimits(summary_max_characters=160, bullet_max_characters=120),
    )

    assert "at most 160 characters" in client.last_call.user_prompt
    assert "at most 120 characters" in client.last_call.user_prompt


def test_the_prompt_protects_the_substance(generated_resume, client):
    """Shortening must not cost a technology, a number, or the claim itself."""
    condense_resume(generated_resume, client=client)

    system_prompt = client.last_call.system_prompt
    assert "Any technology, protocol, or system name." in system_prompt
    assert "Any number, and any unit attached to it." in system_prompt
    assert "You are editing, not rewriting." in system_prompt


# --- failure handling -----------------------------------------------------


@pytest.mark.parametrize("delta", [-1, 1])
def test_a_mismatched_bullet_count_fails_rather_than_guessing(
    generated_resume, make_llm_client, delta
):
    """Without a position-for-position match the text cannot be put back."""
    count = len(generated_resume.all_bullets) + delta
    response = shortened(
        generated_resume,
        bullets=tuple(f"Short bullet number {index} about Go and NVMe." for index in range(count)),
    )

    with pytest.raises(LLMGenerationError, match="cannot be matched back"):
        condense_resume(generated_resume, client=make_llm_client([response]))


def test_a_generation_failure_propagates(generated_resume, make_llm_client):
    failing = make_llm_client([LLMGenerationError("could not shorten further")])

    with pytest.raises(LLMGenerationError, match="could not shorten further"):
        condense_resume(generated_resume, client=failing)


# --- logging --------------------------------------------------------------


def test_the_saving_is_logged(generated_resume, client, caplog):
    with caplog.at_level(logging.INFO, logger="resumelab.pipeline.condenser"):
        condense_resume(generated_resume, client=client)

    assert "condensing resume" in caplog.text
    assert "saved=" in caplog.text
