"""Tests for experience transformation and the bullet schema.

The LLM is a recording fake; no network call is made.
"""

import logging
import re

import pytest
from openai.lib._pydantic import to_strict_json_schema
from pydantic import ValidationError

from resumelab.exceptions import LLMGenerationError
from resumelab.llm.prompts import EXPERIENCE_PROMPT
from resumelab.models.resume import (
    MAX_BULLET_CHARACTERS,
    MIN_BULLET_CHARACTERS,
    REQUIRED_EXPERIENCE_BULLET_COUNT,
    ExperienceBullets,
    GeneratedExperience,
)
from resumelab.pipeline import transform_experiences

BULLETS = (
    "Built a replication controller in Go placing volumes across 3,000 nodes, "
    "cutting rebalance time to ten minutes.",
    "Instrumented the NVMe write path with per-device histograms, surfacing tail "
    "regressions before release.",
    "Designed an erasure-coded storage tier on Linux, cutting capacity overhead "
    "40% with p99 read latency flat.",
)


def bullets(*texts: str) -> ExperienceBullets:
    """Build a response; pass no arguments to test the empty case."""
    return ExperienceBullets(bullets=texts)


@pytest.fixture
def client(make_llm_client):
    return make_llm_client([bullets(*BULLETS)])


def run(profile, analysis, strategy, client):
    return transform_experiences(profile, analysis, strategy, client=client)


# --- the bullet schema ----------------------------------------------------


def test_the_schema_survives_strict_mode_conversion():
    strict = to_strict_json_schema(ExperienceBullets)

    assert strict["additionalProperties"] is False
    assert set(strict["required"]) == set(strict["properties"])


@pytest.mark.parametrize("count", [0, 1, 2, 4, 5])
def test_the_bullet_count_is_fixed(count):
    with pytest.raises(ValidationError, match="exactly 3 bullets"):
        bullets(*(f"{'x' * 60} number {index}" for index in range(count)))


def test_three_bullets_are_accepted():
    assert len(bullets(*BULLETS).bullets) == REQUIRED_EXPERIENCE_BULLET_COUNT


@pytest.mark.parametrize("glyph", ["- ", "\u2022 ", "* ", "1. ", "2) ", "\u2013 "])
def test_a_leading_bullet_glyph_is_stripped(glyph):
    """The renderer draws its own glyph; a second one is a formatting slip."""
    written = bullets(*(f"{glyph}{bullet}" for bullet in BULLETS))

    assert written.bullets == BULLETS


def test_whitespace_is_collapsed():
    wrapped = tuple(bullet.replace(" ", "\n   ", 1) for bullet in BULLETS)

    assert bullets(*wrapped).bullets == BULLETS


def test_an_over_long_bullet_is_rejected_so_the_model_rewrites_it():
    with pytest.raises(ValidationError, match="at most 130 characters"):
        bullets("x" * (MAX_BULLET_CHARACTERS + 1), BULLETS[1], BULLETS[2])


def test_a_bullet_too_short_to_carry_detail_and_impact_is_rejected():
    with pytest.raises(ValidationError, match="at least 40 characters"):
        bullets("Wrote some Go.", BULLETS[1], BULLETS[2])


@pytest.mark.parametrize("blank", ["", "   ", "- "])
def test_a_blank_bullet_is_rejected(blank):
    with pytest.raises(ValidationError):
        bullets(blank, BULLETS[1], BULLETS[2])


def test_repeated_bullets_are_rejected():
    """A duplicate wastes a third of the section, which is worth a repair attempt."""
    with pytest.raises(ValidationError, match="must not repeat"):
        bullets(BULLETS[0], BULLETS[0].upper(), BULLETS[1])


def test_bounds_leave_room_for_a_real_bullet():
    for bullet in BULLETS:
        assert MIN_BULLET_CHARACTERS < len(bullet) < MAX_BULLET_CHARACTERS


# --- anchors are not generated --------------------------------------------


def test_the_factual_anchors_come_from_the_source_profile(
    candidate_profile, job_analysis, transformation_strategy, client
):
    """A rewrite must not be able to change where someone worked or when."""
    transformed = run(candidate_profile, job_analysis, transformation_strategy, client)

    source = candidate_profile.experiences[0]
    assert transformed[0].company == source.company
    assert transformed[0].title == source.title
    assert transformed[0].location == source.location
    assert transformed[0].start_date == source.start_date
    assert transformed[0].end_date == source.end_date


def test_the_model_is_never_asked_for_the_anchors(
    candidate_profile, job_analysis, transformation_strategy, client
):
    run(candidate_profile, job_analysis, transformation_strategy, client)

    assert set(ExperienceBullets.model_fields) == {"bullets"}
    assert client.last_call.response_model is ExperienceBullets


def test_the_generated_bullets_are_used(
    candidate_profile, job_analysis, transformation_strategy, client
):
    transformed = run(candidate_profile, job_analysis, transformation_strategy, client)

    assert transformed[0].bullets == BULLETS


# --- what the stage asks for ----------------------------------------------


def test_the_versioned_prompt_is_used_verbatim(
    candidate_profile, job_analysis, transformation_strategy, client
):
    run(candidate_profile, job_analysis, transformation_strategy, client)

    assert client.last_call.system_prompt == EXPERIENCE_PROMPT.system
    assert client.last_call.purpose == "experience"


def test_the_role_and_its_own_direction_are_supplied(
    candidate_profile, job_analysis, transformation_strategy, client
):
    run(candidate_profile, job_analysis, transformation_strategy, client)

    user_prompt = client.last_call.user_prompt
    assert "SOURCE EXPERIENCE:" in user_prompt
    assert "DIRECTION FOR THIS ENTRY:" in user_prompt
    assert "Reframe the ingestion work as a storage data path." in user_prompt


def test_the_source_bullets_are_supplied_as_raw_material(
    candidate_profile, job_analysis, transformation_strategy, client
):
    run(candidate_profile, job_analysis, transformation_strategy, client)

    assert "Cut p99 latency by 40%" in client.last_call.user_prompt


def test_the_whole_plan_is_supplied_for_global_coherence(
    candidate_profile, job_analysis, transformation_strategy, client
):
    run(candidate_profile, job_analysis, transformation_strategy, client)

    user_prompt = client.last_call.user_prompt
    assert "TRANSFORMATION STRATEGY:" in user_prompt
    assert "JOB ANALYSIS:" in user_prompt


@pytest.mark.parametrize("secret", ["ada@example.edu", "+1 555 0100", "Ada Lovelace"])
def test_personal_details_never_reach_the_model(
    candidate_profile, job_analysis, transformation_strategy, client, secret
):
    run(candidate_profile, job_analysis, transformation_strategy, client)

    assert secret not in client.last_call.user_prompt


# --- cross-entry awareness ------------------------------------------------


def test_the_first_role_is_not_told_about_earlier_bullets(
    candidate_profile, job_analysis, transformation_strategy, client
):
    run(candidate_profile, job_analysis, transformation_strategy, client)

    assert "BULLETS ALREADY WRITTEN" not in client.last_call.user_prompt


def test_later_roles_see_what_has_already_been_written(
    profile_data, job_analysis, transformation_strategy, make_llm_client
):
    """Without this, separate calls converge on the same verbs and the same impact."""
    profile = _profile_with_second_role(profile_data)
    strategy = _strategy_covering_second_role(transformation_strategy)
    second = bullets(
        *(f"Rearchitected component {index} of the data path." * 2 for index in range(3))
    )
    client = make_llm_client([bullets(*BULLETS), second])

    transform_experiences(profile, job_analysis, strategy, client=client)

    assert len(client.calls) == 2
    later_prompt = client.calls[1].user_prompt
    assert "BULLETS ALREADY WRITTEN" in later_prompt
    for bullet in BULLETS:
        assert bullet in later_prompt


def test_every_role_is_transformed_in_profile_order(
    profile_data, job_analysis, transformation_strategy, make_llm_client
):
    profile = _profile_with_second_role(profile_data)
    strategy = _strategy_covering_second_role(transformation_strategy)
    client = make_llm_client([bullets(*BULLETS), bullets(*BULLETS)])

    transformed = transform_experiences(profile, job_analysis, strategy, client=client)

    assert [entry.company for entry in transformed] == [
        experience.company for experience in profile.experiences
    ]


# --- failure handling -----------------------------------------------------


def test_a_role_with_no_direction_fails_the_run(
    candidate_profile, job_analysis, transformation_strategy, make_llm_client
):
    renamed = transformation_strategy.experience_directions[0].model_copy(
        update={"experience": "Somewhere Else"}
    )
    strategy = transformation_strategy.model_copy(update={"experience_directions": (renamed,)})

    with pytest.raises(LLMGenerationError, match=re.escape("Analytical Engines Inc.")):
        run(candidate_profile, job_analysis, strategy, make_llm_client([bullets(*BULLETS)]))


def test_a_generation_failure_propagates(
    candidate_profile, job_analysis, transformation_strategy, make_llm_client
):
    failing = make_llm_client([LLMGenerationError("schema repair exhausted")])

    with pytest.raises(LLMGenerationError, match="schema repair exhausted"):
        run(candidate_profile, job_analysis, transformation_strategy, failing)


# --- logging --------------------------------------------------------------


def test_each_role_is_logged_by_company(
    candidate_profile, job_analysis, transformation_strategy, client, caplog
):
    with caplog.at_level(logging.INFO, logger="resumelab.pipeline.experience_transformer"):
        run(candidate_profile, job_analysis, transformation_strategy, client)

    assert "transforming experience company=Analytical Engines Inc." in caplog.text


# --- helpers --------------------------------------------------------------


def _profile_with_second_role(profile_data):
    from resumelab.models.candidate import CandidateProfile

    second = dict(profile_data["experiences"][0], company="Babbage Systems")
    return CandidateProfile.model_validate(
        profile_data | {"experiences": [profile_data["experiences"][0], second]}
    )


def _strategy_covering_second_role(strategy):
    first = strategy.experience_directions[0]
    second = first.model_copy(update={"experience": "Babbage Systems"})
    return strategy.model_copy(update={"experience_directions": (first, second)})


def test_generated_experience_carries_optional_anchors_as_none():
    entry = GeneratedExperience(
        company="Acme",
        title="Engineer",
        location=None,
        start_date=None,
        end_date=None,
        bullets=BULLETS,
    )

    assert entry.location is None
    assert entry.bullets == BULLETS
