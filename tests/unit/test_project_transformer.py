"""Tests for project transformation and its schema.

The LLM is a recording fake; no network call is made.
"""

import logging

import pytest
from openai.lib._pydantic import to_strict_json_schema
from pydantic import ValidationError

from resumelab.exceptions import LLMGenerationError
from resumelab.llm.prompts import PROJECT_PROMPT
from resumelab.models.candidate import REQUIRED_PROJECT_BULLET_COUNT
from resumelab.models.resume import (
    MAX_PROJECT_HEADING_CHARACTERS,
    MAX_PROJECT_TECHNOLOGIES,
    MAX_SUBTITLE_CHARACTERS,
    MIN_PROJECT_TECHNOLOGIES,
    MIN_SUBTITLE_CHARACTERS,
    GeneratedProject,
    ProjectContent,
)
from resumelab.pipeline import transform_projects

SUBTITLE = "NVMe-oF Event Processing Engine"
TECHNOLOGIES = ("Go", "Linux", "NVMe-oF")
BULLETS = (
    "Designed a shared-nothing ingestion path fanning writes across NVMe-oF "
    "targets at 40k events per second.",
    "Implemented an idempotent replay log with content-addressed segments, making "
    "failure recovery deterministic.",
    "Measured durability under injected disk faults, holding p99 commit latency "
    "under 12ms across 200 runs.",
)


def content(**overrides) -> ProjectContent:
    fields = {"subtitle": SUBTITLE, "technologies": TECHNOLOGIES, "bullets": BULLETS}
    return ProjectContent.model_validate(fields | overrides)


@pytest.fixture
def client(make_llm_client):
    return make_llm_client([content(), content(), content()])


def run(profile, analysis, strategy, client, **kwargs):
    return transform_projects(profile, analysis, strategy, client=client, **kwargs)


# --- the schema -----------------------------------------------------------


def test_the_schema_survives_strict_mode_conversion():
    strict = to_strict_json_schema(ProjectContent)

    assert strict["additionalProperties"] is False
    assert set(strict["required"]) == set(strict["properties"])


def test_the_project_name_is_not_part_of_the_generated_content():
    """The name anchors each generated project to its source for comparison."""
    assert "name" not in ProjectContent.model_fields
    assert set(ProjectContent.model_fields) == {"subtitle", "technologies", "bullets"}


@pytest.mark.parametrize("count", [0, 1, 2, 4])
def test_the_bullet_count_is_fixed(count):
    with pytest.raises(ValidationError, match="exactly 3 bullets"):
        content(bullets=tuple(f"{'x' * 60} number {index}" for index in range(count)))


def test_three_bullets_are_accepted():
    assert len(content().bullets) == REQUIRED_PROJECT_BULLET_COUNT


def test_an_over_long_subtitle_is_rejected():
    with pytest.raises(ValidationError, match="at most 45 characters"):
        content(subtitle="x" * (MAX_SUBTITLE_CHARACTERS + 1))


def test_a_subtitle_too_short_to_say_anything_is_rejected():
    with pytest.raises(ValidationError, match="at least 10 characters"):
        content(subtitle="Backend")


def test_subtitle_whitespace_is_collapsed():
    assert content(subtitle=f"  {SUBTITLE}\n ").subtitle == SUBTITLE


def test_a_project_needs_more_than_one_technology():
    with pytest.raises(ValidationError, match="at least 2 technologies"):
        content(technologies=("Go",))


def test_an_overstuffed_technology_list_is_rejected():
    too_many = tuple(f"Tech{index}" for index in range(MAX_PROJECT_TECHNOLOGIES + 1))

    with pytest.raises(ValidationError, match="at most 5 technologies"):
        content(technologies=too_many)


def test_duplicate_technologies_are_cleaned_rather_than_rejected():
    assert content(technologies=("Go", "go", "  Linux  ")).technologies == ("Go", "Linux")


def test_deduplication_can_push_a_list_below_the_floor():
    """Two entries that are the same technology are one technology."""
    with pytest.raises(ValidationError, match="at least 2 technologies"):
        content(technologies=("Go", "GO", "go"))


def test_repeated_bullets_are_rejected():
    with pytest.raises(ValidationError, match="must not repeat"):
        content(bullets=(BULLETS[0], BULLETS[0].upper(), BULLETS[1]))


def test_the_bounds_leave_room_for_a_real_project():
    assert MIN_SUBTITLE_CHARACTERS < len(SUBTITLE) < MAX_SUBTITLE_CHARACTERS
    assert MIN_PROJECT_TECHNOLOGIES <= len(TECHNOLOGIES) <= MAX_PROJECT_TECHNOLOGIES


# --- what is anchored and what is repositioned ----------------------------


def test_the_project_name_and_date_come_from_the_source(
    candidate_profile, job_analysis, transformation_strategy, client
):
    transformed = run(candidate_profile, job_analysis, transformation_strategy, client)

    for generated, source in zip(transformed, candidate_profile.projects, strict=True):
        assert generated.name == source.name
        assert generated.date == source.date


def test_the_subtitle_and_stack_are_replaced_wholesale(
    candidate_profile, job_analysis, transformation_strategy, client
):
    """What a project appears to be is a framing decision, not a fact."""
    transformed = run(candidate_profile, job_analysis, transformation_strategy, client)

    source = candidate_profile.projects[0]
    assert source.subtitle == "Subtitle 1"
    assert transformed[0].subtitle == SUBTITLE
    assert source.technologies == ("Python", "PostgreSQL")
    assert transformed[0].technologies == TECHNOLOGIES


def test_all_three_projects_are_transformed(
    candidate_profile, job_analysis, transformation_strategy, client
):
    transformed = run(candidate_profile, job_analysis, transformation_strategy, client)

    assert len(transformed) == REQUIRED_PROJECT_BULLET_COUNT
    assert [project.name for project in transformed] == ["Project 1", "Project 2", "Project 3"]
    assert len(client.calls) == 3


# --- what the stage asks for ----------------------------------------------


def test_the_versioned_prompt_is_used_verbatim(
    candidate_profile, job_analysis, transformation_strategy, client
):
    run(candidate_profile, job_analysis, transformation_strategy, client)

    assert client.last_call.system_prompt == PROJECT_PROMPT.system
    assert client.last_call.purpose == "project"
    assert client.last_call.response_model is ProjectContent


def test_each_project_gets_its_own_direction(
    candidate_profile, job_analysis, transformation_strategy, client
):
    run(candidate_profile, job_analysis, transformation_strategy, client)

    for index, call in enumerate(client.calls, start=1):
        assert "SOURCE PROJECT:" in call.user_prompt
        assert "DIRECTION FOR THIS ENTRY:" in call.user_prompt
        assert f"Positioning {index} toward distributed storage." in call.user_prompt


def test_the_proposed_title_direction_reaches_the_model(
    candidate_profile, job_analysis, transformation_strategy, client
):
    run(candidate_profile, job_analysis, transformation_strategy, client)

    assert "Project 1 — Storage Engine" in client.calls[0].user_prompt


def test_the_source_project_is_supplied_as_the_anchor(
    candidate_profile, job_analysis, transformation_strategy, client
):
    run(candidate_profile, job_analysis, transformation_strategy, client)

    first_prompt = client.calls[0].user_prompt
    assert "Designed component 1A." in first_prompt
    assert "PostgreSQL" in first_prompt


@pytest.mark.parametrize("secret", ["ada@example.edu", "+1 555 0100", "Ada Lovelace"])
def test_personal_details_never_reach_the_model(
    candidate_profile, job_analysis, transformation_strategy, client, secret
):
    run(candidate_profile, job_analysis, transformation_strategy, client)

    assert secret not in client.last_call.user_prompt


# --- cross-section awareness ----------------------------------------------


def test_the_first_project_sees_bullets_written_by_the_experience_stage(
    candidate_profile, job_analysis, transformation_strategy, client
):
    """Projects run after experiences, so they must not repeat their claims."""
    earlier = ("Built a replication controller in Go across 3,000 storage nodes.",)

    run(candidate_profile, job_analysis, transformation_strategy, client, already_written=earlier)

    assert "BULLETS ALREADY WRITTEN" in client.calls[0].user_prompt
    assert earlier[0] in client.calls[0].user_prompt


def test_with_nothing_written_yet_the_section_is_omitted(
    candidate_profile, job_analysis, transformation_strategy, client
):
    run(candidate_profile, job_analysis, transformation_strategy, client)

    assert "BULLETS ALREADY WRITTEN" not in client.calls[0].user_prompt


def test_each_project_sees_the_projects_written_before_it(
    candidate_profile, job_analysis, transformation_strategy, client
):
    run(candidate_profile, job_analysis, transformation_strategy, client)

    assert "BULLETS ALREADY WRITTEN" in client.calls[1].user_prompt
    assert BULLETS[0] in client.calls[1].user_prompt


# --- failure handling -----------------------------------------------------


def test_a_project_with_no_direction_fails_the_run(
    candidate_profile, job_analysis, transformation_strategy, make_llm_client
):
    first, second, third = transformation_strategy.project_directions
    strategy = transformation_strategy.model_copy(
        update={
            "project_directions": (
                first.model_copy(update={"project": "Renamed"}),
                second,
                third,
            )
        }
    )

    with pytest.raises(LLMGenerationError, match="Project 1"):
        run(candidate_profile, job_analysis, strategy, make_llm_client([content()]))


def test_a_generation_failure_propagates(
    candidate_profile, job_analysis, transformation_strategy, make_llm_client
):
    failing = make_llm_client([LLMGenerationError("schema repair exhausted")])

    with pytest.raises(LLMGenerationError, match="schema repair exhausted"):
        run(candidate_profile, job_analysis, transformation_strategy, failing)


# --- logging --------------------------------------------------------------


def test_each_project_is_logged_by_name(
    candidate_profile, job_analysis, transformation_strategy, client, caplog
):
    with caplog.at_level(logging.INFO, logger="resumelab.pipeline.project_transformer"):
        run(candidate_profile, job_analysis, transformation_strategy, client)

    assert "transforming project project=Project 1" in caplog.text
    assert "transformed projects count=3" in caplog.text


def test_the_repositioning_is_visible_in_debug_logs(
    candidate_profile, job_analysis, transformation_strategy, client, caplog
):
    """A researcher reading the log should see what each project became."""
    with caplog.at_level(logging.DEBUG, logger="resumelab.pipeline.project_transformer"):
        run(candidate_profile, job_analysis, transformation_strategy, client)

    assert SUBTITLE in caplog.text


def test_generated_project_accepts_a_missing_date():
    project = GeneratedProject(
        name="LoanFlow",
        subtitle=SUBTITLE,
        date=None,
        technologies=TECHNOLOGIES,
        bullets=BULLETS,
    )

    assert project.date is None


def test_an_overlong_heading_is_trimmed_rather_than_rejected():
    """The failure this replaces: a whole run lost to a heading that would have fit.

    The model cannot tell which of two fields to shorten, so it rewrites both, breaks
    something else, and the retry budget goes. Trimming is deterministic and free.
    """
    fitted = content(
        subtitle="Enterprise Secrets Integration Platform",
        technologies=("TypeScript", "Node.js", "PostgreSQL", "Redis", "Kubernetes"),
    )

    assert len(fitted.subtitle) + len(", ".join(fitted.technologies)) <= (
        MAX_PROJECT_HEADING_CHARACTERS
    )
    # Trimmed from the end: the technologies are in priority order.
    assert fitted.technologies[0] == "TypeScript"
    assert len(fitted.technologies) < 5


def test_a_heading_within_budget_is_left_exactly_as_written():
    stack = ("Go", "Linux", "NVMe-oF")

    assert content(subtitle="NVMe-oF Event Engine", technologies=stack).technologies == stack


def test_trimming_never_goes_below_the_minimum_stack():
    """A heading that still will not fit wraps. That costs one line, not the run."""
    fitted = content(
        subtitle="x" * MAX_SUBTITLE_CHARACTERS,
        technologies=("PostgreSQL", "Kubernetes", "OpenTelemetry"),
    )

    assert len(fitted.technologies) == MIN_PROJECT_TECHNOLOGIES


def test_trimming_is_not_silently_skipped(recwarn):
    """The trap this replaces: a model validator returning a new instance.

    Pydantic ignores that when the model is built through ``__init__`` and only warns,
    so the heading budget looked enforced and did nothing. Building the model the way
    the SDKs build it must both apply the trim and raise no warning.
    """
    fitted = ProjectContent(
        subtitle="Enterprise Secrets Integration Platform",
        technologies=("TypeScript", "Node.js", "PostgreSQL", "Kubernetes", "OpenTelemetry"),
        bullets=BULLETS,
    )

    assert len(fitted.technologies) < 5
    assert [str(w.message) for w in recwarn.list] == []
