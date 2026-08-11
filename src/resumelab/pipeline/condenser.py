"""Shortening a resume that does not fit, without truncating it.

Cutting strings to length is the obvious fix and the wrong one: it severs sentences
mid-clause and silently drops the end of whatever claim was being made. This stage
asks the model to shorten the prose instead, keeping every technology, number, and
claim while removing the words that were not carrying them.

The whole document is shortened in one call. Which bullet gives up a clause depends
on what the others are already carrying, so shortening them independently would take
the same words out of all of them.
"""

from __future__ import annotations

import logging

from resumelab.exceptions import LLMGenerationError
from resumelab.llm.client import LLMClient
from resumelab.llm.prompts import CONDENSE_PROMPT, Section
from resumelab.models.resume import (
    CondensedContent,
    GeneratedExperience,
    GeneratedProject,
    GeneratedResume,
    ResumeLimits,
)

logger = logging.getLogger(__name__)


def condense_resume(
    resume: GeneratedResume,
    *,
    client: LLMClient,
    limits: ResumeLimits | None = None,
) -> GeneratedResume:
    """Return a shorter version of ``resume`` with the same structure and claims.

    Args:
        resume: The resume that did not fit.
        client: The LLM client to use, injected by the caller.
        limits: The length budget to aim at. Defaults to the standard limits.

    Returns:
        A resume with the same sections, entries, and bullet counts, shortened.

    Raises:
        LLMGenerationError: If generation fails, or the response does not line up
            with the resume it was asked to shorten.
    """
    budget = limits or ResumeLimits()
    original_bullets = resume.all_bullets
    before = _character_count(resume)

    logger.info("condensing resume bullets=%d characters=%d", len(original_bullets), before)

    condensed = client.generate_structured(
        system_prompt=CONDENSE_PROMPT.system,
        user_prompt=CONDENSE_PROMPT.user(
            Section(label="SUMMARY", content=resume.summary),
            Section(label="BULLETS, IN ORDER", content=_numbered(original_bullets)),
            Section(label="LENGTH BUDGET", content=_budget_text(budget, len(original_bullets))),
        ),
        response_model=CondensedContent,
        purpose=CONDENSE_PROMPT.name,
    )

    if len(condensed.bullets) != len(original_bullets):
        raise LLMGenerationError(
            f"Condensation returned {len(condensed.bullets)} bullets for a resume with "
            f"{len(original_bullets)}. The shortened text cannot be matched back to "
            "the sections it came from."
        )

    shortened = _rebuild(resume, condensed)
    logger.info(
        "condensed resume characters=%d saved=%d",
        _character_count(shortened),
        before - _character_count(shortened),
    )
    return shortened


def _rebuild(resume: GeneratedResume, condensed: CondensedContent) -> GeneratedResume:
    """Put the shortened bullets back where they came from, in order."""
    remaining = list(condensed.bullets)

    experiences: list[GeneratedExperience] = []
    for experience in resume.experiences:
        taken = [remaining.pop(0) for _ in experience.bullets]
        experiences.append(experience.model_copy(update={"bullets": tuple(taken)}))

    projects: list[GeneratedProject] = []
    for project in resume.projects:
        taken = [remaining.pop(0) for _ in project.bullets]
        projects.append(project.model_copy(update={"bullets": tuple(taken)}))

    return resume.model_copy(
        update={
            "summary": condensed.summary,
            "experiences": tuple(experiences),
            "projects": tuple(projects),
        }
    )


def _numbered(bullets: tuple[str, ...]) -> str:
    """Number the bullets so the response can be matched back position by position."""
    return "\n".join(f"{index}. {bullet}" for index, bullet in enumerate(bullets, start=1))


def _budget_text(budget: ResumeLimits, bullet_count: int) -> str:
    return (
        f"Return exactly {bullet_count} bullets.\n"
        f"Summary: at most {budget.summary_max_characters} characters.\n"
        f"Each bullet: at most {budget.bullet_max_characters} characters, and "
        "meaningfully shorter than the one it replaces."
    )


def _character_count(resume: GeneratedResume) -> int:
    return len(resume.summary) + sum(len(bullet) for bullet in resume.all_bullets)
