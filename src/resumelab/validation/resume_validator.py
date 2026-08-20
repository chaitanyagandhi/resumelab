"""Deterministic checks run on a generated resume before it is rendered.

This is the last gate before a run produces an artifact, and it is deliberately
separate from the models. The models validate each piece as it arrives from the
provider; this validates the finished document, and catches the failures that only
exist at the whole-resume level — a section that ended up empty, a project that lost
a bullet during assembly, a control character that survived every earlier step.

Every problem is collected and reported together. Fixing one failure only to be shown
the next is a poor way to debug a research run, and nothing here is expensive enough
to justify stopping early.
"""

from __future__ import annotations

import logging

from resumelab.models.candidate import REQUIRED_PROJECT_COUNT
from resumelab.models.resume import (
    MIN_BULLET_CHARACTERS,
    GeneratedResume,
    ResumeLimits,
)
from resumelab.utils.text import control_characters

logger = logging.getLogger(__name__)


def inspect_resume(resume: GeneratedResume, limits: ResumeLimits | None = None) -> list[str]:
    """Report everything about ``resume`` that falls outside the run's budget.

    Returns the problems rather than raising them. Nothing found here stops a resume
    being drawn: an over-long bullet takes a second line, a thin section looks thin,
    and a reader can see both and fix them. Refusing to render is the one outcome
    that helps nobody, and it throws away every stage the run already paid for.

    Args:
        resume: The assembled resume.
        limits: The run's length budget. Defaults to the standard limits, which match
            the bounds the generation stages already enforce.

    Returns:
        One line per problem, in reading order. Empty when there is nothing to say.
    """
    budget = limits or ResumeLimits()
    logger.info("inspecting generated resume")

    problems: list[str] = []
    _check_identity(resume, problems)
    _check_sections_exist(resume, problems)
    _check_summary_length(resume, budget, problems)
    _check_projects(resume, budget, problems)
    _check_bullets(resume, budget, problems)
    _check_text_hygiene(resume, problems)

    for problem in problems:
        logger.warning("resume: %s", problem)

    logger.debug(
        "resume inspected experiences=%d projects=%d bullets=%d problems=%d",
        len(resume.experiences),
        len(resume.projects),
        len(resume.all_bullets),
        len(problems),
    )
    return problems


def _check_identity(resume: GeneratedResume, problems: list[str]) -> None:
    """A resume nobody can be contacted about is not a resume."""
    if not resume.personal.name.strip():
        problems.append("the candidate has no name")
    if not (resume.personal.email or resume.personal.phone):
        problems.append("there is no contact information: an email or phone is required")


def _check_sections_exist(resume: GeneratedResume, problems: list[str]) -> None:
    if not resume.summary.strip():
        problems.append("the summary is empty")
    if not resume.education:
        problems.append("there are no education entries")
    if not resume.experiences:
        problems.append("there are no experience entries")
    if not resume.skills:
        problems.append("there are no skills")


def _check_summary_length(
    resume: GeneratedResume,
    budget: ResumeLimits,
    problems: list[str],
) -> None:
    if len(resume.summary) > budget.summary_max_characters:
        problems.append(
            f"the summary is {len(resume.summary)} characters, over the "
            f"{budget.summary_max_characters} allowed for this run"
        )


def _check_projects(
    resume: GeneratedResume,
    budget: ResumeLimits,
    problems: list[str],
) -> None:
    """Project counts are the research design, so a drift here invalidates the run."""
    if len(resume.projects) != REQUIRED_PROJECT_COUNT:
        problems.append(
            f"expected exactly {REQUIRED_PROJECT_COUNT} projects, found {len(resume.projects)}"
        )
    for project in resume.projects:
        if len(project.bullets) != budget.project_bullet_count:
            problems.append(
                f"project {project.name!r} has {len(project.bullets)} bullets, "
                f"expected exactly {budget.project_bullet_count}"
            )
        if not project.subtitle.strip():
            problems.append(f"project {project.name!r} has no subtitle")
        if not project.technologies:
            problems.append(f"project {project.name!r} names no technologies")


def _check_bullets(
    resume: GeneratedResume,
    budget: ResumeLimits,
    problems: list[str],
) -> None:
    for experience in resume.experiences:
        where = f"experience {experience.company!r}"
        if not experience.bullets:
            problems.append(f"{where} has no bullets")
        elif len(experience.bullets) != budget.experience_bullet_count:
            problems.append(
                f"{where} has {len(experience.bullets)} bullets, "
                f"expected exactly {budget.experience_bullet_count}"
            )
        _check_bullet_texts(experience.bullets, where, budget, problems)
    for project in resume.projects:
        _check_bullet_texts(project.bullets, f"project {project.name!r}", budget, problems)


def _check_bullet_texts(
    bullets: tuple[str, ...],
    where: str,
    budget: ResumeLimits,
    problems: list[str],
) -> None:
    for index, bullet in enumerate(bullets, start=1):
        if not bullet.strip():
            problems.append(f"{where} bullet {index} is empty")
        elif len(bullet) > budget.bullet_max_characters:
            problems.append(
                f"{where} bullet {index} runs to {len(bullet)} characters and will "
                f"wrap onto a second line; {budget.bullet_max_characters} fits one"
            )
        elif len(bullet) < MIN_BULLET_CHARACTERS:
            problems.append(
                f"{where} bullet {index} is only {len(bullet)} characters and says "
                f"little; {MIN_BULLET_CHARACTERS} is the least that carries an outcome"
            )


def _check_text_hygiene(resume: GeneratedResume, problems: list[str]) -> None:
    """Control characters break text extraction from the PDF, invisibly."""
    for label, text in _rendered_text(resume):
        found = control_characters(text)
        if found:
            problems.append(f"{label} contains control characters: {', '.join(found)}")


def _rendered_text(resume: GeneratedResume) -> list[tuple[str, str]]:
    """Every string that will be drawn onto the page, with a label for reporting."""
    fields: list[tuple[str, str]] = [
        ("the candidate name", resume.personal.name),
        ("the summary", resume.summary),
    ]
    fields += [(f"achievement {index}", text) for index, text in enumerate(resume.achievements, 1)]
    for experience in resume.experiences:
        where = f"experience {experience.company!r}"
        fields += [(f"{where} bullet {i}", b) for i, b in enumerate(experience.bullets, 1)]
    for project in resume.projects:
        where = f"project {project.name!r}"
        fields.append((f"{where} subtitle", project.subtitle))
        fields += [(f"{where} technology {i}", t) for i, t in enumerate(project.technologies, 1)]
        fields += [(f"{where} bullet {i}", b) for i, b in enumerate(project.bullets, 1)]
    fields += [(f"skill {skill!r}", skill) for skill in resume.skills]
    return fields
