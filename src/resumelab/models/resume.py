"""Models for the generated resume.

These are LLM structured-output targets; see :mod:`resumelab.models.common` for what
that constrains. Length limits are enforced by validators rather than JSON-Schema
keywords, which means an over-long response is handed back to the model to shorten
rather than being truncated mid-sentence.
"""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, model_validator

from resumelab.models.candidate import (
    REQUIRED_PROJECT_BULLET_COUNT,
    Education,
    PersonalDetails,
)
from resumelab.models.common import GENERATED_MODEL_CONFIG, clean_items, require_content

REQUIRED_EXPERIENCE_BULLET_COUNT = 3
"""Bullets emitted per role. A fixed count keeps runs comparable."""

MIN_SUBTITLE_CHARACTERS = 10
"""Below this a subtitle says nothing about what the project is."""

MAX_SUBTITLE_CHARACTERS = 90
"""Long enough to reposition a project, short enough to stay on one line."""

MIN_PROJECT_TECHNOLOGIES = 2
"""A project presented as using one technology reads as unfinished."""

MAX_PROJECT_TECHNOLOGIES = 10
"""Beyond this the list stops being read and starts looking padded."""

MIN_SKILL_GROUPS = 2
"""One group is a wall of text; the section exists to be scannable."""

MAX_SKILL_GROUPS = 6
"""Enough to organize a stack, few enough to stay compact on one page."""

MAX_SKILLS_PER_GROUP = 12
"""A longer row wraps and stops being read."""

MAX_TOTAL_SKILLS = 40
"""A ceiling on the whole section, which is where keyword stuffing shows up."""

MIN_BULLET_CHARACTERS = 40
"""Below this a bullet cannot carry implementation, detail, and impact."""

MAX_BULLET_CHARACTERS = 220
"""Roughly two lines. Longer bullets are skimmed past on a one-page resume."""

_LIST_MARKER = re.compile("^\\s*(?:[-*\\u2022\\u2013\\u2014]|\\d+[.)])\\s+")
"""A leading bullet glyph or numbering the renderer would draw a second time.

The escapes are bullet, en dash, and em dash, which models reach for interchangeably.
"""

MIN_SUMMARY_CHARACTERS = 60
"""Below this a summary establishes no technical identity at all."""

MAX_SUMMARY_CHARACTERS = 300
"""Roughly two lines at resume body size. Longer summaries stop being read."""


class ResumeLimits(BaseModel):
    """The length budget a run targets.

    These are the limits a finished resume is checked against before rendering, and
    the targets a condensation pass works toward. They are configurable so a study
    can vary how much room the format allows without touching code.

    The response schemas carry their own hard bounds, which is what stops a provider
    returning something structurally unusable. These are the tighter, run-specific
    budget layered on top.
    """

    model_config = ConfigDict(frozen=True)

    summary_max_characters: int = MAX_SUMMARY_CHARACTERS
    bullet_max_characters: int = MAX_BULLET_CHARACTERS
    experience_bullet_count: int = REQUIRED_EXPERIENCE_BULLET_COUNT
    project_bullet_count: int = REQUIRED_PROJECT_BULLET_COUNT


def _normalize_summary(value: str) -> str:
    """Collapse the summary to one line and hold it to a readable length.

    Whitespace is fixed silently, since a stray newline is not worth an API call.
    Length is enforced by rejection, so the model rewrites to fit instead of having
    a sentence cut off mid-clause.
    """
    collapsed = " ".join(value.split())
    if not collapsed:
        raise ValueError("must not be empty")
    if len(collapsed) < MIN_SUMMARY_CHARACTERS:
        raise ValueError(
            f"must be at least {MIN_SUMMARY_CHARACTERS} characters, got {len(collapsed)}"
        )
    if len(collapsed) > MAX_SUMMARY_CHARACTERS:
        raise ValueError(
            f"must be at most {MAX_SUMMARY_CHARACTERS} characters, got {len(collapsed)}"
        )
    return collapsed


SummaryText = Annotated[str, AfterValidator(_normalize_summary)]
"""A one-line professional summary, bounded to what a reader will actually read."""


class GeneratedSummary(BaseModel):
    """The professional summary that opens the resume."""

    model_config = GENERATED_MODEL_CONFIG

    summary: SummaryText


def _normalize_bullet(value: str) -> str:
    """Clean a bullet and hold it to one readable line.

    A leading glyph is stripped rather than rejected: the renderer draws its own, and
    a model that adds one has made a formatting slip, not a content error.
    """
    collapsed = " ".join(_LIST_MARKER.sub("", value).split())
    if not collapsed:
        raise ValueError("must not be empty")
    if len(collapsed) < MIN_BULLET_CHARACTERS:
        raise ValueError(
            f"must be at least {MIN_BULLET_CHARACTERS} characters, got {len(collapsed)}"
        )
    if len(collapsed) > MAX_BULLET_CHARACTERS:
        raise ValueError(
            f"must be at most {MAX_BULLET_CHARACTERS} characters, got {len(collapsed)}"
        )
    return collapsed


BulletText = Annotated[str, AfterValidator(_normalize_bullet)]
"""One resume bullet, cleaned and bounded to a readable length."""


def _reject_repeated_bullets(bullets: tuple[str, ...]) -> tuple[str, ...]:
    """Two identical bullets waste a third of a section, so they are worth repairing."""
    seen = {bullet.casefold() for bullet in bullets}
    if len(seen) != len(bullets):
        raise ValueError("bullets must not repeat")
    return bullets


class ExperienceBullets(BaseModel):
    """The rewritten bullets for one role.

    Only the bullets are generated. Company, title, dates, and location are factual
    anchors copied from the source profile, so they cannot drift in a rewrite.
    """

    model_config = GENERATED_MODEL_CONFIG

    bullets: Annotated[tuple[BulletText, ...], AfterValidator(_reject_repeated_bullets)]

    @model_validator(mode="after")
    def _check_bullet_count(self) -> ExperienceBullets:
        if len(self.bullets) != REQUIRED_EXPERIENCE_BULLET_COUNT:
            raise ValueError(
                f"must contain exactly {REQUIRED_EXPERIENCE_BULLET_COUNT} bullets, "
                f"got {len(self.bullets)}"
            )
        return self


def _normalize_subtitle(value: str) -> str:
    """Hold the subtitle to something that fits on the title line."""
    collapsed = " ".join(value.split())
    if len(collapsed) < MIN_SUBTITLE_CHARACTERS:
        raise ValueError(
            f"must be at least {MIN_SUBTITLE_CHARACTERS} characters, got {len(collapsed)}"
        )
    if len(collapsed) > MAX_SUBTITLE_CHARACTERS:
        raise ValueError(
            f"must be at most {MAX_SUBTITLE_CHARACTERS} characters, got {len(collapsed)}"
        )
    return collapsed


SubtitleText = Annotated[str, AfterValidator(_normalize_subtitle)]
"""A project subtitle, short enough to sit beside the project name on one line."""


def _check_technologies(values: tuple[str, ...]) -> tuple[str, ...]:
    """Require a credible, readable technology list."""
    cleaned = clean_items(values)
    if len(cleaned) < MIN_PROJECT_TECHNOLOGIES:
        raise ValueError(
            f"must name at least {MIN_PROJECT_TECHNOLOGIES} technologies, got {len(cleaned)}"
        )
    if len(cleaned) > MAX_PROJECT_TECHNOLOGIES:
        raise ValueError(
            f"must name at most {MAX_PROJECT_TECHNOLOGIES} technologies, got {len(cleaned)}"
        )
    return cleaned


ProjectTechnologies = Annotated[tuple[str, ...], AfterValidator(_check_technologies)]
"""The stack a project is presented as being built on."""


class ProjectContent(BaseModel):
    """The rewritten presentation of one project.

    The project name is not here: it is the anchor that lets a researcher line the
    generated project up against its source. Everything about what the project
    appears to *be* — its subtitle, its stack, its bullets — is generated.
    """

    model_config = GENERATED_MODEL_CONFIG

    subtitle: SubtitleText
    technologies: ProjectTechnologies
    bullets: Annotated[tuple[BulletText, ...], AfterValidator(_reject_repeated_bullets)]

    @model_validator(mode="after")
    def _check_bullet_count(self) -> ProjectContent:
        if len(self.bullets) != REQUIRED_PROJECT_BULLET_COUNT:
            raise ValueError(
                f"must contain exactly {REQUIRED_PROJECT_BULLET_COUNT} bullets, "
                f"got {len(self.bullets)}"
            )
        return self


class GeneratedProject(BaseModel):
    """One project as it appears on the generated resume."""

    model_config = GENERATED_MODEL_CONFIG

    name: str
    subtitle: str
    date: str | None
    technologies: tuple[str, ...]
    bullets: tuple[str, ...]


def _check_group_skills(values: tuple[str, ...]) -> tuple[str, ...]:
    """Require a group to carry a readable number of skills."""
    cleaned = clean_items(values)
    if not cleaned:
        raise ValueError("must list at least one skill")
    if len(cleaned) > MAX_SKILLS_PER_GROUP:
        raise ValueError(f"must list at most {MAX_SKILLS_PER_GROUP} skills, got {len(cleaned)}")
    return cleaned


class SkillGroup(BaseModel):
    """One labeled row of the skills section, e.g. ``Languages: Go, Java, C``."""

    model_config = GENERATED_MODEL_CONFIG

    label: Annotated[str, AfterValidator(require_content)]
    skills: Annotated[tuple[str, ...], AfterValidator(_check_group_skills)]


class GeneratedSkills(BaseModel):
    """The skills section, grouped as this role's reader would expect to see it.

    The labels are chosen per run rather than inherited from the source profile: which
    groupings make a candidate look aligned is itself part of the repositioning.
    """

    model_config = GENERATED_MODEL_CONFIG

    groups: tuple[SkillGroup, ...]

    @model_validator(mode="after")
    def _check_groups(self) -> GeneratedSkills:
        if not MIN_SKILL_GROUPS <= len(self.groups) <= MAX_SKILL_GROUPS:
            raise ValueError(
                f"must contain between {MIN_SKILL_GROUPS} and {MAX_SKILL_GROUPS} groups, "
                f"got {len(self.groups)}"
            )
        _reject_duplicate_labels(self.groups)
        _reject_duplicate_skills(self.groups)
        if self.skill_count > MAX_TOTAL_SKILLS:
            raise ValueError(
                f"must list at most {MAX_TOTAL_SKILLS} skills in total, got {self.skill_count}"
            )
        return self

    @property
    def skill_count(self) -> int:
        """Total skills across every group."""
        return sum(len(group.skills) for group in self.groups)


def _reject_duplicate_labels(groups: tuple[SkillGroup, ...]) -> None:
    labels = [group.label.casefold() for group in groups]
    if len(set(labels)) != len(labels):
        raise ValueError("group labels must be distinct")


def _reject_duplicate_skills(groups: tuple[SkillGroup, ...]) -> None:
    """The same skill in two groups reads as carelessness."""
    skills = [skill.casefold() for group in groups for skill in group.skills]
    if len(set(skills)) != len(skills):
        raise ValueError("a skill must not appear in more than one group")


class GeneratedExperience(BaseModel):
    """One role as it appears on the generated resume.

    Assembled rather than generated: the anchors come straight from the source
    profile and the bullets from :class:`ExperienceBullets`.
    """

    model_config = GENERATED_MODEL_CONFIG

    company: str
    title: str
    location: str | None
    start_date: str | None
    end_date: str | None
    bullets: tuple[str, ...]


class CondensedContent(BaseModel):
    """Shortened replacements for a resume's prose, in the order they appear.

    The bullets come back as one flat list covering the whole document, because
    shortening is a whole-page decision: which bullet gives up a clause depends on
    what the others are already carrying.
    """

    model_config = GENERATED_MODEL_CONFIG

    summary: SummaryText
    bullets: tuple[BulletText, ...]


class GeneratedResume(BaseModel):
    """The complete resume, ready to render.

    Deliberately permissive. Its parts were validated as they were generated, but the
    checks that matter before rendering are run by
    :mod:`resumelab.validation.resume_validator`, which reports every problem at once
    instead of failing on the first. Duplicating them here would make an invalid
    resume unconstructible and the validator untestable.
    """

    model_config = GENERATED_MODEL_CONFIG

    personal: PersonalDetails
    summary: str
    education: tuple[Education, ...]
    experiences: tuple[GeneratedExperience, ...]
    projects: tuple[GeneratedProject, ...]
    skills: tuple[SkillGroup, ...]
    achievements: tuple[str, ...] = ()

    @property
    def all_bullets(self) -> tuple[str, ...]:
        """Every bullet on the resume, for whole-document checks."""
        from_experiences = [b for experience in self.experiences for b in experience.bullets]
        from_projects = [b for project in self.projects for b in project.bullets]
        return tuple(from_experiences + from_projects)
