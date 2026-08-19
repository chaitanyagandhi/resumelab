"""Models for the generated resume.

These are LLM structured-output targets; see :mod:`resumelab.models.common` for what
that constrains. Length limits are enforced by validators rather than JSON-Schema
keywords, which means an over-long response is handed back to the model to shorten
rather than being truncated mid-sentence.
"""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    ValidationInfo,
    field_validator,
    model_validator,
)

from resumelab.models.candidate import (
    REQUIRED_PROJECT_BULLET_COUNT,
    Education,
    PersonalDetails,
)
from resumelab.models.common import GENERATED_MODEL_CONFIG, clean_items
from resumelab.utils.text import soften_dashes

REQUIRED_EXPERIENCE_BULLET_COUNT = 3
"""Bullets emitted per role. A fixed count keeps runs comparable."""

MIN_SUBTITLE_CHARACTERS = 10
"""Below this a subtitle says nothing about what the project is."""

MAX_SUBTITLE_CHARACTERS = 45
"""Long enough to reposition a project, short enough to stay on the title line."""

MAX_PROJECT_HEADING_CHARACTERS = 78
"""Budget for the subtitle and the stack together, which share the title line.

Bounding them separately is not enough: either can be within its own limit while the
pair still wraps. Measured with :func:`pdfmetrics.stringWidth` against the date column
at the longest project name in a full profile, not estimated from character counts.

Enforced by trimming, never by rejection — see :meth:`ProjectContent._fit_heading`.
"""

MIN_PROJECT_TECHNOLOGIES = 1
"""The floor the heading trim is allowed to degrade to.

Not the target. Two is what the prompt asks for and what a project normally names;
this is how far :meth:`ProjectContent._fit_heading` may drop when a long subtitle
would otherwise wrap the line. A single technology reads thin, and a wrapped heading
costs a line, so the trim spends the first before the second.
"""

MAX_PROJECT_TECHNOLOGIES = 2
"""Beyond this the list stops being read and starts looking padded.

It also has to share the title line with the project name and subtitle, so the cap is
what keeps that line from wrapping.
"""

MIN_SKILL_COUNT = 10
"""Below this the section reads as a thin candidate rather than a focused one."""

MAX_SKILL_COUNT = 20
"""Beyond this the section stops being a selection and becomes a keyword dump.

The ceiling is the point: a skills list that names everything says nothing about
what this candidate is being presented as, which is the thing under study.
"""

MIN_BULLET_CHARACTERS = 40
"""Below this a bullet cannot carry implementation, detail, and impact."""

TARGET_BULLET_CHARACTERS = 95
"""What the prompts ask for, well inside a line.

A bullet line is :data:`~resumelab.rendering.styles.CONTENT_WIDTH` less the hanging
indent, and at body size that holds about 116 characters of ordinary prose. The target
sits below that rather than at it, because a bullet one word over the line does not
lose a word, it gains a whole line.

It sits far below the cap on purpose. Models write to the stated limit rather than
the stated target, so the limit is what has to be right; the gap between the two is
what stops a bullet that overshoots by a few words from costing an API call.
"""

MAX_BULLET_CHARACTERS = 118
"""What a bullet line holds, and the length a bullet is held to.

Removing this bound was a mistake worth recording. The reasoning was that a length is
a layout constraint and a layout constraint should never cost an API call, which is
true as far as it goes. What it missed is that the bound is also the only thing that
makes the model comply: with a cap of 130 in force a run came back with a median
bullet of 113 and twelve of eighteen fitting a line; with the cap removed the next
run came back at 140 and none of them fitting, condensed itself, and still rendered
at the smallest type the renderer allows.

So the bound stays, and the slack lives between here and
:data:`TARGET_BULLET_CHARACTERS` instead. Multi-sentence overshoots are shortened
before this is consulted, which costs nothing and catches the easy cases.
"""

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


_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


def _drop_trailing_sentences(text: str, budget: int, floor: int) -> str:
    """Shorten ``text`` toward ``budget`` by dropping whole trailing sentences.

    Whole sentences only. Cutting at a character count is how a bullet ends up
    reading "... 25,000+ transactions/day via.", which states less than the long
    version and looks like a bug rather than an edit.

    Most bullets are a single sentence, so usually there is nothing to drop and the
    text comes back unchanged. That is the intended outcome: it wraps onto a second
    line, the renderer tightens, and the run continues. Nothing here rejects.
    """
    if len(text) <= budget:
        return text
    sentences = _SENTENCE_BREAK.split(text)
    while len(sentences) > 1:
        shorter = " ".join(sentences[:-1])
        if len(shorter) < floor:
            break
        sentences = sentences[:-1]
        if len(shorter) <= budget:
            return shorter
    kept = " ".join(sentences)
    return kept if len(kept) >= floor else text


def _normalize_summary(value: str) -> str:
    """Collapse the summary to one line and hold it to a readable length.

    Whitespace is fixed silently, since a stray newline is not worth an API call.
    Length is enforced by rejection, so the model rewrites to fit instead of having
    a sentence cut off mid-clause.
    """
    collapsed = soften_dashes(" ".join(value.split()))
    if not collapsed:
        raise ValueError("must not be empty")
    if len(collapsed) < MIN_SUMMARY_CHARACTERS:
        raise ValueError(
            f"must be at least {MIN_SUMMARY_CHARACTERS} characters, got {len(collapsed)}"
        )
    # Same rule as the bullets: shorten cleanly if possible, never reject. A summary
    # rejected at 308 characters against a limit of 300 has ended a run before.
    return _drop_trailing_sentences(collapsed, MAX_SUMMARY_CHARACTERS, MIN_SUMMARY_CHARACTERS)


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
    collapsed = soften_dashes(" ".join(_LIST_MARKER.sub("", value).split()))
    if not collapsed:
        raise ValueError("must not be empty")
    if len(collapsed) < MIN_BULLET_CHARACTERS:
        raise ValueError(
            f"must be at least {MIN_BULLET_CHARACTERS} characters, got {len(collapsed)}"
        )
    # Length is a layout constraint. Shorten it here if that can be done cleanly,
    # otherwise let it wrap: a second line costs the page a line, and a rejection
    # costs an API call and can cost the whole run.
    collapsed = _drop_trailing_sentences(collapsed, MAX_BULLET_CHARACTERS, MIN_BULLET_CHARACTERS)
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
    collapsed = soften_dashes(" ".join(value.split()))
    if len(collapsed) < MIN_SUBTITLE_CHARACTERS:
        raise ValueError(
            f"must be at least {MIN_SUBTITLE_CHARACTERS} characters, got {len(collapsed)}"
        )
    # Not rejected either. An over-long subtitle is absorbed by the heading trim,
    # which drops technologies until the line fits, and a heading that still will not
    # fit wraps. Both cost less than losing the run.
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

    @field_validator("technologies")
    @classmethod
    def _fit_heading(cls, technologies: tuple[str, ...], info: ValidationInfo) -> tuple[str, ...]:
        """Drop trailing technologies until the title line fits.

        Sanitized, never rejected. Whether a heading wraps is a layout question with a
        deterministic answer — the technologies are in priority order, so the last one
        is the cheapest thing on the line — and the house rule is that an API call is
        spent on content errors only. Rejecting it instead cost a user a whole run:
        the model cannot tell which of two fields to shorten, so it rewrites both,
        breaks something else, and the retry budget goes on a heading that would have
        rendered fine.

        A heading still over budget at :data:`MIN_PROJECT_TECHNOLOGIES` is left alone
        and wraps to a second line. That costs one line; failing the run costs every
        token spent on it.

        A field validator rather than a model validator because this has to *change* a
        value: pydantic ignores a model validator that returns anything but ``self``
        when the model is built through ``__init__``, so trimming there silently did
        nothing. ``subtitle`` is declared first, so it is already validated and
        available on ``info.data``.
        """
        subtitle = info.data.get("subtitle")
        if not isinstance(subtitle, str):
            return technologies  # subtitle failed its own validation; report that

        trimmed = list(technologies)
        while (
            len(trimmed) > MIN_PROJECT_TECHNOLOGIES
            and len(subtitle) + len(", ".join(trimmed)) > MAX_PROJECT_HEADING_CHARACTERS
        ):
            trimmed.pop()
        return tuple(trimmed)

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


def _check_skills(values: tuple[str, ...]) -> tuple[str, ...]:
    """Hold the section to a selected, readable number of skills.

    Duplicates are sanitized away by :func:`clean_items` before counting, so a model
    that repeats a term is not charged an API call for it. The count is checked after
    that, because it is a content decision: too few or too many means the model chose
    badly, and choosing again is what the repair loop is for.
    """
    cleaned = clean_items(values)
    if not MIN_SKILL_COUNT <= len(cleaned) <= MAX_SKILL_COUNT:
        raise ValueError(
            f"must list between {MIN_SKILL_COUNT} and {MAX_SKILL_COUNT} skills, got {len(cleaned)}"
        )
    return cleaned


class GeneratedSkills(BaseModel):
    """The skills section: one selected, ordered list.

    Deliberately flat. Categories were tried and removed: a grouped section invites
    the model to fill every group it invents, which turns selection into coverage and
    buries what the candidate is being presented as. One line, in priority order, is
    also how the section is actually read.
    """

    model_config = GENERATED_MODEL_CONFIG

    skills: Annotated[tuple[str, ...], AfterValidator(_check_skills)]

    @property
    def skill_count(self) -> int:
        """How many skills the section carries."""
        return len(self.skills)


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
    skills: tuple[str, ...]
    achievements: tuple[str, ...] = ()

    @property
    def all_bullets(self) -> tuple[str, ...]:
        """Every bullet on the resume, for whole-document checks."""
        from_experiences = [b for experience in self.experiences for b in experience.bullets]
        from_projects = [b for project in self.projects for b in project.bullets]
        return tuple(from_experiences + from_projects)
