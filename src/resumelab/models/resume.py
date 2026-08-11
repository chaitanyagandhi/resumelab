"""Models for the generated resume.

These are LLM structured-output targets; see :mod:`resumelab.models.common` for what
that constrains. Length limits are enforced by validators rather than JSON-Schema
keywords, which means an over-long response is handed back to the model to shorten
rather than being truncated mid-sentence.
"""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import AfterValidator, BaseModel, model_validator

from resumelab.models.common import GENERATED_MODEL_CONFIG

REQUIRED_EXPERIENCE_BULLET_COUNT = 3
"""Bullets emitted per role. A fixed count keeps runs comparable."""

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
