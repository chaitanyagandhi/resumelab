"""Models for the generated resume.

These are LLM structured-output targets; see :mod:`resumelab.models.common` for what
that constrains. Length limits are enforced by validators rather than JSON-Schema
keywords, which means an over-long response is handed back to the model to shorten
rather than being truncated mid-sentence.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import AfterValidator, BaseModel

from resumelab.models.common import GENERATED_MODEL_CONFIG

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
