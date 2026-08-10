"""Schema for a target job description.

A job description is **untrusted data**. It is supplied by a researcher from an
arbitrary posting and may contain text that looks like instructions to a language
model. This model's job is to carry that text safely and verbatim; the prompt layer
is responsible for presenting it to the LLM as data rather than as instructions.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from resumelab.utils.text import normalize_text

MIN_JOB_DESCRIPTION_CHARACTERS = 50
"""Defensive floor. Rejects an accidental empty paste, not short postings."""

MAX_JOB_DESCRIPTION_CHARACTERS = 50_000
"""Roughly 12k tokens. Guards against an entire careers page being piped in."""


def _normalize(value: object) -> object:
    return normalize_text(value) if isinstance(value, str) else value


JobDescriptionText = Annotated[
    str,
    BeforeValidator(_normalize),
    Field(
        min_length=MIN_JOB_DESCRIPTION_CHARACTERS,
        max_length=MAX_JOB_DESCRIPTION_CHARACTERS,
    ),
]


class JobDescriptionSource(StrEnum):
    """Where the job description text came from, recorded for reproducibility."""

    FILE = "file"
    TEXT = "text"


class JobDescription(BaseModel):
    """A target job description, normalized and length-checked.

    Frozen, so the exact text analyzed is the text persisted into the run's
    artifacts.
    """

    model_config = ConfigDict(frozen=True)

    text: JobDescriptionText
    source: JobDescriptionSource
    source_path: Path | None = None

    @model_validator(mode="after")
    def _check_source_path_matches_source(self) -> JobDescription:
        """Keep the provenance record honest."""
        if self.source is JobDescriptionSource.FILE and self.source_path is None:
            raise ValueError("source_path is required when the source is a file")
        if self.source is JobDescriptionSource.TEXT and self.source_path is not None:
            raise ValueError("source_path must be omitted when the source is inline text")
        return self

    @property
    def character_count(self) -> int:
        """Length of the normalized text, logged instead of the text itself."""
        return len(self.text)
