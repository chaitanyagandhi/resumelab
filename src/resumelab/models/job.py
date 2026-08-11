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

from pydantic import AfterValidator, BaseModel, ConfigDict, model_validator

from resumelab.utils.text import normalize_text

MIN_JOB_DESCRIPTION_CHARACTERS = 50
"""Defensive floor. Rejects an accidental empty paste, not short postings."""

MAX_JOB_DESCRIPTION_CHARACTERS = 50_000
"""Roughly 12k tokens. Guards against an entire careers page being piped in."""


def _normalize_and_bound(value: str) -> str:
    """Normalize the posting, then hold it to a usable length.

    Length is judged after normalization, so a posting made only of invisible
    characters is reported as empty rather than as mysteriously too short.
    """
    normalized = normalize_text(value)
    if not normalized:
        raise ValueError("must not be empty")
    if len(normalized) < MIN_JOB_DESCRIPTION_CHARACTERS:
        raise ValueError(
            f"must be at least {MIN_JOB_DESCRIPTION_CHARACTERS} characters, got {len(normalized)}"
        )
    if len(normalized) > MAX_JOB_DESCRIPTION_CHARACTERS:
        raise ValueError(
            f"must be at most {MAX_JOB_DESCRIPTION_CHARACTERS} characters, got {len(normalized)}"
        )
    return normalized


JobDescriptionText = Annotated[str, AfterValidator(_normalize_and_bound)]


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
