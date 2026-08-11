"""Resolution of a job description from a file path or inline text.

Exactly one input source must be supplied. Both together, or neither, is a usage
error reported before any work begins.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import ValidationError

from resumelab.exceptions import JDAnalysisError
from resumelab.llm.prompts import injection_markers
from resumelab.models.job import JobDescription, JobDescriptionSource
from resumelab.utils.errors import describe_validation_error
from resumelab.utils.files import read_text_file
from resumelab.utils.text import normalize_text

logger = logging.getLogger(__name__)


def load_job_description(
    *,
    path: Path | None = None,
    text: str | None = None,
) -> JobDescription:
    """Build a :class:`JobDescription` from whichever input was supplied.

    Args:
        path: Path to a UTF-8 job description file.
        text: Job description supplied directly, e.g. from ``--jd-text``.

    Returns:
        The normalized, validated job description.

    Raises:
        JDAnalysisError: If both or neither input is supplied, if the file cannot be
            read, or if the resulting text fails validation.
    """
    if path is not None and text is not None:
        raise JDAnalysisError(
            "Provide a job description file or inline text, not both "
            "(--jd and --jd-text are mutually exclusive)."
        )
    if path is not None:
        raw = read_text_file(path, subject="Job description", error_type=JDAnalysisError)
        source = JobDescriptionSource.FILE
    elif text is not None:
        raw = text
        source = JobDescriptionSource.TEXT
    else:
        raise JDAnalysisError("A job description is required: pass --jd PATH or --jd-text TEXT.")

    if not normalize_text(raw):
        raise JDAnalysisError(f"Job description is empty: {_describe_origin(source, path)}")

    try:
        job_description = JobDescription(text=raw, source=source, source_path=path)
    except ValidationError as exc:
        message = describe_validation_error(
            exc,
            f"Invalid job description: {_describe_origin(source, path)}",
        )
        raise JDAnalysisError(message) from exc

    logger.info(
        "loaded job description source=%s characters=%d",
        source.value,
        job_description.character_count,
    )
    _warn_about_instruction_like_content(job_description)
    return job_description


def _warn_about_instruction_like_content(job_description: JobDescription) -> None:
    """Note when a posting appears to address the model.

    The content is still analyzed, fenced as data. This only makes such a run
    identifiable afterwards, which matters when the output looks unusual.
    """
    markers = injection_markers(job_description.text)
    if markers:
        logger.warning(
            "job description contains instruction-like text; it will be analyzed as "
            "data, never obeyed: %s",
            "; ".join(repr(marker) for marker in markers),
        )


def _describe_origin(source: JobDescriptionSource, path: Path | None) -> str:
    """Name the input in error messages without quoting its contents."""
    return str(path) if source is JobDescriptionSource.FILE else "inline text"
