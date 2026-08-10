"""Resolution of a job description from a file path or inline text.

Exactly one input source must be supplied. Both together, or neither, is a usage
error reported before any work begins.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import ValidationError

from resumelab.exceptions import JDAnalysisError
from resumelab.models.job import JobDescription, JobDescriptionSource
from resumelab.utils.errors import describe_validation_error
from resumelab.utils.files import read_text_file

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

    if not raw.strip():
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
    return job_description


def _describe_origin(source: JobDescriptionSource, path: Path | None) -> str:
    """Name the input in error messages without quoting its contents."""
    return str(path) if source is JobDescriptionSource.FILE else "inline text"
