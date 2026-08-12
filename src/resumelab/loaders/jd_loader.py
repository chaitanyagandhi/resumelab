"""Resolution of a job description from a file path, inline text, or a URL.

Exactly one input source must be supplied. Two together, or none, is a usage error
reported before any work begins.

Whichever source is used, what comes out is the same thing: normalized, bounded,
untrusted text. A fetched posting is checked for instruction-like content exactly as
a pasted one is — arriving over the network makes it no more and no less trustworthy
than a file a researcher saved from the same page.
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx
from pydantic import ValidationError

from resumelab.exceptions import JDAnalysisError
from resumelab.fetching import fetch_posting
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
    url: str | None = None,
    http_client: httpx.Client | None = None,
) -> JobDescription:
    """Build a :class:`JobDescription` from whichever input was supplied.

    Args:
        path: Path to a UTF-8 job description file.
        text: Job description supplied directly, e.g. from ``--jd-text``.
        url: Link to a posting, fetched and reduced to text.
        http_client: HTTP client used when fetching. Tests supply one wired to a
            mock transport; a real run leaves it unset.

    Returns:
        The normalized, validated job description.

    Raises:
        JDAnalysisError: If the inputs are not exactly one source, if the file cannot
            be read, if the posting cannot be fetched, or if the resulting text fails
            validation. :class:`~resumelab.exceptions.JDFetchError` is a subclass, so
            a caller that handles this handles a failed fetch too.
    """
    supplied = {"--jd": path, "--jd-text": text, "--jd-url": url}
    given = [flag for flag, value in supplied.items() if value is not None]
    if len(given) > 1:
        raise JDAnalysisError(
            f"Provide one job description source, not {len(given)} "
            f"({', '.join(given)} are mutually exclusive)."
        )
    if not given:
        raise JDAnalysisError(
            "A job description is required: pass --jd PATH, --jd-text TEXT, or --jd-url URL."
        )

    raw, source, label = _read_source(path, text, url, http_client)

    if not normalize_text(raw):
        raise JDAnalysisError(f"Job description is empty: {_describe_origin(source, path, url)}")

    try:
        job_description = JobDescription(
            text=raw,
            source=source,
            source_path=path,
            source_url=url,
            source_label=label,
        )
    except ValidationError as exc:
        message = describe_validation_error(
            exc,
            f"Invalid job description: {_describe_origin(source, path, url)}",
        )
        raise JDAnalysisError(message) from exc

    logger.info(
        "loaded job description source=%s characters=%d",
        source.value,
        job_description.character_count,
    )
    _warn_about_instruction_like_content(job_description)
    return job_description


def _read_source(
    path: Path | None,
    text: str | None,
    url: str | None,
    http_client: httpx.Client | None,
) -> tuple[str, JobDescriptionSource, str | None]:
    """Retrieve the raw posting, and say where it came from and what to call it.

    Reached only after the caller has been shown to have supplied exactly one source,
    so inline text is what remains once a path and a URL are ruled out.
    """
    if path is not None:
        raw = read_text_file(path, subject="Job description", error_type=JDAnalysisError)
        return raw, JobDescriptionSource.FILE, None
    if url is not None:
        posting = fetch_posting(url, client=http_client)
        return posting.text, JobDescriptionSource.URL, posting.label
    # An empty string is a supplied source, and is reported as an empty posting
    # rather than as a missing argument.
    return text or "", JobDescriptionSource.TEXT, None


def _warn_about_instruction_like_content(job_description: JobDescription) -> None:
    """Note when a posting appears to address the model.

    The content is still analyzed, fenced as data. This only makes such a run
    identifiable afterwards, which matters when the output looks unusual — and
    matters more for a fetched posting, where nobody read the text before it was sent.
    """
    markers = injection_markers(job_description.text)
    if markers:
        logger.warning(
            "job description contains instruction-like text; it will be analyzed as "
            "data, never obeyed: %s",
            "; ".join(repr(marker) for marker in markers),
        )


def _describe_origin(source: JobDescriptionSource, path: Path | None, url: str | None) -> str:
    """Name the input in error messages without quoting its contents."""
    if source is JobDescriptionSource.FILE:
        return str(path)
    if source is JobDescriptionSource.URL:
        return str(url)
    return "inline text"
