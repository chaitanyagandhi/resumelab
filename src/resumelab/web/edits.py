"""Hand edits to a generated resume, and the PDF they produce.

An edit never touches what the run produced. ``generated_resume.json`` and
``resume.pdf`` are the research record: the whole design compares a source profile
against a posting against *what the model wrote*, and a file that quietly became
half-model and half-human would make every later comparison a lie. Edits land beside
them under their own names, so a run directory says plainly which is which.

Nothing here calls a model. Re-rendering is deterministic and free, which is what
makes it safe to do on every keystroke the editor settles on. In particular an edited
resume is never condensed: condensing spends an API call, and a page count is
something to report to whoever is typing, not something to fix behind them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from resumelab.exceptions import PDFRenderingError
from resumelab.models.resume import GeneratedResume
from resumelab.rendering import RenderOptions, RenderResult, render_resume

logger = logging.getLogger(__name__)

EDITED_RESUME_FILE = "edited_resume.json"
EDITED_PDF_FILE = "edited_resume.pdf"

_PENDING_PDF_FILE = ".edited_resume.pdf.tmp"
"""Where a re-render is built before it replaces the file the browser is reading."""


@dataclass(frozen=True, slots=True)
class EditOutcome:
    """What the edit did to the page.

    Reported rather than acted on. A human writing their own resume is allowed a
    third page; they are not allowed to be surprised by one.
    """

    page_count: int
    scale: float
    fits_on_one_page: bool


def save_edit(
    directory: Path,
    resume: GeneratedResume,
    *,
    options: RenderOptions | None = None,
) -> EditOutcome:
    """Record an edited resume in ``directory`` and render it.

    Args:
        directory: The run directory, already checked to be one.
        resume: The edited content.
        options: What to show and in what order.

    Returns:
        How the edit came out on the page.

    Raises:
        PDFRenderingError: If the edit could not be laid out or written.
    """
    rendered = _render_atomically(resume, directory, options=options)
    _write_json(directory / EDITED_RESUME_FILE, resume)

    logger.info(
        "saved edit directory=%s pages=%d scale=%.3f",
        directory,
        rendered.page_count,
        rendered.scale,
    )
    return EditOutcome(
        page_count=rendered.page_count,
        scale=rendered.scale,
        fits_on_one_page=rendered.fits_on_one_page,
    )


def _render_atomically(
    resume: GeneratedResume,
    directory: Path,
    *,
    options: RenderOptions | None,
) -> RenderResult:
    """Render to a scratch file, then move it into place in one step.

    The editor re-renders as fast as someone types, so the browser is very likely to
    ask for the PDF while the next one is being written. Replacing a complete file is
    atomic; writing over a file that is being read is a torn page.
    """
    pending = directory / _PENDING_PDF_FILE
    try:
        rendered = render_resume(resume, pending, options=options)
        pending.replace(directory / EDITED_PDF_FILE)
    except OSError as exc:
        raise PDFRenderingError(f"Could not save the edited resume: {exc}") from exc
    finally:
        pending.unlink(missing_ok=True)
    return rendered


def _write_json(path: Path, resume: GeneratedResume) -> None:
    try:
        path.write_text(resume.model_dump_json(indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        raise PDFRenderingError(f"Could not save the edited resume: {exc}") from exc
