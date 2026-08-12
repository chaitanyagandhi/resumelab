"""Rendering of a generated resume to PDF.

The model never produces the document; it produces structured content, and this
module owns every layout decision. That separation is what makes two runs visually
comparable — any difference on the page is a difference in the content.

The layout is built for machine reading as well as human reading. There are no
images, no icons, and no multi-column tables: everything is a single linear flow of
real text, so extraction returns the resume in reading order. Fields that a
conventional resume right-aligns are instead placed inline, separated by a middle
dot, because the only clean way to right-align them is a table, and tables are what
break extraction.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Flowable, HRFlowable, Paragraph, SimpleDocTemplate
from reportlab.platypus.doctemplate import LayoutError

from resumelab.exceptions import PDFRenderingError
from resumelab.models.candidate import Education, PersonalDetails
from resumelab.models.resume import (
    GeneratedExperience,
    GeneratedProject,
    GeneratedResume,
)
from resumelab.rendering import styles

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RenderResult:
    """What rendering produced, and what it had to do to get there."""

    path: Path
    scale: float
    """The layout scale used. 1.0 means nothing had to be tightened."""

    page_count: int

    @property
    def fits_on_one_page(self) -> bool:
        return self.page_count == 1

    @property
    def was_tightened(self) -> bool:
        return self.scale < 1.0


def render_resume(resume: GeneratedResume, output_path: Path) -> RenderResult:
    """Render ``resume`` to a PDF at ``output_path``, fitting one page if it can.

    Progressively tighter layouts are tried until the content fits a single page.
    Each attempt is built in memory and only the chosen one is written, so the file
    on disk is never a discarded draft.

    Content that still overflows at the tightest permitted layout is written at that
    layout and reported as overflowing. Shrinking further would trade a readable
    two-page resume for an unreadable one-page resume, which is the wrong trade; the
    caller decides whether to condense the content instead.

    Args:
        resume: The validated resume to draw.
        output_path: Where to write the PDF. Parent directories are created.

    Returns:
        A :class:`RenderResult` describing the layout that was used.

    Raises:
        PDFRenderingError: If the document could not be built or written.
    """
    logger.info("rendering PDF output=%s", output_path)

    payload, page_count, scale = b"", 0, styles.LAYOUT_SCALES[0]
    for candidate in styles.LAYOUT_SCALES:
        scale = candidate
        payload, page_count = _build_document(resume, candidate)
        if page_count == 1:
            break
        logger.debug("layout overflowed pages=%d scale=%.3f", page_count, candidate)

    if page_count > 1:
        logger.warning(
            "resume does not fit one page at the tightest permitted layout "
            "pages=%d scale=%.3f: condense the content rather than shrinking it further",
            page_count,
            scale,
        )
    elif scale < 1.0:
        logger.info("tightened layout to fit one page scale=%.3f", scale)

    _write(payload, output_path)
    logger.info(
        "rendered PDF output=%s bytes=%d pages=%d scale=%.3f",
        output_path,
        output_path.stat().st_size,
        page_count,
        scale,
    )
    return RenderResult(path=output_path, scale=scale, page_count=page_count)


def _build_document(resume: GeneratedResume, scale: float) -> tuple[bytes, int]:
    """Build the document in memory at ``scale``, returning its bytes and page count."""
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=styles.PAGE_SIZE,
        leftMargin=styles.MARGIN_HORIZONTAL,
        rightMargin=styles.MARGIN_HORIZONTAL,
        topMargin=styles.MARGIN_TOP,
        bottomMargin=styles.MARGIN_BOTTOM,
        title=f"{resume.personal.name} — Resume",
        author=resume.personal.name,
    )
    try:
        document.build(list(_build_story(resume, styles.build_stylesheet(scale))))
    except LayoutError as exc:
        raise PDFRenderingError(
            f"The resume could not be laid out at scale {scale}: {exc}"
        ) from exc
    return buffer.getvalue(), document.page


def _write(payload: bytes, output_path: Path) -> None:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(payload)
    except OSError as exc:
        raise PDFRenderingError(f"Could not write the resume to {output_path}: {exc}") from exc


def _build_story(
    resume: GeneratedResume,
    stylesheet: dict[str, ParagraphStyle],
) -> Iterable[Flowable]:
    """Produce the flowables for the whole document, in reading order."""
    yield from _header(resume.personal, stylesheet)
    yield from _section(
        "Summary", [Paragraph(_text(resume.summary), stylesheet["body"])], stylesheet
    )
    yield from _section("Education", list(_education(resume.education, stylesheet)), stylesheet)
    yield from _section(
        "Experience", list(_experiences(resume.experiences, stylesheet)), stylesheet
    )
    yield from _section("Projects", list(_projects(resume.projects, stylesheet)), stylesheet)
    yield from _section("Skills", list(_skills(resume.skills, stylesheet)), stylesheet)
    if resume.achievements:
        yield from _section(
            "Achievements",
            [_bullet(text, stylesheet) for text in resume.achievements],
            stylesheet,
        )


def _header(
    personal: PersonalDetails,
    stylesheet: dict[str, ParagraphStyle],
) -> Iterable[Flowable]:
    """Name and contact details, the only centered part of the document."""
    yield Paragraph(_text(personal.name), stylesheet["name"])
    contact = _contact_line(personal)
    if contact:
        yield Paragraph(contact, stylesheet["contact"])


def _contact_line(personal: PersonalDetails) -> str:
    """Build the contact line, linking the parts that are addressable."""
    parts = [
        _text(personal.location) if personal.location else "",
        _link(f"mailto:{personal.email}", personal.email) if personal.email else "",
        _text(personal.phone) if personal.phone else "",
        _link(_url(personal.linkedin), personal.linkedin) if personal.linkedin else "",
        _link(_url(personal.github), personal.github) if personal.github else "",
    ]
    return styles.SEPARATOR.join(part for part in parts if part)


def _education(
    entries: Iterable[Education],
    stylesheet: dict[str, ParagraphStyle],
) -> Iterable[Flowable]:
    for entry in entries:
        yield Paragraph(
            _joined(
                _bold(entry.institution), entry.location, _dates(entry.start_date, entry.end_date)
            ),
            stylesheet["entry"],
        )
        qualification = " ".join(part for part in (entry.degree, entry.field) if part)
        gpa = f"GPA: {entry.gpa}" if entry.gpa else ""
        detail = _joined(qualification, gpa)
        if detail:
            yield Paragraph(detail, stylesheet["detail"])


def _experiences(
    entries: Iterable[GeneratedExperience],
    stylesheet: dict[str, ParagraphStyle],
) -> Iterable[Flowable]:
    for entry in entries:
        yield Paragraph(
            _joined(
                _bold(entry.company),
                entry.title,
                entry.location,
                _dates(entry.start_date, entry.end_date),
            ),
            stylesheet["entry"],
        )
        for bullet in entry.bullets:
            yield _bullet(bullet, stylesheet)


def _projects(
    entries: Iterable[GeneratedProject],
    stylesheet: dict[str, ParagraphStyle],
) -> Iterable[Flowable]:
    for entry in entries:
        heading = f"{_bold(entry.name)} — {_text(entry.subtitle)}"
        yield Paragraph(_joined(heading, entry.date), stylesheet["entry"])
        if entry.technologies:
            yield Paragraph(_text(", ".join(entry.technologies)), stylesheet["detail"])
        for bullet in entry.bullets:
            yield _bullet(bullet, stylesheet)


def _skills(
    skills: Iterable[str],
    stylesheet: dict[str, ParagraphStyle],
) -> Iterable[Flowable]:
    """Render the section as one wrapped line, in the order the model chose.

    The same middle dot that separates a role's company, location, and dates, so the
    document has one separator rather than one per section.
    """
    yield Paragraph(styles.SEPARATOR.join(_text(skill) for skill in skills), stylesheet["body"])


def _section(
    title: str,
    body: list[Flowable],
    stylesheet: dict[str, ParagraphStyle],
) -> Iterable[Flowable]:
    """A titled section with a rule under its heading."""
    yield Paragraph(_text(title.upper()), stylesheet["section"])
    yield HRFlowable(
        width="100%",
        thickness=styles.RULE_THICKNESS,
        color=styles.RULE_INK,
        spaceBefore=styles.RULE_SPACE_BEFORE,
        spaceAfter=styles.RULE_SPACE_AFTER,
    )
    yield from body


def _bullet(text: str, stylesheet: dict[str, ParagraphStyle]) -> Paragraph:
    return Paragraph(_text(text), stylesheet["bullet"], bulletText=styles.BULLET_CHARACTER)


def _joined(*parts: str | None) -> str:
    """Join present fields with the separator, escaping any that are raw text."""
    return styles.SEPARATOR.join(
        part if _is_markup(part) else _text(part) for part in parts if part
    )


def _is_markup(part: str) -> bool:
    """Whether ``part`` has already been escaped and wrapped in tags."""
    return part.startswith("<")


def _dates(start: str | None, end: str | None) -> str:
    """Render a date range, tolerating either end being absent."""
    if start and end:
        return f"{_text(start)}{styles.DATE_RANGE_SEPARATOR}{_text(end)}"
    return _text(start or end or "")


def _bold(text: str) -> str:
    return f"<b>{_text(text)}</b>"


def _link(href: str, label: str) -> str:
    return f'<a href="{_text(href)}" color="{styles.LINK_INK}">{_text(label)}</a>'


def _url(value: str) -> str:
    """Make a profile reference clickable without assuming it carries a scheme."""
    return value if value.startswith(("http://", "https://")) else f"https://{value}"


def _text(value: str) -> str:
    """Escape text for ReportLab's inline markup.

    Resume content routinely contains ``&`` and comparison operators, which would
    otherwise be read as markup and either vanish or fail the build.
    """
    return escape(value)
