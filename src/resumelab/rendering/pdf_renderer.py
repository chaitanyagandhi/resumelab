"""Rendering of a generated resume to PDF.

The model never produces the document; it produces structured content, and this
module owns every layout decision. That separation is what makes two runs visually
comparable — any difference on the page is a difference in the content.

The layout is built for machine reading as well as human reading. There are no
images and no icons, and the document is a single linear flow of real text, so
extraction returns the resume in reading order.

Dates are set flush right, which needs a two-cell row. That is the one place a table
appears, and it is bounded deliberately: one line, two cells, no rules, each cell a
single paragraph. What damages extraction is a multi-column *layout*, where a parser
must guess how columns interleave. Here there is nothing to guess — the row extracts
as heading, then date, then the bullets beneath it, which is the reading order. A
test asserts exactly that, because it is the property the whole format exists to
protect.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import Flowable, HRFlowable, Paragraph, SimpleDocTemplate, Table
from reportlab.platypus.doctemplate import LayoutError

from resumelab.exceptions import PDFRenderingError
from resumelab.models.candidate import Education, PersonalDetails
from resumelab.models.resume import (
    GeneratedExperience,
    GeneratedProject,
    GeneratedResume,
)
from resumelab.rendering import styles
from resumelab.rendering.options import DEFAULT_RENDER_OPTIONS, RenderOptions, ResumeSection

logger = logging.getLogger(__name__)

DEGREE_ABBREVIATIONS = {
    "bachelor of arts": "BA",
    "bachelor of engineering": "BE",
    "bachelor of science": "BS",
    "bachelor of technology": "BTech",
    "master of arts": "MA",
    "master of business administration": "MBA",
    "master of engineering": "MEng",
    "master of science": "MS",
    "doctor of philosophy": "PhD",
}
"""How a degree is written on a resume, keyed by how it is written on a transcript.

Display only, and matched on the whole field rather than by pattern: a partial match
would turn an unfamiliar degree into a plausible-looking wrong one, and the profile
is the record of what was actually awarded.
"""


@dataclass(frozen=True, slots=True)
class RenderResult:
    """What rendering produced, and what it had to do to get there."""

    path: Path
    scale: float
    """The layout scale used. 1.0 means nothing had to be tightened."""

    page_count: int

    spacing: float = 1.0
    """How far the vertical gaps were opened to fill the page. 1.0 means not at all."""

    @property
    def fits_on_one_page(self) -> bool:
        return self.page_count == 1

    @property
    def was_tightened(self) -> bool:
        return self.scale < 1.0


def render_resume(
    resume: GeneratedResume,
    output_path: Path,
    *,
    options: RenderOptions | None = None,
) -> RenderResult:
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
        options: What to show and in what order. Defaults to everything, in the
            default order, which is what the pipeline renders.

    Returns:
        A :class:`RenderResult` describing the layout that was used.

    Raises:
        PDFRenderingError: If the document could not be built or written.
    """
    chosen = options if options is not None else DEFAULT_RENDER_OPTIONS
    logger.info("rendering PDF output=%s", output_path)

    payload, page_count, scale = b"", 0, styles.LAYOUT_SCALES[0]
    for candidate in styles.LAYOUT_SCALES:
        scale = candidate
        payload, page_count = _build_document(resume, candidate, chosen)
        if page_count == 1:
            break
        logger.debug("layout overflowed pages=%d scale=%.3f", page_count, candidate)

    spacing = 1.0
    if page_count == 1:
        payload, spacing = _fill_the_page(resume, scale, chosen, payload)

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
        "rendered PDF output=%s bytes=%d pages=%d scale=%.3f spacing=%.2f",
        output_path,
        output_path.stat().st_size,
        page_count,
        scale,
        spacing,
    )
    return RenderResult(path=output_path, scale=scale, page_count=page_count, spacing=spacing)


def _fill_the_page(
    resume: GeneratedResume,
    scale: float,
    options: RenderOptions,
    tight: bytes,
) -> tuple[bytes, float]:
    """Open the vertical gaps until the content reaches the foot of the page.

    Content that stops two thirds of the way down reads as unfinished rather than
    concise. The type size is already settled by the time this runs and is not
    touched: it decides where every line wraps, so raising it can add lines and cost
    a page. A wider gap only ever moves content down, so this search cannot surprise.

    Falls back to the layout it was given, which already fits.
    """
    for spacing in styles.SPACING_SCALES:
        payload, page_count = _build_document(resume, scale, options, spacing=spacing)
        if page_count == 1:
            logger.debug("opened the page out spacing=%.2f", spacing)
            return payload, spacing
    return tight, 1.0


def _build_document(
    resume: GeneratedResume,
    scale: float,
    options: RenderOptions,
    spacing: float = 1.0,
) -> tuple[bytes, int]:
    """Build the document in memory at ``scale``, returning its bytes and page count."""
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=styles.PAGE_SIZE,
        leftMargin=styles.MARGIN_HORIZONTAL,
        rightMargin=styles.MARGIN_HORIZONTAL,
        topMargin=styles.MARGIN_TOP,
        bottomMargin=styles.MARGIN_BOTTOM,
        # No em dash: this is document metadata a reader can surface in a PDF viewer.
        title=f"{resume.personal.name} Resume",
        author=resume.personal.name,
    )
    try:
        document.build(list(_build_story(resume, styles.build_stylesheet(scale, spacing), options)))
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
    options: RenderOptions,
) -> Iterable[Flowable]:
    """Produce the flowables for the whole document, in reading order.

    The header opens the document, with the summary directly beneath the name where
    it is read or skipped in one glance. Below it the four body sections are drawn in
    whatever order was asked for; the dispatch is a lookup rather than a branch so
    that no order is a special case.

    Achievements are not drawn. The profile still records them and a run still keeps
    them, but a one-page resume spends a heading, a rule, and a line on what is
    almost always a restatement of something already above it.
    """
    yield from _header(resume.personal, stylesheet)
    if options.include_summary:
        yield from _section(
            "Summary", [Paragraph(_text(resume.summary), stylesheet["body"])], stylesheet
        )
    for section in options.section_order:
        yield from _SECTION_BUILDERS[section](resume, stylesheet, options)


def _education_section(
    resume: GeneratedResume,
    stylesheet: dict[str, ParagraphStyle],
    options: RenderOptions,
) -> Iterable[Flowable]:
    body = list(_education(resume.education, stylesheet, include_gpa=options.include_gpa))
    yield from _section("Education", body, stylesheet)


def _experience_section(
    resume: GeneratedResume,
    stylesheet: dict[str, ParagraphStyle],
    _options: RenderOptions,
) -> Iterable[Flowable]:
    yield from _section(
        "Experience", list(_experiences(resume.experiences, stylesheet)), stylesheet
    )


def _projects_section(
    resume: GeneratedResume,
    stylesheet: dict[str, ParagraphStyle],
    _options: RenderOptions,
) -> Iterable[Flowable]:
    yield from _section("Projects", list(_projects(resume.projects, stylesheet)), stylesheet)


def _skills_section(
    resume: GeneratedResume,
    stylesheet: dict[str, ParagraphStyle],
    _options: RenderOptions,
) -> Iterable[Flowable]:
    yield from _section("Skills", list(_skills(resume.skills, stylesheet)), stylesheet)


_SectionBuilder = Callable[
    [GeneratedResume, dict[str, ParagraphStyle], RenderOptions], Iterable[Flowable]
]

_SECTION_BUILDERS: dict[ResumeSection, _SectionBuilder] = {
    ResumeSection.EDUCATION: _education_section,
    ResumeSection.EXPERIENCE: _experience_section,
    ResumeSection.PROJECTS: _projects_section,
    ResumeSection.SKILLS: _skills_section,
}
"""Every section has an entry, which is what makes the order above total."""


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
        _link(_url(personal.linkedin), _profile_label(personal.linkedin))
        if personal.linkedin
        else "",
        _link(_url(personal.github), _profile_label(personal.github)) if personal.github else "",
    ]
    return styles.SEPARATOR.join(part for part in parts if part)


def _education(
    entries: Iterable[Education],
    stylesheet: dict[str, ParagraphStyle],
    *,
    include_gpa: bool,
) -> Iterable[Flowable]:
    """Three lines per degree: where it came from, what it was, what it covered.

    The institution leads and the qualification sits beneath it, which is the order
    the section had before the coursework line was added and the order it keeps.
    """
    for entry in entries:
        yield _flush_right_row(
            _institution(entry),
            _date_text(entry.start_date, entry.end_date),
            stylesheet,
        )
        # Withheld and absent are the same thing on the page: the line is set without
        # a right-hand column rather than with an empty one.
        gpa = f"GPA: {entry.gpa}" if include_gpa and entry.gpa else ""
        qualification = _qualification(entry)
        if qualification or gpa:
            yield _flush_right_row(
                _text(qualification),
                gpa,
                stylesheet,
                leading_style="detail",
                trailing_style="detail_right",
            )
        if entry.coursework:
            yield Paragraph(_text(f"Coursework: {', '.join(entry.coursework)}"), stylesheet["note"])


def _qualification(entry: Education) -> str:
    """The degree as a reader scans for it: ``MS Computer Science``.

    Abbreviated because the long form is three words of boilerplate in front of the
    one word that carries meaning, and because the heading has to share its line with
    a date range. A degree with no known abbreviation is left as written rather than
    shortened by guesswork.
    """
    degree = DEGREE_ABBREVIATIONS.get(entry.degree.strip().lower(), entry.degree)
    return " ".join(part for part in (degree, entry.field) if part)


def _institution(entry: Education) -> str:
    """The institution in bold, with its location alongside it in plain text."""
    parts = []
    if entry.institution:
        parts.append(_bold(entry.institution))
    if entry.location:
        parts.append(_text(entry.location))
    return styles.SEPARATOR.join(parts)


def _experiences(
    entries: Iterable[GeneratedExperience],
    stylesheet: dict[str, ParagraphStyle],
) -> Iterable[Flowable]:
    for entry in entries:
        # One line per role: the title carries the weight, the employer qualifies it.
        # Location is deliberately absent — on a one-page resume it is the field a
        # reader is least likely to need and the first worth spending on something else.
        heading = _bold(entry.title)
        if entry.company:
            heading = f"{heading}, <i>{_text(entry.company)}</i>"
        yield _flush_right_row(
            heading,
            _date_text(entry.start_date, entry.end_date),
            stylesheet,
        )
        for bullet in entry.bullets:
            yield _bullet(bullet, stylesheet)


def _projects(
    entries: Iterable[GeneratedProject],
    stylesheet: dict[str, ParagraphStyle],
) -> Iterable[Flowable]:
    for entry in entries:
        heading = f"{_bold(entry.name)}{styles.PROJECT_TITLE_SEPARATOR}{_text(entry.subtitle)}"
        if entry.technologies:
            # On the title line rather than beneath it: the stack qualifies the
            # subtitle, and a line of its own spends a line saying so.
            stack = _text(", ".join(entry.technologies))
            # The separator stays outside the italic run, which is what keeps the
            # bar upright instead of letting the oblique font lean it with the stack.
            heading += f"{styles.PROJECT_STACK_SEPARATOR}<i>{stack}</i>"
        yield _flush_right_row(heading, entry.date or "", stylesheet)
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


# ReportLab ships no type information, so its Flowable is an untyped base.
class TrackedHeading(Flowable):  # type: ignore[misc]
    """One line of text drawn with extra space between its letters.

    ReportLab's paragraph styles cannot letter-space, and doing it by putting spaces
    between the characters would make the heading extract as ``S U M M A R Y``. This
    sets the character spacing on the canvas instead, so the heading is still drawn
    and still extracted as a single word.
    """

    def __init__(self, text: str, style: ParagraphStyle, tracking: float) -> None:
        super().__init__()
        self.text = text
        self.style = style
        self.tracking = tracking
        self.spaceBefore = style.spaceBefore
        self.spaceAfter = style.spaceAfter

    def wrap(self, available_width: float, _available_height: float) -> tuple[float, float]:
        self.height = self.style.leading
        return available_width, self.height

    def draw(self) -> None:
        # Character spacing belongs to a text object rather than the canvas, which is
        # also what keeps it from leaking into everything drawn afterwards.
        # The baseline sits a font size below the top of the line, leaving the rest
        # of the leading beneath it as the paragraph styles do.
        text = self.canv.beginText(0, self.height - self.style.fontSize)
        text.setFont(self.style.fontName, self.style.fontSize)
        text.setFillColor(self.style.textColor)
        text.setCharSpace(self.tracking)
        text.textOut(self.text)
        self.canv.drawText(text)


def _section(
    title: str,
    body: list[Flowable],
    stylesheet: dict[str, ParagraphStyle],
) -> Iterable[Flowable]:
    """A titled section with a rule under its heading."""
    yield TrackedHeading(
        title.upper(), stylesheet["section"], _scaled_tracking(stylesheet["section"])
    )
    yield HRFlowable(
        width="100%",
        thickness=styles.RULE_THICKNESS,
        color=styles.RULE_INK,
        spaceBefore=styles.RULE_SPACE_BEFORE,
        spaceAfter=styles.RULE_SPACE_AFTER,
    )
    yield from body


def _scaled_tracking(section: ParagraphStyle) -> float:
    """Tracking scaled with the type, so the headings keep their proportions.

    Expressed against the heading's own size rather than the layout scale, which the
    stylesheet has already applied by the time this sees it.
    """
    return float(styles.SECTION_TRACKING * section.fontSize / styles.SECTION_FONT_SIZE)


def _bullet(text: str, stylesheet: dict[str, ParagraphStyle]) -> Paragraph:
    return Paragraph(_text(text), stylesheet["bullet"], bulletText=styles.BULLET_CHARACTER)


def _date_text(start: str | None, end: str | None) -> str:
    """A date range as plain text, tolerating either end being absent.

    Plain rather than escaped markup because it is measured before it is drawn, and a
    width taken from escaped text would be the width of ``&amp;`` rather than ``&``.
    """
    if start and end:
        return f"{start}{styles.DATE_RANGE_SEPARATOR}{end}"
    return start or end or ""


def _flush_right_row(
    leading: str,
    trailing: str,
    stylesheet: dict[str, ParagraphStyle],
    *,
    leading_style: str = "entry",
    trailing_style: str = "date",
) -> Flowable:
    """Set ``leading`` against ``trailing``, with the latter on the right margin.

    The right column is measured to its own content rather than fixed, so the trailing
    field lands on the margin at every layout scale and the left column keeps
    everything else. With nothing to trail, this is a plain paragraph — an empty
    column would still consume its width and pull the left side in for no reason.
    """
    left_style = stylesheet[leading_style]
    if not trailing:
        return Paragraph(leading, left_style)

    right_style = stylesheet[trailing_style]
    right_width = pdfmetrics.stringWidth(trailing, right_style.fontName, right_style.fontSize)
    left_width = max(styles.CONTENT_WIDTH - right_width, styles.MIN_HEADING_WIDTH)

    row = Table(
        [[Paragraph(leading, left_style), Paragraph(_text(trailing), right_style)]],
        colWidths=[left_width, right_width],
        style=styles.heading_row_style(),
        # A Table defaults to centring itself in the frame. Left is what makes the row
        # start on the same margin as every paragraph and section rule around it.
        hAlign="LEFT",
    )
    # A Table is not a Paragraph, so the paragraph spacing has to be carried over.
    row.spaceBefore = left_style.spaceBefore
    row.spaceAfter = left_style.spaceAfter
    return row


def _bold(text: str) -> str:
    return f"<b>{_text(text)}</b>"


def _link(href: str, label: str) -> str:
    return f'<a href="{_text(href)}" color="{styles.LINK_INK}">{_text(label)}</a>'


def _url(value: str) -> str:
    """Make a profile reference clickable without assuming it carries a scheme."""
    return value if value.startswith(("http://", "https://")) else f"https://{value}"


def _profile_label(value: str) -> str:
    """Shorten a profile reference to the part worth reading.

    ``https://www.linkedin.com/in/cg10/`` becomes ``linkedin.com/in/cg10``. The
    scheme, the ``www.``, and a trailing slash are noise on a contact line: nobody
    types them, and on a one-page resume they cost room that the address itself
    needs. The link still points at the full URL, so the shortening is display only.
    """
    label = value.strip()
    for scheme in ("https://", "http://"):
        label = label.removeprefix(scheme)
    return label.removeprefix("www.").rstrip("/")


def _text(value: str) -> str:
    """Escape text for ReportLab's inline markup.

    Resume content routinely contains ``&`` and comparison operators, which would
    otherwise be read as markup and either vanish or fail the build.
    """
    return escape(value)
