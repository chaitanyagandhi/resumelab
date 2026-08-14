"""Layout constants and paragraph styles for the rendered resume.

Every measurement the renderer uses lives here, so the page can be re-proportioned
without touching the code that decides what goes on it.

Two constraints shape these choices. The resume must survive machine reading, which
rules out images, icons, and multi-column tables, and keeps the text in a single
linear flow. And it must fit one page, which is why the spacing values are small and
expressed relative to the body size rather than as independent magic numbers.
"""

from __future__ import annotations

from pathlib import Path

import reportlab
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import TableStyle

# --- page -------------------------------------------------------------------
PAGE_SIZE = LETTER
"""US Letter, the expected format for the audience this resume targets."""

MARGIN_HORIZONTAL = 0.6 * inch
MARGIN_TOP = 0.5 * inch
MARGIN_BOTTOM = 0.5 * inch

CONTENT_WIDTH = PAGE_SIZE[0] - 2 * MARGIN_HORIZONTAL
"""Width available between the margins, which a right-aligned date is measured against."""

MIN_HEADING_WIDTH = 1.5 * inch
"""Floor for the heading column, so an absurd date cannot squeeze the title away."""

# --- type -------------------------------------------------------------------
FONT_REGULAR = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
FONT_ITALIC = "Helvetica-Oblique"
"""The standard PDF fonts: no embedding, and universally extractable."""

GLYPH_FONT = "ResumeLabGlyph"
"""Font used for the bullet and the field separator, and for nothing else.

The body text stays in the base-14 Helvetica, which every PDF reader has and no
document has to carry. But Helvetica's dots are square, and its built-in encoding
mangles the round bullet on extraction. Registering one embedded font and tagging
just those two glyphs into it buys a round bullet that extracts as U+2022, without
embedding a typeface for the whole document.

Bitstream Vera Sans ships inside ReportLab under a permissive license, so this adds
no dependency and no file that has to be found at run time.
"""

GLYPH_FONT_BOLD = "ResumeLabGlyph-Bold"
"""The bold weight of :data:`GLYPH_FONT`, used for the list bullet.

A bullet drawn at the regular weight is a thin ring of ink at this size and reads as
faint beside the text it introduces. The bold weight is a denser dot at the same
diameter, which is what makes a list scan as a list.
"""

_GLYPH_FONT_FILES = {
    GLYPH_FONT: "Vera.ttf",
    GLYPH_FONT_BOLD: "VeraBd.ttf",
}

_FONT_DIRECTORY = Path(reportlab.__file__).parent / "fonts"


def register_fonts() -> None:
    """Make the glyph fonts available to the renderer.

    Idempotent: rendering several resumes in one process registers once. ReportLab's
    font registry is process-global, which is why this is a function rather than an
    import-time side effect — a caller that never renders should not pay for it.
    """
    registered = pdfmetrics.getRegisteredFontNames()
    for name, filename in _GLYPH_FONT_FILES.items():
        if name not in registered:
            pdfmetrics.registerFont(TTFont(name, str(_FONT_DIRECTORY / filename)))


BODY_FONT_SIZE = 9.5
NAME_FONT_SIZE = 19.0
CONTACT_FONT_SIZE = 8.5
SECTION_FONT_SIZE = 9.5
ENTRY_FONT_SIZE = 9.5
DETAIL_FONT_SIZE = 8.8

LEADING_RATIO = 1.22
"""Line spacing as a multiple of font size. Tight, but still readable."""

MIN_BODY_FONT_SIZE = 8.5
"""The floor. Below this a resume is harder to read than it is to fit."""

LAYOUT_SCALES = (1.0, 0.96, 0.92, 0.895)
"""Progressively tighter layouts, tried in order until the content fits one page.

The last entry is the tightest permitted: it holds the body text at
:data:`MIN_BODY_FONT_SIZE`. Content that still overflows there is left to overflow
rather than shrunk into something nobody will read.
"""

# --- colour -----------------------------------------------------------------
INK = HexColor("#111111")
"""Near-black rather than pure black; easier to read and prints identically."""

MUTED_INK = HexColor("#444444")
LINK_INK = HexColor("#1F4E79")
RULE_INK = HexColor("#999999")

# --- spacing ----------------------------------------------------------------
SPACE_AFTER_NAME = 1.5
SPACE_AFTER_CONTACT = 7.0
SPACE_BEFORE_SECTION = 7.5
SPACE_AFTER_SECTION_HEADING = 3.0
SPACE_BETWEEN_ENTRIES = 4.5
SPACE_AFTER_ENTRY_HEADING = 1.0
SPACE_BETWEEN_BULLETS = 1.0

# --- rules and bullets ------------------------------------------------------
RULE_THICKNESS = 0.6
RULE_SPACE_BEFORE = 1.0
RULE_SPACE_AFTER = 3.0

BULLET_CHARACTER = "•"
"""The conventional round bullet, drawn from :data:`GLYPH_FONT`.

In the standard PDF fonts this character extracts as U+007F — a control character
every parser reading this resume would see, and one the validator rejects in content.
That is a property of the base-14 fonts' built-in encoding, not of the character: an
embedded font carries a ToUnicode map, so the same glyph extracts as U+2022.

Round rather than square is not only cosmetic. The square alternatives that survive
the base-14 encoding (``▪``, ``∙``) are drawn from a different part of the
repertoire, and several of them extract as U+25A0 regardless of what was asked for.
"""

BULLET_FONT_SIZE = 9.5
"""Body size, at the bold weight. Smaller than this the marker reads as a speck."""

BULLET_OFFSET_Y = 0.0
"""No lift needed: this glyph is already drawn at mid-height within its em.

The square it replaced sat on the baseline and had to be raised. Keeping that
correction here would push a round bullet up into the ascenders.
"""

BULLET_INDENT = 9.0
BULLET_TEXT_INDENT = 18.0
"""Text hangs at this indent so wrapped bullet lines align under the first."""

SEPARATOR_SIZE_DELTA = -3.0
"""Points below the surrounding text, so the separator stays a dot.

Expressed as a delta rather than a size because the separator appears at three
different text sizes, and because the whole layout is scaled to fit the page —
an absolute size here would grow relative to the text every time the page tightened.
"""

SEPARATOR = f' <font name="{GLYPH_FONT}" size="{SEPARATOR_SIZE_DELTA:g}">•</font> '
"""Separates fields on one line. A single flow extracts far better than columns.

The same round glyph as the list bullet, drawn small. It is a divider between terms
rather than a marker in front of them, so at bullet size it would read as a list laid
sideways and would out-shout the words it separates.

Tagged into :data:`GLYPH_FONT` while the surrounding text stays Helvetica, whose own
middle dot is square.
"""

DATE_RANGE_SEPARATOR = " \u2013 "
"""En dash, the typographic convention for a span of dates."""


def heading_row_style() -> TableStyle:
    """Styling for the two-cell row that sets a heading against its date.

    A row, not a layout: one line, two cells, no rules and no padding, so the pair
    occupies exactly the space the paragraph would have. Text still extracts in
    reading order \u2014 heading, then date, then the bullets beneath \u2014 which is the
    property that decides whether a resume survives being parsed.
    """
    return TableStyle(
        [
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]
    )


def _scaled(points: float, scale: float) -> float:
    """Apply the layout scale to a measurement, rounded to a stable value."""
    return round(points * scale, 2)


def _leading(size: float) -> float:
    return round(size * LEADING_RATIO, 2)


def build_stylesheet(scale: float = 1.0) -> dict[str, ParagraphStyle]:
    """Build every paragraph style the renderer uses.

    Args:
        scale: Multiplier applied to every font size and every vertical space, used
            to tighten the page when content slightly overflows. Type and spacing
            scale together, so the proportions of the page are preserved rather than
            the text being squeezed into unchanged whitespace.

    Returns:
        Styles keyed by role: ``name``, ``contact``, ``section``, ``entry``,
        ``detail``, ``body``, and ``bullet``.
    """
    register_fonts()
    body = _scaled(BODY_FONT_SIZE, scale)
    detail = _scaled(DETAIL_FONT_SIZE, scale)
    entry = _scaled(ENTRY_FONT_SIZE, scale)

    def style(name: str, **overrides: object) -> ParagraphStyle:
        """Apply the defaults shared by every style, so each entry states only its own."""
        return ParagraphStyle(name, **{"textColor": INK, "alignment": TA_LEFT, **overrides})

    return {
        "name": style(
            "name",
            fontName=FONT_BOLD,
            fontSize=_scaled(NAME_FONT_SIZE, scale),
            leading=_leading(_scaled(NAME_FONT_SIZE, scale)),
            alignment=TA_CENTER,
            spaceAfter=_scaled(SPACE_AFTER_NAME, scale),
        ),
        "contact": style(
            "contact",
            fontName=FONT_REGULAR,
            fontSize=_scaled(CONTACT_FONT_SIZE, scale),
            leading=_leading(_scaled(CONTACT_FONT_SIZE, scale)),
            alignment=TA_CENTER,
            textColor=MUTED_INK,
            spaceAfter=_scaled(SPACE_AFTER_CONTACT, scale),
        ),
        "section": style(
            "section",
            fontName=FONT_BOLD,
            fontSize=_scaled(SECTION_FONT_SIZE, scale),
            leading=_leading(_scaled(SECTION_FONT_SIZE, scale)),
            spaceBefore=_scaled(SPACE_BEFORE_SECTION, scale),
            spaceAfter=_scaled(SPACE_AFTER_SECTION_HEADING, scale),
        ),
        "entry": style(
            "entry",
            fontName=FONT_REGULAR,
            fontSize=entry,
            leading=_leading(entry),
            spaceBefore=_scaled(SPACE_BETWEEN_ENTRIES, scale),
            spaceAfter=_scaled(SPACE_AFTER_ENTRY_HEADING, scale),
        ),
        "date": style(
            "date",
            fontName=FONT_REGULAR,
            fontSize=entry,
            leading=_leading(entry),
            alignment=TA_RIGHT,
            textColor=MUTED_INK,
        ),
        "detail": style(
            "detail",
            fontName=FONT_ITALIC,
            fontSize=detail,
            leading=_leading(detail),
            textColor=MUTED_INK,
            spaceAfter=_scaled(SPACE_AFTER_ENTRY_HEADING, scale),
        ),
        "body": style(
            "body",
            fontName=FONT_REGULAR,
            fontSize=body,
            leading=_leading(body),
        ),
        "bullet": style(
            "bullet",
            fontName=FONT_REGULAR,
            fontSize=body,
            leading=_leading(body),
            leftIndent=_scaled(BULLET_TEXT_INDENT, scale),
            bulletIndent=_scaled(BULLET_INDENT, scale),
            bulletFontName=GLYPH_FONT_BOLD,
            bulletFontSize=_scaled(BULLET_FONT_SIZE, scale),
            bulletOffsetY=_scaled(BULLET_OFFSET_Y, scale),
            spaceAfter=_scaled(SPACE_BETWEEN_BULLETS, scale),
        ),
    }
