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

MARGIN_HORIZONTAL = 0.5 * inch
"""Half an inch, which puts :data:`CONTENT_WIDTH` at 528pt.

Measured off the reference resume's section rules, which span exactly that. The extra
fourteen points over the old six-tenths of an inch is two more characters on every
line, and a bullet that fits one line instead of two saves far more than that."""
MARGIN_TOP = 0.5 * inch
MARGIN_BOTTOM = 0.5 * inch

FRAME_PADDING = 6.0
"""ReportLab's frame padding, applied inside the page margin on every side.

``SimpleDocTemplate`` builds its frame with this padding, so text begins this far
inside the margin. Paragraphs pick it up for free because they are laid out *by* the
frame. A table is different: it is given an explicit width, so the padding has to be
subtracted here or the row hangs outside the section rules everything else aligns to.
"""

CONTENT_WIDTH = PAGE_SIZE[0] - 2 * MARGIN_HORIZONTAL - 2 * FRAME_PADDING
"""Width text actually occupies, which a right-aligned field is measured against."""

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


BODY_FONT_SIZE = 10.0
NAME_FONT_SIZE = 18.0
CONTACT_FONT_SIZE = 9.5
SECTION_FONT_SIZE = 11.0
ENTRY_FONT_SIZE = 10.0
DATE_FONT_SIZE = 9.0
DETAIL_FONT_SIZE = 9.0
"""The type scale, in points at full size.

Read off a reference resume rather than chosen: the section headings are set a point
above the body, the dates and the secondary lines a point below it, and the name is
the only thing on the page that is much larger. The previous scale had the headings
and the dates all at body size, which left the page with one texture and no
hierarchy.

The contact line is the exception. The reference sets it at the heading size, which
it can afford because it lists four fields; this one lists five, and a location
pushes the line past :data:`CONTENT_WIDTH` at anything above nine points. A wrapped
contact line costs a whole line of the page and looks like an accident, so the size
is held to what fits. Dropping the location would buy the larger size back.
"""

SECTION_TRACKING = 0.8
"""Extra space between the letters of a section heading, in points at full size.

Measured off the reference at :data:`SECTION_FONT_SIZE`, where it works out at about
0.07em. Tracking is what makes a short word in capitals read as a label rather than
as a shouted word, and it is the reason the headings there look deliberate.
"""

LEADING_RATIO = 1.22
"""Line spacing as a multiple of font size. Tight, but still readable."""

MIN_BODY_FONT_SIZE = 8.5
"""The floor. Below this a resume is harder to read than it is to fit."""

_TIGHTEST_SCALE = MIN_BODY_FONT_SIZE / BODY_FONT_SIZE

LAYOUT_SCALES = (1.0, 0.96, 0.92, _TIGHTEST_SCALE)
"""Progressively tighter layouts, tried in order until the content fits one page.

The last entry is the tightest permitted, and is derived rather than written down so
that it always holds the body text at exactly :data:`MIN_BODY_FONT_SIZE`. It used to
be a literal, which meant that changing the body size silently moved the floor.
Content that still overflows there is left to overflow rather than shrunk into
something nobody will read.
"""

# --- colour -----------------------------------------------------------------
INK = HexColor("#111111")
"""Near-black rather than pure black; easier to read and prints identically."""

MUTED_INK = HexColor("#444444")
LINK_INK = INK
"""Links are drawn in the body colour, as the reference does.

They stay clickable: the annotation is what makes a link, not the colour. Blue link
text on a resume reads as a web page, and the addresses are short enough to be typed
out by anyone reading it on paper."""
RULE_INK = HexColor("#999999")

# --- spacing ----------------------------------------------------------------
SPACE_AFTER_NAME = 1.5
SPACE_AFTER_CONTACT = 8.0
SPACE_BEFORE_SECTION = 9.0
SPACE_AFTER_SECTION_HEADING = 3.0
SPACE_BETWEEN_ENTRIES = 6.5
SPACE_AFTER_ENTRY_HEADING = 1.0
SPACE_BETWEEN_BULLETS = 1.5
"""Vertical rhythm. Entries are separated more than the lines inside them.

The gap between two roles has to be clearly larger than the gap between a role's own
bullets, or the section reads as one undifferentiated list. That relationship, not
the absolute numbers, is what the values are chosen to hold.
"""

# --- rules and bullets ------------------------------------------------------
RULE_THICKNESS = 0.75
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
"""Where the marker sits, measured in from the text margin."""

BULLET_PADDING_RATIO = 0.6
"""ReportLab's own gap after a bullet, as a multiple of the bullet font size.

Mirrored from ``reportlab.platypus.paragraph._handleBulletWidth``, which starts the
first line of a bulleted paragraph at ``bulletIndent + bulletWidth + 0.6 *
bulletFontSize`` whenever that exceeds ``leftIndent``, and every later line at
``leftIndent``. ReportLab exposes no accessor for the factor, so it is named here.
:func:`bullet_text_indent` has to agree with it exactly or the hanging indent is
silently lost; a test measures a rendered page rather than trusting this number.
"""


def bullet_text_indent(scale: float = 1.0) -> float:
    """Where bullet text hangs, measured from the glyph rather than guessed.

    A round bullet at body size is wider than the 9pt gap the marker is given, so a
    fixed indent smaller than ReportLab's own overrun leaves the first line starting
    to the right of every line that follows it. That is invisible until a bullet
    wraps, and every long bullet on the page wraps.

    Args:
        scale: The layout scale, applied here rather than by the caller because the
            glyph is measured at the scaled size it will actually be drawn at.
    """
    register_fonts()
    size = _scaled(BULLET_FONT_SIZE, scale)
    # ReportLab ships no type information, so the measurement arrives untyped.
    marker = float(pdfmetrics.stringWidth(BULLET_CHARACTER, GLYPH_FONT_BOLD, size))
    return _scaled(BULLET_INDENT, scale) + marker + BULLET_PADDING_RATIO * size


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

PROJECT_TITLE_SEPARATOR = " - "
"""Divides a project's name from its subtitle.

A hyphen, not an em dash. Nothing on this page uses an em dash: it is the most
recognizable signature of machine-written text, and a document whose whole purpose is
to read as though a person wrote it cannot afford one.
"""

PROJECT_STACK_SEPARATOR = " | "
"""Divides a project's subtitle from the stack sharing its line.

Deliberately not :data:`SEPARATOR`. That line already carries a hyphen between the
name and the subtitle, and a dot after it would read as one more field in the same
run rather than a shift to the technologies.

A pipe rather than a slash. This reverses an earlier decision that preferred the
slash for leaning with the italic stack it introduces; an upright bar is what the
reference format uses, and against a line that already slopes it reads as a divider
rather than as punctuation belonging to either side. Drawn outside the italic run,
which is what keeps it upright. Plain ASCII either way, so it extracts as ``|``
everywhere and the shape costs a parser nothing.
"""


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
    date = _scaled(DATE_FONT_SIZE, scale)

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
            fontSize=date,
            leading=_leading(entry),
            alignment=TA_RIGHT,
            textColor=MUTED_INK,
        ),
        "detail_right": style(
            "detail_right",
            fontName=FONT_REGULAR,
            fontSize=detail,
            leading=_leading(detail),
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
        # Upright, because a whole line of italics is read as an aside. The
        # qualification above it is the aside; the coursework is a list of facts.
        "note": style(
            "note",
            fontName=FONT_REGULAR,
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
            leftIndent=bullet_text_indent(scale),
            bulletIndent=_scaled(BULLET_INDENT, scale),
            bulletFontName=GLYPH_FONT_BOLD,
            bulletFontSize=_scaled(BULLET_FONT_SIZE, scale),
            bulletOffsetY=_scaled(BULLET_OFFSET_Y, scale),
            spaceAfter=_scaled(SPACE_BETWEEN_BULLETS, scale),
        ),
    }
