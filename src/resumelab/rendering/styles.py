"""Layout constants and paragraph styles for the rendered resume.

Every measurement the renderer uses lives here, so the page can be re-proportioned
without touching the code that decides what goes on it.

Two constraints shape these choices. The resume must survive machine reading, which
rules out images, icons, and multi-column tables, and keeps the text in a single
linear flow. And it must fit one page, which is why the spacing values are small and
expressed relative to the body size rather than as independent magic numbers.
"""

from __future__ import annotations

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch

# --- page -------------------------------------------------------------------
PAGE_SIZE = LETTER
"""US Letter, the expected format for the audience this resume targets."""

MARGIN_HORIZONTAL = 0.6 * inch
MARGIN_TOP = 0.5 * inch
MARGIN_BOTTOM = 0.5 * inch

# --- type -------------------------------------------------------------------
FONT_REGULAR = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
FONT_ITALIC = "Helvetica-Oblique"
"""The standard PDF fonts: no embedding, and universally extractable."""

BODY_FONT_SIZE = 9.5
NAME_FONT_SIZE = 19.0
CONTACT_FONT_SIZE = 8.5
SECTION_FONT_SIZE = 9.5
ENTRY_FONT_SIZE = 9.5
DETAIL_FONT_SIZE = 8.8

LEADING_RATIO = 1.22
"""Line spacing as a multiple of font size. Tight, but still readable."""

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

BULLET_CHARACTER = "▪"
"""A small square rather than a round bullet.

U+2022 is the conventional choice, but in the standard PDF fonts it extracts as
U+007F — a control character. Every parser reading this resume would see it, and the
validator rejects control characters in content for the same reason. This glyph
renders as a bullet and extracts as itself.
"""
BULLET_FONT_SIZE = 5.5
"""Drawn smaller than the body text, so the square reads as a bullet."""

BULLET_OFFSET_Y = -1.6
"""Lifts the smaller glyph off the baseline to sit optically centered."""

BULLET_INDENT = 9.0
BULLET_TEXT_INDENT = 18.0
"""Text hangs at this indent so wrapped bullet lines align under the first."""

SEPARATOR = " · "
"""Separates fields on one line. A single flow extracts far better than columns."""

DATE_RANGE_SEPARATOR = " \u2013 "
"""En dash, the typographic convention for a span of dates."""


def _leading(size: float) -> float:
    return round(size * LEADING_RATIO, 2)


def build_stylesheet() -> dict[str, ParagraphStyle]:
    """Build every paragraph style the renderer uses.

    Returns:
        Styles keyed by role: ``name``, ``contact``, ``section``, ``entry``,
        ``detail``, ``body``, and ``bullet``.
    """
    return {
        "name": ParagraphStyle(
            "name",
            fontName=FONT_BOLD,
            fontSize=NAME_FONT_SIZE,
            leading=_leading(NAME_FONT_SIZE),
            alignment=TA_CENTER,
            textColor=INK,
            spaceAfter=SPACE_AFTER_NAME,
        ),
        "contact": ParagraphStyle(
            "contact",
            fontName=FONT_REGULAR,
            fontSize=CONTACT_FONT_SIZE,
            leading=_leading(CONTACT_FONT_SIZE),
            alignment=TA_CENTER,
            textColor=MUTED_INK,
            spaceAfter=SPACE_AFTER_CONTACT,
        ),
        "section": ParagraphStyle(
            "section",
            fontName=FONT_BOLD,
            fontSize=SECTION_FONT_SIZE,
            leading=_leading(SECTION_FONT_SIZE),
            alignment=TA_LEFT,
            textColor=INK,
            spaceBefore=SPACE_BEFORE_SECTION,
            spaceAfter=SPACE_AFTER_SECTION_HEADING,
        ),
        "entry": ParagraphStyle(
            "entry",
            fontName=FONT_REGULAR,
            fontSize=ENTRY_FONT_SIZE,
            leading=_leading(ENTRY_FONT_SIZE),
            alignment=TA_LEFT,
            textColor=INK,
            spaceAfter=SPACE_AFTER_ENTRY_HEADING,
        ),
        "detail": ParagraphStyle(
            "detail",
            fontName=FONT_ITALIC,
            fontSize=DETAIL_FONT_SIZE,
            leading=_leading(DETAIL_FONT_SIZE),
            alignment=TA_LEFT,
            textColor=MUTED_INK,
            spaceAfter=SPACE_AFTER_ENTRY_HEADING,
        ),
        "body": ParagraphStyle(
            "body",
            fontName=FONT_REGULAR,
            fontSize=BODY_FONT_SIZE,
            leading=_leading(BODY_FONT_SIZE),
            alignment=TA_LEFT,
            textColor=INK,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            fontName=FONT_REGULAR,
            fontSize=BODY_FONT_SIZE,
            leading=_leading(BODY_FONT_SIZE),
            alignment=TA_LEFT,
            textColor=INK,
            leftIndent=BULLET_TEXT_INDENT,
            bulletIndent=BULLET_INDENT,
            bulletFontSize=BULLET_FONT_SIZE,
            bulletOffsetY=BULLET_OFFSET_Y,
            spaceAfter=SPACE_BETWEEN_BULLETS,
        ),
    }
