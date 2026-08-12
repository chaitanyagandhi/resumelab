"""Turning posting HTML into the plain text the pipeline analyzes.

Job boards return HTML in three different shapes: real pages, HTML fragments inside a
JSON field, and — Greenhouse — HTML that has been entity-escaped a second time on its
way into that field. All three end up here.

Structure is preserved only where it carries meaning for a posting: paragraph breaks
and list items. A requirements list read as one run-on paragraph loses the fact that
it was a list, and the analysis stage reads these as separate requirements.
"""

from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser
from typing import Final

_DROPPED_ELEMENTS: Final = frozenset(
    {"script", "style", "noscript", "svg", "head", "nav", "footer", "template", "iframe"}
)
"""Elements whose text is never posting content. Their contents are skipped entirely."""

_BLOCK_ELEMENTS: Final = frozenset(
    {
        "address", "article", "aside", "blockquote", "div", "dd", "dl", "dt", "fieldset",
        "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr",
        "main", "nav", "ol", "p", "pre", "section", "table", "tbody", "td", "th", "thead",
        "tr", "ul",
    }
)  # fmt: skip
"""Elements that end the current line. Inline elements deliberately do not."""

_LIST_ITEM: Final = "li"
_LINE_BREAK: Final = "br"

_BULLET_PREFIX: Final = "- "
"""A neutral marker. The renderer draws its own glyphs; this only survives into the
prompt, where it tells the model that these were separate list items."""

_SPACES = re.compile(r"[^\S\n]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_ANY_WHITESPACE = re.compile(r"\s+")


class _TextExtractor(HTMLParser):
    """Collect readable text, remembering only the structure a posting needs.

    When ``prefer`` names any elements, text inside them is collected separately as
    well. A page that has a ``<main>`` is telling us where its content is, and taking
    that over the whole document is what keeps a site's header out of the posting.
    """

    def __init__(self, prefer: frozenset[str] = frozenset()) -> None:
        # convert_charrefs resolves entities in text for us, so `&amp;` arrives as `&`.
        super().__init__(convert_charrefs=True)
        self._prefer = prefer
        self._parts: list[str] = []
        self._preferred: list[str] = []
        self._preferred_depth = 0
        self._skip_depth = 0
        self._dropped_tag: str | None = None

    def _emit(self, value: str) -> None:
        self._parts.append(value)
        if self._preferred_depth:
            self._preferred.append(value)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if self._dropped_tag is not None:
            if tag == self._dropped_tag:
                self._skip_depth += 1
            return
        if tag in _DROPPED_ELEMENTS:
            self._dropped_tag = tag
            self._skip_depth = 1
            return
        if tag in self._prefer:
            self._preferred_depth += 1
        if tag == _LINE_BREAK:
            self._emit("\n")
        elif tag == _LIST_ITEM:
            self._emit("\n" + _BULLET_PREFIX)
        elif tag in _BLOCK_ELEMENTS:
            self._emit("\n")

    def handle_endtag(self, tag: str) -> None:
        if self._dropped_tag is not None:
            if tag == self._dropped_tag:
                self._skip_depth -= 1
                if self._skip_depth == 0:
                    self._dropped_tag = None
            return
        # `</li>` deliberately emits nothing: the next `<li>` opens its own line, and
        # closing one here too would put a blank line between every list item.
        if tag in _BLOCK_ELEMENTS:
            self._emit("\n")
        if tag in self._prefer and self._preferred_depth:
            self._preferred_depth -= 1

    def handle_data(self, data: str) -> None:
        """Collect text, treating the whitespace inside it as insignificant.

        Newlines in the source are formatting of the markup, not of the posting; only
        the newlines this parser emits for block elements mean anything. Collapsing
        here is what keeps a paragraph wrapped across source lines as one line.
        """
        if self._dropped_tag is None:
            self._emit(_ANY_WHITESPACE.sub(" ", data))

    @property
    def text(self) -> str:
        """The preferred region when the page marked one, the whole document otherwise."""
        if tidy("".join(self._preferred)):
            return "".join(self._preferred)
        return "".join(self._parts)


def html_to_text(html: str, *, prefer: frozenset[str] = frozenset()) -> str:
    """Reduce an HTML document or fragment to readable plain text.

    Args:
        html: A document or a fragment. Handles the double-escaped case
            transparently: a value that is entity-escaped markup rather than markup is
            unescaped first, so Greenhouse's ``&lt;p&gt;`` does not survive into the
            posting as literal angle brackets.
        prefer: Elements whose contents should be taken alone if the page has any,
            such as ``main`` on a full page. Ignored when nothing matches.
    """
    parser = _TextExtractor(prefer)
    parser.feed(_decode_escaped_markup(html))
    parser.close()
    return tidy(parser.text)


def _decode_escaped_markup(value: str) -> str:
    """Unescape a string that holds markup as entities rather than as tags.

    Greenhouse's ``content`` field is HTML that was escaped on the way into the JSON,
    so it arrives as ``&lt;h2&gt;``. Detected rather than assumed: a posting that
    merely mentions ``&lt;`` in its text is left alone, because it also has real tags.
    """
    if "<" not in value and "&lt;" in value:
        return unescape(value)
    return value


def tidy(text: str) -> str:
    """Collapse the whitespace that markup-to-text conversion leaves behind.

    Runs of spaces become one, trailing space is dropped, and any number of blank
    lines becomes at most one — a posting laid out with nested divs otherwise arrives
    mostly made of newlines.
    """
    collapsed = _SPACES.sub(" ", text)
    lines = [line.strip() for line in collapsed.split("\n")]
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()
