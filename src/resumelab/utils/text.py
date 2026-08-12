"""Text normalization shared by every input path.

Inputs arrive from files a researcher edited on an unknown platform and from shell
arguments. Normalizing once, at the boundary, keeps everything downstream — hashing,
prompting, validation, and PDF rendering — working on consistent UTF-8 text.
"""

from __future__ import annotations

import re
import unicodedata

_PRESERVED_CONTROL_CHARACTERS = frozenset("\n\t")

_HIDDEN_CATEGORIES = frozenset({"Cc", "Cf"})
"""Control and format characters.

``Cf`` matters as much as ``Cc`` here: zero-width joiners and bidirectional
overrides are invisible, survive copy-paste, and can make text render in an order
that differs from the order it will be read in.
"""


MAX_SLUG_LENGTH = 40
"""Keeps run directory names readable and well inside filesystem limits."""

FALLBACK_SLUG = "run"
"""Used when a label reduces to nothing, so a directory always has a name."""

_NON_SLUG = re.compile(r"[^a-z0-9]+")


def slugify(value: str, *, max_length: int = MAX_SLUG_LENGTH) -> str:
    """Reduce ``value`` to a safe file name component.

    Everything outside ``a-z0-9`` becomes a hyphen, so a label taken from user input
    cannot introduce a path separator, a parent reference, or a hidden file. A label
    that reduces to nothing falls back to :data:`FALLBACK_SLUG`.

    An over-long label is cut back to a word boundary where one is close enough to
    the limit, because these name run directories and directories get read: a run
    named after a fetched posting should end at ``…-software-engineer`` rather than
    at ``…-software-engineer-clou``.
    """
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = _NON_SLUG.sub("-", normalized.lower()).strip("-")
    if len(slug) > max_length:
        slug = _trim_to_word(slug, max_length)
    return slug or FALLBACK_SLUG


def _trim_to_word(slug: str, max_length: int) -> str:
    """Shorten ``slug``, preferring the last word boundary in the second half.

    A boundary in the first half would throw away more of the name than the ragged
    edge costs, so a label with no late boundary — one very long word — is simply cut.
    """
    clipped = slug[:max_length].strip("-")
    boundary = clipped.rfind("-")
    if boundary >= max_length // 2:
        return clipped[:boundary]
    return clipped


def control_characters(value: str) -> list[str]:
    """Return the control characters in ``value``, as escaped literals.

    Nothing on a finished resume should contain one. They survive copy-paste, break
    text extraction from the rendered PDF, and are invisible while doing it.
    """
    return sorted(
        {repr(character) for character in value if unicodedata.category(character) == "Cc"}
    )


def normalize_text(value: str) -> str:
    """Return ``value`` as clean, consistently encoded text.

    Line endings are unified to ``\\n``, invisible control and format characters
    other than newline and tab are dropped, the result is NFC-normalized so visually
    identical strings hash identically, and surrounding whitespace is removed.
    """
    unified_newlines = value.replace("\r\n", "\n").replace("\r", "\n")
    without_controls = "".join(
        character
        for character in unified_newlines
        if character in _PRESERVED_CONTROL_CHARACTERS
        or unicodedata.category(character) not in _HIDDEN_CATEGORIES
    )
    return unicodedata.normalize("NFC", without_controls).strip()
