"""Text normalization shared by every input path.

Inputs arrive from files a researcher edited on an unknown platform and from shell
arguments. Normalizing once, at the boundary, keeps everything downstream — hashing,
prompting, validation, and PDF rendering — working on consistent UTF-8 text.
"""

from __future__ import annotations

import unicodedata

_PRESERVED_CONTROL_CHARACTERS = frozenset("\n\t")


def normalize_text(value: str) -> str:
    """Return ``value`` as clean, consistently encoded text.

    Line endings are unified to ``\\n``, control characters other than newline and
    tab are dropped, the result is NFC-normalized so visually identical strings hash
    identically, and surrounding whitespace is removed.
    """
    unified_newlines = value.replace("\r\n", "\n").replace("\r", "\n")
    without_controls = "".join(
        character
        for character in unified_newlines
        if character in _PRESERVED_CONTROL_CHARACTERS or unicodedata.category(character) != "Cc"
    )
    return unicodedata.normalize("NFC", without_controls).strip()
