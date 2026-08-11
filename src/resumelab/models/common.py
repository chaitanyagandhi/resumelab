"""Field types shared by the models the LLM fills in.

These models are structured-output targets, which rules out JSON-Schema constraints:
strict modes reject schemas carrying ``minLength``, ``maxItems``, defaults, and the
like. Validation therefore lives in Python, where it never reaches the schema and a
violation is fed back to the model for repair instead.

The split below is about cost. Rejecting a response costs an API call, so it is
reserved for content the pipeline genuinely cannot proceed without; ordinary
extraction noise is cleaned up in place.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import AfterValidator, ConfigDict

MAX_LIST_ITEMS = 30
"""Ceiling per list. Keeps an over-eager response from flooding later prompts."""

GENERATED_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    str_strip_whitespace=True,
    frozen=True,
)
"""Shared config for LLM-filled models: no extra keys, stripped, immutable."""


def clean_items(values: tuple[str, ...]) -> tuple[str, ...]:
    """Drop blanks, remove case-insensitive duplicates, and cap the length.

    A duplicated keyword is not worth another API call, so this sanitizes rather
    than rejects.
    """
    seen: set[str] = set()
    cleaned: list[str] = []
    for value in values:
        item = value.strip()
        key = item.casefold()
        if not item or key in seen:
            continue
        seen.add(key)
        cleaned.append(item)
    return tuple(cleaned[:MAX_LIST_ITEMS])


def require_content(value: str) -> str:
    """Reject a field the pipeline cannot proceed without."""
    if not value.strip():
        raise ValueError("must not be empty")
    return value


TermList = Annotated[tuple[str, ...], AfterValidator(clean_items)]
"""A deduplicated, bounded list of terms."""

RequiredText = Annotated[str, AfterValidator(require_content)]
"""Free text the pipeline depends on, so an empty value is worth a repair attempt."""
