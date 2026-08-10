"""Structured reading of a target job description.

:class:`JobAnalysis` is an LLM structured-output target, which constrains its shape:
every field is required and no JSON-Schema string or array constraints are used,
because strict structured-output modes reject schemas that carry them. Hygiene is
enforced by validators instead — those run in Python and never reach the schema, so a
response that violates one is fed back to the model for repair rather than silently
accepted.

The most important field is :attr:`JobAnalysis.technical_identity`: the engineering
identity a maximally-aligned candidate would present. Every later stage transforms the
candidate toward it, so the whole run is downstream of getting it right.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict

MAX_LIST_ITEMS = 30
"""Ceiling per list. Keeps an over-eager extraction from flooding later prompts."""


def _clean_items(values: tuple[str, ...]) -> tuple[str, ...]:
    """Drop blanks, remove case-insensitive duplicates, and cap the length.

    Extraction noise is sanitized rather than rejected: a duplicated keyword is not a
    reason to spend another API call on a repair attempt.
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


def _require_content(value: str) -> str:
    """Reject a field the pipeline cannot proceed without."""
    if not value.strip():
        raise ValueError("must not be empty")
    return value


TermList = Annotated[tuple[str, ...], AfterValidator(_clean_items)]
"""A deduplicated, bounded list of extracted terms."""

RequiredText = Annotated[str, AfterValidator(_require_content)]
"""Free text the pipeline depends on, so an empty value is worth a repair attempt."""


class JobAnalysis(BaseModel):
    """What the job description asks for, and who it is asking for."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        frozen=True,
    )

    # --- identification -----------------------------------------------------
    company: str
    role_title: RequiredText
    role_archetype: RequiredText
    seniority: str

    # --- technology surface -------------------------------------------------
    core_languages: TermList
    frameworks: TermList
    infrastructure: TermList
    databases: TermList
    ai_ml_concepts: TermList

    # --- conceptual surface -------------------------------------------------
    domain_concepts: TermList
    engineering_concepts: TermList

    # --- what the role involves ---------------------------------------------
    responsibilities: TermList
    high_priority_requirements: TermList
    bonus_requirements: TermList
    soft_traits: TermList
    high_value_keywords: TermList

    # --- the target the pipeline aims at ------------------------------------
    technical_identity: RequiredText
    ideal_candidate_profile: RequiredText
