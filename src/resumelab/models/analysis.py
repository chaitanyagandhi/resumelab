"""Structured reading of a target job description.

:class:`JobAnalysis` is an LLM structured-output target; see
:mod:`resumelab.models.common` for what that constrains.

The most important field is :attr:`JobAnalysis.technical_identity`: the engineering
identity a maximally-aligned candidate would present. Every later stage transforms the
candidate toward it, so the whole run is downstream of getting it right.
"""

from __future__ import annotations

from pydantic import BaseModel

from resumelab.models.common import GENERATED_MODEL_CONFIG, RequiredText, TermList


class JobAnalysis(BaseModel):
    """What the job description asks for, and who it is asking for."""

    model_config = GENERATED_MODEL_CONFIG

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
