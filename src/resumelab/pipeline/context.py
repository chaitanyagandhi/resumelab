"""Rendering of pipeline state into prompt sections.

Stages share the same inputs — the source profile, the job analysis, the strategy —
so they are rendered once, here, and identically everywhere. JSON is used because it
is lossless and byte-stable for a given model, which matters when two runs are meant
to be comparable.

**Personal details are never sent to a model.** Name, email, phone, and profile links
play no part in deciding how to reposition a candidate, and the assembler copies them
onto the resume directly from the source profile. Withholding them keeps a researcher's
contact details out of every provider's logs.
"""

from __future__ import annotations

from pydantic import BaseModel

from resumelab.llm.prompts import Section
from resumelab.models.analysis import JobAnalysis
from resumelab.models.candidate import CandidateProfile
from resumelab.models.strategy import TransformationStrategy

EXCLUDED_FROM_PROMPTS: set[str] = {"personal"}
"""Profile sections withheld from every prompt."""


def _as_json(model: BaseModel, *, exclude: set[str] | None = None) -> str:
    return model.model_dump_json(indent=2, exclude=exclude)


def profile_section(profile: CandidateProfile) -> Section:
    """Render the source profile, minus personal details."""
    return Section(
        label="CANDIDATE PROFILE",
        content=_as_json(profile, exclude=EXCLUDED_FROM_PROMPTS),
    )


def analysis_section(analysis: JobAnalysis) -> Section:
    """Render the structured job analysis produced by the previous stage."""
    return Section(label="JOB ANALYSIS", content=_as_json(analysis))


def strategy_section(strategy: TransformationStrategy) -> Section:
    """Render the global plan every rewriting stage works from."""
    return Section(label="TRANSFORMATION STRATEGY", content=_as_json(strategy))
