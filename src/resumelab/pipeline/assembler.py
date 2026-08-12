"""Stage 8 — combine the generated sections into one validated resume.

Assembly is deterministic and involves no model. The generated sections are placed
alongside the parts of the profile that are carried through untouched — identity,
education, achievements — and the result is validated before it is handed on to be
rendered.

Personal details reach the resume here, having been withheld from every prompt.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from resumelab.models.candidate import CandidateProfile
from resumelab.models.resume import (
    GeneratedExperience,
    GeneratedProject,
    GeneratedResume,
)
from resumelab.validation.resume_validator import validate_resume

logger = logging.getLogger(__name__)


def assemble_resume(
    profile: CandidateProfile,
    *,
    summary: str,
    experiences: Sequence[GeneratedExperience],
    projects: Sequence[GeneratedProject],
    skills: Sequence[str],
) -> GeneratedResume:
    """Build the final resume and check it is fit to render.

    Args:
        profile: The immutable source profile, supplying identity, education, and
            achievements verbatim.
        summary: The generated professional summary.
        experiences: The transformed roles, in the order they should appear.
        projects: The repositioned projects, in the order they should appear.
        skills: The generated skill groups, in render order.

    Returns:
        A validated :class:`GeneratedResume`.

    Raises:
        ResumeValidationError: If the assembled resume is not fit to render.
    """
    logger.info("assembling resume")

    resume = GeneratedResume(
        personal=profile.personal,
        summary=summary,
        education=profile.education,
        experiences=tuple(experiences),
        projects=tuple(projects),
        skills=tuple(skills),
        achievements=profile.achievements,
    )
    validate_resume(resume)

    logger.info(
        "assembled resume experiences=%d projects=%d skills=%d",
        len(resume.experiences),
        len(resume.projects),
        len(resume.skills),
    )
    return resume
