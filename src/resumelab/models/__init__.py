"""Domain models for ResumeLab."""

from resumelab.models.candidate import (
    MIN_EXPERIENCE_BULLET_COUNT,
    REQUIRED_PROJECT_BULLET_COUNT,
    REQUIRED_PROJECT_COUNT,
    CandidateProfile,
    Education,
    Experience,
    PersonalDetails,
    Project,
    Skills,
)
from resumelab.models.job import (
    MAX_JOB_DESCRIPTION_CHARACTERS,
    MIN_JOB_DESCRIPTION_CHARACTERS,
    JobDescription,
    JobDescriptionSource,
)

__all__ = [
    "MAX_JOB_DESCRIPTION_CHARACTERS",
    "MIN_EXPERIENCE_BULLET_COUNT",
    "MIN_JOB_DESCRIPTION_CHARACTERS",
    "REQUIRED_PROJECT_BULLET_COUNT",
    "REQUIRED_PROJECT_COUNT",
    "CandidateProfile",
    "Education",
    "Experience",
    "JobDescription",
    "JobDescriptionSource",
    "PersonalDetails",
    "Project",
    "Skills",
]
