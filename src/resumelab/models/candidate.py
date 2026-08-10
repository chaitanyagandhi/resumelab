"""Schema for the immutable source candidate profile.

The candidate profile is the experimental control: every run compares *original
profile* against *target JD* against *generated resume*, so the profile must be
stable, fully validated, and impossible to mutate by accident.

Immutability is enforced structurally rather than by convention. Models are frozen
and every collection is a ``tuple``, so no pipeline stage can append to a bullet list
or reassign a field on the object it was handed.

The counts required by the research design live in module constants
(:data:`REQUIRED_PROJECT_COUNT`, :data:`REQUIRED_PROJECT_BULLET_COUNT`) so that an
ablation study can change them in one place.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

REQUIRED_PROJECT_COUNT = 3
"""The research design fixes the profile at exactly three projects."""

REQUIRED_PROJECT_BULLET_COUNT = 3
"""Each source project carries exactly three bullets."""

MIN_EXPERIENCE_BULLET_COUNT = 1
"""Experience bullet counts are not fixed; generation decides how many to emit."""


def _blank_to_none(value: object) -> object:
    """Treat an unpopulated YAML field (``""`` or whitespace) as absent."""
    if isinstance(value, str) and not value.strip():
        return None
    return value


RequiredText = Annotated[str, Field(min_length=1)]
"""Text that must be present and non-blank."""

OptionalText = Annotated[str | None, BeforeValidator(_blank_to_none)]
"""Text that may be omitted; blank strings normalize to ``None``."""


class ProfileModel(BaseModel):
    """Base for every profile model.

    ``extra="forbid"`` matters for a hand-maintained YAML file: a mistyped key such
    as ``bullet:`` becomes a loud error instead of a silently dropped section.
    """

    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        extra="forbid",
    )


class PersonalDetails(ProfileModel):
    """Candidate identity and contact details, reproduced verbatim on the resume."""

    name: RequiredText
    email: RequiredText
    phone: OptionalText = None
    linkedin: OptionalText = None
    github: OptionalText = None
    location: OptionalText = None


class Education(ProfileModel):
    """A degree program. Carried through to the resume without transformation."""

    institution: RequiredText
    degree: RequiredText
    field: OptionalText = None
    location: OptionalText = None
    start_date: OptionalText = None
    end_date: OptionalText = None
    gpa: OptionalText = None
    coursework: tuple[str, ...] = ()


class Experience(ProfileModel):
    """A role held by the candidate.

    Company, title, dates, and location are factual anchors that survive
    transformation. ``bullets`` are the source material that later stages rewrite.
    """

    company: RequiredText
    title: RequiredText
    location: OptionalText = None
    start_date: OptionalText = None
    end_date: OptionalText = None
    description: OptionalText = None
    bullets: tuple[RequiredText, ...] = Field(min_length=MIN_EXPERIENCE_BULLET_COUNT)


class Project(ProfileModel):
    """A project.

    Only ``name`` acts as an anchor. The subtitle, technologies, and bullets are all
    open to aggressive JD-driven repositioning downstream.
    """

    name: RequiredText
    subtitle: OptionalText = None
    date: OptionalText = None
    technologies: tuple[str, ...] = ()
    description: OptionalText = None
    bullets: tuple[RequiredText, ...] = Field(
        min_length=REQUIRED_PROJECT_BULLET_COUNT,
        max_length=REQUIRED_PROJECT_BULLET_COUNT,
    )


class Skills(ProfileModel):
    """The candidate's declared skills, grouped by category.

    Categories are the source vocabulary. The skills transformation stage may
    reorder, prune, or flatten them for the generated resume.
    """

    programming_languages: tuple[str, ...] = ()
    frameworks: tuple[str, ...] = ()
    databases: tuple[str, ...] = ()
    cloud_devops: tuple[str, ...] = ()
    ai_ml: tuple[str, ...] = ()
    other: tuple[str, ...] = ()


class CandidateProfile(ProfileModel):
    """The complete source profile, loaded once per run and never written back."""

    personal: PersonalDetails
    education: tuple[Education, ...] = Field(min_length=1)
    experiences: tuple[Experience, ...] = Field(min_length=1)
    projects: tuple[Project, ...] = Field(
        min_length=REQUIRED_PROJECT_COUNT,
        max_length=REQUIRED_PROJECT_COUNT,
    )
    skills: Skills = Field(default_factory=Skills)
    achievements: tuple[str, ...] = ()
