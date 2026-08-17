"""What a reader chose to show on the page, and in what order.

These are presentation choices, not content choices. Nothing here changes what a run
generated — the same :class:`~resumelab.models.resume.GeneratedResume` rendered under
two different sets of options is the same resume shown two ways, which is what keeps
runs comparable even when they were drawn differently.

That is also why the options live in ``rendering`` rather than travelling with the
resume: the model never sees them, and a run's recorded content is unaffected by how
someone later chose to lay it out.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, field_validator


class ResumeSection(StrEnum):
    """A body section whose position on the page is the reader's to choose.

    The summary and the achievements are deliberately absent. Neither is a section a
    reader scans *for*: the summary is read first or not at all, and achievements are
    a footnote to everything above them. Pinning them costs two degrees of freedom
    nobody wants and saves every caller from having to order them.
    """

    EDUCATION = "education"
    EXPERIENCE = "experience"
    PROJECTS = "projects"
    SKILLS = "skills"


DEFAULT_SECTION_ORDER: tuple[ResumeSection, ...] = (
    ResumeSection.EDUCATION,
    ResumeSection.EXPERIENCE,
    ResumeSection.PROJECTS,
    ResumeSection.SKILLS,
)
"""Education first, as a resume whose strongest claim is a degree in progress."""


class RenderOptions(BaseModel):
    """How to lay out a resume that has already been generated.

    Validated rather than trusted because these arrive from outside the pipeline —
    a caller choosing what to show. An order that named a section twice would draw it
    twice, and one that omitted a section would silently drop it from the page; both
    are worth rejecting where they can still be reported.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    include_summary: bool = True
    include_gpa: bool = True
    section_order: tuple[ResumeSection, ...] = DEFAULT_SECTION_ORDER

    @field_validator("section_order")
    @classmethod
    def _check_section_order(cls, value: tuple[ResumeSection, ...]) -> tuple[ResumeSection, ...]:
        """Require a permutation: every body section named exactly once."""
        if len(set(value)) != len(value):
            raise ValueError("must not name a section more than once")
        missing = sorted(section.value for section in set(ResumeSection) - set(value))
        if missing:
            raise ValueError(f"must name every section, missing: {', '.join(missing)}")
        return value


DEFAULT_RENDER_OPTIONS = RenderOptions()
"""Everything shown, in the default order. Shared because the model is frozen."""
