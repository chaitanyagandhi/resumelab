"""The global plan for repositioning the candidate.

Bullets are never rewritten independently. A resume whose sections were each tailored
in isolation reads as a set of unrelated claims; the transformation being studied
produces a coherent professional identity, so one plan is produced first and every
later stage works from it.

The directions are keyed by name to entries in the source profile. Matching is
normalized, because a model asked to echo ``"LoanFlow"`` will sometimes return
``"loanflow "``.
"""

from __future__ import annotations

from pydantic import BaseModel, model_validator

from resumelab.models.candidate import REQUIRED_PROJECT_COUNT
from resumelab.models.common import GENERATED_MODEL_CONFIG, RequiredText, TermList


def match_key(name: str) -> str:
    """Normalize a name so it can be matched across the model's echo of it."""
    return " ".join(name.split()).casefold()


class ExperienceDirection(BaseModel):
    """How one role should be reframed."""

    model_config = GENERATED_MODEL_CONFIG

    experience: RequiredText
    """The company, echoed from the profile, identifying which role this applies to."""

    target_framing: RequiredText
    concepts_to_emphasize: TermList
    jd_terms_to_incorporate: TermList


class ProjectDirection(BaseModel):
    """How one project should be repositioned."""

    model_config = GENERATED_MODEL_CONFIG

    project: RequiredText
    """The project name, echoed from the profile."""

    new_positioning: RequiredText
    possible_title_direction: RequiredText
    concepts_to_incorporate: TermList


class TransformationStrategy(BaseModel):
    """One coherent plan for transforming the candidate toward the target identity."""

    model_config = GENERATED_MODEL_CONFIG

    target_identity: RequiredText
    summary_direction: RequiredText
    experience_directions: tuple[ExperienceDirection, ...]
    project_directions: tuple[ProjectDirection, ...]
    skills_priority: TermList
    tone: RequiredText
    overall_strategy: RequiredText

    @model_validator(mode="after")
    def _check_directions_are_usable(self) -> TransformationStrategy:
        """Reject shapes the transformation stages could not consume."""
        if not self.experience_directions:
            raise ValueError("experience_directions must cover at least one role")
        if len(self.project_directions) != REQUIRED_PROJECT_COUNT:
            raise ValueError(
                f"project_directions must contain exactly {REQUIRED_PROJECT_COUNT} entries, "
                f"got {len(self.project_directions)}"
            )
        _reject_duplicates(
            [direction.experience for direction in self.experience_directions],
            "experience_directions",
        )
        _reject_duplicates(
            [direction.project for direction in self.project_directions],
            "project_directions",
        )
        return self

    def direction_for_experience(self, company: str) -> ExperienceDirection | None:
        """Find the direction for a role, or ``None`` if the plan omits it."""
        key = match_key(company)
        return next(
            (d for d in self.experience_directions if match_key(d.experience) == key),
            None,
        )

    def direction_for_project(self, name: str) -> ProjectDirection | None:
        """Find the direction for a project, or ``None`` if the plan omits it."""
        key = match_key(name)
        return next(
            (d for d in self.project_directions if match_key(d.project) == key),
            None,
        )


def _reject_duplicates(names: list[str], field: str) -> None:
    """Two directions for the same entry make the lookup ambiguous."""
    keys = [match_key(name) for name in names]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{field} contains more than one direction for the same entry")
