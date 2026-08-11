"""Stage 3 — decide how the candidate gets repositioned, before anything is rewritten.

Producing one plan first is what makes the finished resume coherent. Every later
stage reads its direction from here rather than deciding for itself, so the summary,
the experience bullets, and the projects tell one story instead of three.

The plan is checked against the source profile before it is returned: a direction the
transformers cannot find, or a profile entry the plan forgot, fails the run rather
than silently producing an untransformed section.
"""

from __future__ import annotations

import logging

from resumelab.exceptions import LLMGenerationError
from resumelab.llm.client import LLMClient
from resumelab.llm.prompts import STRATEGY_PROMPT
from resumelab.models.analysis import JobAnalysis
from resumelab.models.candidate import CandidateProfile
from resumelab.models.strategy import TransformationStrategy, match_key
from resumelab.pipeline.context import analysis_section, profile_section

logger = logging.getLogger(__name__)


def build_transformation_strategy(
    profile: CandidateProfile,
    analysis: JobAnalysis,
    *,
    client: LLMClient,
) -> TransformationStrategy:
    """Produce the global plan for transforming ``profile`` toward ``analysis``.

    Args:
        profile: The immutable source profile.
        analysis: The structured reading of the target job description.
        client: The LLM client to use, injected by the caller.

    Returns:
        A :class:`TransformationStrategy` covering every experience and project.

    Raises:
        LLMGenerationError: If generation fails, or the plan does not cover the
            profile it was given.
    """
    logger.info(
        "building transformation strategy archetype=%s experiences=%d projects=%d",
        analysis.role_archetype,
        len(profile.experiences),
        len(profile.projects),
    )

    strategy = client.generate_structured(
        system_prompt=STRATEGY_PROMPT.system,
        user_prompt=STRATEGY_PROMPT.user(
            profile_section(profile),
            analysis_section(analysis),
        ),
        response_model=TransformationStrategy,
        purpose=STRATEGY_PROMPT.name,
    )
    _check_covers_profile(strategy, profile)

    logger.info("transformation strategy target_identity=%s", strategy.target_identity)
    logger.debug("overall strategy: %s", strategy.overall_strategy)
    return strategy


def _check_covers_profile(
    strategy: TransformationStrategy,
    profile: CandidateProfile,
) -> None:
    """Fail loudly when the plan and the profile do not line up.

    The transformers look their direction up by name. A missing direction would leave
    a section untransformed, which is worse than a failed run: the artifact would
    look like a result.
    """
    missing_experiences = [
        experience.company
        for experience in profile.experiences
        if strategy.direction_for_experience(experience.company) is None
    ]
    missing_projects = [
        project.name
        for project in profile.projects
        if strategy.direction_for_project(project.name) is None
    ]

    if missing_experiences or missing_projects:
        raise LLMGenerationError(
            "The transformation strategy does not cover the whole profile.\n"
            f"  experiences without a direction: {missing_experiences or 'none'}\n"
            f"  projects without a direction: {missing_projects or 'none'}"
        )

    _warn_about_unknown_directions(strategy, profile)


def _warn_about_unknown_directions(
    strategy: TransformationStrategy,
    profile: CandidateProfile,
) -> None:
    """Note directions that match nothing, which the transformers will ignore.

    Only experiences can have stray directions. The schema pins project directions to
    exactly the profile's project count, so once every project is covered there is no
    room left for one that matches nothing.
    """
    known_experiences = {match_key(experience.company) for experience in profile.experiences}

    for direction in strategy.experience_directions:
        if match_key(direction.experience) not in known_experiences:
            logger.warning(
                "strategy names an experience that is not in the profile: %s",
                direction.experience,
            )
