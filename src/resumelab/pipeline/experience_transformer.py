"""Stage 5 — rewrite each role's bullets around the target identity.

Roles are rewritten one at a time, each with its own slice of the strategy. Bullets
already written earlier in the run are passed along, because otherwise the calls
cannot see one another and converge on the same opening verbs and the same claimed
impact — the clearest tell of machine-written bullets.

Company, title, dates, and location never pass through the model. They are copied
from the source profile after generation, so a rewrite cannot alter where someone
worked or when.
"""

from __future__ import annotations

import logging

from resumelab.exceptions import LLMGenerationError
from resumelab.llm.client import LLMClient
from resumelab.llm.prompts import EXPERIENCE_PROMPT
from resumelab.models.analysis import JobAnalysis
from resumelab.models.candidate import CandidateProfile, Experience
from resumelab.models.resume import ExperienceBullets, GeneratedExperience
from resumelab.models.strategy import ExperienceDirection, TransformationStrategy
from resumelab.pipeline.context import (
    already_written_section,
    analysis_section,
    direction_section,
    source_experience_section,
    strategy_section,
)

logger = logging.getLogger(__name__)


def transform_experiences(
    profile: CandidateProfile,
    analysis: JobAnalysis,
    strategy: TransformationStrategy,
    *,
    client: LLMClient,
) -> tuple[GeneratedExperience, ...]:
    """Rewrite every role in ``profile`` according to ``strategy``.

    Args:
        profile: The immutable source profile.
        analysis: The structured reading of the target posting.
        strategy: The global plan, which must carry a direction for every role.
        client: The LLM client to use, injected by the caller.

    Returns:
        One :class:`GeneratedExperience` per source role, in profile order.

    Raises:
        LLMGenerationError: If generation fails, or the plan omits a role.
    """
    transformed: list[GeneratedExperience] = []
    written: list[str] = []

    for experience in profile.experiences:
        direction = strategy.direction_for_experience(experience.company)
        if direction is None:
            raise LLMGenerationError(
                f"The transformation strategy has no direction for {experience.company!r}."
            )

        logger.info("transforming experience company=%s", experience.company)
        bullets = _rewrite_bullets(
            experience,
            direction,
            analysis,
            strategy,
            written=written,
            client=client,
        )
        transformed.append(_assemble(experience, bullets))
        written.extend(bullets)

    logger.info("transformed experiences count=%d", len(transformed))
    return tuple(transformed)


def _rewrite_bullets(
    experience: Experience,
    direction: ExperienceDirection,
    analysis: JobAnalysis,
    strategy: TransformationStrategy,
    *,
    written: list[str],
    client: LLMClient,
) -> tuple[str, ...]:
    """Generate the bullets for one role."""
    sections = [
        source_experience_section(experience),
        direction_section(direction),
        analysis_section(analysis),
        strategy_section(strategy),
    ]
    if written:
        sections.append(already_written_section(written))

    generated = client.generate_structured(
        system_prompt=EXPERIENCE_PROMPT.system,
        user_prompt=EXPERIENCE_PROMPT.user(*sections),
        response_model=ExperienceBullets,
        purpose=EXPERIENCE_PROMPT.name,
    )
    return generated.bullets


def _assemble(experience: Experience, bullets: tuple[str, ...]) -> GeneratedExperience:
    """Combine generated bullets with the factual anchors from the source profile."""
    return GeneratedExperience(
        company=experience.company,
        title=experience.title,
        location=experience.location,
        start_date=experience.start_date,
        end_date=experience.end_date,
        bullets=bullets,
    )
