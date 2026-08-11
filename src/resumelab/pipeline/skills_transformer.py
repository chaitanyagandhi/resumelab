"""Stage 7 — build the skills section around the target identity.

The section is regenerated rather than filtered. Which groupings make a candidate
look aligned is itself part of the repositioning, so the labels are chosen per run:
a storage role and a GenAI role should not produce the same headings from the same
source profile.

Everything written so far is passed in, because the skills section is where a resume
is most easily caught contradicting itself — a technology named in a bullet and then
missing from the skills list is noticed immediately.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from resumelab.llm.client import LLMClient
from resumelab.llm.prompts import SKILLS_PROMPT
from resumelab.models.analysis import JobAnalysis
from resumelab.models.candidate import CandidateProfile
from resumelab.models.resume import GeneratedSkills, SkillGroup
from resumelab.models.strategy import TransformationStrategy
from resumelab.pipeline.context import (
    already_written_section,
    analysis_section,
    profile_section,
    strategy_section,
)

logger = logging.getLogger(__name__)


def transform_skills(
    profile: CandidateProfile,
    analysis: JobAnalysis,
    strategy: TransformationStrategy,
    *,
    client: LLMClient,
    already_written: Sequence[str] = (),
) -> tuple[SkillGroup, ...]:
    """Build the skills section for this run.

    Args:
        profile: The immutable source profile, whose skills are the starting point.
        analysis: The structured reading of the target posting.
        strategy: The global plan, whose ``skills_priority`` decides what leads.
        client: The LLM client to use, injected by the caller.
        already_written: Bullets generated earlier in the run, so the section stays
            consistent with the technologies those bullets claim.

    Returns:
        The labeled skill groups, in the order they should be rendered.

    Raises:
        LLMGenerationError: If no valid skills section could be produced.
    """
    logger.info("transforming skills priority=%s", ", ".join(strategy.skills_priority))

    sections = [
        profile_section(profile),
        analysis_section(analysis),
        strategy_section(strategy),
    ]
    if already_written:
        sections.append(already_written_section(already_written))

    generated = client.generate_structured(
        system_prompt=SKILLS_PROMPT.system,
        user_prompt=SKILLS_PROMPT.user(*sections),
        response_model=GeneratedSkills,
        purpose=SKILLS_PROMPT.name,
    )

    logger.info(
        "transformed skills groups=%d skills=%d",
        len(generated.groups),
        generated.skill_count,
    )
    logger.debug("skill groups: %s", ", ".join(group.label for group in generated.groups))
    return generated.groups
