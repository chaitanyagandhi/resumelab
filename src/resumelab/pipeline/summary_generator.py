"""Stage 4 — write the professional summary.

The summary is the first thing read and frames everything below it, so it is
generated from the strategy rather than from the source profile: its job is to state
the target identity outright, in the first clause.
"""

from __future__ import annotations

import logging

from resumelab.llm.client import LLMClient
from resumelab.llm.prompts import SUMMARY_PROMPT
from resumelab.models.analysis import JobAnalysis
from resumelab.models.candidate import CandidateProfile
from resumelab.models.resume import GeneratedSummary
from resumelab.models.strategy import TransformationStrategy
from resumelab.pipeline.context import analysis_section, profile_section, strategy_section

logger = logging.getLogger(__name__)


def generate_summary(
    profile: CandidateProfile,
    analysis: JobAnalysis,
    strategy: TransformationStrategy,
    *,
    client: LLMClient,
) -> str:
    """Write the summary that opens the generated resume.

    Args:
        profile: The immutable source profile, for the material behind the claim.
        analysis: The structured reading of the target posting.
        strategy: The global plan, whose ``summary_direction`` this stage executes.
        client: The LLM client to use, injected by the caller.

    Returns:
        The summary text, collapsed to one line and length-checked.

    Raises:
        LLMGenerationError: If no valid summary could be produced.
    """
    logger.info("generating summary target_identity=%s", strategy.target_identity)

    generated = client.generate_structured(
        system_prompt=SUMMARY_PROMPT.system,
        user_prompt=SUMMARY_PROMPT.user(
            profile_section(profile),
            analysis_section(analysis),
            strategy_section(strategy),
        ),
        response_model=GeneratedSummary,
        purpose=SUMMARY_PROMPT.name,
    )

    logger.debug("generated summary characters=%d", len(generated.summary))
    return generated.summary
