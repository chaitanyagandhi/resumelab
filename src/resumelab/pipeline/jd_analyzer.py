"""Stage 2 — read the target job description into a structured analysis.

This is the first stage that calls a model, and everything downstream is conditioned
on its output: the transformation strategy aims the candidate at the
``technical_identity`` extracted here.

The job description is passed as untrusted, fenced data. A posting that contains text
resembling instructions is analyzed as evidence about the employer, never obeyed.
"""

from __future__ import annotations

import logging

from resumelab.exceptions import JDAnalysisError, LLMGenerationError
from resumelab.llm.client import LLMClient
from resumelab.llm.prompts import JD_ANALYSIS_PROMPT, Section
from resumelab.models.analysis import JobAnalysis
from resumelab.models.job import JobDescription

logger = logging.getLogger(__name__)


def analyze_job_description(
    job_description: JobDescription,
    *,
    client: LLMClient,
) -> JobAnalysis:
    """Extract a structured analysis of ``job_description``.

    Args:
        job_description: The normalized, validated target posting.
        client: The LLM client to use, injected by the caller.

    Returns:
        The validated :class:`JobAnalysis`.

    Raises:
        JDAnalysisError: If no valid analysis could be produced.
    """
    logger.info(
        "analyzing job description source=%s characters=%d",
        job_description.source.value,
        job_description.character_count,
    )

    try:
        analysis = client.generate_structured(
            system_prompt=JD_ANALYSIS_PROMPT.system,
            user_prompt=JD_ANALYSIS_PROMPT.user(
                Section(
                    label="JOB DESCRIPTION",
                    content=job_description.text,
                    untrusted=True,
                )
            ),
            response_model=JobAnalysis,
            purpose=JD_ANALYSIS_PROMPT.name,
        )
    except LLMGenerationError as exc:
        raise JDAnalysisError(f"Could not analyze the job description.\n  {exc}") from exc

    logger.info(
        "analyzed job description company=%s role=%s archetype=%s",
        analysis.company or "<unnamed>",
        analysis.role_title,
        analysis.role_archetype,
    )
    logger.debug("target technical identity: %s", analysis.technical_identity)
    return analysis
