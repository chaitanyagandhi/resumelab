"""Assembly of a run's metadata record.

Every field is read individually. Settings are never serialized wholesale into
metadata, because settings hold API keys, and a run directory is the thing a
researcher shares when they share a result.
"""

from __future__ import annotations

import logging

from resumelab import __version__
from resumelab.config import LLMProvider, Settings
from resumelab.experiment.recorder import ExperimentRun
from resumelab.llm.client import LLMCallStats
from resumelab.llm.prompts import JD_ANALYSIS_PROMPT_VERSION, TRANSFORMATION_PROMPT_VERSION
from resumelab.models.job import JobDescription
from resumelab.models.metadata import RunMetadata, TokenUsageRecord
from resumelab.utils.files import sha256_of_file

logger = logging.getLogger(__name__)


def build_metadata(
    run: ExperimentRun,
    *,
    settings: Settings,
    provider: LLMProvider,
    model: str,
    job_description: JobDescription,
    stats: LLMCallStats,
) -> RunMetadata:
    """Describe how this run was produced.

    Args:
        run: The run being recorded, supplying its id, start time, and duration.
        settings: Loaded settings, read field by field.
        provider: The provider actually used, which may differ from the configured
            default when the CLI overrode it.
        model: The model actually used.
        job_description: The posting this run targeted.
        stats: Call and token accounting from the client.

    Returns:
        The :class:`RunMetadata` to write into the run directory.
    """
    is_openai = provider is LLMProvider.OPENAI
    metadata = RunMetadata(
        run_id=run.run_id,
        timestamp=run.started_at,
        resumelab_version=__version__,
        provider=provider.value,
        model=model,
        # Current Claude models reject sampling parameters, so temperature is
        # meaningless for them; effort is the corresponding dial.
        temperature=settings.openai_temperature if is_openai else None,
        effort=None if is_openai else settings.anthropic_effort,
        jd_analysis_prompt_version=JD_ANALYSIS_PROMPT_VERSION,
        transformation_prompt_version=TRANSFORMATION_PROMPT_VERSION,
        candidate_profile_path=str(settings.candidate_profile_path),
        candidate_profile_hash=sha256_of_file(settings.candidate_profile_path),
        job_description_source=job_description.source.value,
        job_description_characters=job_description.character_count,
        duration_seconds=round(run.elapsed_seconds(), 3),
        llm_calls=stats.call_count,
        token_usage=TokenUsageRecord(
            prompt_tokens=stats.usage.prompt_tokens,
            completion_tokens=stats.usage.completion_tokens,
            total_tokens=stats.usage.total_tokens,
        ),
    )
    logger.info(
        "run metadata provider=%s model=%s llm_calls=%d total_tokens=%d duration=%.1fs",
        metadata.provider,
        metadata.model,
        metadata.llm_calls,
        metadata.token_usage.total_tokens,
        metadata.duration_seconds,
    )
    return metadata
