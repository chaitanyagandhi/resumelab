"""The record of how a run was produced.

Written to ``metadata.json`` in every run directory. This is what makes a generated
resume a result rather than an anecdote: it says which model produced it, under which
prompt versions, from which source profile, and at what cost.

Fields are listed explicitly and assembled one at a time. Settings are never dumped
wholesale into this model, because settings hold API keys.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class TokenUsageRecord(BaseModel):
    """Provider-reported token usage for a run, when the provider reports it."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class RunMetadata(BaseModel):
    """Everything needed to interpret, compare, and repeat a run."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    """Directory name of the run: a UTC timestamp and a label."""

    timestamp: datetime
    """When the run started, in UTC."""

    resumelab_version: str

    # --- what produced the output ------------------------------------------
    provider: str
    model: str
    temperature: float | None = None
    """Sampling temperature, or ``None`` for providers whose models reject it."""

    effort: str | None = None
    """Reasoning depth, where the provider exposes it instead of temperature."""

    # --- what it was produced from -----------------------------------------
    jd_analysis_prompt_version: str
    transformation_prompt_version: str
    candidate_profile_path: str
    candidate_profile_hash: str
    """SHA-256 of the source profile, so two runs can be shown to share an input."""

    job_description_source: str
    job_description_characters: int

    # --- how it turned out --------------------------------------------------
    layout_scale: float | None = None
    """Layout tightening applied to fit one page; 1.0 means none was needed."""

    page_count: int | None = None
    condensed: bool = False
    """Whether the content had to be shortened before it fit."""

    # --- what it cost -------------------------------------------------------
    duration_seconds: float
    llm_calls: int
    token_usage: TokenUsageRecord = Field(default_factory=TokenUsageRecord)
