"""Application configuration.

Settings are read from the process environment, optionally seeded from a ``.env``
file for local development. Secrets are never hardcoded and never printed: API keys
are held as :class:`~pydantic.SecretStr`, so they are redacted in reprs, logs, and
tracebacks.

Configuration is loaded explicitly via :func:`load_settings` and passed down through
the pipeline by dependency injection. There is deliberately no module-level singleton,
which keeps tests isolated and lets a single process run several configurations — for
example when comparing providers or models in an experiment.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Final, Literal

from pydantic import Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from resumelab.exceptions import ConfigurationError
from resumelab.models.candidate import REQUIRED_PROJECT_BULLET_COUNT
from resumelab.models.resume import (
    MAX_BULLET_CHARACTERS,
    MAX_SUMMARY_CHARACTERS,
    REQUIRED_EXPERIENCE_BULLET_COUNT,
    ResumeLimits,
)
from resumelab.utils.errors import describe_validation_error

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

EffortLevel = Literal["low", "medium", "high", "xhigh", "max"]
"""Anthropic reasoning-depth control, the closest analogue to a temperature dial."""

DEFAULT_ENV_FILE: Final = Path(".env")
"""Local development env file. Real deployments rely on the process environment."""


class LLMProvider(StrEnum):
    """Supported LLM providers, recorded in every run's metadata."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class Settings(BaseSettings):
    """Runtime configuration for a ResumeLab research run.

    Every field maps to an environment variable of the same name, upper-cased; see
    ``.env.example`` for the documented set. Instances are frozen so that a run's
    configuration cannot drift after it has been recorded in experiment metadata.
    """

    model_config = SettingsConfigDict(
        env_file=DEFAULT_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Unrelated variables in the environment or .env file are not our concern.
        extra="ignore",
        frozen=True,
    )

    # --- provider selection -------------------------------------------------
    llm_provider: LLMProvider | None = Field(
        default=None,
        description="Provider to use. When unset, inferred from the configured keys.",
    )

    # --- shared LLM behaviour ----------------------------------------------
    llm_max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Retry budget for transient API failures, using exponential backoff.",
    )
    llm_timeout_seconds: float = Field(
        default=60.0,
        gt=0.0,
        le=600.0,
        description="Per-request timeout applied to every LLM call.",
    )

    # --- OpenAI -------------------------------------------------------------
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI API key. Required when the provider is openai.",
    )
    openai_model: str = Field(
        default="gpt-4o",
        min_length=1,
        description="OpenAI model used for every pipeline stage.",
    )
    openai_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Low by default so repeated research runs stay comparable.",
    )

    # --- Anthropic ----------------------------------------------------------
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description="Anthropic API key. Required when the provider is anthropic.",
    )
    anthropic_model: str = Field(
        default="claude-opus-5",
        min_length=1,
        description="Anthropic model used for every pipeline stage.",
    )
    anthropic_max_tokens: int = Field(
        default=16_000,
        ge=1_024,
        le=128_000,
        description=(
            "Output budget per Anthropic call, which the API requires. It covers "
            "reasoning as well as the response, so leave generous headroom."
        ),
    )
    anthropic_effort: EffortLevel = Field(
        default="high",
        description=(
            "Reasoning depth. Current Claude models reject temperature, so this is "
            "the equivalent quality/cost dial."
        ),
    )

    # --- resume length budget ----------------------------------------------
    summary_max_characters: int = Field(
        default=MAX_SUMMARY_CHARACTERS,
        ge=80,
        le=600,
        description="Longest professional summary this run will accept.",
    )
    bullet_max_characters: int = Field(
        default=MAX_BULLET_CHARACTERS,
        ge=80,
        le=400,
        description="Length a bullet is condensed toward. Not a rejection bound.",
    )
    experience_bullet_count: int = Field(
        default=REQUIRED_EXPERIENCE_BULLET_COUNT,
        ge=1,
        le=6,
        description="Bullets to emit per role.",
    )
    project_bullet_count: int = Field(
        default=REQUIRED_PROJECT_BULLET_COUNT,
        ge=1,
        le=6,
        description="Bullets to emit per project.",
    )

    # --- paths --------------------------------------------------------------
    candidate_profile_path: Path = Field(
        default=Path("data/candidate_profile.yaml"),
        description="Immutable source profile; never written to during generation.",
    )
    output_dir: Path = Field(
        default=Path("output"),
        description="Root directory for generated experiment artifacts.",
    )

    # --- diagnostics --------------------------------------------------------
    log_level: LogLevel = Field(
        default="INFO",
        description="Standard library logging level name.",
    )

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: object) -> object:
        """Accept ``debug``/``Debug`` as well as ``DEBUG``."""
        if isinstance(value, str):
            return value.strip().upper()
        return value

    @field_validator("anthropic_effort", "llm_provider", mode="before")
    @classmethod
    def _normalize_lowercase(cls, value: object) -> object:
        """Accept ``OpenAI``/``HIGH`` as well as the canonical lower-case values."""
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @model_validator(mode="after")
    def _check_a_provider_is_usable(self) -> Settings:
        """Fail at load time rather than at the first API call."""
        if self.openai_api_key is None and self.anthropic_api_key is None:
            raise ValueError(
                "no LLM credentials configured; set OPENAI_API_KEY or ANTHROPIC_API_KEY"
            )
        if self.llm_provider is not None and self._key_for(self.llm_provider) is None:
            raise ValueError(
                f"LLM_PROVIDER is {self.llm_provider.value!r} "
                f"but {self.llm_provider.value.upper()}_API_KEY is not set"
            )
        return self

    @property
    def resolved_provider(self) -> LLMProvider:
        """The provider a run will use.

        An explicit ``LLM_PROVIDER`` always wins. Otherwise the provider is inferred
        from whichever key is configured, so a researcher with a single key does not
        have to set it; OpenAI wins when both are present.
        """
        if self.llm_provider is not None:
            return self.llm_provider
        if self.openai_api_key is not None:
            return LLMProvider.OPENAI
        return LLMProvider.ANTHROPIC

    def api_key_for(self, provider: LLMProvider) -> SecretStr:
        """Return the key for ``provider``.

        Raises:
            ConfigurationError: If that provider has no key configured.
        """
        key = self._key_for(provider)
        if key is None:
            raise ConfigurationError(
                f"Provider {provider.value!r} is selected but "
                f"{provider.value.upper()}_API_KEY is not set."
            )
        return key

    def model_for(self, provider: LLMProvider) -> str:
        """Return the model identifier configured for ``provider``."""
        if provider is LLMProvider.OPENAI:
            return self.openai_model
        return self.anthropic_model

    def _key_for(self, provider: LLMProvider) -> SecretStr | None:
        if provider is LLMProvider.OPENAI:
            return self.openai_api_key
        return self.anthropic_api_key

    @property
    def resume_limits(self) -> ResumeLimits:
        """The length budget this run targets."""
        return ResumeLimits(
            summary_max_characters=self.summary_max_characters,
            bullet_max_characters=self.bullet_max_characters,
            experience_bullet_count=self.experience_bullet_count,
            project_bullet_count=self.project_bullet_count,
        )

    @property
    def runs_dir(self) -> Path:
        """Directory holding one sub-directory per experiment run."""
        return self.output_dir / "runs"


def load_settings(env_file: Path | str | None = DEFAULT_ENV_FILE) -> Settings:
    """Load and validate settings.

    Args:
        env_file: Path to a ``.env`` file to seed values from, or ``None`` to read
            only from the process environment. Values already present in the
            environment take precedence over the file.

    Returns:
        A validated, frozen :class:`Settings` instance.

    Raises:
        ConfigurationError: If any setting is missing or invalid. The message names
            the offending variables without echoing their values.
    """
    try:
        # `_env_file` is a real pydantic-settings init argument, but PEP 681
        # dataclass_transform makes mypy synthesize __init__ from the model fields
        # alone, so it cannot see the private settings kwargs.
        return Settings(_env_file=env_file)  # type: ignore[call-arg]
    except ValidationError as exc:
        message = describe_validation_error(
            exc,
            "Invalid ResumeLab configuration:",
            uppercase_locations=True,
        )
        raise ConfigurationError(message) from exc
