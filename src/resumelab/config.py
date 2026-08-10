"""Application configuration.

Settings are read from the process environment, optionally seeded from a ``.env``
file for local development. Secrets are never hardcoded and never printed: the API
key is held as a :class:`~pydantic.SecretStr`, so it is redacted in reprs, logs, and
tracebacks.

Configuration is loaded explicitly via :func:`load_settings` and passed down through
the pipeline by dependency injection. There is deliberately no module-level singleton,
which keeps tests isolated and lets a single process run several configurations (for
example when comparing models in a future experiment).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from resumelab.exceptions import ConfigurationError
from resumelab.utils.errors import describe_validation_error

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

DEFAULT_ENV_FILE: Final = Path(".env")
"""Local development env file. Real deployments rely on the process environment."""


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

    openai_api_key: SecretStr = Field(
        description="OpenAI API key. Required; never logged or persisted.",
    )
    openai_model: str = Field(
        default="gpt-4o",
        min_length=1,
        description="Model used for every pipeline stage; recorded in run metadata.",
    )
    openai_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Low by default so repeated research runs stay comparable.",
    )
    openai_max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Retry budget for transient API failures, using exponential backoff.",
    )
    openai_timeout_seconds: float = Field(
        default=60.0,
        gt=0.0,
        le=600.0,
        description="Per-request timeout applied to every LLM call.",
    )
    candidate_profile_path: Path = Field(
        default=Path("data/candidate_profile.yaml"),
        description="Immutable source profile; never written to during generation.",
    )
    output_dir: Path = Field(
        default=Path("output"),
        description="Root directory for generated experiment artifacts.",
    )
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
