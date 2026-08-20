"""Shared retry, repair, and accounting behaviour for provider adapters.

Every provider needs the same policy: retry transient failures with backoff, fail
fast on failures that would repeat identically, ask the model to repair a response
that failed schema validation, and never let a credential escape through a log line.
Only the request itself is provider-specific, so that is the only thing subclasses
implement.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar

from pydantic import BaseModel, ValidationError

from resumelab.config import Settings
from resumelab.exceptions import LLMGenerationError
from resumelab.llm.client import LLMCallStats, TokenUsage
from resumelab.utils.errors import describe_validation_error

logger = logging.getLogger(__name__)

BACKOFF_BASE_SECONDS = 1.0
"""First retry waits this long; each further attempt doubles it."""

MAX_BACKOFF_SECONDS = 30.0
"""Ceiling on a single wait, so a long retry budget cannot stall a run."""

MAX_REJECTED_VALUE_CHARACTERS = 4000
"""Cap on how much of a rejected value is quoted back to the model.

Generous, because the whole point is to hand back the text being edited, but bounded
so a runaway response cannot grow the next request without limit.
"""


class MalformedResponseError(Exception):
    """Internal signal that a response was structurally unusable and may be repaired."""


class RetryingLLMClient(ABC):
    """Base for structured-output clients with a shared retry and repair policy."""

    retryable_errors: ClassVar[tuple[type[Exception], ...]] = ()
    """Transient transport and server-side failures worth attempting again."""

    fatal_errors: ClassVar[tuple[type[Exception], ...]] = ()
    """Failures that would fail identically on every retry."""

    def __init__(
        self,
        settings: Settings,
        *,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings
        self._sleeper = sleeper
        self._stats = LLMCallStats()

    @property
    @abstractmethod
    def model(self) -> str:
        """Identifier of the model being used, recorded in run metadata."""

    @property
    def stats(self) -> LLMCallStats:
        """Snapshot of calls and token usage so far."""
        return self._stats

    def generate_structured[T: BaseModel](
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        purpose: str,
        fallback_model: type[T] | None = None,
    ) -> T:
        """Generate a response validated against ``response_model``.

        Transient failures are retried with exponential backoff. A response that
        fails schema validation is retried with the validation errors appended, so
        the model can repair its own output. Authentication and request errors fail
        immediately.

        ``fallback_model`` is asked for on the final attempt instead of
        ``response_model``. It is the same schema with the rules that are worth
        asking twice for relaxed, so a run cannot end over a constraint the model has
        already refused three times to satisfy. It costs no extra call: the last
        attempt happens either way, and this decides what is done with its answer.
        """
        max_attempts = self._settings.llm_max_retries + 1
        repair_hint: str | None = None
        failure = "no attempt was made"

        for attempt in range(1, max_attempts + 1):
            wanted = response_model
            if fallback_model is not None and attempt == max_attempts:
                wanted = fallback_model
                logger.info(
                    "asking for %s on the final attempt purpose=%s", wanted.__name__, purpose
                )
            try:
                return self._request(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    repair_hint=repair_hint,
                    response_model=wanted,
                    purpose=purpose,
                )
            except self.fatal_errors as exc:
                raise LLMGenerationError(self._describe_fatal(exc, purpose)) from exc
            except self.retryable_errors as exc:
                failure = self._scrub(f"{type(exc).__name__}: {exc}")
                logger.warning(
                    "llm call failed purpose=%s attempt=%d/%d error=%s",
                    purpose,
                    attempt,
                    max_attempts,
                    type(exc).__name__,
                )
                if attempt < max_attempts:
                    self._backoff(attempt)
            except ValidationError as exc:
                failure = describe_validation_error(exc, "response did not match the schema:")
                repair_hint = _validation_repair_prompt(exc)
                logger.warning(
                    "llm response failed validation purpose=%s attempt=%d/%d errors=%d",
                    purpose,
                    attempt,
                    max_attempts,
                    exc.error_count(),
                )
            except MalformedResponseError as exc:
                failure = str(exc)
                repair_hint = _repair_prompt(failure)
                logger.warning(
                    "llm response unusable purpose=%s attempt=%d/%d",
                    purpose,
                    attempt,
                    max_attempts,
                )

        raise LLMGenerationError(
            f"Gave up generating {purpose} after {max_attempts} attempt(s).\n  {failure}"
        )

    @abstractmethod
    def _request[T: BaseModel](
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        repair_hint: str | None,
        response_model: type[T],
        purpose: str,
    ) -> T:
        """Perform one structured-output request and return the parsed value."""

    @abstractmethod
    def _describe_fatal(self, exc: Exception, purpose: str) -> str:
        """Explain a non-retryable failure without leaking the credential."""

    @property
    @abstractmethod
    def _api_key(self) -> str:
        """The credential in use, so it can be scrubbed from provider messages."""

    def _record_usage(self, usage: TokenUsage) -> None:
        """Accumulate reported token usage; a call with no usage still counts."""
        self._stats = self._stats.record(usage)

    def _backoff(self, attempt: int) -> None:
        delay = min(BACKOFF_BASE_SECONDS * 2 ** (attempt - 1), MAX_BACKOFF_SECONDS)
        logger.debug("backing off seconds=%.1f before attempt=%d", delay, attempt + 1)
        self._sleeper(delay)

    def _scrub(self, text: str) -> str:
        """Remove the API key from text that came from outside this process."""
        key = self._api_key
        return text.replace(key, "***") if key else text

    def _log_completed(self, purpose: str) -> None:
        logger.info(
            "llm call completed purpose=%s model=%s total_tokens=%d",
            purpose,
            self.model,
            self._stats.usage.total_tokens,
        )


def _repair_prompt(failure: str) -> str:
    """Ask the model to fix a response that was structurally unusable.

    Used when there is nothing to quote back — the response could not be parsed at
    all, so there is no rejected value to show.
    """
    return (
        "Your previous response was rejected because it "
        f"{failure}\n"
        "Return a corrected response that satisfies the required schema exactly. "
        "Do not explain the correction."
    )


def _validation_repair_prompt(exc: ValidationError) -> str:
    """Ask the model to correct a response, quoting what it actually returned.

    Naming the broken rule is not enough on its own. Told only "must be at most 300
    characters, got 308", a model writes a *new* summary of about the same length and
    the run burns its whole retry budget landing in the same place — which is exactly
    how runs were failing. Shown the text it wrote, the task becomes editing, and one
    attempt usually does it.

    This is the only place a rejected value is quoted, and it goes to the model that
    just produced it rather than to a log or the CLI. That is why
    :func:`describe_validation_error` still omits values: those messages are seen by
    people and end up in files.
    """
    lines: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) if error["loc"] else "<root>"
        lines.append(f"- {location}: {error['msg']}")
        rejected = _render_rejected(error.get("input"))
        if rejected:
            lines.append(f"  you returned: {rejected}")

    return (
        "Your previous response was rejected. Fix exactly these problems:\n"
        + "\n".join(lines)
        + "\nEdit what you wrote so it satisfies every rule, keeping the content and "
        "meaning. Do not start over, and do not explain the correction."
    )


def _render_rejected(value: object) -> str:
    """Render a rejected value for the model, bounded so a repair cannot bloat.

    Truncated rather than dropped when long: a model editing the first part of an
    over-long list still gets the correction it needs.
    """
    if value is None:
        return ""
    text = value if isinstance(value, str) else repr(value)
    if len(text) > MAX_REJECTED_VALUE_CHARACTERS:
        return f"{text[:MAX_REJECTED_VALUE_CHARACTERS]}… (truncated)"
    return text
