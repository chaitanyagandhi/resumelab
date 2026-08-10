"""OpenAI adapter for the :class:`~resumelab.llm.client.LLMClient` protocol.

Responsibilities kept here, and nowhere else in the codebase:

* talking to the OpenAI SDK, using structured outputs so responses are parsed into
  Pydantic models rather than scraped out of prose;
* deciding which failures are worth retrying, and waiting between attempts;
* asking the model to repair a response that failed schema validation, as required
  for research runs that must never continue on malformed data;
* making sure the API key cannot escape through a log line or an exception.

Retry policy lives here rather than in the SDK: the underlying client is constructed
with ``max_retries=0`` so this module's budget is the only one in effect and the
recorded call count reflects reality.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ContentFilterFinishReasonError,
    InternalServerError,
    LengthFinishReasonError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel, ValidationError

from resumelab.config import Settings
from resumelab.exceptions import LLMGenerationError
from resumelab.llm.client import LLMCallStats, TokenUsage
from resumelab.utils.errors import describe_validation_error

if TYPE_CHECKING:
    from resumelab.llm.client import LLMClient

logger = logging.getLogger(__name__)

BACKOFF_BASE_SECONDS = 1.0
"""First retry waits this long; each further attempt doubles it."""

MAX_BACKOFF_SECONDS = 30.0
"""Ceiling on a single wait, so a long retry budget cannot stall a run."""

RETRYABLE_ERRORS = (
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    InternalServerError,
)
"""Transient transport and server-side failures worth attempting again."""

FATAL_ERRORS = (
    AuthenticationError,
    PermissionDeniedError,
    BadRequestError,
    NotFoundError,
)
"""Failures that would fail identically on every retry."""


class MalformedResponseError(Exception):
    """Internal signal that a response was structurally unusable and may be repaired."""


class OpenAIClient:
    """Structured-output client backed by the OpenAI Chat Completions API."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: OpenAI | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        """Build an adapter.

        Args:
            settings: Model, temperature, timeout, and retry budget.
            client: Pre-built SDK client, injected by tests. When omitted, one is
                created with the configured key and timeout, and with the SDK's own
                retries disabled.
            sleeper: Backoff sleep function, injected by tests so retry behaviour can
                be verified without real delays.
        """
        self._settings = settings
        self._sleeper = sleeper
        self._stats = LLMCallStats()
        self._client = client or OpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.openai_timeout_seconds,
            max_retries=0,
        )

    @property
    def model(self) -> str:
        return self._settings.openai_model

    @property
    def stats(self) -> LLMCallStats:
        return self._stats

    def generate_structured[T: BaseModel](
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        purpose: str,
    ) -> T:
        """Generate a response validated against ``response_model``.

        Transient failures are retried with exponential backoff. A response that
        fails schema validation is retried with the validation errors appended, so
        the model can repair its own output. Authentication and request errors fail
        immediately.
        """
        max_attempts = self._settings.openai_max_retries + 1
        repair_hint: str | None = None
        failure = "no attempt was made"

        for attempt in range(1, max_attempts + 1):
            messages = self._build_messages(system_prompt, user_prompt, repair_hint)
            try:
                return self._request(messages, response_model, purpose)
            except FATAL_ERRORS as exc:
                raise LLMGenerationError(self._describe_fatal(exc, purpose)) from exc
            except RETRYABLE_ERRORS as exc:
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
                repair_hint = _repair_prompt(failure)
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

    def _request[T: BaseModel](
        self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[T],
        purpose: str,
    ) -> T:
        """Perform one structured-output request and return the parsed value."""
        logger.debug(
            "llm request purpose=%s model=%s schema=%s",
            purpose,
            self.model,
            response_model.__name__,
        )
        try:
            completion = self._client.chat.completions.parse(
                model=self._settings.openai_model,
                temperature=self._settings.openai_temperature,
                messages=messages,
                response_format=response_model,
            )
        except LengthFinishReasonError as exc:
            raise LLMGenerationError(
                f"The model hit its output limit while generating {purpose}. "
                "Shorten the input or raise the model's output budget."
            ) from exc
        except ContentFilterFinishReasonError as exc:
            raise LLMGenerationError(
                f"The provider's content filter blocked the response for {purpose}."
            ) from exc

        self._record_usage(completion)

        message = completion.choices[0].message
        if message.refusal:
            raise LLMGenerationError(
                f"The model refused to produce {purpose}: {self._scrub(message.refusal)}"
            )
        if message.parsed is None:
            raise MalformedResponseError("the response contained no structured content")

        logger.info(
            "llm call completed purpose=%s model=%s total_tokens=%d",
            purpose,
            self.model,
            self._stats.usage.total_tokens,
        )
        return message.parsed

    def _record_usage(self, completion: object) -> None:
        """Accumulate reported token usage; absent usage still counts as a call."""
        usage = getattr(completion, "usage", None)
        self._stats = self._stats.record(
            TokenUsage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(usage, "total_tokens", 0) or 0,
            )
        )

    def _build_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        repair_hint: str | None,
    ) -> list[ChatCompletionMessageParam]:
        messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(role="system", content=system_prompt),
            ChatCompletionUserMessageParam(role="user", content=user_prompt),
        ]
        if repair_hint is not None:
            messages.append(ChatCompletionUserMessageParam(role="user", content=repair_hint))
        return messages

    def _backoff(self, attempt: int) -> None:
        delay = min(BACKOFF_BASE_SECONDS * 2 ** (attempt - 1), MAX_BACKOFF_SECONDS)
        logger.debug("backing off seconds=%.1f before attempt=%d", delay, attempt + 1)
        self._sleeper(delay)

    def _describe_fatal(self, exc: APIStatusError, purpose: str) -> str:
        """Explain a non-retryable failure without quoting the provider's response.

        Authentication failures are the sensitive case: providers echo part of the
        rejected credential back in the error body.
        """
        if isinstance(exc, AuthenticationError):
            return "OpenAI rejected the API key. Check OPENAI_API_KEY in your environment."
        if isinstance(exc, PermissionDeniedError):
            return f"This API key is not permitted to use model {self.model!r}."
        if isinstance(exc, NotFoundError):
            return f"Model {self.model!r} was not found. Check OPENAI_MODEL."
        return self._scrub(f"The request for {purpose} was rejected: {exc}")

    def _scrub(self, text: str) -> str:
        """Remove the API key from text that came from outside this process."""
        key = self._settings.openai_api_key.get_secret_value()
        return text.replace(key, "***") if key else text


def _repair_prompt(failure: str) -> str:
    """Ask the model to fix a response that failed validation."""
    return (
        "Your previous response was rejected because it "
        f"{failure}\n"
        "Return a corrected response that satisfies the required schema exactly. "
        "Do not explain the correction."
    )


if TYPE_CHECKING:

    def _satisfies_protocol(client: OpenAIClient) -> LLMClient:
        """Static assertion that the adapter still implements the protocol.

        Checked by mypy; never executed.
        """
        return client
