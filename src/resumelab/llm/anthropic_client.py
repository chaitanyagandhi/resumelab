"""Anthropic adapter for the :class:`~resumelab.llm.client.LLMClient` protocol.

Uses the Messages API's structured outputs (``messages.parse`` with an
``output_format``), so responses are validated into Pydantic models rather than
scraped out of prose. Retry policy is inherited from
:class:`~resumelab.llm.base.RetryingLLMClient`, and the SDK client is constructed
with ``max_retries=0`` so that policy is the only one in effect.

Two provider differences shape this adapter:

* **No temperature.** Current Claude models reject ``temperature``, ``top_p``, and
  ``top_k`` with a 400, so none are sent. Reasoning depth is controlled by
  ``ANTHROPIC_EFFORT`` instead, which is the equivalent quality/cost dial.
* **``max_tokens`` is required** and bounds reasoning as well as the response, so it
  is configured explicitly and a truncated response is reported as such.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

from anthropic import (
    Anthropic,
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    OverloadedError,
    PermissionDeniedError,
    RateLimitError,
)
from anthropic.types import MessageParam
from pydantic import BaseModel

from resumelab.config import LLMProvider, Settings
from resumelab.exceptions import LLMGenerationError
from resumelab.llm.base import MalformedResponseError, RetryingLLMClient
from resumelab.llm.client import TokenUsage

if TYPE_CHECKING:
    from resumelab.llm.client import LLMClient

logger = logging.getLogger(__name__)


class AnthropicClient(RetryingLLMClient):
    """Structured-output client backed by the Anthropic Messages API."""

    retryable_errors: ClassVar[tuple[type[Exception], ...]] = (
        APITimeoutError,
        APIConnectionError,
        RateLimitError,
        InternalServerError,
        OverloadedError,
    )
    fatal_errors: ClassVar[tuple[type[Exception], ...]] = (
        AuthenticationError,
        PermissionDeniedError,
        BadRequestError,
        NotFoundError,
    )

    def __init__(
        self,
        settings: Settings,
        *,
        client: Anthropic | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        """Build an adapter.

        Args:
            settings: Model, effort, output budget, timeout, and retry budget.
            client: Pre-built SDK client, injected by tests. When omitted, one is
                created with the configured key and timeout, and with the SDK's own
                retries disabled.
            sleeper: Backoff sleep function, injected by tests so retry behaviour can
                be verified without real delays.
        """
        super().__init__(settings, sleeper=sleeper)
        self._client = client or Anthropic(
            api_key=settings.api_key_for(LLMProvider.ANTHROPIC).get_secret_value(),
            timeout=settings.llm_timeout_seconds,
            max_retries=0,
        )

    @property
    def model(self) -> str:
        return self._settings.anthropic_model

    @property
    def _api_key(self) -> str:
        key = self._settings.anthropic_api_key
        return key.get_secret_value() if key is not None else ""

    def _request[T: BaseModel](
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        repair_hint: str | None,
        response_model: type[T],
        purpose: str,
    ) -> T:
        logger.debug(
            "llm request provider=anthropic purpose=%s model=%s schema=%s",
            purpose,
            self.model,
            response_model.__name__,
        )
        message = self._client.messages.parse(
            model=self.model,
            max_tokens=self._settings.anthropic_max_tokens,
            system=system_prompt,
            messages=self._build_messages(user_prompt, repair_hint),
            output_config={"effort": self._settings.anthropic_effort},
            output_format=response_model,
        )

        usage = getattr(message, "usage", None)
        prompt_tokens = getattr(usage, "input_tokens", 0) or 0
        completion_tokens = getattr(usage, "output_tokens", 0) or 0
        self._record_usage(
            TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
        )

        self._check_stop_reason(message, purpose)
        if message.parsed_output is None:
            raise MalformedResponseError("the response contained no structured content")

        self._log_completed(purpose)
        return message.parsed_output

    def _check_stop_reason(self, message: object, purpose: str) -> None:
        """Reject responses that stopped for a reason that invalidates the content."""
        stop_reason = getattr(message, "stop_reason", None)
        if stop_reason == "refusal":
            raise LLMGenerationError(
                f"The model declined to produce {purpose} "
                f"(category: {_refusal_category(message)}). "
                "The job description or profile may have tripped a safety classifier."
            )
        if stop_reason == "max_tokens":
            raise LLMGenerationError(
                f"The model hit its output limit while generating {purpose}. "
                "Raise ANTHROPIC_MAX_TOKENS, or lower ANTHROPIC_EFFORT so that less "
                "of the budget is spent on reasoning."
            )

    def _build_messages(
        self,
        user_prompt: str,
        repair_hint: str | None,
    ) -> list[MessageParam]:
        """Build the user turn.

        The system prompt is a top-level parameter on this API rather than a message.
        Consecutive user messages are permitted and are combined into one turn.
        """
        messages: list[MessageParam] = [MessageParam(role="user", content=user_prompt)]
        if repair_hint is not None:
            messages.append(MessageParam(role="user", content=repair_hint))
        return messages

    def _describe_fatal(self, exc: Exception, purpose: str) -> str:
        """Explain a non-retryable failure without quoting the provider's response.

        Authentication failures are the sensitive case: providers echo part of the
        rejected credential back in the error body.
        """
        if isinstance(exc, AuthenticationError):
            return "Anthropic rejected the API key. Check ANTHROPIC_API_KEY in your environment."
        if isinstance(exc, PermissionDeniedError):
            return f"This API key is not permitted to use model {self.model!r}."
        if isinstance(exc, NotFoundError):
            return f"Model {self.model!r} was not found. Check ANTHROPIC_MODEL."
        return self._scrub(f"The request for {purpose} was rejected: {exc}")


def _refusal_category(message: object) -> str:
    """Read the refusal category, which is informational and often absent."""
    details = getattr(message, "stop_details", None)
    return getattr(details, "category", None) or "unspecified"


if TYPE_CHECKING:

    def _satisfies_protocol(client: AnthropicClient) -> LLMClient:
        """Static assertion that the adapter still implements the protocol.

        Checked by mypy; never executed.
        """
        return client
