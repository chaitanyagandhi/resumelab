"""OpenAI adapter for the :class:`~resumelab.llm.client.LLMClient` protocol.

Structured outputs are used throughout, so responses are parsed into Pydantic models
rather than scraped out of prose. Retry policy lives in
:class:`~resumelab.llm.base.RetryingLLMClient`; the SDK client is constructed with
``max_retries=0`` so that policy is the only one in effect and the recorded call count
reflects reality.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

from openai import (
    APIConnectionError,
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
from pydantic import BaseModel

from resumelab.config import LLMProvider, Settings
from resumelab.exceptions import LLMGenerationError
from resumelab.llm.base import MalformedResponseError, RetryingLLMClient
from resumelab.llm.client import TokenUsage

if TYPE_CHECKING:
    from resumelab.llm.client import LLMClient

logger = logging.getLogger(__name__)


class OpenAIClient(RetryingLLMClient):
    """Structured-output client backed by the OpenAI Chat Completions API."""

    retryable_errors: ClassVar[tuple[type[Exception], ...]] = (
        APITimeoutError,
        APIConnectionError,
        RateLimitError,
        InternalServerError,
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
        super().__init__(settings, sleeper=sleeper)
        self._client = client or OpenAI(
            api_key=settings.api_key_for(LLMProvider.OPENAI).get_secret_value(),
            timeout=settings.llm_timeout_seconds,
            max_retries=0,
        )

    @property
    def model(self) -> str:
        return self._settings.openai_model

    @property
    def _api_key(self) -> str:
        key = self._settings.openai_api_key
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
            "llm request provider=openai purpose=%s model=%s schema=%s",
            purpose,
            self.model,
            response_model.__name__,
        )
        try:
            completion = self._client.chat.completions.parse(
                model=self.model,
                temperature=self._settings.openai_temperature,
                messages=self._build_messages(system_prompt, user_prompt, repair_hint),
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

        usage = getattr(completion, "usage", None)
        self._record_usage(
            TokenUsage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(usage, "total_tokens", 0) or 0,
            )
        )

        message = completion.choices[0].message
        if message.refusal:
            raise LLMGenerationError(
                f"The model refused to produce {purpose}: {self._scrub(message.refusal)}"
            )
        if message.parsed is None:
            raise MalformedResponseError("the response contained no structured content")

        self._log_completed(purpose)
        return message.parsed

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

    def _describe_fatal(self, exc: Exception, purpose: str) -> str:
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


if TYPE_CHECKING:

    def _satisfies_protocol(client: OpenAIClient) -> LLMClient:
        """Static assertion that the adapter still implements the protocol.

        Checked by mypy; never executed.
        """
        return client
