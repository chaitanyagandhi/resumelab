"""The LLM abstraction the pipeline depends on.

Every pipeline stage takes an :class:`LLMClient` rather than constructing a provider
client. That keeps stages unit-testable against fakes, and leaves room for the
provider and model comparison experiments this research is designed to grow into.

The contract is deliberately narrow: given a system prompt, a user prompt, and a
Pydantic model, return a validated instance of that model. Prompt text belongs to the
prompt layer, and retry policy belongs to the adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol

from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token counts reported by the provider, accumulated across a run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass(frozen=True, slots=True)
class LLMCallStats:
    """Per-run call accounting, recorded in experiment metadata.

    ``call_count`` counts provider requests, including retried attempts, so the cost
    of a run is not understated.
    """

    call_count: int = 0
    usage: TokenUsage = field(default_factory=TokenUsage)

    def record(self, usage: TokenUsage) -> LLMCallStats:
        """Return updated stats including one more call."""
        return replace(self, call_count=self.call_count + 1, usage=self.usage + usage)


class LLMClient(Protocol):
    """Structured-output client for a single model."""

    @property
    def model(self) -> str:
        """Identifier of the model being used, recorded in run metadata."""
        ...

    @property
    def stats(self) -> LLMCallStats:
        """Snapshot of calls and token usage so far."""
        ...

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

        Args:
            system_prompt: Developer instructions for the model.
            user_prompt: The task input, including any untrusted data.
            response_model: Schema the response must satisfy.
            purpose: Short stage name such as ``"jd_analysis"``, used in logs.

        Returns:
            A validated instance of ``response_model``.

        Raises:
            LLMGenerationError: If the call fails, or no valid response is obtained
                within the configured retry budget.
        """
        ...
