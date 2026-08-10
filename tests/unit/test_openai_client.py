"""Tests for the OpenAI adapter.

Every test drives a stub SDK client. Nothing here performs a network call.
"""

from types import SimpleNamespace

import httpx
import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ContentFilterFinishReasonError,
    InternalServerError,
    LengthFinishReasonError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError

from resumelab.config import Settings
from resumelab.exceptions import LLMGenerationError
from resumelab.llm import LLMCallStats, OpenAIClient, TokenUsage

API_KEY = "sk-test-not-a-real-key"


class Answer(BaseModel):
    """Schema used as the structured response target."""

    verdict: str


# --- stub SDK -------------------------------------------------------------


class StubCompletions:
    """Replays queued outcomes, recording the arguments it was called with."""

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class StubOpenAI:
    def __init__(self, outcomes):
        self.completions = StubCompletions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)


def completion(parsed=None, refusal=None, usage=(10, 5, 15)):
    """Build a stand-in for a parsed chat completion."""
    token_counts = (
        None
        if usage is None
        else SimpleNamespace(
            prompt_tokens=usage[0],
            completion_tokens=usage[1],
            total_tokens=usage[2],
        )
    )
    message = SimpleNamespace(parsed=parsed, refusal=refusal)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=token_counts)


def api_error(error_type, status):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    if error_type in (APITimeoutError, APIConnectionError):
        return error_type(request=request)
    response = httpx.Response(status_code=status, request=request)
    return error_type("provider said no", response=response, body=None)


def validation_error():
    try:
        Answer.model_validate({"verdict": 123, "extra": True})
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a validation error")


# --- fixtures -------------------------------------------------------------


@pytest.fixture
def settings():
    return Settings(_env_file=None, openai_api_key=API_KEY, openai_max_retries=2)


@pytest.fixture
def sleeps():
    return []


@pytest.fixture
def build_client(settings, sleeps):
    def build(outcomes, **overrides):
        active = settings.model_copy(update=overrides) if overrides else settings
        stub = StubOpenAI(outcomes)
        client = OpenAIClient(active, client=stub, sleeper=sleeps.append)
        return client, stub

    return build


def generate(client):
    return client.generate_structured(
        system_prompt="You analyze job descriptions.",
        user_prompt="Analyze this.",
        response_model=Answer,
        purpose="jd_analysis",
    )


# --- happy path -----------------------------------------------------------


def test_a_parsed_response_is_returned(build_client):
    client, _ = build_client([completion(parsed=Answer(verdict="ok"))])

    assert generate(client) == Answer(verdict="ok")


def test_the_request_carries_the_configured_model_and_temperature(build_client):
    client, stub = build_client([completion(parsed=Answer(verdict="ok"))])

    generate(client)

    request = stub.completions.calls[0]
    assert request["model"] == "gpt-4o"
    assert request["temperature"] == pytest.approx(0.2)
    assert request["response_format"] is Answer


def test_the_prompts_are_sent_as_system_and_user_messages(build_client):
    client, stub = build_client([completion(parsed=Answer(verdict="ok"))])

    generate(client)

    messages = stub.completions.calls[0]["messages"]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"] == "You analyze job descriptions."
    assert messages[1]["content"] == "Analyze this."


def test_the_model_name_is_exposed_for_metadata(build_client):
    client, _ = build_client([completion(parsed=Answer(verdict="ok"))])

    assert client.model == "gpt-4o"


def test_the_adapter_satisfies_the_client_protocol(build_client):
    client, _ = build_client([])

    assert hasattr(client, "generate_structured")
    assert hasattr(client, "stats")
    assert hasattr(client, "model")


# --- usage accounting -----------------------------------------------------


def test_token_usage_accumulates_across_calls(build_client):
    client, _ = build_client(
        [
            completion(parsed=Answer(verdict="a"), usage=(10, 5, 15)),
            completion(parsed=Answer(verdict="b"), usage=(20, 10, 30)),
        ]
    )

    generate(client)
    generate(client)

    assert client.stats.call_count == 2
    assert client.stats.usage == TokenUsage(prompt_tokens=30, completion_tokens=15, total_tokens=45)


def test_a_response_without_usage_still_counts_as_a_call(build_client):
    client, _ = build_client([completion(parsed=Answer(verdict="ok"), usage=None)])

    generate(client)

    assert client.stats.call_count == 1
    assert client.stats.usage == TokenUsage()


def test_retried_attempts_are_counted_so_cost_is_not_understated(build_client):
    client, _ = build_client(
        [
            api_error(RateLimitError, 429),
            completion(parsed=Answer(verdict="ok")),
        ]
    )

    generate(client)

    assert client.stats.call_count == 1  # the failed attempt reported no usage


def test_stats_start_empty(build_client):
    client, _ = build_client([])

    assert client.stats == LLMCallStats()


# --- retry behaviour ------------------------------------------------------


@pytest.mark.parametrize(
    ("error_type", "status"),
    [
        (APITimeoutError, 408),
        (APIConnectionError, 500),
        (RateLimitError, 429),
        (InternalServerError, 500),
    ],
)
def test_transient_failures_are_retried(build_client, error_type, status):
    client, stub = build_client(
        [api_error(error_type, status), completion(parsed=Answer(verdict="ok"))]
    )

    assert generate(client) == Answer(verdict="ok")
    assert len(stub.completions.calls) == 2


def test_backoff_grows_exponentially(build_client, sleeps):
    client, _ = build_client(
        [api_error(RateLimitError, 429)] * 4,
        openai_max_retries=3,
    )

    with pytest.raises(LLMGenerationError):
        generate(client)

    assert sleeps == [1.0, 2.0, 4.0]


def test_backoff_is_capped(build_client, sleeps):
    client, _ = build_client([api_error(RateLimitError, 429)] * 10, openai_max_retries=9)

    with pytest.raises(LLMGenerationError):
        generate(client)

    assert max(sleeps) <= 30.0


def test_the_retry_budget_is_honoured(build_client):
    client, stub = build_client([api_error(RateLimitError, 429)] * 3)

    with pytest.raises(LLMGenerationError, match="3 attempt"):
        generate(client)

    assert len(stub.completions.calls) == 3


def test_zero_retries_means_a_single_attempt(build_client):
    client, stub = build_client([api_error(RateLimitError, 429)], openai_max_retries=0)

    with pytest.raises(LLMGenerationError):
        generate(client)

    assert len(stub.completions.calls) == 1


def test_exhausted_retries_report_the_underlying_failure(build_client):
    client, _ = build_client([api_error(RateLimitError, 429)] * 3)

    with pytest.raises(LLMGenerationError) as exc_info:
        generate(client)

    assert "RateLimitError" in str(exc_info.value)


# --- fatal failures -------------------------------------------------------


@pytest.mark.parametrize(
    ("error_type", "status"),
    [
        (AuthenticationError, 401),
        (PermissionDeniedError, 403),
        (BadRequestError, 400),
        (NotFoundError, 404),
    ],
)
def test_fatal_failures_are_not_retried(build_client, error_type, status):
    client, stub = build_client([api_error(error_type, status)] * 3)

    with pytest.raises(LLMGenerationError):
        generate(client)

    assert len(stub.completions.calls) == 1


def test_an_authentication_failure_points_at_the_environment_variable(build_client):
    client, _ = build_client([api_error(AuthenticationError, 401)])

    with pytest.raises(LLMGenerationError, match="OPENAI_API_KEY"):
        generate(client)


def test_a_missing_model_points_at_the_model_variable(build_client):
    client, _ = build_client([api_error(NotFoundError, 404)])

    with pytest.raises(LLMGenerationError, match="OPENAI_MODEL"):
        generate(client)


def test_a_permission_failure_names_the_model(build_client):
    client, _ = build_client([api_error(PermissionDeniedError, 403)])

    with pytest.raises(LLMGenerationError, match="gpt-4o"):
        generate(client)


# --- secret handling ------------------------------------------------------


def test_the_api_key_never_reaches_an_authentication_error_message(build_client):
    """Providers echo part of the rejected credential back in the error body."""
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status_code=401, request=request)
    leak = AuthenticationError(
        f"Incorrect API key provided: {API_KEY}",
        response=response,
        body=None,
    )
    client, _ = build_client([leak])

    with pytest.raises(LLMGenerationError) as exc_info:
        generate(client)

    assert API_KEY not in str(exc_info.value)


def test_the_api_key_is_scrubbed_from_other_provider_messages(build_client):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status_code=400, request=request)
    leak = BadRequestError(f"bad request for {API_KEY}", response=response, body=None)
    client, _ = build_client([leak])

    with pytest.raises(LLMGenerationError) as exc_info:
        generate(client)

    assert API_KEY not in str(exc_info.value)
    assert "***" in str(exc_info.value)


def test_the_api_key_is_scrubbed_from_exhausted_retry_messages(build_client):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status_code=429, request=request)
    leaks = [RateLimitError(f"slow down {API_KEY}", response=response, body=None)] * 3
    client, _ = build_client(leaks)

    with pytest.raises(LLMGenerationError) as exc_info:
        generate(client)

    assert API_KEY not in str(exc_info.value)


# --- unusable responses ---------------------------------------------------


def test_a_refusal_fails_immediately(build_client):
    client, stub = build_client([completion(refusal="I cannot help with that.")])

    with pytest.raises(LLMGenerationError, match="refused"):
        generate(client)

    assert len(stub.completions.calls) == 1


def test_a_response_without_structured_content_is_repaired(build_client):
    client, stub = build_client([completion(parsed=None), completion(parsed=Answer(verdict="ok"))])

    assert generate(client) == Answer(verdict="ok")

    repair = stub.completions.calls[1]["messages"][-1]["content"]
    assert "no structured content" in repair


def test_a_schema_failure_is_retried_with_the_errors_appended(build_client):
    client, stub = build_client([validation_error(), completion(parsed=Answer(verdict="ok"))])

    assert generate(client) == Answer(verdict="ok")

    messages = stub.completions.calls[1]["messages"]
    repair = messages[-1]["content"]
    assert [message["role"] for message in messages] == ["system", "user", "user"]
    assert "verdict" in repair
    assert "corrected response" in repair


def test_schema_repair_does_not_wait_between_attempts(build_client, sleeps):
    """A malformed response is not a rate problem; there is nothing to wait for."""
    client, _ = build_client([validation_error(), completion(parsed=Answer(verdict="ok"))])

    generate(client)

    assert sleeps == []


def test_persistent_schema_failures_fail_the_run(build_client):
    client, _ = build_client([validation_error()] * 3)

    with pytest.raises(LLMGenerationError) as exc_info:
        generate(client)

    assert "did not match the schema" in str(exc_info.value)


def test_a_truncated_response_explains_the_output_limit(build_client):
    truncated = LengthFinishReasonError(completion=SimpleNamespace(choices=[], usage=None))
    client, stub = build_client([truncated])

    with pytest.raises(LLMGenerationError, match="output limit"):
        generate(client)

    assert len(stub.completions.calls) == 1


def test_a_content_filtered_response_is_reported(build_client):
    client, _ = build_client([ContentFilterFinishReasonError()])

    with pytest.raises(LLMGenerationError, match="content filter"):
        generate(client)


# --- value objects --------------------------------------------------------


def test_token_usage_adds_componentwise():
    total = TokenUsage(1, 2, 3) + TokenUsage(10, 20, 30)

    assert total == TokenUsage(11, 22, 33)


def test_recording_a_call_returns_new_stats():
    stats = LLMCallStats()

    updated = stats.record(TokenUsage(1, 2, 3))

    assert stats == LLMCallStats()
    assert updated.call_count == 1
    assert updated.usage == TokenUsage(1, 2, 3)
