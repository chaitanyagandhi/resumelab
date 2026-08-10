"""Tests for the Anthropic adapter.

Every test drives a stub SDK client. Nothing here performs a network call.
"""

from types import SimpleNamespace

import httpx
import pytest
from anthropic import (
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
from pydantic import BaseModel, ValidationError

from resumelab.config import Settings
from resumelab.exceptions import LLMGenerationError
from resumelab.llm import AnthropicClient, LLMCallStats, TokenUsage

API_KEY = "sk-ant-test-not-a-real-key"


class Answer(BaseModel):
    """Schema used as the structured response target."""

    verdict: str


# --- stub SDK -------------------------------------------------------------


class StubMessages:
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


class StubAnthropic:
    def __init__(self, outcomes):
        self.messages = StubMessages(outcomes)


def message(parsed=None, stop_reason="end_turn", usage=(10, 5), category=None):
    """Build a stand-in for a parsed Anthropic message."""
    token_counts = (
        None if usage is None else SimpleNamespace(input_tokens=usage[0], output_tokens=usage[1])
    )
    stop_details = None if category is None else SimpleNamespace(category=category)
    return SimpleNamespace(
        parsed_output=parsed,
        stop_reason=stop_reason,
        stop_details=stop_details,
        usage=token_counts,
    )


def api_error(error_type, status):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
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
    return Settings(_env_file=None, anthropic_api_key=API_KEY, llm_max_retries=2)


@pytest.fixture
def sleeps():
    return []


@pytest.fixture
def build_client(settings, sleeps):
    def build(outcomes, **overrides):
        active = settings.model_copy(update=overrides) if overrides else settings
        stub = StubAnthropic(outcomes)
        client = AnthropicClient(active, client=stub, sleeper=sleeps.append)
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
    client, _ = build_client([message(parsed=Answer(verdict="ok"))])

    assert generate(client) == Answer(verdict="ok")


def test_the_request_carries_the_configured_model_budget_and_effort(build_client):
    client, stub = build_client([message(parsed=Answer(verdict="ok"))])

    generate(client)

    request = stub.messages.calls[0]
    assert request["model"] == "claude-opus-5"
    assert request["max_tokens"] == 16_000
    assert request["output_config"] == {"effort": "high"}
    assert request["output_format"] is Answer


def test_no_sampling_parameters_are_sent(build_client):
    """Current Claude models reject temperature, top_p, and top_k with a 400."""
    client, stub = build_client([message(parsed=Answer(verdict="ok"))])

    generate(client)

    request = stub.messages.calls[0]
    assert "temperature" not in request
    assert "top_p" not in request
    assert "top_k" not in request


def test_the_system_prompt_is_a_parameter_not_a_message(build_client):
    client, stub = build_client([message(parsed=Answer(verdict="ok"))])

    generate(client)

    request = stub.messages.calls[0]
    assert request["system"] == "You analyze job descriptions."
    assert [entry["role"] for entry in request["messages"]] == ["user"]
    assert request["messages"][0]["content"] == "Analyze this."


def test_the_model_name_is_exposed_for_metadata(build_client):
    client, _ = build_client([message(parsed=Answer(verdict="ok"))])

    assert client.model == "claude-opus-5"


def test_the_adapter_satisfies_the_client_protocol(build_client):
    client, _ = build_client([])

    assert hasattr(client, "generate_structured")
    assert hasattr(client, "stats")
    assert hasattr(client, "model")


# --- usage accounting -----------------------------------------------------


def test_token_usage_accumulates_across_calls(build_client):
    client, _ = build_client(
        [
            message(parsed=Answer(verdict="a"), usage=(10, 5)),
            message(parsed=Answer(verdict="b"), usage=(20, 10)),
        ]
    )

    generate(client)
    generate(client)

    assert client.stats.call_count == 2
    assert client.stats.usage == TokenUsage(prompt_tokens=30, completion_tokens=15, total_tokens=45)


def test_total_tokens_are_derived_because_the_provider_reports_only_the_parts(build_client):
    client, _ = build_client([message(parsed=Answer(verdict="ok"), usage=(7, 3))])

    generate(client)

    assert client.stats.usage.total_tokens == 10


def test_a_response_without_usage_still_counts_as_a_call(build_client):
    client, _ = build_client([message(parsed=Answer(verdict="ok"), usage=None)])

    generate(client)

    assert client.stats.call_count == 1
    assert client.stats.usage == TokenUsage()


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
        (OverloadedError, 529),
    ],
)
def test_transient_failures_are_retried(build_client, error_type, status):
    client, stub = build_client(
        [api_error(error_type, status), message(parsed=Answer(verdict="ok"))]
    )

    assert generate(client) == Answer(verdict="ok")
    assert len(stub.messages.calls) == 2


def test_an_overloaded_provider_is_retried_rather_than_failing_the_run(build_client):
    """529 is Anthropic-specific and would be a fatal 5xx if not classified."""
    client, _ = build_client(
        [api_error(OverloadedError, 529), message(parsed=Answer(verdict="ok"))]
    )

    assert generate(client) == Answer(verdict="ok")


def test_backoff_grows_exponentially(build_client, sleeps):
    client, _ = build_client([api_error(RateLimitError, 429)] * 4, llm_max_retries=3)

    with pytest.raises(LLMGenerationError):
        generate(client)

    assert sleeps == [1.0, 2.0, 4.0]


def test_the_retry_budget_is_honoured(build_client):
    client, stub = build_client([api_error(RateLimitError, 429)] * 3)

    with pytest.raises(LLMGenerationError, match="3 attempt"):
        generate(client)

    assert len(stub.messages.calls) == 3


def test_zero_retries_means_a_single_attempt(build_client):
    client, stub = build_client([api_error(RateLimitError, 429)], llm_max_retries=0)

    with pytest.raises(LLMGenerationError):
        generate(client)

    assert len(stub.messages.calls) == 1


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

    assert len(stub.messages.calls) == 1


def test_an_authentication_failure_points_at_the_environment_variable(build_client):
    client, _ = build_client([api_error(AuthenticationError, 401)])

    with pytest.raises(LLMGenerationError, match="ANTHROPIC_API_KEY"):
        generate(client)


def test_a_missing_model_points_at_the_model_variable(build_client):
    client, _ = build_client([api_error(NotFoundError, 404)])

    with pytest.raises(LLMGenerationError, match="ANTHROPIC_MODEL"):
        generate(client)


def test_a_permission_failure_names_the_model(build_client):
    client, _ = build_client([api_error(PermissionDeniedError, 403)])

    with pytest.raises(LLMGenerationError, match="claude-opus-5"):
        generate(client)


# --- secret handling ------------------------------------------------------


def test_the_api_key_never_reaches_an_authentication_error_message(build_client):
    """Providers echo part of the rejected credential back in the error body."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code=401, request=request)
    leak = AuthenticationError(f"invalid x-api-key: {API_KEY}", response=response, body=None)
    client, _ = build_client([leak])

    with pytest.raises(LLMGenerationError) as exc_info:
        generate(client)

    assert API_KEY not in str(exc_info.value)


def test_the_api_key_is_scrubbed_from_other_provider_messages(build_client):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code=400, request=request)
    leak = BadRequestError(f"bad request for {API_KEY}", response=response, body=None)
    client, _ = build_client([leak])

    with pytest.raises(LLMGenerationError) as exc_info:
        generate(client)

    assert API_KEY not in str(exc_info.value)
    assert "***" in str(exc_info.value)


# --- unusable responses ---------------------------------------------------


def test_a_refusal_fails_immediately(build_client):
    client, stub = build_client([message(stop_reason="refusal", category="cyber")])

    with pytest.raises(LLMGenerationError, match="declined"):
        generate(client)

    assert len(stub.messages.calls) == 1


def test_a_refusal_reports_its_category(build_client):
    client, _ = build_client([message(stop_reason="refusal", category="cyber")])

    with pytest.raises(LLMGenerationError, match="cyber"):
        generate(client)


def test_a_refusal_without_details_is_still_reported(build_client):
    """stop_details is informational and is often absent."""
    client, _ = build_client([message(stop_reason="refusal")])

    with pytest.raises(LLMGenerationError, match="unspecified"):
        generate(client)


def test_a_truncated_response_points_at_the_output_budget(build_client):
    client, stub = build_client([message(stop_reason="max_tokens")])

    with pytest.raises(LLMGenerationError, match="ANTHROPIC_MAX_TOKENS"):
        generate(client)

    assert len(stub.messages.calls) == 1


def test_truncation_also_suggests_lowering_effort(build_client):
    """max_tokens bounds reasoning as well as the response on current models."""
    client, _ = build_client([message(stop_reason="max_tokens")])

    with pytest.raises(LLMGenerationError, match="ANTHROPIC_EFFORT"):
        generate(client)


def test_a_response_without_structured_content_is_repaired(build_client):
    client, stub = build_client([message(parsed=None), message(parsed=Answer(verdict="ok"))])

    assert generate(client) == Answer(verdict="ok")

    repair = stub.messages.calls[1]["messages"][-1]["content"]
    assert "no structured content" in repair


def test_a_schema_failure_is_retried_with_the_errors_appended(build_client):
    client, stub = build_client([validation_error(), message(parsed=Answer(verdict="ok"))])

    assert generate(client) == Answer(verdict="ok")

    messages = stub.messages.calls[1]["messages"]
    assert [entry["role"] for entry in messages] == ["user", "user"]
    assert "verdict" in messages[-1]["content"]
    assert "corrected response" in messages[-1]["content"]


def test_schema_repair_does_not_wait_between_attempts(build_client, sleeps):
    client, _ = build_client([validation_error(), message(parsed=Answer(verdict="ok"))])

    generate(client)

    assert sleeps == []


def test_persistent_schema_failures_fail_the_run(build_client):
    client, _ = build_client([validation_error()] * 3)

    with pytest.raises(LLMGenerationError) as exc_info:
        generate(client)

    assert "did not match the schema" in str(exc_info.value)
