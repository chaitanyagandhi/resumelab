"""Tests for token accounting, driven through a real SDK over a fake transport.

The adapters' other tests hand the SDK a stub object, which is the right shape for
checking how a provider's errors are mapped. It is the wrong shape here: usage is
metered on the HTTP response, below anything a stub SDK would reach. These build the
genuine client and answer it with :class:`httpx.MockTransport`, which is how the
fetching suite has always faked a network.
"""

import json

import httpx
import pytest
from anthropic import Anthropic
from openai import OpenAI
from pydantic import BaseModel

from resumelab.config import Settings
from resumelab.exceptions import LLMGenerationError
from resumelab.llm import AnthropicClient, OpenAIClient, TokenUsage
from resumelab.llm.usage import metering_http_client, usage_of

API_KEY = "sk-test-not-a-real-key"


class Answer(BaseModel):
    verdict: str


def openai_body(verdict="ok", usage=(4000, 120, 4120)):
    body = {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps({"verdict": verdict})},
            }
        ],
    }
    if usage is not None:
        body["usage"] = dict(
            zip(("prompt_tokens", "completion_tokens", "total_tokens"), usage, strict=True)
        )
    return body


def anthropic_body(verdict="ok", usage=(4000, 120)):
    body = {
        "id": "x",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5",
        "content": [{"type": "text", "text": json.dumps({"verdict": verdict})}],
        "stop_reason": "end_turn",
    }
    if usage is not None:
        body["usage"] = dict(zip(("input_tokens", "output_tokens"), usage, strict=True))
    return body


def metered(client, responses):
    """Point a built adapter's SDK at a queue of canned HTTP responses."""
    queue = list(responses)

    def handler(_request):
        return queue.pop(0)

    transport = httpx.MockTransport(handler)
    http = metering_http_client(client._record_usage, timeout=10)
    http._transport = transport
    return http


@pytest.fixture
def openai_client():
    def build(responses):
        settings = Settings(_env_file=None, openai_api_key=API_KEY, llm_max_retries=2)
        client = OpenAIClient(settings)
        client._client = OpenAI(
            api_key=API_KEY, max_retries=0, http_client=metered(client, responses)
        )
        return client

    return build


@pytest.fixture
def anthropic_client():
    def build(responses):
        settings = Settings(_env_file=None, anthropic_api_key=API_KEY, llm_max_retries=2)
        client = AnthropicClient(settings)
        client._client = Anthropic(
            api_key=API_KEY, max_retries=0, http_client=metered(client, responses)
        )
        return client

    return build


def ask(client):
    return client.generate_structured(
        system_prompt="s", user_prompt="u", response_model=Answer, purpose="test"
    )


# --- the bug this exists for ----------------------------------------------


def test_a_response_that_fails_the_schema_is_still_counted(openai_client):
    """The whole reason accounting moved onto the wire.

    Both SDKs validate a structured response inside the call that fetches it, so a
    schema failure escapes before any accounting written afterwards can run. The
    attempt is billed regardless: the model read the prompt and wrote an answer.

    Measured against a provider console on one run, 73,778 tokens were reported
    against 112,393 charged, and every missing token was a repair like this one.
    """
    rejected = httpx.Response(200, json=openai_body(verdict=None))
    client = openai_client([rejected, rejected, rejected])

    with pytest.raises(LLMGenerationError):
        ask(client)

    assert client.stats.call_count == 3
    assert client.stats.usage.total_tokens == 3 * 4120


def test_the_same_holds_for_the_other_provider(anthropic_client):
    rejected = httpx.Response(200, json=anthropic_body(verdict=None))
    client = anthropic_client([rejected, rejected, rejected])

    with pytest.raises(LLMGenerationError):
        ask(client)

    assert client.stats.usage.total_tokens == 3 * 4120


# --- ordinary accounting --------------------------------------------------


def test_usage_accumulates_across_calls(openai_client):
    client = openai_client(
        [
            httpx.Response(200, json=openai_body(usage=(10, 5, 15))),
            httpx.Response(200, json=openai_body(usage=(20, 10, 30))),
        ]
    )

    ask(client)
    ask(client)

    assert client.stats.call_count == 2
    assert client.stats.usage == TokenUsage(prompt_tokens=30, completion_tokens=15, total_tokens=45)


def test_a_success_without_usage_still_counts_as_a_call(openai_client):
    """The provider said nothing about the size; the call still happened."""
    client = openai_client([httpx.Response(200, json=openai_body(usage=None))])

    ask(client)

    assert client.stats.call_count == 1
    assert client.stats.usage == TokenUsage()


def test_a_refused_request_is_not_counted(openai_client):
    """A 429 bills nothing, and counting it would overstate a run the other way."""
    client = openai_client(
        [
            httpx.Response(429, json={"error": {"message": "slow down"}}),
            httpx.Response(200, json=openai_body(usage=(10, 5, 15))),
        ]
    )

    ask(client)

    assert client.stats.call_count == 1


def test_the_total_is_derived_when_a_provider_reports_only_the_parts(anthropic_client):
    client = anthropic_client([httpx.Response(200, json=anthropic_body(usage=(70, 30)))])

    ask(client)

    assert client.stats.usage == TokenUsage(
        prompt_tokens=70, completion_tokens=30, total_tokens=100
    )


# --- reading a body -------------------------------------------------------


def test_both_providers_field_names_are_understood():
    assert usage_of(
        {"usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}
    ) == (TokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10))
    assert usage_of({"usage": {"input_tokens": 7, "output_tokens": 3}}) == TokenUsage(
        prompt_tokens=7, completion_tokens=3, total_tokens=10
    )


@pytest.mark.parametrize("payload", ["not an object", 42, None, ["a"]])
def test_a_body_that_is_not_an_object_reports_nothing(payload):
    """No evidence a provider call produced it, so it is not counted as one."""
    assert usage_of(payload) is None


def test_an_unfamiliar_usage_shape_still_counts_the_call():
    """A provider that renames its fields should cost a known call, not a silent one."""
    assert usage_of({"usage": {"tokens_consumed": 12}}) == TokenUsage()


def test_a_json_body_that_is_not_an_object_records_nothing():
    """Valid JSON, but not a provider response; the hook leaves the totals alone."""
    seen: list[TokenUsage] = []
    http = metering_http_client(seen.append, timeout=10)
    http._transport = httpx.MockTransport(lambda _request: httpx.Response(200, json=["a", "b"]))

    http.get("https://provider.test/v1/messages")

    assert seen == []


@pytest.mark.parametrize(
    "reported", [{"prompt_tokens": None}, {"prompt_tokens": -5}, {"prompt_tokens": "many"}]
)
def test_a_nonsense_count_reads_as_zero_rather_than_crashing(reported):
    assert usage_of({"usage": reported}) == TokenUsage()


def test_a_body_that_will_not_decode_is_passed_over():
    """A hook that raised would turn a provider hiccup into a failed request.

    Driven at the hook rather than through a generation, because an SDK handed a
    body it cannot understand fails on its own account, which is a separate matter
    and not something the accounting should be asked to survive.
    """
    seen: list[TokenUsage] = []
    http = metering_http_client(seen.append, timeout=10)
    http._transport = httpx.MockTransport(
        lambda _request: httpx.Response(200, content=b"<html>maintenance</html>")
    )

    response = http.get("https://provider.test/v1/messages")

    assert response.status_code == 200
    assert seen == []


def test_a_decodable_body_reaches_the_recorder():
    """The other half of the same seam: what does decode is passed on."""
    seen: list[TokenUsage] = []
    http = metering_http_client(seen.append, timeout=10)
    http._transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200, json={"usage": {"input_tokens": 9, "output_tokens": 1}}
        )
    )

    http.get("https://provider.test/v1/messages")

    assert seen == [TokenUsage(prompt_tokens=9, completion_tokens=1, total_tokens=10)]
