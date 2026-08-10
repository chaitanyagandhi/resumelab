"""Tests for selecting and constructing the configured provider client."""

import logging

import pytest

from resumelab.config import LLMProvider, Settings
from resumelab.exceptions import ConfigurationError
from resumelab.llm import AnthropicClient, OpenAIClient, create_llm_client

OPENAI_KEY = "sk-test-not-a-real-key"
ANTHROPIC_KEY = "sk-ant-test-not-a-real-key"


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_a_lone_openai_key_builds_an_openai_client():
    client = create_llm_client(settings(openai_api_key=OPENAI_KEY))

    assert isinstance(client, OpenAIClient)
    assert client.model == "gpt-4o"


def test_a_lone_anthropic_key_builds_an_anthropic_client():
    client = create_llm_client(settings(anthropic_api_key=ANTHROPIC_KEY))

    assert isinstance(client, AnthropicClient)
    assert client.model == "claude-opus-5"


def test_the_configured_provider_wins_over_inference():
    client = create_llm_client(
        settings(
            openai_api_key=OPENAI_KEY,
            anthropic_api_key=ANTHROPIC_KEY,
            llm_provider="anthropic",
        )
    )

    assert isinstance(client, AnthropicClient)


@pytest.mark.parametrize(
    ("provider", "expected"),
    [(LLMProvider.OPENAI, OpenAIClient), (LLMProvider.ANTHROPIC, AnthropicClient)],
)
def test_an_explicit_provider_argument_overrides_configuration(provider, expected):
    """This is the hook the CLI uses when a researcher picks a provider per run."""
    configured = settings(
        openai_api_key=OPENAI_KEY,
        anthropic_api_key=ANTHROPIC_KEY,
        llm_provider="openai",
    )

    assert isinstance(create_llm_client(configured, provider=provider), expected)


def test_choosing_a_provider_without_its_key_fails_before_any_network_call():
    with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
        create_llm_client(
            settings(openai_api_key=OPENAI_KEY),
            provider=LLMProvider.ANTHROPIC,
        )


def test_both_clients_expose_the_same_protocol_surface():
    for configured in (
        settings(openai_api_key=OPENAI_KEY),
        settings(anthropic_api_key=ANTHROPIC_KEY),
    ):
        client = create_llm_client(configured)

        assert callable(client.generate_structured)
        assert client.stats.call_count == 0
        assert isinstance(client.model, str)


def test_the_chosen_provider_is_logged(caplog):
    with caplog.at_level(logging.INFO, logger="resumelab.llm.factory"):
        create_llm_client(settings(anthropic_api_key=ANTHROPIC_KEY))

    assert "provider=anthropic" in caplog.text
    assert "model=claude-opus-5" in caplog.text


def test_the_api_key_is_not_logged(caplog):
    with caplog.at_level(logging.DEBUG):
        create_llm_client(settings(anthropic_api_key=ANTHROPIC_KEY))

    assert ANTHROPIC_KEY not in caplog.text
