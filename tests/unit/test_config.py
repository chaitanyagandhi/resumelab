"""Tests for environment-driven application configuration."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from resumelab.config import LLMProvider, Settings, load_settings
from resumelab.exceptions import ConfigurationError, ResumeLabError

OPENAI_KEY = "sk-test-not-a-real-key"
ANTHROPIC_KEY = "sk-ant-test-not-a-real-key"


def load(**env: str) -> Settings:
    """Load settings from an explicit environment, ignoring any local .env file."""
    return Settings(_env_file=None, **env)


def openai_settings(**overrides: str) -> Settings:
    return load(openai_api_key=OPENAI_KEY, **overrides)


def test_defaults_are_applied_when_only_an_api_key_is_present():
    settings = openai_settings()

    assert settings.llm_provider is None
    assert settings.llm_max_retries == 3
    assert settings.llm_timeout_seconds == pytest.approx(60.0)
    assert settings.openai_model == "gpt-4o"
    assert settings.openai_temperature == pytest.approx(0.2)
    assert settings.anthropic_model == "claude-opus-5"
    assert settings.anthropic_max_tokens == 16_000
    assert settings.anthropic_effort == "high"
    assert settings.candidate_profile_path == Path("data/candidate_profile.yaml")
    assert settings.output_dir == Path("output")
    assert settings.log_level == "INFO"


def test_every_setting_can_be_overridden_from_the_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MAX_RETRIES", "5")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("OPENAI_TEMPERATURE", "0.9")
    monkeypatch.setenv("ANTHROPIC_API_KEY", ANTHROPIC_KEY)
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "32000")
    monkeypatch.setenv("ANTHROPIC_EFFORT", "medium")
    monkeypatch.setenv("CANDIDATE_PROFILE_PATH", "profiles/other.yaml")
    monkeypatch.setenv("OUTPUT_DIR", "artifacts")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = load_settings(env_file=None)

    assert settings.llm_provider is LLMProvider.ANTHROPIC
    assert settings.llm_max_retries == 5
    assert settings.llm_timeout_seconds == pytest.approx(12.5)
    assert settings.openai_api_key.get_secret_value() == OPENAI_KEY
    assert settings.openai_model == "gpt-4.1-mini"
    assert settings.openai_temperature == pytest.approx(0.9)
    assert settings.anthropic_api_key.get_secret_value() == ANTHROPIC_KEY
    assert settings.anthropic_model == "claude-sonnet-5"
    assert settings.anthropic_max_tokens == 32_000
    assert settings.anthropic_effort == "medium"
    assert settings.candidate_profile_path == Path("profiles/other.yaml")
    assert settings.output_dir == Path("artifacts")
    assert settings.log_level == "DEBUG"


def test_runs_dir_is_derived_from_the_output_directory():
    assert openai_settings(output_dir="artifacts").runs_dir == Path("artifacts/runs")


@pytest.mark.parametrize("raw_level", ["debug", "Debug", " debug "])
def test_log_level_is_case_and_whitespace_insensitive(raw_level):
    assert openai_settings(log_level=raw_level).log_level == "DEBUG"


def test_a_non_string_log_level_is_passed_through_and_rejected():
    """The normalizer must not crash on non-string input; validation still rejects it."""
    with pytest.raises(ValidationError):
        openai_settings(log_level=10)


@pytest.mark.parametrize("raw", ["ANTHROPIC", " Anthropic "])
def test_provider_names_are_case_and_whitespace_insensitive(raw):
    settings = load(anthropic_api_key=ANTHROPIC_KEY, llm_provider=raw)

    assert settings.llm_provider is LLMProvider.ANTHROPIC


@pytest.mark.parametrize("raw", ["HIGH", " High "])
def test_effort_levels_are_case_and_whitespace_insensitive(raw):
    assert openai_settings(anthropic_effort=raw).anthropic_effort == "high"


def test_a_non_string_provider_is_passed_through_and_rejected():
    with pytest.raises(ValidationError):
        openai_settings(llm_provider=7)


# --- provider resolution --------------------------------------------------


def test_the_provider_is_inferred_from_a_lone_openai_key():
    assert openai_settings().resolved_provider is LLMProvider.OPENAI


def test_the_provider_is_inferred_from_a_lone_anthropic_key():
    settings = load(anthropic_api_key=ANTHROPIC_KEY)

    assert settings.resolved_provider is LLMProvider.ANTHROPIC


def test_openai_wins_when_both_keys_are_configured_and_no_provider_is_chosen():
    settings = load(openai_api_key=OPENAI_KEY, anthropic_api_key=ANTHROPIC_KEY)

    assert settings.resolved_provider is LLMProvider.OPENAI


def test_an_explicit_provider_overrides_inference():
    settings = load(
        openai_api_key=OPENAI_KEY,
        anthropic_api_key=ANTHROPIC_KEY,
        llm_provider="anthropic",
    )

    assert settings.resolved_provider is LLMProvider.ANTHROPIC


def test_configuring_no_credentials_at_all_is_rejected():
    with pytest.raises(ConfigurationError) as exc_info:
        load_settings(env_file=None)

    message = str(exc_info.value)
    assert "OPENAI_API_KEY" in message
    assert "ANTHROPIC_API_KEY" in message


def test_choosing_a_provider_without_its_key_is_rejected_at_load_time(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")

    with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
        load_settings(env_file=None)


# --- per-provider lookups -------------------------------------------------


def test_api_key_for_returns_the_matching_key():
    settings = load(openai_api_key=OPENAI_KEY, anthropic_api_key=ANTHROPIC_KEY)

    assert settings.api_key_for(LLMProvider.OPENAI).get_secret_value() == OPENAI_KEY
    assert settings.api_key_for(LLMProvider.ANTHROPIC).get_secret_value() == ANTHROPIC_KEY


def test_api_key_for_an_unconfigured_provider_names_the_variable():
    with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
        openai_settings().api_key_for(LLMProvider.ANTHROPIC)


def test_model_for_returns_the_matching_model():
    settings = openai_settings(openai_model="gpt-4.1", anthropic_model="claude-sonnet-5")

    assert settings.model_for(LLMProvider.OPENAI) == "gpt-4.1"
    assert settings.model_for(LLMProvider.ANTHROPIC) == "claude-sonnet-5"


# --- secret handling ------------------------------------------------------


def test_api_keys_are_not_exposed_by_repr_or_str():
    settings = load(openai_api_key=OPENAI_KEY, anthropic_api_key=ANTHROPIC_KEY)

    for rendered in (repr(settings), str(settings)):
        assert OPENAI_KEY not in rendered
        assert ANTHROPIC_KEY not in rendered
    assert settings.openai_api_key.get_secret_value() == OPENAI_KEY


def test_validation_errors_name_the_variable_without_echoing_its_value(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.setenv("OPENAI_TEMPERATURE", "9.9")

    with pytest.raises(ConfigurationError) as exc_info:
        load_settings(env_file=None)

    message = str(exc_info.value)
    assert "OPENAI_TEMPERATURE" in message
    assert "9.9" not in message


def test_configuration_error_is_a_resumelab_error():
    with pytest.raises(ResumeLabError):
        load_settings(env_file=None)


# --- invalid configuration ------------------------------------------------


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("OPENAI_TEMPERATURE", "-0.1"),
        ("OPENAI_TEMPERATURE", "2.1"),
        ("OPENAI_TEMPERATURE", "hot"),
        ("OPENAI_MODEL", ""),
        ("LLM_MAX_RETRIES", "-1"),
        ("LLM_MAX_RETRIES", "11"),
        ("LLM_TIMEOUT_SECONDS", "0"),
        ("LLM_TIMEOUT_SECONDS", "601"),
        ("ANTHROPIC_MODEL", ""),
        ("ANTHROPIC_MAX_TOKENS", "512"),
        ("ANTHROPIC_MAX_TOKENS", "200000"),
        ("ANTHROPIC_EFFORT", "turbo"),
        ("LLM_PROVIDER", "gemini"),
        ("LOG_LEVEL", "VERBOSE"),
    ],
)
def test_out_of_range_settings_are_rejected(monkeypatch, variable, value):
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.setenv("ANTHROPIC_API_KEY", ANTHROPIC_KEY)
    monkeypatch.setenv(variable, value)

    with pytest.raises(ConfigurationError) as exc_info:
        load_settings(env_file=None)

    assert variable in str(exc_info.value)


def test_settings_are_frozen_so_a_recorded_run_cannot_drift():
    settings = openai_settings()

    with pytest.raises(ValidationError):
        settings.openai_model = "gpt-4.1"


# --- .env file support ----------------------------------------------------


def test_values_are_read_from_an_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"ANTHROPIC_API_KEY={ANTHROPIC_KEY}\nANTHROPIC_EFFORT=low\nLOG_LEVEL=warning\n",
        encoding="utf-8",
    )

    settings = load_settings(env_file=env_file)

    assert settings.anthropic_api_key.get_secret_value() == ANTHROPIC_KEY
    assert settings.anthropic_effort == "low"
    assert settings.log_level == "WARNING"
    assert settings.resolved_provider is LLMProvider.ANTHROPIC


def test_environment_variables_take_precedence_over_the_env_file(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(f"OPENAI_API_KEY={OPENAI_KEY}\nOPENAI_MODEL=from-file\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_MODEL", "from-environment")

    assert load_settings(env_file=env_file).openai_model == "from-environment"


def test_unrelated_entries_in_the_env_file_are_ignored(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(f"OPENAI_API_KEY={OPENAI_KEY}\nUNRELATED_TOOL_FLAG=1\n", encoding="utf-8")

    assert load_settings(env_file=env_file).openai_model == "gpt-4o"


def test_a_missing_env_file_falls_back_to_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)

    assert load_settings(env_file=tmp_path / "absent.env").openai_model == "gpt-4o"
