"""Tests for environment-driven application configuration."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from resumelab.config import Settings, load_settings
from resumelab.exceptions import ConfigurationError, ResumeLabError

API_KEY = "sk-test-not-a-real-key"


def load(**env: str) -> Settings:
    """Load settings from an explicit environment, ignoring any local .env file."""
    return Settings(_env_file=None, **env)


def test_defaults_are_applied_when_only_the_api_key_is_present():
    settings = load(openai_api_key=API_KEY)

    assert settings.openai_model == "gpt-4o"
    assert settings.openai_temperature == pytest.approx(0.2)
    assert settings.openai_max_retries == 3
    assert settings.openai_timeout_seconds == pytest.approx(60.0)
    assert settings.candidate_profile_path == Path("data/candidate_profile.yaml")
    assert settings.output_dir == Path("output")
    assert settings.log_level == "INFO"


def test_every_setting_can_be_overridden_from_the_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", API_KEY)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("OPENAI_TEMPERATURE", "0.9")
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "5")
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("CANDIDATE_PROFILE_PATH", "profiles/other.yaml")
    monkeypatch.setenv("OUTPUT_DIR", "artifacts")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = load_settings(env_file=None)

    assert settings.openai_api_key.get_secret_value() == API_KEY
    assert settings.openai_model == "gpt-4.1-mini"
    assert settings.openai_temperature == pytest.approx(0.9)
    assert settings.openai_max_retries == 5
    assert settings.openai_timeout_seconds == pytest.approx(12.5)
    assert settings.candidate_profile_path == Path("profiles/other.yaml")
    assert settings.output_dir == Path("artifacts")
    assert settings.log_level == "DEBUG"


def test_runs_dir_is_derived_from_the_output_directory():
    settings = load(openai_api_key=API_KEY, output_dir="artifacts")

    assert settings.runs_dir == Path("artifacts/runs")


@pytest.mark.parametrize("raw_level", ["debug", "Debug", " debug "])
def test_log_level_is_case_and_whitespace_insensitive(raw_level):
    assert load(openai_api_key=API_KEY, log_level=raw_level).log_level == "DEBUG"


def test_a_non_string_log_level_is_passed_through_and_rejected():
    """The normalizer must not crash on non-string input; validation still rejects it."""
    with pytest.raises(ValidationError):
        load(openai_api_key=API_KEY, log_level=10)


# --- secret handling ------------------------------------------------------


def test_api_key_is_not_exposed_by_repr_or_str():
    settings = load(openai_api_key=API_KEY)

    assert API_KEY not in repr(settings)
    assert API_KEY not in str(settings)
    assert API_KEY not in str(settings.openai_api_key)
    assert settings.openai_api_key.get_secret_value() == API_KEY


def test_validation_errors_name_the_variable_without_echoing_its_value(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", API_KEY)
    monkeypatch.setenv("OPENAI_TEMPERATURE", "9.9")

    with pytest.raises(ConfigurationError) as exc_info:
        load_settings(env_file=None)

    message = str(exc_info.value)
    assert "OPENAI_TEMPERATURE" in message
    assert "9.9" not in message


# --- invalid configuration ------------------------------------------------


def test_missing_api_key_is_reported_as_a_configuration_error():
    with pytest.raises(ConfigurationError) as exc_info:
        load_settings(env_file=None)

    assert "OPENAI_API_KEY" in str(exc_info.value)


def test_configuration_error_is_a_resumelab_error():
    with pytest.raises(ResumeLabError):
        load_settings(env_file=None)


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("OPENAI_TEMPERATURE", "-0.1"),
        ("OPENAI_TEMPERATURE", "2.1"),
        ("OPENAI_TEMPERATURE", "hot"),
        ("OPENAI_MAX_RETRIES", "-1"),
        ("OPENAI_MAX_RETRIES", "11"),
        ("OPENAI_TIMEOUT_SECONDS", "0"),
        ("OPENAI_TIMEOUT_SECONDS", "601"),
        ("OPENAI_MODEL", ""),
        ("LOG_LEVEL", "VERBOSE"),
    ],
)
def test_out_of_range_settings_are_rejected(monkeypatch, variable, value):
    monkeypatch.setenv("OPENAI_API_KEY", API_KEY)
    monkeypatch.setenv(variable, value)

    with pytest.raises(ConfigurationError) as exc_info:
        load_settings(env_file=None)

    assert variable in str(exc_info.value)


def test_settings_are_frozen_so_a_recorded_run_cannot_drift():
    settings = load(openai_api_key=API_KEY)

    with pytest.raises(ValidationError):
        settings.openai_model = "gpt-4.1"


# --- .env file support ----------------------------------------------------


def test_values_are_read_from_an_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"OPENAI_API_KEY={API_KEY}\nOPENAI_MODEL=gpt-4.1\nLOG_LEVEL=warning\n",
        encoding="utf-8",
    )

    settings = load_settings(env_file=env_file)

    assert settings.openai_api_key.get_secret_value() == API_KEY
    assert settings.openai_model == "gpt-4.1"
    assert settings.log_level == "WARNING"


def test_environment_variables_take_precedence_over_the_env_file(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(f"OPENAI_API_KEY={API_KEY}\nOPENAI_MODEL=from-file\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_MODEL", "from-environment")

    assert load_settings(env_file=env_file).openai_model == "from-environment"


def test_unrelated_entries_in_the_env_file_are_ignored(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(f"OPENAI_API_KEY={API_KEY}\nUNRELATED_TOOL_FLAG=1\n", encoding="utf-8")

    assert load_settings(env_file=env_file).openai_model == "gpt-4o"


def test_a_missing_env_file_falls_back_to_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", API_KEY)

    assert load_settings(env_file=tmp_path / "absent.env").openai_model == "gpt-4o"
