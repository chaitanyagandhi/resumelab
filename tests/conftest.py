"""Shared pytest fixtures."""

import pytest

SETTINGS_ENV_VARS = (
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_TEMPERATURE",
    "OPENAI_MAX_RETRIES",
    "OPENAI_TIMEOUT_SECONDS",
    "CANDIDATE_PROFILE_PATH",
    "OUTPUT_DIR",
    "LOG_LEVEL",
)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    """Hide the developer's real configuration from every test.

    Without this, a shell that exports ``OPENAI_API_KEY`` would silently satisfy
    tests that assert configuration is missing.
    """
    for name in SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
