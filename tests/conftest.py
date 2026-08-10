"""Shared pytest fixtures."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

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


@pytest.fixture(scope="session")
def profile_template_path():
    """Path to the unpopulated candidate profile template tracked in the repository."""
    return REPO_ROOT / "data" / "candidate_profile.example.yaml"


@pytest.fixture
def profile_data():
    """A minimal but fully valid candidate profile, as parsed YAML would supply it.

    Deliberately built as plain data so tests can corrupt one key at a time.
    """
    return {
        "personal": {
            "name": "Ada Lovelace",
            "email": "ada@example.edu",
            "phone": "+1 555 0100",
            "linkedin": "linkedin.com/in/ada",
            "github": "github.com/ada",
            "location": "Los Angeles, CA",
        },
        "education": [
            {
                "institution": "University of Southern California",
                "degree": "M.S.",
                "field": "Computer Science",
                "location": "Los Angeles, CA",
                "start_date": "Aug 2024",
                "end_date": "May 2026",
                "gpa": "3.9",
                "coursework": ["Distributed Systems", "Machine Learning"],
            }
        ],
        "experiences": [
            {
                "company": "Analytical Engines Inc.",
                "title": "Software Engineer Intern",
                "location": "Remote",
                "start_date": "May 2025",
                "end_date": "Aug 2025",
                "description": "Backend services team.",
                "bullets": [
                    "Built an ingestion service handling 2M events per day.",
                    "Cut p99 latency by 40% by restructuring the write path.",
                    "Added contract tests covering 30 internal endpoints.",
                ],
            }
        ],
        "projects": [
            {
                "name": f"Project {index}",
                "subtitle": f"Subtitle {index}",
                "date": "2025",
                "technologies": ["Python", "PostgreSQL"],
                "description": f"Description {index}.",
                "bullets": [
                    f"Designed component {index}A.",
                    f"Implemented component {index}B.",
                    f"Measured component {index}C.",
                ],
            }
            for index in range(1, 4)
        ],
        "skills": {
            "programming_languages": ["Python", "Go"],
            "frameworks": ["FastAPI"],
            "databases": ["PostgreSQL"],
            "cloud_devops": ["AWS"],
            "ai_ml": ["PyTorch"],
            "other": ["Git"],
        },
        "achievements": ["Dean's List"],
    }
