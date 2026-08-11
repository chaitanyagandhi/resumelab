"""Shared pytest fixtures."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from resumelab.llm.client import LLMCallStats, TokenUsage
from resumelab.models.analysis import JobAnalysis
from resumelab.models.candidate import CandidateProfile
from resumelab.models.job import JobDescription, JobDescriptionSource
from resumelab.models.resume import (
    GeneratedExperience,
    GeneratedProject,
    GeneratedResume,
    SkillGroup,
)
from resumelab.models.strategy import (
    ExperienceDirection,
    ProjectDirection,
    TransformationStrategy,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPERIENCE_BULLETS = (
    "Built a replication controller in Go that places volumes across 3,000 nodes, "
    "cutting rebalance time from hours to under ten minutes.",
    "Instrumented the NVMe write path with per-device latency histograms, surfacing "
    "tail regressions before they reached customers.",
    "Designed an erasure-coded storage tier on Linux that cut capacity overhead by "
    "40% while holding p99 read latency flat.",
)

PROJECT_BULLETS = (
    "Architected a shared-nothing ingestion path that fans writes across NVMe-oF "
    "targets at 40k events per second per node.",
    "Implemented an idempotent replay log with content-addressed segments, making "
    "broker-failure recovery deterministic.",
    "Measured durability under injected disk faults, holding p99 commit latency "
    "under 12ms across 200 simulated failures.",
)

SETTINGS_ENV_VARS = (
    "LLM_PROVIDER",
    "LLM_MAX_RETRIES",
    "LLM_TIMEOUT_SECONDS",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_TEMPERATURE",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_MAX_TOKENS",
    "ANTHROPIC_EFFORT",
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


@dataclass
class RecordedCall:
    """One captured request to the fake client."""

    system_prompt: str
    user_prompt: str
    response_model: type
    purpose: str


class RecordingLLMClient:
    """A deterministic :class:`~resumelab.llm.client.LLMClient` for pipeline tests.

    Returns queued responses in order and records every request, so a stage can be
    tested on what it asked for as well as what it did with the answer. Raises queued
    exceptions, which is how failure paths are exercised.
    """

    def __init__(self, responses, *, model="fake-model-1"):
        self._responses = list(responses)
        self._model = model
        self.calls: list[RecordedCall] = []
        self.stats = LLMCallStats()

    @property
    def model(self) -> str:
        return self._model

    def generate_structured(self, *, system_prompt, user_prompt, response_model, purpose):
        self.calls.append(
            RecordedCall(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=response_model,
                purpose=purpose,
            )
        )
        self.stats = self.stats.record(TokenUsage(1, 1, 2))
        if not self._responses:
            raise AssertionError(f"no queued response for purpose={purpose!r}")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    @property
    def last_call(self) -> RecordedCall:
        return self.calls[-1]


@pytest.fixture
def generated_resume(candidate_profile):
    """A complete, valid resume, as the assembler would produce it."""
    return GeneratedResume(
        personal=candidate_profile.personal,
        summary=(
            "Storage infrastructure engineer who builds distributed data-path services "
            "in Go and Java on Linux, close to NVMe devices and network protocols."
        ),
        education=candidate_profile.education,
        experiences=(
            GeneratedExperience(
                company="Analytical Engines Inc.",
                title="Software Engineer Intern",
                location="Remote",
                start_date="May 2025",
                end_date="Aug 2025",
                bullets=EXPERIENCE_BULLETS,
            ),
        ),
        projects=tuple(
            GeneratedProject(
                name=f"Project {index}",
                subtitle=f"Distributed Storage Engine {index}",
                date="2025",
                technologies=("Go", "Linux", "NVMe-oF"),
                bullets=tuple(f"{bullet} Variant {index}." for bullet in PROJECT_BULLETS),
            )
            for index in range(1, 4)
        ),
        skills=(
            SkillGroup(label="Languages", skills=("Go", "Java", "Python")),
            SkillGroup(label="Storage & Systems", skills=("Linux", "NVMe-oF", "NFS")),
        ),
        achievements=candidate_profile.achievements,
    )


@pytest.fixture
def make_llm_client():
    """Factory for :class:`RecordingLLMClient`, so tests need no cross-module import."""
    return RecordingLLMClient


@pytest.fixture
def job_description():
    """A storage-infrastructure posting, matching the shipped example's archetype."""
    return JobDescription(
        text=(
            "Software Engineer, Cloud Storage Infrastructure at Northlake Systems. "
            "Design services in Go and Java that manage volume placement and "
            "replication across storage clusters. Work close to the Linux storage "
            "stack, NVMe devices, and network storage protocols including NFS, SMB, "
            "iSCSI, and NVMe-oF."
        ),
        source=JobDescriptionSource.TEXT,
    )


@pytest.fixture
def job_analysis():
    """A plausible analysis of the storage posting above."""
    return JobAnalysis(
        company="Northlake Systems",
        role_title="Software Engineer, Cloud Storage Infrastructure",
        role_archetype="storage infrastructure engineer",
        seniority="early-career",
        core_languages=("Go", "Java", "C"),
        frameworks=(),
        infrastructure=("Linux", "NVMe", "Kubernetes"),
        databases=(),
        ai_ml_concepts=(),
        domain_concepts=("network storage protocols", "NFS", "SMB", "iSCSI", "NVMe-oF"),
        engineering_concepts=("distributed consensus", "latency profiling"),
        responsibilities=("Design volume placement and replication services",),
        high_priority_requirements=("Proficiency in a systems language", "Linux fundamentals"),
        bonus_requirements=("Filesystem internals",),
        soft_traits=("Communicates through design documents",),
        high_value_keywords=("NVMe-oF", "distributed storage", "Go"),
        technical_identity=(
            "Early-career storage infrastructure engineer experienced with Go, Java, "
            "Linux, distributed storage systems, NVMe and network storage protocols."
        ),
        ideal_candidate_profile=(
            "An engineer who has worked close to the operating system and reasons "
            "about tail latency and failure modes in distributed storage."
        ),
    )


@pytest.fixture
def candidate_profile(profile_data):
    """The validated profile corresponding to ``profile_data``."""
    return CandidateProfile.model_validate(profile_data)


@pytest.fixture
def transformation_strategy():
    """A strategy covering the fixture profile: one role and three projects."""
    return TransformationStrategy(
        target_identity="Early-career storage infrastructure engineer.",
        summary_direction="Lead with distributed storage systems work in Go on Linux.",
        experience_directions=(
            ExperienceDirection(
                experience="Analytical Engines Inc.",
                target_framing="Reframe the ingestion work as a storage data path.",
                concepts_to_emphasize=("write path latency", "replication"),
                jd_terms_to_incorporate=("NVMe", "Go"),
            ),
        ),
        project_directions=tuple(
            ProjectDirection(
                project=f"Project {index}",
                new_positioning=f"Positioning {index} toward distributed storage.",
                possible_title_direction=f"Project {index} — Storage Engine",
                concepts_to_incorporate=("erasure coding", "tail latency"),
            )
            for index in range(1, 4)
        ),
        skills_priority=("Go", "Linux", "Distributed Systems"),
        tone="Direct and systems-oriented.",
        overall_strategy="Present the candidate as a storage systems engineer.",
    )


@pytest.fixture(scope="session")
def repo_root():
    """Root of the repository, for tests that read tracked example files."""
    return REPO_ROOT


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
