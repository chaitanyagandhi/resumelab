"""A deterministic fake LLM, and the fixtures that drive a whole pipeline run.

The unit tests script a fake with a fixed queue of responses. That proves each stage
does the right thing with an answer, but it cannot prove the stages fit together,
because a queue does not care what it was asked.

This fake reads its prompts the way a model would: it parses the profile out of the
prompt to learn which roles and projects exist, and it draws its vocabulary from the
job description. So the strategy really does have to echo names the transformers can
match, and two different postings really do produce two different resumes.

It is deterministic: the same inputs produce byte-identical output, which is what
lets a test assert that a change in the resume came from a change in the input.
"""

from __future__ import annotations

import hashlib
import json
import re

import pytest
import yaml

from resumelab.llm.client import LLMCallStats, TokenUsage
from resumelab.models.analysis import JobAnalysis
from resumelab.models.resume import (
    CondensedContent,
    ExperienceBullets,
    GeneratedSkills,
    GeneratedSummary,
    ProjectContent,
    SkillGroup,
)
from resumelab.models.strategy import (
    ExperienceDirection,
    ProjectDirection,
    TransformationStrategy,
)

WORD = re.compile(r"\b[A-Za-z][A-Za-z0-9+#.-]{1,20}\b")

DISTINCTIVE = re.compile(r"(?<=.)[A-Z]|[0-9]|[.+#-]")
"""What makes a token unmistakably a technology name rather than a job-ad word.

An interior capital, a digit, or punctuation inside the word: NVMe, NVMe-oF, iSCSI,
GenAI, Next.js. Plain capitalised words like "Software" or "Engineer" fail this and
are only used to pad the list out.
"""

STOP_WORDS = frozenset(
    {
        "The",
        "This",
        "That",
        "We",
        "You",
        "Your",
        "Our",
        "About",
        "What",
        "Design",
        "Build",
        "Work",
        "Extend",
        "Improve",
        "Experience",
        "Software",
        "Engineer",
        "Full",
        "Stack",
        "Services",
        "Enterprise",
        "Large",
        "Document",
        "Corpora",
        "SOURCE",
        "JOB",
        "CANDIDATE",
        "PROFILE",
        "DESCRIPTION",
        "ANALYSIS",
        "BEGIN",
        "END",
        "TRANSFORMATION",
        "STRATEGY",
        "DIRECTION",
        "ENTRY",
        "BULLETS",
        "ALREADY",
        "WRITTEN",
        "ELSEWHERE",
        "RESUME",
        "SUMMARY",
        "LENGTH",
        "BUDGET",
    }
)


class DeterministicLLM:
    """A fake that answers from its inputs, reproducibly."""

    model = "deterministic-fake-1"

    def __init__(self) -> None:
        self.stats = LLMCallStats()
        self.calls: list[tuple[str, str]] = []

    def generate_structured(self, *, system_prompt, user_prompt, response_model, purpose):
        self.calls.append((purpose, user_prompt))
        self.stats = self.stats.record(TokenUsage(120, 60, 180))

        builders = {
            JobAnalysis: self._analysis,
            TransformationStrategy: self._strategy,
            GeneratedSummary: self._summary,
            ExperienceBullets: self._experience_bullets,
            ProjectContent: self._project_content,
            GeneratedSkills: self._skills,
            CondensedContent: self._condensed,
        }
        try:
            build = builders[response_model]
        except KeyError:  # pragma: no cover - a new stage would need a fake response
            raise AssertionError(
                f"the fake has no response for {response_model.__name__}"
            ) from None
        return build(user_prompt)

    # --- stages -----------------------------------------------------------

    def _analysis(self, prompt: str) -> JobAnalysis:
        terms = _terms(prompt)
        return JobAnalysis(
            company=terms[0] if terms else "",
            role_title=f"{terms[0]} Engineer" if terms else "Engineer",
            role_archetype=f"{_lower(terms, 0)} engineer",
            seniority="early-career",
            core_languages=tuple(terms[:3]),
            frameworks=tuple(terms[3:5]),
            infrastructure=tuple(terms[5:8]),
            databases=tuple(terms[8:9]),
            ai_ml_concepts=(),
            domain_concepts=tuple(terms[:6]),
            engineering_concepts=("latency profiling", "failure recovery"),
            responsibilities=(f"Build {_lower(terms, 1)} systems",),
            high_priority_requirements=tuple(terms[:2]),
            bonus_requirements=tuple(terms[2:4]),
            soft_traits=("Writes design documents",),
            high_value_keywords=tuple(terms[:5]),
            technical_identity=(
                f"Early-career {_lower(terms, 0)} engineer experienced with "
                f"{', '.join(terms[:4])} and the systems built around them."
            ),
            ideal_candidate_profile=(
                f"An engineer who has worked closely with {_lower(terms, 1)} and reasons "
                "about failure modes in production."
            ),
        )

    def _strategy(self, prompt: str) -> TransformationStrategy:
        profile = _json_section(prompt, "CANDIDATE PROFILE")
        analysis = _json_section(prompt, "JOB ANALYSIS")
        terms = tuple(analysis["high_value_keywords"]) or ("systems",)
        return TransformationStrategy(
            target_identity=analysis["technical_identity"],
            summary_direction=f"Lead with {terms[0]} and the systems around it.",
            experience_directions=tuple(
                ExperienceDirection(
                    experience=entry["company"],
                    target_framing=f"Reframe {entry['company']} as {terms[0]} work.",
                    concepts_to_emphasize=terms[:2],
                    jd_terms_to_incorporate=terms[:3],
                )
                for entry in profile["experiences"]
            ),
            project_directions=tuple(
                ProjectDirection(
                    project=project["name"],
                    new_positioning=f"{project['name']} as {terms[0]} infrastructure.",
                    possible_title_direction=f"{terms[0]} Platform",
                    concepts_to_incorporate=terms[:2],
                )
                for project in profile["projects"]
            ),
            skills_priority=terms[:4],
            tone="Direct and systems-oriented.",
            overall_strategy=f"Present the candidate as a {analysis['role_archetype']}.",
        )

    def _summary(self, prompt: str) -> GeneratedSummary:
        strategy = _json_section(prompt, "TRANSFORMATION STRATEGY")
        return GeneratedSummary(
            summary=(
                f"{strategy['target_identity']} Builds and measures the systems that "
                "carry the load."
            )[:300]
        )

    def _experience_bullets(self, prompt: str) -> ExperienceBullets:
        direction = _json_section(prompt, "DIRECTION FOR THIS ENTRY")
        source = _json_section(prompt, "SOURCE EXPERIENCE")
        return ExperienceBullets(bullets=_bullets(direction, seed=source["company"]))

    def _project_content(self, prompt: str) -> ProjectContent:
        direction = _json_section(prompt, "DIRECTION FOR THIS ENTRY")
        source = _json_section(prompt, "SOURCE PROJECT")
        terms = list(direction["concepts_to_incorporate"]) or ["systems"]
        return ProjectContent(
            subtitle=f"{direction['possible_title_direction']} for {source['name']}"[:90],
            technologies=tuple(dict.fromkeys([*terms, "Linux", "Go"]))[:6],
            bullets=_bullets(direction, seed=source["name"]),
        )

    def _skills(self, prompt: str) -> GeneratedSkills:
        strategy = _json_section(prompt, "TRANSFORMATION STRATEGY")
        priority = list(dict.fromkeys(strategy["skills_priority"])) or ["Systems"]
        return GeneratedSkills(
            groups=(
                SkillGroup(label="Core", skills=tuple(priority[:4])),
                SkillGroup(label="Systems", skills=("Linux", "Distributed Systems")),
            )
        )

    def _condensed(self, prompt: str) -> CondensedContent:
        bullets = re.findall(r"^\d+\. (.+)$", prompt, flags=re.MULTILINE)
        return CondensedContent(
            summary=(
                "Systems engineer who builds and measures the infrastructure that carries the load."
            ),
            bullets=tuple(f"{bullet[:120].rstrip()} (short)" for bullet in bullets),
        )


# --- helpers --------------------------------------------------------------


def _terms(prompt: str) -> tuple[str, ...]:
    """The names a posting is actually about, most distinctive first.

    Technology names lead, so a storage posting yields NVMe and NFS rather than
    "Software" and "Engineer". Ordinary capitalised words pad the list out.
    """
    words = [
        token
        for token in WORD.findall(prompt)
        if token not in STOP_WORDS and (token[0].isupper() or DISTINCTIVE.search(token))
    ]
    technical = [token for token in words if DISTINCTIVE.search(token)]
    ordinary = [token for token in words if token not in technical and token not in STOP_WORDS]
    return tuple(dict.fromkeys([*technical, *ordinary]))[:12]


def _lower(terms: tuple[str, ...], index: int) -> str:
    return terms[index].lower() if len(terms) > index else "systems"


def _json_section(prompt: str, label: str) -> dict:
    """Read back a section the prompt layer rendered as labelled JSON."""
    for block in prompt.split("\n\n"):
        if block.startswith(f"{label}:\n"):
            return json.loads(block.split("\n", 1)[1])
    raise AssertionError(f"prompt has no {label!r} section")


def _bullets(direction: dict, *, seed: str) -> tuple[str, ...]:
    """Three distinct, plausibly-sized bullets derived from one entry's direction."""
    concepts = (
        list(direction.get("concepts_to_incorporate") or [])
        or list(direction.get("concepts_to_emphasize") or [])
        or ["throughput"]
    )
    digest = hashlib.sha256(seed.encode()).hexdigest()
    scale = int(digest[:4], 16)
    verbs = ("Built", "Instrumented", "Measured")
    return tuple(
        f"{verb} the {concepts[index % len(concepts)]} path for {seed}, "
        f"cutting tail latency by {scale % 40 + 10}% across {scale % 900 + 100} nodes."
        for index, verb in enumerate(verbs)
    )


# --- fixtures -------------------------------------------------------------


@pytest.fixture
def fake_llm():
    return DeterministicLLM()


@pytest.fixture
def profile_path(tmp_path, profile_data):
    path = tmp_path / "candidate_profile.yaml"
    path.write_text(yaml.safe_dump(profile_data, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def storage_jd(tmp_path):
    path = tmp_path / "storage.txt"
    path.write_text(
        "Software Engineer, Cloud Storage Infrastructure. Design services in Go and "
        "Java that manage volume placement and replication across storage clusters. "
        "Work close to the Linux storage stack, NVMe devices, and network storage "
        "protocols including NFS, SMB, iSCSI, and NVMe-oF.",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def genai_jd(tmp_path):
    path = tmp_path / "genai.txt"
    path.write_text(
        "Full Stack GenAI Engineer. Build enterprise search over React and Next.js "
        "with Python services. Work on semantic retrieval, knowledge graphs, vector "
        "search, embeddings, and agent orchestration for large document corpora.",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def settings(tmp_path, profile_path):
    from resumelab.config import Settings

    return Settings(
        _env_file=None,
        openai_api_key="sk-test-not-a-real-key",
        candidate_profile_path=profile_path,
        output_dir=tmp_path / "output",
    )
