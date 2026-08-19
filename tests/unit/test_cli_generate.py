"""Tests for the generate command and the pipeline it drives.

The LLM is a scripted fake returning one response per stage, so a full run happens
without a network call.
"""

import json

import pytest
import yaml
from pypdf import PdfReader
from typer.testing import CliRunner

from resumelab.cli import app
from resumelab.config import LLMProvider
from resumelab.exceptions import LLMGenerationError
from resumelab.models.resume import (
    CondensedContent,
    ExperienceBullets,
    GeneratedSkills,
    GeneratedSummary,
    ProjectContent,
)

OPENAI_KEY = "sk-test-not-a-real-key"
ANTHROPIC_KEY = "sk-ant-test-not-a-real-key"

SUMMARY = (
    "Storage infrastructure engineer who builds distributed data-path services in Go "
    "and Java on Linux, close to NVMe devices and network storage protocols."
)
BULLETS = (
    "Built a replication controller in Go placing volumes across 3,000 nodes, "
    "cutting rebalance to ten minutes.",
    "Instrumented the NVMe write path with per-device histograms, surfacing tail "
    "regressions before release.",
    "Designed an erasure-coded storage tier on Linux, cutting capacity overhead "
    "40% with p99 read latency flat.",
)

runner = CliRunner()


def project_content(index: int) -> ProjectContent:
    return ProjectContent(
        subtitle=f"NVMe-oF Backed Storage Engine {index}",
        technologies=("Go", "NVMe-oF"),
        bullets=tuple(f"{bullet} Variant {index}." for bullet in BULLETS),
    )


def pipeline_responses(job_analysis, transformation_strategy):
    """One response per stage, in the order the pipeline calls them."""
    return [
        job_analysis,
        transformation_strategy,
        GeneratedSummary(summary=SUMMARY),
        ExperienceBullets(bullets=BULLETS),
        project_content(1),
        project_content(2),
        project_content(3),
        GeneratedSkills(skills=SKILLS),
    ]


SKILLS = (
    "Go",
    "Java",
    "Python",
    "Linux",
    "NVMe-oF",
    "NFS",
    "Distributed Systems",
    "Replication",
    "Kubernetes",
    "PostgreSQL",
)


@pytest.fixture
def workspace(monkeypatch, tmp_path, profile_data):
    """An isolated working directory holding a profile and a job description."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)

    profile = tmp_path / "candidate_profile.yaml"
    profile.write_text(yaml.safe_dump(profile_data, sort_keys=False), encoding="utf-8")
    monkeypatch.setenv("CANDIDATE_PROFILE_PATH", str(profile))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))

    jd = tmp_path / "crusoe.txt"
    jd.write_text(
        "Storage Infrastructure Engineer. Build distributed storage services in Go "
        "and Java on Linux, working with NVMe devices and network storage protocols.",
        encoding="utf-8",
    )
    return tmp_path, jd


@pytest.fixture
def fake_llm(monkeypatch, make_llm_client, job_analysis, transformation_strategy):
    client = make_llm_client(pipeline_responses(job_analysis, transformation_strategy))
    monkeypatch.setattr("resumelab.cli.create_llm_client", lambda *_a, **_k: client)
    return client


def invoke(*args):
    return runner.invoke(app, list(args))


def run_directory(tmp_path):
    return next((tmp_path / "output" / "runs").iterdir())


# --- a complete run -------------------------------------------------------


def test_a_resume_is_generated(workspace, fake_llm):
    tmp_path, jd = workspace

    result = invoke("generate", "--jd", str(jd))

    assert result.exit_code == 0, result.output
    assert (run_directory(tmp_path) / "resume.pdf").exists()


def test_every_stage_runs_in_order(workspace, fake_llm):
    _, jd = workspace

    invoke("generate", "--jd", str(jd))

    assert [call.purpose for call in fake_llm.calls] == [
        "jd_analysis",
        "transformation_strategy",
        "summary",
        "experience",
        "project",
        "project",
        "project",
        "skills",
    ]


def test_the_run_directory_holds_every_artifact(workspace, fake_llm):
    tmp_path, jd = workspace

    invoke("generate", "--jd", str(jd))

    written = {path.name for path in run_directory(tmp_path).iterdir()}
    assert written == {
        "jd.txt",
        "jd_analysis.json",
        "transformation_strategy.json",
        "generated_resume.json",
        "metadata.json",
        "resume.pdf",
    }


def test_the_run_is_named_after_the_posting(workspace, fake_llm):
    tmp_path, jd = workspace

    invoke("generate", "--jd", str(jd))

    assert run_directory(tmp_path).name.endswith("_crusoe")


def test_the_recorded_resume_holds_the_transformed_content(workspace, fake_llm):
    tmp_path, jd = workspace

    invoke("generate", "--jd", str(jd))

    recorded = json.loads((run_directory(tmp_path) / "generated_resume.json").read_text())
    assert recorded["summary"] == SUMMARY
    assert recorded["projects"][0]["subtitle"] == "NVMe-oF Backed Storage Engine 1"


def test_the_pdf_contains_the_generated_content(workspace, fake_llm):
    tmp_path, jd = workspace

    invoke("generate", "--jd", str(jd))

    text = " ".join(
        " ".join(page.extract_text().split())
        for page in PdfReader(run_directory(tmp_path) / "resume.pdf").pages
    )
    assert SUMMARY in text
    assert "Ada Lovelace" in text


def test_personal_details_reach_the_pdf_but_never_the_model(workspace, fake_llm):
    """They are withheld from every prompt and copied in at assembly."""
    tmp_path, jd = workspace

    invoke("generate", "--jd", str(jd))

    for call in fake_llm.calls:
        assert "ada@example.edu" not in call.user_prompt
    text = "".join(
        page.extract_text() for page in PdfReader(run_directory(tmp_path) / "resume.pdf").pages
    )
    assert "ada@example.edu" in text


def test_the_source_profile_is_not_modified(workspace, fake_llm):
    tmp_path, jd = workspace
    profile = tmp_path / "candidate_profile.yaml"
    before = profile.read_bytes()

    invoke("generate", "--jd", str(jd))

    assert profile.read_bytes() == before


# --- metadata -------------------------------------------------------------


def test_metadata_describes_the_run(workspace, fake_llm):
    tmp_path, jd = workspace

    invoke("generate", "--jd", str(jd))

    metadata = json.loads((run_directory(tmp_path) / "metadata.json").read_text())
    assert metadata["provider"] == "openai"
    assert metadata["llm_calls"] == 8
    assert metadata["candidate_profile_hash"]
    assert metadata["page_count"] == 1
    assert metadata["condensed"] is False


def test_no_api_key_reaches_the_run_directory(workspace, fake_llm):
    tmp_path, jd = workspace

    invoke("generate", "--jd", str(jd))

    for path in run_directory(tmp_path).iterdir():
        if path.suffix != ".pdf":
            assert OPENAI_KEY not in path.read_text(encoding="utf-8")


# --- output ---------------------------------------------------------------


def test_the_pdf_can_also_be_written_somewhere_convenient(workspace, fake_llm):
    tmp_path, jd = workspace
    target = tmp_path / "resumes" / "crusoe.pdf"

    result = invoke("generate", "--jd", str(jd), "--output", str(target))

    assert result.exit_code == 0
    assert target.read_bytes().startswith(b"%PDF-")
    assert target.read_bytes() == (run_directory(tmp_path) / "resume.pdf").read_bytes()


def test_the_summary_reports_where_everything_went(workspace, fake_llm):
    tmp_path, jd = workspace

    result = invoke("generate", "--jd", str(jd))

    assert str(run_directory(tmp_path)) in result.stdout
    assert "resume.pdf" in result.stdout
    assert "openai" in result.stdout
    assert "1 page(s)" in result.stdout


# --- condensation ---------------------------------------------------------


def test_content_that_overflows_is_condensed_and_re_rendered(
    monkeypatch, workspace, make_llm_client, job_analysis, transformation_strategy
):
    """The renderer tightens what it can; beyond that the only honest fix is less text."""
    tmp_path, jd = workspace
    responses = pipeline_responses(job_analysis, transformation_strategy)
    responses.append(CondensedContent(summary=SUMMARY, bullets=BULLETS * 4))
    client = make_llm_client(responses)
    monkeypatch.setattr("resumelab.cli.create_llm_client", lambda *_a, **_k: client)
    monkeypatch.setattr(
        "resumelab.pipeline.generator.render_resume",
        _render_that_overflows_once(),
    )

    result = invoke("generate", "--jd", str(jd))

    assert result.exit_code == 0, result.output
    assert [call.purpose for call in client.calls][-1] == "condense"
    assert json.loads((run_directory(tmp_path) / "metadata.json").read_text())["condensed"] is True


def _render_that_overflows_once():
    """Report a two-page result the first time, then defer to the real renderer."""
    from resumelab.rendering import RenderResult, render_resume

    calls: list[int] = []

    def render(resume, path):
        real = render_resume(resume, path)
        calls.append(1)
        if len(calls) == 1:
            return RenderResult(path=real.path, scale=real.scale, page_count=2)
        return real

    return render


# --- provider selection ---------------------------------------------------


def test_the_provider_flag_wins(monkeypatch, workspace, fake_llm):
    _, jd = workspace
    chosen: list[LLMProvider | None] = []
    monkeypatch.setattr(
        "resumelab.cli.create_llm_client",
        lambda _s, *, provider=None: (chosen.append(provider), fake_llm)[1],
    )

    invoke("generate", "--jd", str(jd), "--provider", "anthropic")

    assert chosen == [LLMProvider.ANTHROPIC]


def test_with_one_key_configured_no_question_is_asked(workspace, fake_llm):
    """Nothing to choose between, so nothing to ask."""
    _, jd = workspace

    result = invoke("generate", "--jd", str(jd))

    assert "Which provider" not in result.stdout


def test_with_both_keys_the_researcher_is_asked(monkeypatch, workspace, fake_llm):
    _, jd = workspace
    monkeypatch.setenv("ANTHROPIC_API_KEY", ANTHROPIC_KEY)
    monkeypatch.setattr("resumelab.cli._is_interactive", lambda: True)

    result = runner.invoke(app, ["generate", "--jd", str(jd)], input="anthropic\n")

    assert "Which provider" in result.stdout
    assert result.exit_code == 0, result.output


def test_the_answer_selects_the_provider(
    monkeypatch, workspace, make_llm_client, job_analysis, transformation_strategy
):
    _, jd = workspace
    monkeypatch.setenv("ANTHROPIC_API_KEY", ANTHROPIC_KEY)
    monkeypatch.setattr("resumelab.cli._is_interactive", lambda: True)
    chosen: list[LLMProvider | None] = []
    client = make_llm_client(pipeline_responses(job_analysis, transformation_strategy))
    monkeypatch.setattr(
        "resumelab.cli.create_llm_client",
        lambda _s, *, provider=None: (chosen.append(provider), client)[1],
    )

    runner.invoke(app, ["generate", "--jd", str(jd)], input="anthropic\n")

    assert chosen == [LLMProvider.ANTHROPIC]


def test_an_unrecognised_answer_is_asked_again(monkeypatch, workspace, fake_llm):
    _, jd = workspace
    monkeypatch.setenv("ANTHROPIC_API_KEY", ANTHROPIC_KEY)
    monkeypatch.setattr("resumelab.cli._is_interactive", lambda: True)

    result = runner.invoke(app, ["generate", "--jd", str(jd)], input="gemini\nopenai\n")

    assert "Choose one of" in result.stderr
    assert result.exit_code == 0, result.output


def test_a_non_interactive_run_is_never_blocked_by_the_prompt(monkeypatch, workspace, fake_llm):
    """A batch run must not hang waiting for an answer nobody is there to give."""
    _, jd = workspace
    monkeypatch.setenv("ANTHROPIC_API_KEY", ANTHROPIC_KEY)
    monkeypatch.setattr("resumelab.cli._is_interactive", lambda: False)

    result = invoke("generate", "--jd", str(jd))

    assert result.exit_code == 0, result.output
    assert "Which provider" not in result.stdout


# --- failures -------------------------------------------------------------


def test_a_missing_profile_is_reported_readably(monkeypatch, workspace, fake_llm):
    tmp_path, jd = workspace
    monkeypatch.setenv("CANDIDATE_PROFILE_PATH", str(tmp_path / "absent.yaml"))

    result = invoke("generate", "--jd", str(jd))

    assert result.exit_code == 1
    assert "Candidate profile not found" in result.stderr
    assert "Traceback" not in result.stderr


def test_a_stage_failure_leaves_the_earlier_artifacts_behind(
    monkeypatch, workspace, make_llm_client, job_analysis
):
    """A failed run should still show the reasoning that led to the failure."""
    tmp_path, jd = workspace
    client = make_llm_client([job_analysis, LLMGenerationError("the model was unreachable")])
    monkeypatch.setattr("resumelab.cli.create_llm_client", lambda *_a, **_k: client)

    result = invoke("generate", "--jd", str(jd))

    assert result.exit_code == 1
    written = {path.name for path in run_directory(tmp_path).iterdir()}
    assert written == {"jd.txt", "jd_analysis.json"}


def test_the_help_carries_the_research_disclaimer():
    result = invoke("generate", "--help")

    assert "not in the source profile" in " ".join(result.stdout.split())


def test_an_inline_posting_produces_a_named_run(workspace, fake_llm):
    tmp_path, _ = workspace

    result = invoke(
        "generate",
        "--jd-text",
        "Storage engineer wanted for distributed systems work in Go on Linux at scale.",
    )

    assert result.exit_code == 0, result.output
    assert run_directory(tmp_path).name.endswith("_inline")


def test_a_resume_that_still_overflows_after_condensing_is_kept_readable(
    monkeypatch, workspace, make_llm_client, job_analysis, transformation_strategy
):
    """Two readable pages beat one unreadable page, and the artifacts say so."""
    from resumelab.rendering import RenderResult, render_resume

    tmp_path, jd = workspace
    responses = pipeline_responses(job_analysis, transformation_strategy)
    responses.append(CondensedContent(summary=SUMMARY, bullets=BULLETS * 4))
    client = make_llm_client(responses)
    monkeypatch.setattr("resumelab.cli.create_llm_client", lambda *_a, **_k: client)
    monkeypatch.setattr(
        "resumelab.pipeline.generator.render_resume",
        lambda resume, path: RenderResult(
            path=render_resume(resume, path).path, scale=0.895, page_count=2
        ),
    )

    result = invoke("generate", "--jd", str(jd))

    assert result.exit_code == 0, result.output
    assert "still spans 2 pages" in result.stderr
    assert json.loads((run_directory(tmp_path) / "metadata.json").read_text())["page_count"] == 2


def test_the_interactive_check_reads_the_real_terminal():
    """The seam the tests patch must still be wired to stdin."""
    import sys as real_sys

    from resumelab.cli import _is_interactive

    assert _is_interactive() == real_sys.stdin.isatty()
