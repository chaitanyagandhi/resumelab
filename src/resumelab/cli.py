"""The ResumeLab command line.

This layer is deliberately thin. It parses arguments, wires the services together,
and formats what comes back; every decision about what a stage does lives in the
stage. That keeps the pipeline callable from a notebook or a batch script without
going through argument parsing.

Expected failures are reported as a message and a non-zero exit code. A researcher
who mistypes a path should see one sentence, not a traceback — ``--debug`` re-raises
for when the traceback is the thing you want.
"""

from __future__ import annotations

import textwrap
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer

from resumelab import __version__
from resumelab.config import LLMProvider, Settings, load_settings
from resumelab.exceptions import ResumeLabError
from resumelab.llm.factory import create_llm_client
from resumelab.loaders import load_job_description
from resumelab.logging_setup import configure_logging
from resumelab.models.analysis import JobAnalysis
from resumelab.pipeline import analyze_job_description

app = typer.Typer(
    name="resumelab",
    help=(
        "Study how AI resume tailoring transforms a candidate for a target job. "
        "Generated output may contain claims not present in the source profile."
    ),
    no_args_is_help=True,
    add_completion=False,
)

JdOption = Annotated[
    Path | None,
    typer.Option("--jd", help="Path to a job description file.", show_default=False),
]
JdTextOption = Annotated[
    str | None,
    typer.Option("--jd-text", help="Job description supplied directly.", show_default=False),
]
ProviderOption = Annotated[
    LLMProvider | None,
    typer.Option("--provider", help="Override the configured LLM provider."),
]
OutputOption = Annotated[
    Path | None,
    typer.Option("--output", "-o", help="Write the analysis as JSON to this path."),
]
DebugOption = Annotated[
    bool,
    typer.Option("--debug", help="Log at DEBUG and show tracebacks on failure."),
]


@app.callback()
def main() -> None:
    """ResumeLab: a research prototype for JD-conditioned resume transformation."""


@app.command()
def version() -> None:
    """Print the installed ResumeLab version."""
    typer.echo(__version__)


@app.command()
def analyze(
    jd: JdOption = None,
    jd_text: JdTextOption = None,
    provider: ProviderOption = None,
    output: OutputOption = None,
    debug: DebugOption = False,
) -> None:
    """Analyze a job description without generating a resume.

    Useful on its own: the analysis is what every later stage is conditioned on, so
    reading it is how you tell whether a disappointing resume came from a bad plan
    or a bad reading of the posting.
    """
    settings = _load_settings(debug)
    configure_logging(settings.log_level, debug=debug)

    with _reported_failures(debug):
        job_description = load_job_description(path=jd, text=jd_text)
        client = create_llm_client(settings, provider=provider)
        analysis = analyze_job_description(job_description, client=client)

    if output is not None:
        _write_json(output, analysis, debug=debug)
        typer.echo(f"Wrote analysis to {output}")
    else:
        typer.echo(_format_analysis(analysis))


def _load_settings(debug: bool) -> Settings:
    """Load settings before logging exists, so a config error still reads well."""
    with _reported_failures(debug):
        return load_settings()


@contextmanager
def _reported_failures(debug: bool) -> Iterator[None]:
    """Turn an expected failure into a message and a non-zero exit."""
    try:
        yield
    except ResumeLabError as exc:
        if debug:
            raise
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


def _write_json(path: Path, analysis: JobAnalysis, *, debug: bool) -> None:
    """Write the analysis where a later comparison can read it."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(analysis.model_dump_json(indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        if debug:
            raise
        typer.secho(f"Could not write the analysis to {path}: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


def _format_analysis(analysis: JobAnalysis) -> str:
    """Render the analysis for reading, leading with what drives the pipeline."""
    lines = [
        _heading("TARGET"),
        f"  {analysis.role_title or '<untitled>'} at {analysis.company or '<unnamed company>'}",
        f"  archetype: {analysis.role_archetype}",
        f"  seniority: {analysis.seniority or 'unstated'}",
        "",
        _heading("TECHNICAL IDENTITY"),
        _wrapped(analysis.technical_identity),
        "",
        _heading("IDEAL CANDIDATE"),
        _wrapped(analysis.ideal_candidate_profile),
    ]
    for label, values in _term_sections(analysis):
        if values:
            lines += ["", _heading(label), _wrapped(", ".join(values))]
    return "\n".join(lines)


def _term_sections(analysis: JobAnalysis) -> list[tuple[str, tuple[str, ...]]]:
    return [
        ("LANGUAGES", analysis.core_languages),
        ("FRAMEWORKS", analysis.frameworks),
        ("INFRASTRUCTURE", analysis.infrastructure),
        ("DATABASES", analysis.databases),
        ("AI / ML", analysis.ai_ml_concepts),
        ("DOMAIN CONCEPTS", analysis.domain_concepts),
        ("ENGINEERING CONCEPTS", analysis.engineering_concepts),
        ("RESPONSIBILITIES", analysis.responsibilities),
        ("HIGH PRIORITY REQUIREMENTS", analysis.high_priority_requirements),
        ("BONUS REQUIREMENTS", analysis.bonus_requirements),
        ("SOFT TRAITS", analysis.soft_traits),
        ("HIGH VALUE KEYWORDS", analysis.high_value_keywords),
    ]


def _heading(text: str) -> str:
    return typer.style(text, bold=True)


def _wrapped(text: str, *, width: int = 78) -> str:
    """Indent and wrap a block so long fields stay readable in a terminal."""
    return textwrap.fill(text, width=width, initial_indent="  ", subsequent_indent="  ")
