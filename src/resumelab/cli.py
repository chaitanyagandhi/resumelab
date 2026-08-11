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

import sys
import textwrap
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer

from resumelab import __version__
from resumelab.config import LLMProvider, Settings, load_settings
from resumelab.exceptions import ResumeLabError, UnsafePathError
from resumelab.llm.factory import create_llm_client
from resumelab.loaders import load_job_description
from resumelab.logging_setup import configure_logging
from resumelab.models.analysis import JobAnalysis
from resumelab.pipeline import (
    GenerationResult,
    analyze_job_description,
    copy_pdf,
    generate_resume,
)
from resumelab.utils.paths import prepare_output_file

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
PdfOutputOption = Annotated[
    Path | None,
    typer.Option("--output", "-o", help="Also write the resume PDF here."),
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
    """Turn a failure into a message and a non-zero exit.

    Expected failures are reported as themselves. Anything else is a bug, and gets a
    short line naming it plus a pointer at --debug — a wall of traceback tells an
    ordinary user nothing and buries the one line that does.
    """
    try:
        yield
    except ResumeLabError as exc:
        if debug:
            raise
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    except KeyboardInterrupt:
        typer.secho("Interrupted.", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=130) from None
    except Exception as exc:
        if debug:
            raise
        typer.secho(
            f"ResumeLab failed unexpectedly: {type(exc).__name__}: {exc}\n"
            "Re-run with --debug for the full traceback.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from exc


def _write_json(path: Path, analysis: JobAnalysis, *, debug: bool) -> None:
    """Write the analysis where a later comparison can read it."""
    with _reported_failures(debug):
        target = prepare_output_file(path, subject="The analysis output path")
        try:
            target.write_text(analysis.model_dump_json(indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            raise UnsafePathError(f"Could not write the analysis to {target}: {exc}") from exc


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


@app.command()
def generate(
    jd: JdOption = None,
    jd_text: JdTextOption = None,
    provider: ProviderOption = None,
    output: PdfOutputOption = None,
    debug: DebugOption = False,
) -> None:
    """Generate a resume tailored to a job description.

    Every run writes a self-contained directory under the configured output path,
    holding the posting, the analysis, the strategy, the generated resume, the
    metadata, and the PDF.

    The generated resume may present technologies, metrics, and project framings
    that are not in the source profile. That transformation is what this tool exists
    to study; the source profile is never modified.
    """
    settings = _load_settings(debug)
    configure_logging(settings.log_level, debug=debug)

    with _reported_failures(debug):
        job_description = load_job_description(path=jd, text=jd_text)
        chosen = _choose_provider(settings, provider)
        client = create_llm_client(settings, provider=chosen)
        result = generate_resume(
            job_description,
            settings=settings,
            provider=chosen or settings.resolved_provider,
            client=client,
        )
        if output is not None:
            copy_pdf(result.render, prepare_output_file(output, subject="The resume output path"))

    typer.echo(_format_result(result, output))


def _choose_provider(settings: Settings, provider: LLMProvider | None) -> LLMProvider | None:
    """Decide which provider to use, asking when the choice is genuinely open.

    The flag wins. Otherwise the question is only worth asking when both providers
    are usable and someone is there to answer: in a script or a batch run the
    configured provider is used without a prompt.
    """
    if provider is not None:
        return provider
    if not (settings.openai_api_key and settings.anthropic_api_key):
        return None
    if not _is_interactive():
        return None

    options = [member.value for member in LLMProvider]
    answer = typer.prompt(
        f"Which provider should generate this resume? ({'/'.join(options)})",
        default=settings.resolved_provider.value,
    )
    while answer not in options:
        typer.secho(f"Choose one of: {', '.join(options)}", fg=typer.colors.YELLOW, err=True)
        answer = typer.prompt("Provider", default=settings.resolved_provider.value)
    return LLMProvider(answer)


def _is_interactive() -> bool:
    """Whether someone is at the keyboard to answer a question."""
    return sys.stdin.isatty()


def _format_result(result: GenerationResult, output: Path | None) -> str:
    """Summarize the run, leading with the file the researcher wants to open."""
    metadata = result.metadata
    lines = [
        _heading("GENERATED"),
        f"  resume:    {output or result.render.path}",
        f"  run:       {result.run.directory}",
        "",
        _heading("RUN"),
        f"  provider:  {metadata.provider} ({metadata.model})",
        f"  identity:  {result.resume.summary[:70]}...",
        f"  calls:     {metadata.llm_calls}"
        f"  tokens: {metadata.token_usage.total_tokens}"
        f"  duration: {metadata.duration_seconds:.1f}s",
        f"  layout:    {result.render.page_count} page(s)"
        f" at scale {result.render.scale:g}"
        f"{', condensed to fit' if result.condensed else ''}",
    ]
    return "\n".join(lines)
