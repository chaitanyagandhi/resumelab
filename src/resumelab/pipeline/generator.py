"""The full generation run, from job description to rendered resume.

This is the orchestration §2 describes, in order: analyze the posting, plan the
repositioning, execute that plan section by section, assemble, validate, render.
Every intermediate is written into the run directory as it is produced, so a run
that fails at any point still leaves behind the reasoning that led there.

Nothing here decides *how* a stage works; each stage owns that. What this owns is
the order, what is handed between stages, and what happens when the finished resume
does not fit on a page.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from resumelab.config import LLMProvider, Settings
from resumelab.experiment import ExperimentRun, build_metadata, create_run
from resumelab.llm.client import LLMClient
from resumelab.loaders import load_candidate_profile
from resumelab.models.job import JobDescription, JobDescriptionSource
from resumelab.models.metadata import RunMetadata
from resumelab.models.resume import GeneratedResume
from resumelab.pipeline.assembler import assemble_resume
from resumelab.pipeline.condenser import condense_resume
from resumelab.pipeline.experience_transformer import transform_experiences
from resumelab.pipeline.jd_analyzer import analyze_job_description
from resumelab.pipeline.project_transformer import transform_projects
from resumelab.pipeline.skills_transformer import transform_skills
from resumelab.pipeline.strategist import build_transformation_strategy
from resumelab.pipeline.summary_generator import generate_summary
from resumelab.rendering import RenderResult, render_resume
from resumelab.validation import validate_resume

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Everything a completed run produced."""

    run: ExperimentRun
    resume: GeneratedResume
    render: RenderResult
    metadata: RunMetadata
    condensed: bool
    """Whether the resume had to be shortened to fit."""


def generate_resume(
    job_description: JobDescription,
    *,
    settings: Settings,
    provider: LLMProvider,
    client: LLMClient,
) -> GenerationResult:
    """Run the whole pipeline and record it.

    Args:
        job_description: The target posting.
        settings: Loaded settings, supplying the profile path and length budget.
        provider: The provider actually in use, for the run's metadata.
        client: The LLM client every stage will use.

    Returns:
        A :class:`GenerationResult` describing the run and its artifacts.

    Raises:
        ResumeLabError: If any stage fails. The run directory keeps whatever was
            produced before the failure.
    """
    run = create_run(settings.runs_dir, label=_run_label(job_description))
    run.record_job_description(job_description)

    profile = load_candidate_profile(settings.candidate_profile_path)
    analysis = analyze_job_description(job_description, client=client)
    run.record_analysis(analysis)

    strategy = build_transformation_strategy(profile, analysis, client=client)
    run.record_strategy(strategy)

    summary = generate_summary(profile, analysis, strategy, client=client)
    experiences = transform_experiences(profile, analysis, strategy, client=client)
    written = [bullet for entry in experiences for bullet in entry.bullets]
    projects = transform_projects(
        profile, analysis, strategy, client=client, already_written=written
    )
    written += [bullet for project in projects for bullet in project.bullets]
    skills = transform_skills(profile, analysis, strategy, client=client, already_written=written)

    resume = assemble_resume(
        profile,
        summary=summary,
        experiences=experiences,
        projects=projects,
        skills=skills,
    )
    resume, rendered, condensed = _render_to_fit(resume, run, client=client, settings=settings)
    run.record_resume(resume)

    metadata = build_metadata(
        run,
        settings=settings,
        provider=provider,
        model=client.model,
        job_description=job_description,
        stats=client.stats,
        layout_scale=rendered.scale,
        page_count=rendered.page_count,
        condensed=condensed,
    )
    run.record_metadata(metadata)

    logger.info("generation completed output=%s", rendered.path)
    return GenerationResult(
        run=run,
        resume=resume,
        render=rendered,
        metadata=metadata,
        condensed=condensed,
    )


def _render_to_fit(
    resume: GeneratedResume,
    run: ExperimentRun,
    *,
    client: LLMClient,
    settings: Settings,
) -> tuple[GeneratedResume, RenderResult, bool]:
    """Render, and shorten the content once if the page still overflows.

    The renderer already tightens the layout as far as it safely can. Beyond that
    the only honest fix is less text, so the content is condensed and rendered
    again — once. A resume that still overflows is written as it is: two readable
    pages beat one unreadable page, and the run's artifacts say what happened.
    """
    rendered = render_resume(resume, run.pdf_path)
    if rendered.fits_on_one_page:
        return resume, rendered, False

    logger.info("condensing to fit one page pages=%d", rendered.page_count)
    shortened = condense_resume(resume, client=client, limits=settings.resume_limits)
    validate_resume(shortened, settings.resume_limits)

    rendered = render_resume(shortened, run.pdf_path)
    if not rendered.fits_on_one_page:
        logger.warning(
            "resume still spans %d pages after condensing; leaving it readable",
            rendered.page_count,
        )
    return shortened, rendered, True


def _run_label(job_description: JobDescription) -> str:
    """Name the run after where the posting came from, so directories are findable."""
    if job_description.source is JobDescriptionSource.FILE and job_description.source_path:
        return job_description.source_path.stem
    return "inline"


def copy_pdf(rendered: RenderResult, destination: Path) -> Path:
    """Also place the rendered PDF at ``destination``.

    The run directory is the record; this is the copy a researcher actually opens.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(rendered.path.read_bytes())
    logger.info("copied resume to %s", destination)
    return destination
