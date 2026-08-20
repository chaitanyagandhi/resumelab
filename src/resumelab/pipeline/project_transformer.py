"""Stage 6 — reposition each project around the target identity.

This is where the transformation under study goes furthest. A role's dates are facts;
what a project *was about* is a framing decision, so the subtitle, the stack, and the
bullets are all regenerated. Only the project name is carried over, and only because
it is what lets a researcher line each generated project up against its source.

Projects are repositioned one at a time. Bullets written earlier in the run — from
the experience section as well as from earlier projects — are passed along so the
sections do not repeat each other's verbs and claims.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from resumelab.exceptions import LLMGenerationError
from resumelab.llm.client import LLMClient
from resumelab.llm.prompts import PROJECT_PROMPT
from resumelab.models.analysis import JobAnalysis
from resumelab.models.candidate import CandidateProfile, Project
from resumelab.models.resume import (
    GeneratedProject,
    ProjectContent,
    TolerantProjectContent,
)
from resumelab.models.strategy import ProjectDirection, TransformationStrategy
from resumelab.pipeline.context import (
    already_written_section,
    analysis_section,
    direction_section,
    source_project_section,
    strategy_section,
)

logger = logging.getLogger(__name__)


def transform_projects(
    profile: CandidateProfile,
    analysis: JobAnalysis,
    strategy: TransformationStrategy,
    *,
    client: LLMClient,
    already_written: Sequence[str] = (),
) -> tuple[GeneratedProject, ...]:
    """Reposition every project in ``profile`` according to ``strategy``.

    Args:
        profile: The immutable source profile.
        analysis: The structured reading of the target posting.
        strategy: The global plan, which must carry a direction for every project.
        client: The LLM client to use, injected by the caller.
        already_written: Bullets written earlier in the run, so this stage does not
            repeat them. The caller passes the experience bullets in here.

    Returns:
        One :class:`GeneratedProject` per source project, in profile order.

    Raises:
        LLMGenerationError: If generation fails, or the plan omits a project.
    """
    transformed: list[GeneratedProject] = []
    written = list(already_written)

    for project in profile.projects:
        direction = strategy.direction_for_project(project.name)
        if direction is None:
            raise LLMGenerationError(
                f"The transformation strategy has no direction for project {project.name!r}."
            )

        logger.info("transforming project project=%s", project.name)
        content = _reposition(
            project,
            direction,
            analysis,
            strategy,
            written=written,
            client=client,
        )
        transformed.append(_assemble(project, content))
        written.extend(content.bullets)
        logger.debug(
            "repositioned project=%s subtitle=%s technologies=%s",
            project.name,
            content.subtitle,
            ", ".join(content.technologies),
        )

    logger.info("transformed projects count=%d", len(transformed))
    return tuple(transformed)


def _reposition(
    project: Project,
    direction: ProjectDirection,
    analysis: JobAnalysis,
    strategy: TransformationStrategy,
    *,
    written: list[str],
    client: LLMClient,
) -> ProjectContent:
    """Generate the new presentation of one project."""
    sections = [
        source_project_section(project),
        direction_section(direction),
        analysis_section(analysis),
        strategy_section(strategy),
    ]
    if written:
        sections.append(already_written_section(written))

    return client.generate_structured(
        system_prompt=PROJECT_PROMPT.system,
        user_prompt=PROJECT_PROMPT.user(*sections),
        response_model=ProjectContent,
        purpose=PROJECT_PROMPT.name,
        fallback_model=TolerantProjectContent,
    )


def _assemble(project: Project, content: ProjectContent) -> GeneratedProject:
    """Combine the generated presentation with the project's anchoring name and date."""
    return GeneratedProject(
        name=project.name,
        subtitle=content.subtitle,
        date=project.date,
        technologies=content.technologies,
        bullets=content.bullets,
    )
