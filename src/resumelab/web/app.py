"""The HTTP layer behind the local review UI.

This exists so a generated resume can be looked at and adjusted without leaving the
machine that produced it. It is a view onto the same pipeline the CLI drives, not a
second way of doing the work: every route here delegates, and nothing in this package
decides what a stage does.

It is a **local** tool. There is no authentication, no multi-user state, and no
notion of a session, because there is exactly one candidate profile and one API
budget, both belonging to whoever started the server. That is also why the CLI binds
to the loopback interface by default.

Run identifiers arrive from the browser and are used to build filesystem paths, so
every one of them is resolved and checked against the configured runs directory
before anything is opened. A run that does not resolve inside it does not exist.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, model_validator

from resumelab import __version__
from resumelab.config import LLMProvider, Settings, load_settings
from resumelab.exceptions import ResumeLabError, UnsafePathError
from resumelab.experiment.recorder import PDF_FILE, RESUME_FILE
from resumelab.llm.factory import create_llm_client
from resumelab.loaders import load_job_description
from resumelab.models.resume import GeneratedResume
from resumelab.pipeline import GenerationResult, StageReporter, generate_resume
from resumelab.utils.paths import ensure_within
from resumelab.web.jobs import GenerationJob, JobRegistry

logger = logging.getLogger(__name__)

STATIC_DIRECTORY = Path(__file__).parent / "static"
"""The front end, served as files. There is no build step and no bundler."""

INDEX_FILE = STATIC_DIRECTORY / "index.html"


class Health(BaseModel):
    """Enough for the page to tell a running server from a stale tab."""

    status: str
    version: str


class GenerateRequest(BaseModel):
    """A posting to generate against, as a link or as the text itself."""

    model_config = ConfigDict(extra="forbid")

    url: str | None = None
    text: str | None = None
    provider: LLMProvider | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> GenerateRequest:
        """One posting per run. Two sources is a mistake worth reporting up front."""
        if (self.url is None) == (self.text is None):
            raise ValueError("supply exactly one of url and text")
        return self


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application.

    A factory rather than a module-level singleton so tests can hold an app of their
    own, and so importing this module never starts anything.

    Args:
        settings: Configuration to serve. Loaded from the environment when omitted,
            which is what the CLI does; tests pass their own.
    """
    config = settings if settings is not None else load_settings()
    registry = JobRegistry()

    app = FastAPI(
        title="ResumeLab",
        summary="Local review UI for JD-conditioned resume generation.",
        version=__version__,
    )
    app.state.settings = config
    app.state.jobs = registry

    @app.get("/api/health")
    def health() -> Health:
        return Health(status="ok", version=__version__)

    @app.post("/api/generate", status_code=202)
    def generate(request: GenerateRequest) -> GenerationJob:
        """Start a run and return the job that tracks it.

        The client is built here rather than in the worker, so a missing API key is
        an immediate error instead of a job that fails a moment later. Fetching the
        posting is not: it reaches the network, and a slow board should show up as a
        run in progress rather than a request that hangs.
        """
        try:
            client = create_llm_client(config, provider=request.provider)
        except ResumeLabError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        provider = request.provider if request.provider is not None else config.resolved_provider

        def work(on_stage: StageReporter) -> GenerationResult:
            posting = load_job_description(text=request.text, url=request.url)
            return generate_resume(
                posting,
                settings=config,
                provider=provider,
                client=client,
                on_stage=on_stage,
            )

        return registry.start(work)

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str) -> GenerationJob:
        found = registry.get(job_id)
        if found is None:
            raise HTTPException(status_code=404, detail=f"No such job: {job_id}")
        return found

    @app.get("/api/runs/{run_id}/resume")
    def run_resume(run_id: str) -> GeneratedResume:
        """The generated content, as the editor will load and send it back."""
        path = _artifact(config, run_id, RESUME_FILE)
        try:
            return GeneratedResume.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("unreadable resume run=%s: %s", run_id, exc)
            raise HTTPException(
                status_code=500, detail=f"The resume for run {run_id} could not be read."
            ) from exc

    @app.get("/api/runs/{run_id}/resume.pdf")
    def run_pdf(run_id: str) -> FileResponse:
        return FileResponse(
            _artifact(config, run_id, PDF_FILE),
            media_type="application/pdf",
            # Shown in the page rather than downloaded; the browser's own viewer is
            # a better preview than anything worth building here.
            content_disposition_type="inline",
            filename=f"{run_id}.pdf",
        )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(INDEX_FILE)

    # Mounted after the routes above, so a static file can never shadow the API.
    app.mount("/static", StaticFiles(directory=STATIC_DIRECTORY), name="static")

    logger.debug("web application created runs=%s", config.runs_dir)
    return app


def _artifact(settings: Settings, run_id: str, name: str) -> Path:
    """Resolve one file inside a run directory, or refuse to say it exists.

    ``run_id`` comes from the browser. A traversal attempt and a genuine typo get the
    same answer, so the endpoint cannot be used to probe the filesystem.
    """
    try:
        directory = ensure_within(
            settings.runs_dir / run_id, settings.runs_dir, subject="The run directory"
        )
    except UnsafePathError as exc:
        logger.warning("rejected run id=%r: %s", run_id, exc)
        raise HTTPException(status_code=404, detail=f"No such run: {run_id}") from exc

    path = directory / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"No such run: {run_id}")
    return path
