"""The HTTP layer behind the local review UI.

This exists so a generated resume can be looked at and adjusted without leaving the
machine that produced it. It is a view onto the same pipeline the CLI drives, not a
second way of doing the work: every route here delegates, and nothing in this package
decides what a stage does.

It is a **local** tool. There is no authentication, no multi-user state, and no
notion of a session, because there is exactly one candidate profile and one API
budget, both belonging to whoever started the server. That is also why the CLI binds
to the loopback interface by default.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from resumelab import __version__

logger = logging.getLogger(__name__)

STATIC_DIRECTORY = Path(__file__).parent / "static"
"""The front end, served as files. There is no build step and no bundler."""

INDEX_FILE = STATIC_DIRECTORY / "index.html"


class Health(BaseModel):
    """Enough for the page to tell a running server from a stale tab."""

    status: str
    version: str


def create_app() -> FastAPI:
    """Build the application.

    A factory rather than a module-level singleton so tests can hold an app of their
    own, and so importing this module never starts anything.
    """
    app = FastAPI(
        title="ResumeLab",
        summary="Local review UI for JD-conditioned resume generation.",
        version=__version__,
    )

    @app.get("/api/health")
    def health() -> Health:
        return Health(status="ok", version=__version__)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(INDEX_FILE)

    # Mounted after the routes above, so a static file can never shadow the API.
    app.mount("/static", StaticFiles(directory=STATIC_DIRECTORY), name="static")

    logger.debug("web application created static=%s", STATIC_DIRECTORY)
    return app
