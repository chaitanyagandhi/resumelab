"""Per-run artifact directories.

Every generation writes a self-contained directory holding the exact inputs, every
intermediate structure, the final resume, and the metadata describing how it was
produced. A run that cannot be inspected afterwards is not a research result, so the
intermediates are written as they are produced rather than at the end — a run that
fails at the renderer still leaves behind the analysis and strategy that led there.

Run directories are named from a UTC timestamp and a caller-supplied label. The label
is slugified, so a label taken from a filename or a job title cannot escape the
configured output directory.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from resumelab.models.analysis import JobAnalysis
from resumelab.models.job import JobDescription
from resumelab.models.metadata import RunMetadata
from resumelab.models.resume import GeneratedResume
from resumelab.models.strategy import TransformationStrategy
from resumelab.utils.text import slugify

logger = logging.getLogger(__name__)

JOB_DESCRIPTION_FILE = "jd.txt"
ANALYSIS_FILE = "jd_analysis.json"
STRATEGY_FILE = "transformation_strategy.json"
RESUME_FILE = "generated_resume.json"
METADATA_FILE = "metadata.json"
PDF_FILE = "resume.pdf"

RUN_TIMESTAMP_FORMAT = "%Y-%m-%dT%H%M%S"
"""Sorts chronologically as text, which is how run directories get read."""

MAX_COLLISION_ATTEMPTS = 100


class ExperimentRun:
    """One run's directory, and the writes that populate it."""

    def __init__(self, directory: Path, run_id: str, started_at: datetime) -> None:
        self.directory = directory
        self.run_id = run_id
        self.started_at = started_at

    @property
    def pdf_path(self) -> Path:
        """Where the renderer should write the resume."""
        return self.directory / PDF_FILE

    def record_job_description(self, job_description: JobDescription) -> Path:
        """Write the exact text that was analyzed, not the file it came from."""
        return self._write_text(JOB_DESCRIPTION_FILE, job_description.text)

    def record_analysis(self, analysis: JobAnalysis) -> Path:
        return self._write_json(ANALYSIS_FILE, analysis)

    def record_strategy(self, strategy: TransformationStrategy) -> Path:
        return self._write_json(STRATEGY_FILE, strategy)

    def record_resume(self, resume: GeneratedResume) -> Path:
        return self._write_json(RESUME_FILE, resume)

    def record_metadata(self, metadata: RunMetadata) -> Path:
        return self._write_json(METADATA_FILE, metadata)

    def elapsed_seconds(self, *, now: datetime | None = None) -> float:
        """Wall-clock duration so far, for the metadata record."""
        return ((now or datetime.now(UTC)) - self.started_at).total_seconds()

    def _write_json(self, name: str, model: BaseModel) -> Path:
        return self._write_text(name, model.model_dump_json(indent=2))

    def _write_text(self, name: str, content: str) -> Path:
        path = self.directory / name
        path.write_text(content if content.endswith("\n") else f"{content}\n", encoding="utf-8")
        logger.debug("recorded artifact=%s bytes=%d", name, len(content))
        return path


def create_run(
    runs_dir: Path,
    *,
    label: str,
    now: datetime | None = None,
) -> ExperimentRun:
    """Create a fresh run directory under ``runs_dir``.

    Args:
        runs_dir: Root holding one sub-directory per run.
        label: Human-meaningful name for the run, such as the job description's
            filename or the target company. Slugified before use.
        now: Start time, injected by tests. Defaults to the current UTC time.

    Returns:
        The :class:`ExperimentRun` for the created directory.

    Raises:
        OSError: If the directory cannot be created.
    """
    started_at = now or datetime.now(UTC)
    stamp = started_at.strftime(RUN_TIMESTAMP_FORMAT)
    base = f"{stamp}_{slugify(label)}"

    directory = _make_unique_directory(runs_dir, base)
    logger.info("recording run id=%s directory=%s", directory.name, directory)
    return ExperimentRun(directory=directory, run_id=directory.name, started_at=started_at)


def _make_unique_directory(runs_dir: Path, base: str) -> Path:
    """Create ``base`` under ``runs_dir``, suffixing if that name is taken.

    Two runs started in the same second must not write into each other's directory,
    which would interleave their artifacts and silently corrupt both.
    """
    runs_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(MAX_COLLISION_ATTEMPTS):
        name = base if attempt == 0 else f"{base}-{attempt + 1}"
        directory = runs_dir / name
        try:
            directory.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return directory
    raise OSError(f"Could not create a unique run directory for {base!r} in {runs_dir}")
