"""Generation runs, tracked while they are in flight.

A run takes a minute or more and spends real money, which rules out doing it inside
a request. The browser starts one, gets an identifier back immediately, and asks how
it is going until it finishes.

Polling rather than a streamed response, deliberately. A stream ties the run's
observability to one connection: a reloaded tab or a laptop lid closing at the wrong
moment loses all sight of a run that is still burning API budget. Here the run
outlives the request that started it, and anything that can ask for a job id can
pick it back up.

This is in-memory and single-process, which is the right size for a local tool with
one candidate profile and one API budget. Nothing here survives a restart; the run
directory on disk is the durable record, as it has always been.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from uuid import uuid4

from resumelab.exceptions import ResumeLabError
from resumelab.pipeline import GenerationResult, GenerationStage, StageReporter

logger = logging.getLogger(__name__)

DEFAULT_CAPACITY = 20
"""Finished jobs kept before the oldest is forgotten. A session runs a handful."""

UNEXPECTED_FAILURE = "The run failed unexpectedly. Check the server log for the full traceback."
"""What an unexpected failure tells the browser.

Domain errors are written to be read by whoever caused them and are already free of
credentials. Anything else is a bug, and its message is an arbitrary string from an
arbitrary library - it goes to the log, where it is useful, and not to a page.
"""

Work = Callable[[StageReporter], GenerationResult]
"""A run, waiting for somewhere to report its progress to."""


class JobState(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class GenerationJob:
    """One run's progress, as a value that can be handed out safely.

    Frozen because it is read from the request thread while the worker thread is
    still writing: the registry swaps a whole new job in rather than mutating one
    that somebody may be halfway through serializing.
    """

    id: str
    state: JobState
    stage: GenerationStage | None = None
    run_id: str | None = None
    error: str | None = None

    @property
    def is_finished(self) -> bool:
        return self.state is not JobState.RUNNING


@dataclass
class JobRegistry:
    """Every run this process has started, newest last."""

    capacity: int = DEFAULT_CAPACITY
    _jobs: OrderedDict[str, GenerationJob] = field(default_factory=OrderedDict)
    _threads: dict[str, threading.Thread] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def start(self, work: Work) -> GenerationJob:
        """Begin ``work`` on a background thread and return the job watching it."""
        job = GenerationJob(id=uuid4().hex, state=JobState.RUNNING)
        with self._lock:
            self._jobs[job.id] = job
            self._forget_oldest()

        thread = threading.Thread(
            target=self._execute,
            args=(job.id, work),
            name=f"resumelab-job-{job.id}",
            daemon=True,
        )
        with self._lock:
            self._threads[job.id] = thread
        thread.start()
        logger.info("generation job started id=%s", job.id)
        return job

    def get(self, job_id: str) -> GenerationJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def wait(self, job_id: str, *, timeout: float) -> GenerationJob | None:
        """Block until the job finishes, for tests and for an orderly shutdown."""
        with self._lock:
            thread = self._threads.get(job_id)
        if thread is not None:
            thread.join(timeout)
        return self.get(job_id)

    def _execute(self, job_id: str, work: Work) -> None:
        """Run the work, recording where it got to and how it ended."""
        try:
            result = work(lambda stage: self._advance(job_id, stage))
        except ResumeLabError as exc:
            logger.warning("generation job failed id=%s: %s", job_id, exc)
            self._replace(job_id, state=JobState.FAILED, error=str(exc))
        except Exception:
            logger.exception("generation job failed unexpectedly id=%s", job_id)
            self._replace(job_id, state=JobState.FAILED, error=UNEXPECTED_FAILURE)
        else:
            logger.info("generation job completed id=%s run=%s", job_id, result.run.run_id)
            self._replace(job_id, state=JobState.COMPLETED, run_id=result.run.run_id)

    def _advance(self, job_id: str, stage: GenerationStage) -> None:
        self._replace(job_id, stage=stage)

    def _replace(
        self,
        job_id: str,
        *,
        state: JobState | None = None,
        stage: GenerationStage | None = None,
        run_id: str | None = None,
        error: str | None = None,
    ) -> None:
        """Swap in a new job value, keeping whatever this update does not set.

        A job evicted mid-run is simply not updated. Dropping the write is the whole
        point of eviction, and resurrecting the entry would defeat the bound.
        """
        with self._lock:
            current = self._jobs.get(job_id)
            if current is None:
                return
            self._jobs[job_id] = GenerationJob(
                id=current.id,
                state=state if state is not None else current.state,
                stage=stage if stage is not None else current.stage,
                run_id=run_id if run_id is not None else current.run_id,
                error=error if error is not None else current.error,
            )

    def _forget_oldest(self) -> None:
        """Bound the registry, never discarding a run that is still going.

        Called with the lock held.
        """
        while len(self._jobs) > self.capacity:
            for job_id, job in self._jobs.items():
                if job.is_finished:
                    del self._jobs[job_id]
                    self._threads.pop(job_id, None)
                    break
            else:
                # Every job is still running, which at this depth means something is
                # wrong. Let the registry grow rather than lose sight of live work.
                logger.warning("job registry above capacity with none finished")
                return
