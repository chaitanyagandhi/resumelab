"""Tests for the in-flight job registry.

No LLM and no pipeline here: the registry takes a callable, and these tests supply
one that does exactly what the test needs. What is under test is the bookkeeping
around a run, not the run.
"""

import threading

import pytest

from resumelab.exceptions import JDAnalysisError
from resumelab.pipeline import GenerationStage
from resumelab.web.jobs import UNEXPECTED_FAILURE, GenerationJob, JobRegistry, JobState

TIMEOUT = 5.0
"""Generous: these jobs do nothing, and a hang should fail rather than wedge CI."""


@pytest.fixture
def registry():
    return JobRegistry()


class FakeResult:
    """Stands in for a GenerationResult, which only its run id is read from."""

    def __init__(self, run_id="2026-01-01T000000_run"):
        self.run = type("Run", (), {"run_id": run_id})()


def finishes(run_id="2026-01-01T000000_run"):
    """Work that succeeds without reporting a stage."""
    return lambda _on_stage: FakeResult(run_id)


# --- starting -------------------------------------------------------------


def test_a_started_job_is_running_immediately(registry):
    """The point of the registry: the caller is not kept waiting for the run."""
    job = registry.start(lambda _on_stage: FakeResult())

    assert job.state is JobState.RUNNING
    assert job.id


def test_each_job_gets_its_own_identifier(registry):
    first = registry.start(finishes())
    second = registry.start(finishes())

    assert first.id != second.id


def test_a_started_job_can_be_looked_up(registry):
    job = registry.start(finishes())

    assert registry.get(job.id) is not None


def test_an_unknown_job_is_not_found(registry):
    assert registry.get("no-such-job") is None


# --- finishing ------------------------------------------------------------


def test_a_completed_job_carries_the_run_it_produced(registry):
    job = registry.start(finishes("2026-03-04T120000_acme"))

    finished = registry.wait(job.id, timeout=TIMEOUT)

    assert finished.state is JobState.COMPLETED
    assert finished.run_id == "2026-03-04T120000_acme"
    assert finished.error is None


def test_progress_is_recorded_as_the_run_moves_through_stages(registry):
    """What the browser polls for: a minute of waiting with something to show."""
    released = threading.Event()

    def work(on_stage):
        on_stage(GenerationStage.ANALYSIS)
        on_stage(GenerationStage.SKILLS)
        released.wait(TIMEOUT)
        return FakeResult()

    job = registry.start(work)
    _wait_for(lambda: registry.get(job.id).stage is GenerationStage.SKILLS)

    assert registry.get(job.id).state is JobState.RUNNING
    released.set()
    registry.wait(job.id, timeout=TIMEOUT)


def test_the_last_stage_survives_completion(registry):
    """A finished job still says where it got to, which the UI shows as done."""

    def work(on_stage):
        on_stage(GenerationStage.RENDERING)
        return FakeResult()

    job = registry.start(work)

    assert registry.wait(job.id, timeout=TIMEOUT).stage is GenerationStage.RENDERING


def test_waiting_on_an_unknown_job_returns_nothing(registry):
    assert registry.wait("no-such-job", timeout=TIMEOUT) is None


# --- failing --------------------------------------------------------------


def test_a_domain_failure_is_reported_to_the_browser(registry):
    """These messages are written for whoever caused them and carry no credentials."""

    def work(_on_stage):
        raise JDAnalysisError("That posting could not be fetched.")

    job = registry.start(work)
    finished = registry.wait(job.id, timeout=TIMEOUT)

    assert finished.state is JobState.FAILED
    assert finished.error == "That posting could not be fetched."


def test_an_unexpected_failure_is_not_repeated_to_the_browser(registry):
    """An arbitrary library's exception text is a log entry, not a page message."""

    def work(_on_stage):
        raise RuntimeError("connection to sk-secret-key-material failed")

    job = registry.start(work)
    finished = registry.wait(job.id, timeout=TIMEOUT)

    assert finished.state is JobState.FAILED
    assert finished.error == UNEXPECTED_FAILURE
    assert "sk-secret" not in finished.error


def test_an_unexpected_failure_is_logged_in_full(registry, caplog):
    def work(_on_stage):
        raise RuntimeError("the real reason")

    with caplog.at_level("ERROR", logger="resumelab.web.jobs"):
        job = registry.start(work)
        registry.wait(job.id, timeout=TIMEOUT)

    assert "the real reason" in caplog.text


def test_a_failed_job_names_no_run(registry):
    def work(_on_stage):
        raise JDAnalysisError("nope")

    job = registry.start(work)

    assert registry.wait(job.id, timeout=TIMEOUT).run_id is None


# --- staying bounded ------------------------------------------------------


def test_finished_jobs_are_forgotten_once_the_registry_is_full():
    registry = JobRegistry(capacity=2)

    jobs = [registry.start(finishes()) for _ in range(4)]
    for job in jobs:
        registry.wait(job.id, timeout=TIMEOUT)

    assert registry.get(jobs[0].id) is None
    assert registry.get(jobs[-1].id) is not None


def test_a_running_job_is_never_forgotten_to_make_room():
    """Losing sight of a run that is still spending API budget is the worst outcome."""
    registry = JobRegistry(capacity=1)
    released = threading.Event()

    def blocks(_on_stage):
        released.wait(TIMEOUT)
        return FakeResult()

    running = registry.start(blocks)
    for _ in range(3):
        registry.wait(registry.start(finishes()).id, timeout=TIMEOUT)

    assert registry.get(running.id).state is JobState.RUNNING
    released.set()
    registry.wait(running.id, timeout=TIMEOUT)


def test_a_registry_of_only_running_jobs_grows_rather_than_losing_one(caplog):
    registry = JobRegistry(capacity=1)
    released = threading.Event()

    def blocks(_on_stage):
        released.wait(TIMEOUT)
        return FakeResult()

    with caplog.at_level("WARNING", logger="resumelab.web.jobs"):
        first = registry.start(blocks)
        second = registry.start(blocks)

    assert registry.get(first.id) is not None
    assert registry.get(second.id) is not None
    assert "above capacity" in caplog.text
    released.set()
    for job in (first, second):
        registry.wait(job.id, timeout=TIMEOUT)


def test_a_write_for_a_forgotten_job_is_dropped(registry):
    """Defence in depth, reached directly because nothing else can reach it.

    Eviction only ever takes a finished job, and a finished job's thread has already
    made its last write, so a late update cannot currently happen. The guard is here
    because the alternative is a KeyError raised on a background thread nobody is
    watching, and resurrecting the entry would defeat the bound it was evicted for.
    """
    registry._replace("never-existed", state=JobState.COMPLETED)

    assert registry.get("never-existed") is None


def test_an_evicted_job_does_not_come_back_when_its_work_finishes():
    """The eviction bound has to hold against a late write from a slow thread."""
    registry = JobRegistry(capacity=1)
    evicted = registry.start(finishes())
    registry.wait(evicted.id, timeout=TIMEOUT)

    for _ in range(2):
        registry.wait(registry.start(finishes()).id, timeout=TIMEOUT)

    assert registry.get(evicted.id) is None


# --- the job value --------------------------------------------------------


def test_a_running_job_is_not_finished():
    assert GenerationJob(id="x", state=JobState.RUNNING).is_finished is False


@pytest.mark.parametrize("state", [JobState.COMPLETED, JobState.FAILED])
def test_a_settled_job_is_finished(state):
    assert GenerationJob(id="x", state=state).is_finished is True


def _wait_for(condition, *, timeout=TIMEOUT):
    """Spin until ``condition`` holds, so a thread race fails loudly and quickly."""
    deadline = threading.Event()
    for _ in range(int(timeout * 200)):
        if condition():
            return
        deadline.wait(0.005)
    raise AssertionError("condition was never met")
