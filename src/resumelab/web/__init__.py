"""The local review UI: an HTTP view onto the generation pipeline."""

from resumelab.web.app import EditRequest, GenerateRequest, Health, create_app
from resumelab.web.edits import EditOutcome, save_edit
from resumelab.web.jobs import GenerationJob, JobRegistry, JobState

__all__ = [
    "EditOutcome",
    "EditRequest",
    "GenerateRequest",
    "GenerationJob",
    "Health",
    "JobRegistry",
    "JobState",
    "create_app",
    "save_edit",
]
