"""Domain-specific exceptions for ResumeLab.

Every error raised by ResumeLab derives from :class:`ResumeLabError`, which lets the
CLI present a readable message for expected failures while still allowing unexpected
exceptions to surface with a full traceback in debug mode.

Exception messages must never contain API keys or other secrets.
"""


class ResumeLabError(Exception):
    """Base class for all ResumeLab errors."""


class CandidateProfileError(ResumeLabError):
    """The candidate profile is missing, unreadable, or fails schema validation."""


class JDAnalysisError(ResumeLabError):
    """The job description could not be loaded or analyzed into a structured form."""


class LLMGenerationError(ResumeLabError):
    """An LLM call failed, or its response could not be validated after retries."""


class ResumeValidationError(ResumeLabError):
    """A generated resume failed deterministic validation before rendering."""


class PDFRenderingError(ResumeLabError):
    """The generated resume could not be rendered to a PDF document."""
