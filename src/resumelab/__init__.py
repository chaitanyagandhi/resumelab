"""ResumeLab — a research prototype for studying JD-conditioned resume transformation.

ResumeLab reproduces the behaviour of aggressive commercial AI resume tailoring
systems: it infers the technical identity implied by a target job description and
transforms a fixed candidate profile toward that identity. The source candidate
profile is always treated as immutable so that transformations remain measurable.
"""

from resumelab.exceptions import (
    CandidateProfileError,
    ConfigurationError,
    JDAnalysisError,
    LLMGenerationError,
    PDFRenderingError,
    ResumeLabError,
    ResumeValidationError,
    UnsafePathError,
)

__version__ = "0.1.0"

__all__ = [
    "CandidateProfileError",
    "ConfigurationError",
    "JDAnalysisError",
    "LLMGenerationError",
    "PDFRenderingError",
    "ResumeLabError",
    "ResumeValidationError",
    "UnsafePathError",
    "__version__",
]
