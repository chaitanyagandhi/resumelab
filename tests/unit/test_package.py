"""Smoke tests for package metadata and the public exception hierarchy."""

from importlib import metadata

import pytest

import resumelab
from resumelab.exceptions import (
    CandidateProfileError,
    JDAnalysisError,
    LLMGenerationError,
    PDFRenderingError,
    ResumeLabError,
    ResumeValidationError,
)

DOMAIN_ERRORS = [
    CandidateProfileError,
    JDAnalysisError,
    LLMGenerationError,
    PDFRenderingError,
    ResumeValidationError,
]


def test_version_is_exposed():
    assert resumelab.__version__ == "0.1.0"


def test_declared_version_matches_installed_distribution():
    """The single source of truth is __init__.py; hatchling reads the version from it."""
    assert metadata.version("resumelab") == resumelab.__version__


@pytest.mark.parametrize("error_type", DOMAIN_ERRORS)
def test_domain_errors_derive_from_base_error(error_type):
    assert issubclass(error_type, ResumeLabError)


@pytest.mark.parametrize("error_type", DOMAIN_ERRORS)
def test_domain_errors_are_catchable_as_base_error(error_type):
    with pytest.raises(ResumeLabError):
        raise error_type("failure detail")


def test_base_error_derives_from_exception():
    assert issubclass(ResumeLabError, Exception)


@pytest.mark.parametrize("error_type", [ResumeLabError, *DOMAIN_ERRORS])
def test_errors_are_exported_from_the_package_root(error_type):
    assert error_type.__name__ in resumelab.__all__
    assert getattr(resumelab, error_type.__name__) is error_type
