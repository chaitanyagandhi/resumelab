"""Loading of the source candidate profile from YAML.

The profile is read, parsed, and validated exactly once per run. Every failure mode
produces a :class:`~resumelab.exceptions.CandidateProfileError` naming the file and
the problem, so a researcher never has to read a traceback to learn that a bullet is
missing.

The file is opened read-only and is never written back: it is the experimental
control for the run.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from resumelab.exceptions import CandidateProfileError
from resumelab.models.candidate import CandidateProfile
from resumelab.utils.errors import describe_validation_error

logger = logging.getLogger(__name__)

EXAMPLE_PROFILE_PATH = Path("data/candidate_profile.example.yaml")
"""Tracked template a researcher copies to create their own profile."""


def load_candidate_profile(path: Path) -> CandidateProfile:
    """Read and validate the candidate profile at ``path``.

    Args:
        path: Location of the profile YAML file.

    Returns:
        The validated, immutable :class:`CandidateProfile`.

    Raises:
        CandidateProfileError: If the file is missing, unreadable, not UTF-8, not
            valid YAML, not a mapping, or does not satisfy the profile schema.
    """
    logger.info("loading candidate profile path=%s", path)

    document = _parse_yaml(_read_text(path), path)
    try:
        profile = CandidateProfile.model_validate(document)
    except ValidationError as exc:
        message = describe_validation_error(exc, f"Invalid candidate profile: {path}")
        raise CandidateProfileError(message) from exc

    logger.debug(
        "loaded candidate profile experiences=%d projects=%d education=%d",
        len(profile.experiences),
        len(profile.projects),
        len(profile.education),
    )
    return profile


def _read_text(path: Path) -> str:
    """Read the profile as UTF-8, translating filesystem failures into domain errors."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise CandidateProfileError(
            f"Candidate profile not found: {path}\n"
            f"  Create it with: cp {EXAMPLE_PROFILE_PATH} {path}"
        ) from exc
    except IsADirectoryError as exc:
        raise CandidateProfileError(
            f"Candidate profile path is a directory, not a file: {path}"
        ) from exc
    except PermissionError as exc:
        raise CandidateProfileError(f"Candidate profile is not readable: {path}") from exc
    except UnicodeDecodeError as exc:
        raise CandidateProfileError(
            f"Candidate profile must be UTF-8 encoded: {path}\n  {exc.reason}"
        ) from exc
    except OSError as exc:
        raise CandidateProfileError(f"Could not read candidate profile {path}: {exc}") from exc


def _parse_yaml(text: str, path: Path) -> dict[Any, Any]:
    """Parse the profile document.

    Uses ``yaml.safe_load``: the profile is data, and must never be able to construct
    arbitrary Python objects through YAML tags.
    """
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise CandidateProfileError(
            f"Candidate profile is not valid YAML: {path}\n  {exc}"
        ) from exc

    if document is None:
        raise CandidateProfileError(f"Candidate profile is empty: {path}")
    if not isinstance(document, dict):
        raise CandidateProfileError(
            f"Candidate profile must be a YAML mapping of sections, "
            f"but the file contains {type(document).__name__}: {path}"
        )
    return document
