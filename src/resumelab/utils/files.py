"""Reading input files with domain-appropriate errors.

Every ResumeLab input is UTF-8 text read from a path a researcher typed. The
filesystem failure modes are identical across inputs, so they are handled once here
and re-raised as whichever domain error the caller owns.
"""

from __future__ import annotations

from pathlib import Path

from resumelab.exceptions import ResumeLabError


def read_text_file(
    path: Path,
    *,
    subject: str,
    error_type: type[ResumeLabError],
    missing_hint: str | None = None,
) -> str:
    """Read ``path`` as UTF-8 text.

    Args:
        path: File to read.
        subject: Human-readable name of the input, used to open every message,
            e.g. ``"Candidate profile"``.
        error_type: Domain exception raised for any failure.
        missing_hint: Optional remediation line appended when the file is absent.

    Returns:
        The decoded file contents.

    Raises:
        error_type: If the file is missing, a directory, unreadable, not valid
            UTF-8, or fails to read for any other reason.
    """
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        message = f"{subject} not found: {path}"
        if missing_hint is not None:
            message = f"{message}\n  {missing_hint}"
        raise error_type(message) from exc
    except IsADirectoryError as exc:
        raise error_type(f"{subject} path is a directory, not a file: {path}") from exc
    except PermissionError as exc:
        raise error_type(f"{subject} is not readable: {path}") from exc
    except UnicodeDecodeError as exc:
        raise error_type(f"{subject} must be UTF-8 encoded: {path}\n  {exc.reason}") from exc
    except OSError as exc:
        raise error_type(f"Could not read {subject.lower()}: {path}\n  {exc}") from exc
