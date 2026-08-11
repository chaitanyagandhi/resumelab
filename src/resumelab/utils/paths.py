"""Validation of the paths this tool writes to.

Two different risks. A path built from input the researcher did not write — a job
description's filename, a company name lifted from a posting — must not be able to
escape the directory it was meant to land in. And a path the researcher did type
should fail with a sentence rather than a traceback when it names a directory, an
unwritable location, or something the filesystem will not accept.
"""

from __future__ import annotations

from pathlib import Path

from resumelab.exceptions import UnsafePathError


def ensure_within(candidate: Path, root: Path, *, subject: str) -> Path:
    """Return ``candidate`` resolved, having checked it stays under ``root``.

    Resolution happens first, so ``..`` segments and symlinks are followed before the
    comparison rather than after it.

    Args:
        candidate: The path to check.
        root: The directory it must remain inside.
        subject: Human-readable name for the path, used in the error message.

    Raises:
        UnsafePathError: If ``candidate`` resolves outside ``root``.
    """
    resolved = _resolved(candidate, subject=subject)
    base = _resolved(root, subject=subject)
    if not resolved.is_relative_to(base):
        raise UnsafePathError(f"{subject} would be written outside {base}: {resolved}")
    return resolved


def prepare_output_file(path: Path, *, subject: str) -> Path:
    """Check ``path`` can be written to as a file, and create its parent directory.

    Args:
        path: Where the caller intends to write.
        subject: Human-readable name for the path, used in error messages.

    Returns:
        The resolved path.

    Raises:
        UnsafePathError: If the path names a directory, is not a usable path, or its
            parent cannot be created.
    """
    resolved = _resolved(path, subject=subject)
    # A path with no filename ("/", ".") is a directory, so the check above has
    # already rejected it.
    if resolved.is_dir():
        raise UnsafePathError(f"{subject} is a directory, not a file: {resolved}")

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise UnsafePathError(f"Could not create the directory for {subject}: {exc}") from exc
    return resolved


def _resolved(path: Path, *, subject: str) -> Path:
    """Resolve a path, turning what the filesystem rejects into a domain error."""
    try:
        return path.expanduser().resolve()
    except (OSError, ValueError, RuntimeError) as exc:
        raise UnsafePathError(f"{subject} is not a usable path: {exc}") from exc
