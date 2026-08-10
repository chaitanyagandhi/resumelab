"""Rendering of pydantic validation failures into readable, safe messages.

Pydantic's own string form embeds the offending input value. That is unacceptable
here: the same machinery validates API credentials and personal contact details, and
these messages end up in logs and CLI output. Only the location and the reason are
reported.
"""

from __future__ import annotations

from pydantic import ValidationError


def describe_validation_error(
    exc: ValidationError,
    header: str,
    *,
    uppercase_locations: bool = False,
) -> str:
    """Summarize every error in ``exc`` as an indented ``location: reason`` line.

    Args:
        exc: The validation failure to describe.
        header: First line of the message, naming what failed to validate.
        uppercase_locations: Render locations in upper case, for settings whose
            locations are field names that map to environment variables.

    Returns:
        A multi-line message that never contains the rejected values.
    """
    lines = []
    for error in exc.errors():
        location = _format_location(error["loc"])
        lines.append(f"  {location.upper() if uppercase_locations else location}: {error['msg']}")
    return "\n".join([header, *lines])


def _format_location(loc: tuple[int | str, ...]) -> str:
    """Render an error location as a dotted path, e.g. ``projects.0.bullets``."""
    return ".".join(str(part) for part in loc) if loc else "<root>"
