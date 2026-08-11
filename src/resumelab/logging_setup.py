"""Logging configuration for the command line.

Logs go to stderr so that stdout carries only the command's output, which keeps the
tool usable in a pipeline. The format is deliberately plain — a level and a message —
because the pipeline's log lines are already written as structured ``key=value``
text that reads well without decoration.
"""

from __future__ import annotations

import logging
import sys

LOG_FORMAT = "%(levelname)-8s %(message)s"
DEBUG_LOG_FORMAT = "%(levelname)-8s %(name)s %(message)s"
"""Debug adds the logger name, which is what tells you which stage spoke."""


def configure_logging(level: str, *, debug: bool = False) -> None:
    """Send ResumeLab's logs to stderr at ``level``.

    Args:
        level: Standard library level name, such as ``"INFO"``.
        debug: Force ``DEBUG`` and include logger names, overriding ``level``.
    """
    effective = "DEBUG" if debug else level
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(DEBUG_LOG_FORMAT if debug else LOG_FORMAT))

    root = logging.getLogger("resumelab")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(effective)
    root.propagate = False
