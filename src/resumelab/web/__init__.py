"""The local review UI: an HTTP view onto the generation pipeline."""

from resumelab.web.app import Health, create_app

__all__ = ["Health", "create_app"]
