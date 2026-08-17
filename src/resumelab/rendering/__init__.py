"""PDF rendering of the generated resume."""

from resumelab.rendering.options import (
    DEFAULT_RENDER_OPTIONS,
    DEFAULT_SECTION_ORDER,
    RenderOptions,
    ResumeSection,
)
from resumelab.rendering.pdf_renderer import RenderResult, render_resume

__all__ = [
    "DEFAULT_RENDER_OPTIONS",
    "DEFAULT_SECTION_ORDER",
    "RenderOptions",
    "RenderResult",
    "ResumeSection",
    "render_resume",
]
