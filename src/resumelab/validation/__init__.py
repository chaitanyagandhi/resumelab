"""Deterministic checks over the finished resume.

Advisory, not a gate. Everything here reports; nothing here stops a resume being
drawn. A rendered resume with a long bullet can be read and edited, and the editor
exists for exactly that; a refused one is a message about work already paid for.
"""

from resumelab.validation.resume_validator import inspect_resume

__all__ = ["inspect_resume"]
