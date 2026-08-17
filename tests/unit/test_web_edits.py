"""Tests for saving a hand-edited resume.

The property that matters most here is what an edit does *not* touch. A run
directory is a research record, and an edit that overwrote what the model wrote
would make every later comparison between profile, posting, and output a lie.
"""

import pytest
from pypdf import PdfReader

from resumelab.exceptions import PDFRenderingError
from resumelab.experiment.recorder import PDF_FILE, RESUME_FILE
from resumelab.models.resume import GeneratedResume
from resumelab.rendering import RenderOptions, ResumeSection, render_resume
from resumelab.web.edits import EDITED_PDF_FILE, EDITED_RESUME_FILE, save_edit


@pytest.fixture
def run(tmp_path, generated_resume):
    """A finished run on disk, as the pipeline would have left it."""
    directory = tmp_path / "2026-03-04T120000_acme"
    directory.mkdir(parents=True)
    (directory / RESUME_FILE).write_text(generated_resume.model_dump_json(), encoding="utf-8")
    render_resume(generated_resume, directory / PDF_FILE)
    return directory


def edited(resume: GeneratedResume, summary: str) -> GeneratedResume:
    return resume.model_copy(update={"summary": summary})


# --- what an edit leaves alone --------------------------------------------


def test_the_generated_resume_is_not_overwritten(run, generated_resume):
    """The run records what the model wrote. An edit is a separate document."""
    original = (run / RESUME_FILE).read_text(encoding="utf-8")

    save_edit(run, edited(generated_resume, "Something else entirely, at some length."))

    assert (run / RESUME_FILE).read_text(encoding="utf-8") == original


def test_the_generated_pdf_is_not_overwritten(run, generated_resume):
    original = (run / PDF_FILE).read_bytes()

    save_edit(run, edited(generated_resume, "Something else entirely, at some length."))

    assert (run / PDF_FILE).read_bytes() == original


# --- what an edit writes --------------------------------------------------


def test_the_edit_is_recorded_beside_the_original(run, generated_resume):
    save_edit(run, edited(generated_resume, "A hand-written summary line."))

    saved = GeneratedResume.model_validate_json(
        (run / EDITED_RESUME_FILE).read_text(encoding="utf-8")
    )
    assert saved.summary == "A hand-written summary line."


def test_the_edit_is_rendered(run, generated_resume):
    save_edit(run, edited(generated_resume, "A hand-written summary line."))

    text = "\n".join(page.extract_text() for page in PdfReader(run / EDITED_PDF_FILE).pages)
    assert "A hand-written summary line." in text


def test_the_outcome_reports_the_page(run, generated_resume):
    outcome = save_edit(run, generated_resume)

    assert outcome.page_count == 1
    assert outcome.fits_on_one_page is True
    assert outcome.scale == 1.0


def test_editing_twice_leaves_only_the_later_edit(run, generated_resume):
    save_edit(run, edited(generated_resume, "The first attempt at a summary line."))
    save_edit(run, edited(generated_resume, "The second attempt at a summary line."))

    text = "\n".join(page.extract_text() for page in PdfReader(run / EDITED_PDF_FILE).pages)
    assert "second attempt" in text
    assert "first attempt" not in text


def test_no_scratch_file_is_left_behind(run, generated_resume):
    """The PDF is built beside its target and moved, so the browser never reads a
    half-written page. The scratch file is not part of the record."""
    save_edit(run, generated_resume)

    assert [path.name for path in run.iterdir() if path.name.startswith(".")] == []


# --- what an edit is allowed to be ----------------------------------------


def test_an_overlong_edit_is_rendered_rather_than_refused(run, generated_resume):
    """The length budget exists to hold a model inside a format. Someone editing
    their own resume is told the page count and left to decide."""
    sprawling = generated_resume.model_copy(
        update={"experiences": generated_resume.experiences * 12}
    )

    outcome = save_edit(run, sprawling)

    assert outcome.page_count > 1
    assert outcome.fits_on_one_page is False
    assert (run / EDITED_PDF_FILE).is_file()


def test_render_options_are_applied_to_the_edit(run, generated_resume):
    save_edit(run, generated_resume, options=RenderOptions(include_summary=False))

    text = "\n".join(page.extract_text() for page in PdfReader(run / EDITED_PDF_FILE).pages)
    assert "SUMMARY" not in text


def test_the_section_order_is_applied_to_the_edit(run, generated_resume):
    order = (
        ResumeSection.SKILLS,
        ResumeSection.PROJECTS,
        ResumeSection.EXPERIENCE,
        ResumeSection.EDUCATION,
    )

    save_edit(run, generated_resume, options=RenderOptions(section_order=order))

    text = "\n".join(page.extract_text() for page in PdfReader(run / EDITED_PDF_FILE).pages)
    assert text.index("SKILLS") < text.index("EDUCATION")


# --- when it cannot be written --------------------------------------------


def test_an_unwritable_directory_is_reported_as_a_rendering_error(run, generated_resume):
    """The renderer reports the write it could not do; this only has to not bury it."""
    run.chmod(0o500)
    try:
        with pytest.raises(PDFRenderingError, match="Could not write the resume"):
            save_edit(run, generated_resume)
    finally:
        run.chmod(0o700)


def test_a_failed_move_into_place_is_reported(run, generated_resume, monkeypatch):
    """The one bare syscall here. Everything else already raises a domain error."""

    def refuse(*_args, **_kwargs):
        raise OSError("cross-device link")

    monkeypatch.setattr("pathlib.Path.replace", refuse)

    with pytest.raises(PDFRenderingError, match="Could not save"):
        save_edit(run, generated_resume)


def test_a_failed_render_leaves_the_previous_edit_in_place(run, generated_resume, monkeypatch):
    """Render first, record second: content that cannot be drawn is not the edit."""
    save_edit(run, edited(generated_resume, "The edit that worked, at some length."))
    kept = (run / EDITED_RESUME_FILE).read_text(encoding="utf-8")

    def refuse(*_args, **_kwargs):
        raise PDFRenderingError("nope")

    monkeypatch.setattr("resumelab.web.edits.render_resume", refuse)

    with pytest.raises(PDFRenderingError):
        save_edit(run, edited(generated_resume, "The edit that did not, at some length."))
    assert (run / EDITED_RESUME_FILE).read_text(encoding="utf-8") == kept


def test_an_unwritable_json_file_is_reported(run, generated_resume, monkeypatch):
    def refuse(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("pathlib.Path.write_text", refuse)

    with pytest.raises(PDFRenderingError, match="Could not save"):
        save_edit(run, generated_resume)
