"""Tests for PDF rendering.

Rendered output is checked by reading the PDF back: a resume that looks right but
does not extract as text has failed the requirement that matters most to its
audience.
"""

import logging

import pytest
from pypdf import PdfReader

from resumelab.exceptions import PDFRenderingError, ResumeLabError
from resumelab.rendering import render_resume
from resumelab.rendering.styles import PAGE_SIZE, build_stylesheet
from resumelab.utils.text import control_characters


@pytest.fixture
def rendered(tmp_path, generated_resume):
    return render_resume(generated_resume, tmp_path / "resume.pdf")


@pytest.fixture
def reader(rendered):
    return PdfReader(rendered)


@pytest.fixture
def extracted(reader):
    """Extracted text with wrapping collapsed, so whole sentences can be asserted."""
    return _flatten("\n".join(page.extract_text() for page in reader.pages))


# --- the file -------------------------------------------------------------


def test_a_pdf_is_written(rendered):
    assert rendered.exists()
    assert rendered.stat().st_size > 0


def test_the_file_carries_the_pdf_signature(rendered):
    assert rendered.read_bytes().startswith(b"%PDF-")


def test_the_path_is_returned(tmp_path, generated_resume):
    target = tmp_path / "out" / "resume.pdf"

    assert render_resume(generated_resume, target) == target


def test_missing_parent_directories_are_created(tmp_path, generated_resume):
    target = tmp_path / "deeply" / "nested" / "resume.pdf"

    assert render_resume(generated_resume, target).exists()


def test_the_page_is_us_letter(reader):
    box = reader.pages[0].mediabox

    assert (round(float(box.width)), round(float(box.height))) == (
        round(PAGE_SIZE[0]),
        round(PAGE_SIZE[1]),
    )


def test_a_normal_resume_fits_one_page(reader):
    assert len(reader.pages) == 1


# --- the text is real text ------------------------------------------------


def test_the_candidate_is_identifiable(extracted, generated_resume):
    assert generated_resume.personal.name in extracted
    assert generated_resume.personal.email in extracted
    assert generated_resume.personal.phone in extracted


def test_every_section_heading_is_present(extracted):
    for heading in ("SUMMARY", "EDUCATION", "EXPERIENCE", "PROJECTS", "SKILLS"):
        assert heading in extracted


def test_the_summary_is_rendered(extracted, generated_resume):
    assert generated_resume.summary in extracted


def test_education_is_rendered(extracted, generated_resume):
    entry = generated_resume.education[0]

    assert entry.institution in extracted
    assert entry.field in extracted
    assert f"GPA: {entry.gpa}" in extracted


def test_every_bullet_survives_into_the_pdf(extracted, generated_resume):
    """Extraction is how this document is actually read by its first audience."""
    for bullet in generated_resume.all_bullets:
        assert bullet in extracted


def test_experience_anchors_are_rendered(extracted, generated_resume):
    entry = generated_resume.experiences[0]

    assert entry.company in extracted
    assert entry.title in extracted
    assert entry.start_date in extracted
    assert entry.end_date in extracted


def test_projects_render_their_repositioning(extracted, generated_resume):
    for project in generated_resume.projects:
        assert project.name in extracted
        assert project.subtitle in extracted
        for technology in project.technologies:
            assert technology in extracted


def test_skills_render_with_their_group_labels(extracted, generated_resume):
    for group in generated_resume.skills:
        assert group.label in extracted
        for skill in group.skills:
            assert skill in extracted


def test_achievements_are_rendered_when_present(extracted, generated_resume):
    assert generated_resume.achievements[0] in extracted


def test_the_achievements_section_is_omitted_when_there_are_none(tmp_path, generated_resume):
    bare = generated_resume.model_copy(update={"achievements": ()})

    text = _text_of(render_resume(bare, tmp_path / "bare.pdf"))

    assert "ACHIEVEMENTS" not in text


def test_content_appears_in_reading_order(extracted, generated_resume):
    """A single linear flow, so extraction returns the resume in the order it reads."""
    positions = [
        extracted.index(generated_resume.personal.name),
        extracted.index("SUMMARY"),
        extracted.index("EXPERIENCE"),
        extracted.index("PROJECTS"),
        extracted.index("SKILLS"),
    ]

    assert positions == sorted(positions)


# --- machine readability --------------------------------------------------


def test_the_page_contains_no_images(reader):
    """Icons and logos are the classic way a resume becomes unreadable to a parser."""
    resources = reader.pages[0].get("/Resources", {})

    assert not resources.get("/XObject")


def test_the_bullet_glyph_does_not_extract_as_a_control_character(reader):
    """U+2022 extracts as U+007F in the standard PDF fonts, which every parser sees."""
    raw = "\n".join(page.extract_text() for page in reader.pages)

    assert control_characters(raw.replace("\n", "")) == []


def test_contact_details_are_clickable(reader, generated_resume):
    links = [
        annotation.get_object()["/A"]["/URI"] for annotation in reader.pages[0].get("/Annots", [])
    ]

    assert f"mailto:{generated_resume.personal.email}" in links
    assert f"https://{generated_resume.personal.linkedin}" in links
    assert f"https://{generated_resume.personal.github}" in links


def test_a_profile_url_that_already_has_a_scheme_is_not_doubled(tmp_path, generated_resume):
    personal = generated_resume.personal.model_copy(update={"github": "https://github.com/ada"})

    rendered = render_resume(
        generated_resume.model_copy(update={"personal": personal}), tmp_path / "scheme.pdf"
    )

    links = [a.get_object()["/A"]["/URI"] for a in PdfReader(rendered).pages[0]["/Annots"]]
    assert "https://github.com/ada" in links
    assert "https://https://github.com/ada" not in links


# --- content that would otherwise break the build -------------------------


@pytest.mark.parametrize(
    "hazard",
    ["Scaled R&D throughput", "Handled <5ms p99 latency", "Cut cost by >40%", "A & B <c> d"],
)
def test_markup_characters_are_escaped_not_interpreted(tmp_path, generated_resume, hazard):
    """Resume content routinely contains & and comparison operators."""
    summary = f"{hazard} across distributed storage clusters running Go on Linux."
    resume = generated_resume.model_copy(update={"summary": summary})

    assert hazard in _text_of(render_resume(resume, tmp_path / "escaped.pdf"))


def test_accented_names_render(tmp_path, generated_resume):
    personal = generated_resume.personal.model_copy(update={"name": "José Ramírez"})
    resume = generated_resume.model_copy(update={"personal": personal})

    assert "José Ramírez" in _text_of(render_resume(resume, tmp_path / "accents.pdf"))


def test_a_resume_without_optional_contact_fields_renders(tmp_path, generated_resume):
    from resumelab.models.candidate import PersonalDetails

    minimal = PersonalDetails(name="Ada Lovelace", email="ada@example.edu")
    resume = generated_resume.model_copy(update={"personal": minimal})

    assert "Ada Lovelace" in _text_of(render_resume(resume, tmp_path / "minimal.pdf"))


def test_an_education_entry_without_a_qualification_renders(tmp_path, generated_resume):
    sparse = generated_resume.education[0].model_construct(
        institution="Somewhere",
        degree="",
        field=None,
        location=None,
        start_date=None,
        end_date=None,
        gpa=None,
        coursework=(),
    )
    resume = generated_resume.model_copy(update={"education": (sparse,)})

    assert "Somewhere" in _text_of(render_resume(resume, tmp_path / "sparse.pdf"))


def test_a_resume_with_no_contact_details_at_all_still_renders(tmp_path, generated_resume):
    """The validator rejects this, but the renderer must not crash on it."""
    from resumelab.models.candidate import PersonalDetails

    nameless = PersonalDetails.model_construct(
        name="Ada Lovelace", email=None, phone=None, linkedin=None, github=None, location=None
    )
    resume = generated_resume.model_copy(update={"personal": nameless})

    assert "Ada Lovelace" in _text_of(render_resume(resume, tmp_path / "no-contact.pdf"))


def test_a_project_with_no_technologies_renders(tmp_path, generated_resume):
    first, *rest = generated_resume.projects
    bare = first.model_copy(update={"technologies": ()})
    resume = generated_resume.model_copy(update={"projects": (bare, *rest)})

    assert bare.subtitle in _text_of(render_resume(resume, tmp_path / "no-tech.pdf"))


def test_an_experience_with_only_a_start_date_renders(tmp_path, generated_resume):
    entry = generated_resume.experiences[0].model_copy(update={"end_date": None})
    resume = generated_resume.model_copy(update={"experiences": (entry,)})

    assert "May 2025" in _text_of(render_resume(resume, tmp_path / "open-ended.pdf"))


# --- failure handling -----------------------------------------------------


def test_an_unwritable_destination_is_reported_as_a_rendering_error(tmp_path, generated_resume):
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    blocked.chmod(0o500)
    try:
        with pytest.raises(PDFRenderingError, match="Could not write"):
            render_resume(generated_resume, blocked / "sub" / "resume.pdf")
    finally:
        blocked.chmod(0o700)


def test_rendering_errors_are_resumelab_errors(tmp_path, generated_resume):
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    blocked.chmod(0o500)
    try:
        with pytest.raises(ResumeLabError):
            render_resume(generated_resume, blocked / "sub" / "resume.pdf")
    finally:
        blocked.chmod(0o700)


# --- logging and styles ---------------------------------------------------


def test_rendering_is_logged(tmp_path, generated_resume, caplog):
    with caplog.at_level(logging.INFO, logger="resumelab.rendering.pdf_renderer"):
        render_resume(generated_resume, tmp_path / "logged.pdf")

    assert "rendering PDF" in caplog.text
    assert "bytes=" in caplog.text


def test_the_stylesheet_covers_every_role_the_renderer_uses():
    stylesheet = build_stylesheet()

    assert set(stylesheet) == {"name", "contact", "section", "entry", "detail", "body", "bullet"}


def test_leading_is_proportional_to_font_size():
    """Spacing is derived from the body size rather than set as loose magic numbers."""
    stylesheet = build_stylesheet()

    for style in stylesheet.values():
        assert style.leading > style.fontSize


def test_bullets_hang_so_wrapped_lines_align():
    bullet = build_stylesheet()["bullet"]

    assert bullet.leftIndent > bullet.bulletIndent


def _text_of(path):
    return _flatten("\n".join(page.extract_text() for page in PdfReader(path).pages))


def _flatten(text: str) -> str:
    """Collapse the line breaks the renderer introduced by wrapping."""
    return " ".join(text.split())
