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
from resumelab.rendering.styles import (
    LAYOUT_SCALES,
    MIN_BODY_FONT_SIZE,
    PAGE_SIZE,
    build_stylesheet,
)
from resumelab.utils.text import control_characters


@pytest.fixture
def rendered(tmp_path, generated_resume):
    return render_resume(generated_resume, tmp_path / "resume.pdf").path


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

    assert render_resume(generated_resume, target).path == target


def test_missing_parent_directories_are_created(tmp_path, generated_resume):
    target = tmp_path / "deeply" / "nested" / "resume.pdf"

    assert render_resume(generated_resume, target).path.exists()


def test_the_page_is_us_letter(reader):
    box = reader.pages[0].mediabox

    assert (round(float(box.width)), round(float(box.height))) == (
        round(PAGE_SIZE[0]),
        round(PAGE_SIZE[1]),
    )


def test_a_normal_resume_fits_one_page(reader):
    assert len(reader.pages) == 1


# --- fitting one page -----------------------------------------------------


def test_content_that_already_fits_is_not_tightened(tmp_path, generated_resume):
    result = render_resume(generated_resume, tmp_path / "roomy.pdf")

    assert result.scale == 1.0
    assert result.was_tightened is False
    assert result.fits_on_one_page


def test_slightly_overflowing_content_is_tightened_onto_one_page(tmp_path, generated_resume):
    """The conservative auto-fit: shrink type and spacing together, within limits."""
    crowded = _with_extra_roles(generated_resume, count=4)

    result = render_resume(crowded, tmp_path / "crowded.pdf")

    assert result.fits_on_one_page
    assert result.was_tightened
    assert result.scale in LAYOUT_SCALES


def test_the_written_file_matches_the_chosen_layout(tmp_path, generated_resume):
    """Only the accepted attempt is written; the file is never a discarded draft."""
    crowded = _with_extra_roles(generated_resume, count=4)

    result = render_resume(crowded, tmp_path / "crowded.pdf")

    assert len(PdfReader(result.path).pages) == result.page_count


def test_content_is_never_shrunk_below_the_readability_floor(tmp_path, generated_resume):
    """A readable two-page resume beats an unreadable one-page resume."""
    far_too_much = _with_extra_roles(generated_resume, count=30)

    result = render_resume(far_too_much, tmp_path / "overflowing.pdf")

    assert result.scale == LAYOUT_SCALES[-1]
    assert build_stylesheet(result.scale)["body"].fontSize >= MIN_BODY_FONT_SIZE


def test_overflow_is_reported_rather_than_hidden(tmp_path, generated_resume):
    far_too_much = _with_extra_roles(generated_resume, count=30)

    result = render_resume(far_too_much, tmp_path / "overflowing.pdf")

    assert result.fits_on_one_page is False
    assert result.page_count > 1


def test_overflow_is_logged_with_what_to_do_about_it(tmp_path, generated_resume, caplog):
    far_too_much = _with_extra_roles(generated_resume, count=30)

    with caplog.at_level(logging.WARNING, logger="resumelab.rendering.pdf_renderer"):
        render_resume(far_too_much, tmp_path / "overflowing.pdf")

    assert "does not fit one page" in caplog.text
    assert "condense" in caplog.text


def test_tightening_is_logged(tmp_path, generated_resume, caplog):
    crowded = _with_extra_roles(generated_resume, count=4)

    with caplog.at_level(logging.INFO, logger="resumelab.rendering.pdf_renderer"):
        render_resume(crowded, tmp_path / "crowded.pdf")

    assert "tightened layout" in caplog.text


def test_every_layout_scale_keeps_the_body_readable():
    for scale in LAYOUT_SCALES:
        assert build_stylesheet(scale)["body"].fontSize >= MIN_BODY_FONT_SIZE


def test_tighter_scales_shrink_type_and_spacing_together():
    roomy, tight = build_stylesheet(LAYOUT_SCALES[0]), build_stylesheet(LAYOUT_SCALES[-1])

    assert tight["body"].fontSize < roomy["body"].fontSize
    assert tight["section"].spaceBefore < roomy["section"].spaceBefore
    assert tight["bullet"].leftIndent < roomy["bullet"].leftIndent


def test_a_tightened_resume_still_extracts_completely(tmp_path, generated_resume):
    """Fitting must not cost content: shrinking is a layout change, not an edit."""
    crowded = _with_extra_roles(generated_resume, count=4)

    text = _text_of(render_resume(crowded, tmp_path / "crowded.pdf").path)

    for bullet in crowded.all_bullets:
        assert bullet in text


def _with_extra_roles(resume, *, count):
    """Repeat the experience section to push the page past one."""
    original = resume.experiences[0]
    roles = tuple(
        original.model_copy(update={"company": f"Company {index}"}) for index in range(count)
    )
    return resume.model_copy(update={"experiences": roles})


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


def test_every_skill_is_rendered(extracted, generated_resume):
    for skill in generated_resume.skills:
        assert skill in extracted


def test_skills_render_as_one_separated_line(extracted, generated_resume):
    """Flat, not grouped: no labels, and the document's own separator between terms."""
    first, second = generated_resume.skills[:2]

    assert f"{first} \u2022 {second}" in extracted


def test_the_skill_separator_extracts_as_itself(extracted):
    """A separator that renders but extracts as a control character would be
    invisible here and wrong in every ATS."""
    assert "\u2022" in extracted
    assert "\x7f" not in extracted


def test_no_control_characters_survive_anywhere_in_the_document(extracted):
    """The whole point of the embedded glyph font: U+2022 in a base-14 font extracts
    as U+007F, which every parser reading this resume would see."""
    assert not [
        character for character in extracted if ord(character) < 32 or ord(character) == 127
    ]


def test_achievements_are_rendered_when_present(extracted, generated_resume):
    assert generated_resume.achievements[0] in extracted


def test_the_achievements_section_is_omitted_when_there_are_none(tmp_path, generated_resume):
    bare = generated_resume.model_copy(update={"achievements": ()})

    text = _text_of(render_resume(bare, tmp_path / "bare.pdf").path)

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
    ).path

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

    assert hazard in _text_of(render_resume(resume, tmp_path / "escaped.pdf").path)


def test_accented_names_render(tmp_path, generated_resume):
    personal = generated_resume.personal.model_copy(update={"name": "José Ramírez"})
    resume = generated_resume.model_copy(update={"personal": personal})

    assert "José Ramírez" in _text_of(render_resume(resume, tmp_path / "accents.pdf").path)


def test_a_resume_without_optional_contact_fields_renders(tmp_path, generated_resume):
    from resumelab.models.candidate import PersonalDetails

    minimal = PersonalDetails(name="Ada Lovelace", email="ada@example.edu")
    resume = generated_resume.model_copy(update={"personal": minimal})

    assert "Ada Lovelace" in _text_of(render_resume(resume, tmp_path / "minimal.pdf").path)


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

    assert "Somewhere" in _text_of(render_resume(resume, tmp_path / "sparse.pdf").path)


def test_a_resume_with_no_contact_details_at_all_still_renders(tmp_path, generated_resume):
    """The validator rejects this, but the renderer must not crash on it."""
    from resumelab.models.candidate import PersonalDetails

    nameless = PersonalDetails.model_construct(
        name="Ada Lovelace", email=None, phone=None, linkedin=None, github=None, location=None
    )
    resume = generated_resume.model_copy(update={"personal": nameless})

    assert "Ada Lovelace" in _text_of(render_resume(resume, tmp_path / "no-contact.pdf").path)


def test_a_project_with_no_technologies_renders(tmp_path, generated_resume):
    first, *rest = generated_resume.projects
    bare = first.model_copy(update={"technologies": ()})
    resume = generated_resume.model_copy(update={"projects": (bare, *rest)})

    assert bare.subtitle in _text_of(render_resume(resume, tmp_path / "no-tech.pdf").path)


def test_an_experience_with_only_a_start_date_renders(tmp_path, generated_resume):
    entry = generated_resume.experiences[0].model_copy(update={"end_date": None})
    resume = generated_resume.model_copy(update={"experiences": (entry,)})

    assert "May 2025" in _text_of(render_resume(resume, tmp_path / "open-ended.pdf").path)


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


def test_an_unlayoutable_document_is_reported_as_a_rendering_error(
    tmp_path, generated_resume, monkeypatch
):
    """ReportLab raises when a flowable cannot fit at all; that must not escape raw."""
    from reportlab.platypus.doctemplate import LayoutError

    def refuse(*_args, **_kwargs):
        raise LayoutError("flowable too large on page")

    monkeypatch.setattr("resumelab.rendering.pdf_renderer.SimpleDocTemplate.build", refuse)

    with pytest.raises(PDFRenderingError, match="could not be laid out"):
        render_resume(generated_resume, tmp_path / "impossible.pdf")
