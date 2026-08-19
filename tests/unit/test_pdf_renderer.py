"""Tests for PDF rendering.

Rendered output is checked by reading the PDF back: a resume that looks right but
does not extract as text has failed the requirement that matters most to its
audience.
"""

import logging
import re

import pytest
from pypdf import PdfReader
from reportlab.lib.enums import TA_RIGHT
from reportlab.pdfbase import pdfmetrics

from resumelab.exceptions import PDFRenderingError, ResumeLabError
from resumelab.models.resume import (
    MAX_BULLET_CHARACTERS,
    MAX_PROJECT_TECHNOLOGIES,
    TARGET_BULLET_CHARACTERS,
)
from resumelab.rendering import RenderOptions, ResumeSection, pdf_renderer, render_resume, styles
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
    crowded = _with_extra_roles(generated_resume, count=5)

    result = render_resume(crowded, tmp_path / "crowded.pdf")

    assert result.fits_on_one_page
    assert result.was_tightened
    assert result.scale in LAYOUT_SCALES


def test_the_written_file_matches_the_chosen_layout(tmp_path, generated_resume):
    """Only the accepted attempt is written; the file is never a discarded draft."""
    crowded = _with_extra_roles(generated_resume, count=5)

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
    crowded = _with_extra_roles(generated_resume, count=5)

    with caplog.at_level(logging.INFO, logger="resumelab.rendering.pdf_renderer"):
        render_resume(crowded, tmp_path / "crowded.pdf")

    assert "tightened layout" in caplog.text


def test_a_bullet_at_the_target_length_occupies_one_line():
    """The target is a measurement, not a preference.

    This is the number the prompts state, and the reason the page reads the way it
    does. Set it above what a line holds and every bullet wraps, the content
    overflows, a condensing call is spent, and the type still ends up at the
    readability floor. That is exactly what a 220 character budget produced.
    """
    bullet = build_stylesheet()["bullet"]
    line_width = styles.CONTENT_WIDTH - bullet.leftIndent
    # Ordinary prose rather than a repeated character, whose width is not typical.
    prose = (
        "delivered a service that processes requests for teams across the "
        "business, holding latency low "
    )
    sample = (prose * 4)[:TARGET_BULLET_CHARACTERS]

    width = pdfmetrics.stringWidth(sample, bullet.fontName, bullet.fontSize)

    assert width <= line_width
    assert TARGET_BULLET_CHARACTERS < MAX_BULLET_CHARACTERS


def test_the_contact_line_fits_on_one_line(generated_resume):
    """A wrapped contact line costs a line of the page and reads as an accident.

    It carries five fields including a location, which is what bounds its size. The
    check is against a rendered contact line rather than a guess at its length.
    """
    line = pdf_renderer._contact_line(generated_resume.personal)
    # Strip the markup the renderer adds; what is drawn is the text inside it.
    drawn = re.sub(r"<[^>]+>", "", line)
    contact = build_stylesheet()["contact"]

    width = pdfmetrics.stringWidth(drawn, contact.fontName, contact.fontSize)

    assert width <= styles.CONTENT_WIDTH


def test_a_tracked_heading_still_extracts_as_one_word(extracted):
    """The reason tracking is set on the text object rather than typed as spaces.

    Letter-spacing a heading by putting spaces between its characters would make it
    extract as ``S U M M A R Y``, which is exactly the kind of damage this format
    exists to avoid. Character spacing moves the glyphs without touching the string.
    """
    for heading in ("SUMMARY", "EDUCATION", "EXPERIENCE", "PROJECTS", "SKILLS"):
        assert heading in extracted
        assert " ".join(heading) not in extracted


def test_headings_are_actually_tracked(tmp_path, generated_resume):
    """Read off the page's own instructions rather than a picture of it.

    Rasterising to measure the ink would need a converter this project does not
    depend on, and would tie the test to whichever one the machine happened to have.
    The character-spacing operator is what tracking *is*, and it is right there.
    """
    rendered = render_resume(generated_resume, tmp_path / "tracked.pdf").path
    content = PdfReader(rendered).pages[0].get_contents().get_data().decode("latin-1")

    spacings = [float(value) for value in re.findall(r"([-\d.]+)\s+Tc\b", content)]

    assert spacings, "no character spacing was set anywhere on the page"
    assert max(spacings) == pytest.approx(styles.SECTION_TRACKING, abs=0.01)


def test_tracking_scales_with_the_type(tmp_path, generated_resume):
    """A fixed tracking would grow relative to the letters as the page tightened."""
    roomy = pdf_renderer._scaled_tracking(build_stylesheet(1.0)["section"])
    tight = pdf_renderer._scaled_tracking(build_stylesheet(LAYOUT_SCALES[-1])["section"])

    assert tight < roomy
    assert tight / roomy == pytest.approx(LAYOUT_SCALES[-1], abs=0.01)


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
    crowded = _with_extra_roles(generated_resume, count=5)

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


def test_achievements_are_not_drawn_even_when_the_profile_has_them(extracted, generated_resume):
    """The run still records them; the page spends its room on something else.

    A heading, a rule, and a line for what is almost always a restatement of
    something already above it is the worst trade on a one-page resume.
    """
    assert generated_resume.achievements
    assert "ACHIEVEMENTS" not in extracted
    assert generated_resume.achievements[0] not in extracted


def test_the_recorded_resume_still_carries_its_achievements(generated_resume):
    """Not drawing them is a layout decision, not a deletion from the record."""
    assert generated_resume.achievements == ("Dean's List",)


# --- education ------------------------------------------------------------


def test_the_institution_leads_the_entry(extracted, generated_resume):
    """The institution heads the entry and the qualification sits beneath it."""
    entry = generated_resume.education[0]

    assert extracted.index(entry.institution) < extracted.index("MS Computer Science")


def test_the_qualification_sits_above_the_coursework(extracted, generated_resume):
    entry = generated_resume.education[0]

    assert extracted.index("MS Computer Science") < extracted.index("Coursework:")
    assert extracted.index(entry.institution) < extracted.index("Coursework:")


def test_a_degree_with_no_known_abbreviation_is_left_as_written(tmp_path, generated_resume):
    """Guessing at an unfamiliar degree would invent a qualification nobody holds."""
    odd = generated_resume.education[0].model_copy(update={"degree": "Licenciatura en Informatica"})
    resume = generated_resume.model_copy(update={"education": (odd,)})

    text = _flatten(_text_of(render_resume(resume, tmp_path / "odd.pdf").path))

    assert "Licenciatura en Informatica" in text


def test_the_abbreviation_ignores_how_the_degree_was_capitalised(tmp_path, generated_resume):
    shouted = generated_resume.education[0].model_copy(update={"degree": "MASTER OF SCIENCE"})
    resume = generated_resume.model_copy(update={"education": (shouted,)})

    text = _flatten(_text_of(render_resume(resume, tmp_path / "shouted.pdf").path))

    assert "MS Computer Science" in text


def test_coursework_is_rendered(extracted, generated_resume):
    """It is in the profile and was being dropped on the floor."""
    entry = generated_resume.education[0]

    assert f"Coursework: {', '.join(entry.coursework)}" in extracted


def test_an_entry_with_only_an_institution_skips_the_line_beneath_it(tmp_path, generated_resume):
    """No qualification and no GPA leaves nothing to set, so nothing is set."""
    sparse = generated_resume.education[0].model_copy(
        update={"degree": "", "field": None, "gpa": None, "coursework": ()}
    )
    resume = generated_resume.model_copy(update={"education": (sparse,)})

    text = _flatten(_text_of(render_resume(resume, tmp_path / "sparse.pdf").path))

    assert sparse.institution in text
    assert "GPA" not in text


def test_an_entry_without_an_institution_still_renders_its_qualification(
    tmp_path, generated_resume
):
    bare = generated_resume.education[0].model_copy(update={"institution": "", "location": None})
    resume = generated_resume.model_copy(update={"education": (bare,)})

    text = _flatten(_text_of(render_resume(resume, tmp_path / "no_school.pdf").path))

    assert "MS Computer Science" in text


def test_an_entry_without_coursework_skips_the_line(tmp_path, generated_resume):
    bare = generated_resume.education[0].model_copy(update={"coursework": ()})
    resume = generated_resume.model_copy(update={"education": (bare,)})

    text = _flatten(_text_of(render_resume(resume, tmp_path / "no_course.pdf").path))

    assert "Coursework" not in text
    assert bare.institution in text


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

    assert set(stylesheet) == {
        "name",
        "contact",
        "section",
        "entry",
        "date",
        "detail",
        "detail_right",
        "note",
        "body",
        "bullet",
    }


def test_leading_is_proportional_to_font_size():
    """Spacing is derived from the body size rather than set as loose magic numbers."""
    stylesheet = build_stylesheet()

    for style in stylesheet.values():
        assert style.leading > style.fontSize


def test_bullet_text_clears_the_marker_at_every_scale():
    """The condition ReportLab actually applies, not the weaker one it implies.

    ``leftIndent > bulletIndent`` is necessary and nowhere near sufficient: ReportLab
    starts the first line at ``bulletIndent + bulletWidth + 0.6 * bulletFontSize``
    whenever that exceeds ``leftIndent``, and later lines at ``leftIndent``. An indent
    that clears the marker's position but not its width loses the hanging indent.
    """
    for scale in LAYOUT_SCALES:
        bullet = build_stylesheet(scale)["bullet"]
        marker = pdfmetrics.stringWidth(
            styles.BULLET_CHARACTER, bullet.bulletFontName, bullet.bulletFontSize
        )
        overrun = bullet.bulletIndent + marker + styles.BULLET_PADDING_RATIO * bullet.bulletFontSize

        assert bullet.leftIndent == pytest.approx(overrun, abs=0.01)


def test_a_wrapped_bullet_line_starts_where_its_first_line_does(tmp_path, generated_resume):
    """Measured off the page, because this is invisible to every text assertion.

    A bullet that does not wrap looks correct however the indents are set. The defect
    only appears on the second line, and every long bullet has one.
    """
    long_bullet = (
        "Owned end to end delivery of a reservation platform, carrying it from an "
        "ambiguous specification through to production and serving ten thousand "
        "people every day without a regression."
    )
    role = generated_resume.experiences[0].model_copy(update={"bullets": (long_bullet,) * 3})
    resume = generated_resume.model_copy(update={"experiences": (role,)})

    rendered = render_resume(resume, tmp_path / "wrapped.pdf").path
    first_line, wrapped_line = _wrapped_bullet_columns(_text_runs(rendered))

    assert wrapped_line == pytest.approx(first_line, abs=0.05)


def _text_runs(path):
    """Every text run on the page, with where it was actually drawn.

    The text matrix alone is relative to the enclosing form, so it is composed with
    the current transformation matrix to get a position on the page.
    """
    runs = []

    def visit(text, cm, tm, _font_dict, _font_size):
        if text.strip():
            x = tm[4] * cm[0] + tm[5] * cm[2] + cm[4]
            y = tm[4] * cm[1] + tm[5] * cm[3] + cm[5]
            runs.append((round(y, 1), round(x, 2), text.strip()))

    for page in PdfReader(path).pages:
        page.extract_text(visitor_text=visit)
    return runs


def _wrapped_bullet_columns(runs):
    """Where a bullet's first line starts, and where the line under it starts."""
    lines = {}
    for y, x, text in runs:
        lines.setdefault(y, []).append((x, text))

    descending = sorted(lines, reverse=True)
    for index, y in enumerate(descending[:-1]):
        parts = sorted(lines[y])
        below = sorted(lines[descending[index + 1]])
        if parts[0][1] != styles.BULLET_CHARACTER or len(parts) < 2:
            continue
        # A line carrying its own marker is the next bullet, and one starting at or
        # left of the marker is a heading, not a line wrapped out of this bullet.
        if below[0][1] != styles.BULLET_CHARACTER and below[0][0] > parts[0][0]:
            return parts[1][0], below[0][0]
    raise AssertionError("no wrapped bullet was found on the page")


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


def test_the_separator_is_smaller_than_the_list_bullet():
    """They are the same glyph doing opposite jobs.

    A bullet marks the start of a line and should be seen; a separator divides terms
    within one and should not compete with them. Drawn at the same size, the skills
    section reads as a list laid sideways.
    """
    assert styles.SEPARATOR_SIZE_DELTA < 0
    assert styles.BULLET_FONT_SIZE <= styles.BODY_FONT_SIZE
    assert styles.BODY_FONT_SIZE + styles.SEPARATOR_SIZE_DELTA < styles.BULLET_FONT_SIZE


def test_the_list_bullet_is_drawn_at_the_bold_weight():
    """At the regular weight the bullet is a thin ring and reads as faint."""
    bullet = styles.build_stylesheet()["bullet"]

    assert bullet.bulletFontName == styles.GLYPH_FONT_BOLD
    assert styles.GLYPH_FONT_BOLD in pdfmetrics.getRegisteredFontNames()


# --- right-aligned dates --------------------------------------------------


def test_a_right_aligned_date_still_extracts_in_reading_order(extracted, generated_resume):
    """The property the two-cell row had to preserve.

    A multi-column layout forces a parser to guess how columns interleave. This row
    must not: the date belongs to the heading above it and must arrive between that
    heading and the bullets beneath, exactly as a person reads it.
    """
    entry = generated_resume.experiences[0]

    company = extracted.index(entry.company)
    date = extracted.index(entry.start_date, company)
    first_bullet = extracted.index(entry.bullets[0], company)

    assert company < date < first_bullet


def test_the_date_is_set_against_the_right_margin():
    """The heading and its date together span the full content width.

    That is what puts the date on the margin: the date column is exactly as wide as
    the date, and the heading column takes everything else.
    """
    stylesheet = build_stylesheet()
    row = pdf_renderer._flush_right_row("<b>Northlake</b>", "Jul 2025 \u2013 May 2026", stylesheet)

    assert stylesheet["date"].alignment == TA_RIGHT
    assert sum(row._argW) == pytest.approx(styles.CONTENT_WIDTH)


def test_a_long_date_cannot_squeeze_the_heading_away(generated_resume):
    """The heading column has a floor, so an absurd date degrades rather than erases."""
    row = pdf_renderer._flush_right_row("<b>Northlake</b>", "x" * 400, build_stylesheet())

    assert row._argW[0] >= styles.MIN_HEADING_WIDTH


def test_an_entry_without_a_date_is_not_given_an_empty_column(tmp_path, generated_resume):
    """An empty column would still take its width and pull the heading in."""
    undated = generated_resume.projects[0].model_copy(update={"date": None})
    resume = generated_resume.model_copy(update={"projects": (undated,)})

    text = _text_of(render_resume(resume, tmp_path / "undated.pdf").path)

    assert undated.name in _flatten(text)
    assert undated.subtitle in _flatten(text)


def test_the_gpa_is_set_against_the_right_margin(extracted, generated_resume):
    """The second education line carries the GPA the way the first carries the date."""
    entry = generated_resume.education[0]

    assert f"GPA: {entry.gpa}" in extracted
    assert entry.field in extracted


def test_an_education_entry_without_a_gpa_still_renders_its_qualification(
    tmp_path, generated_resume
):
    without = generated_resume.education[0].model_copy(update={"gpa": None})
    resume = generated_resume.model_copy(update={"education": (without,)})

    text = _flatten(_text_of(render_resume(resume, tmp_path / "no_gpa.pdf").path))

    assert without.field in text
    assert "GPA" not in text


def test_the_content_width_matches_the_frame_the_document_actually_uses():
    """Entry rows align to the section rules only if this width is right.

    Paragraphs are laid out by the frame and inherit its padding for free. A table is
    given an explicit width, so the padding has to be subtracted by hand — and getting
    it wrong is invisible to every test that only reads the text back.
    """
    import io

    from reportlab.platypus import SimpleDocTemplate
    from reportlab.platypus.frames import Frame

    document = SimpleDocTemplate(
        io.BytesIO(),
        pagesize=styles.PAGE_SIZE,
        leftMargin=styles.MARGIN_HORIZONTAL,
        rightMargin=styles.MARGIN_HORIZONTAL,
        topMargin=styles.MARGIN_TOP,
        bottomMargin=styles.MARGIN_BOTTOM,
    )
    frame = Frame(document.leftMargin, document.bottomMargin, document.width, document.height)

    assert (
        pytest.approx(frame._width - frame._leftPadding - frame._rightPadding)
        == styles.CONTENT_WIDTH
    )


def test_an_entry_row_starts_on_the_left_margin():
    """A Table centres itself by default, which hangs the row out on both sides."""
    row = pdf_renderer._flush_right_row("<b>Northlake</b>", "Jun 2026", build_stylesheet())

    assert row.hAlign == "LEFT"


# --- profile links --------------------------------------------------------


@pytest.mark.parametrize(
    ("stored", "shown"),
    [
        ("https://www.linkedin.com/in/cg10/", "linkedin.com/in/cg10"),
        ("http://linkedin.com/in/cg10", "linkedin.com/in/cg10"),
        ("https://github.com/chaitanyagandhi", "github.com/chaitanyagandhi"),
        ("www.github.com/ada/", "github.com/ada"),
        ("github.com/ada", "github.com/ada"),
    ],
)
def test_a_profile_link_is_shown_without_its_scheme(stored, shown):
    """Nobody types the scheme, and on one page it costs room the address needs."""
    assert pdf_renderer._profile_label(stored) == shown


def test_shortening_a_profile_link_does_not_break_it(tmp_path, generated_resume):
    """The label is shortened; the link still has to resolve."""
    full = "https://www.linkedin.com/in/cg10/"
    personal = generated_resume.personal.model_copy(update={"linkedin": full})
    resume = generated_resume.model_copy(update={"personal": personal})

    rendered = render_resume(resume, tmp_path / "links.pdf").path
    annotations = [
        annotation.get_object()
        for page in PdfReader(rendered).pages
        for annotation in page.get("/Annots", [])
    ]
    targets = [str(entry["/A"]["/URI"]) for entry in annotations if "/A" in entry]

    assert full in targets
    assert _flatten(_text_of(rendered)).count("https://www.linkedin.com") == 0
    assert "linkedin.com/in/cg10" in _flatten(_text_of(rendered))


def test_an_experience_is_one_line_with_the_title_leading(extracted, generated_resume):
    """Title first and bold, employer qualifying it, dates right, no location."""
    entry = generated_resume.experiences[0]

    assert f"{entry.title}, {entry.company}" in extracted
    assert entry.location not in extracted

    title = extracted.index(entry.title)
    dates = extracted.index(entry.start_date, title)
    assert title < dates < extracted.index(entry.bullets[0], title)


def test_an_experience_without_a_company_still_shows_its_title(tmp_path, generated_resume):
    entry = generated_resume.experiences[0].model_copy(update={"company": ""})
    resume = generated_resume.model_copy(update={"experiences": (entry,)})

    text = _flatten(_text_of(render_resume(resume, tmp_path / "no_company.pdf").path))

    assert entry.title in text


def test_an_experience_with_neither_title_nor_location_skips_the_second_line(
    tmp_path, generated_resume
):
    """No content means no row: an empty line would just be a gap above the bullets."""
    entry = generated_resume.experiences[0].model_copy(update={"title": "", "location": None})
    resume = generated_resume.model_copy(update={"experiences": (entry,)})

    text = _flatten(_text_of(render_resume(resume, tmp_path / "bare.pdf").path))

    assert entry.company in text
    assert entry.bullets[0] in text


def test_project_technologies_sit_on_the_title_line(extracted, generated_resume):
    """The stack qualifies the subtitle; a line of its own spends a line saying so."""
    project = generated_resume.projects[0]

    separator = styles.PROJECT_STACK_SEPARATOR
    assert f"{project.subtitle}{separator}{', '.join(project.technologies)}" in extracted


def test_the_project_stack_uses_its_own_separator(extracted, generated_resume):
    """A dot on that line would read as one more field, not a shift to the stack."""
    assert styles.PROJECT_STACK_SEPARATOR != styles.SEPARATOR
    assert styles.PROJECT_STACK_SEPARATOR.strip() in extracted


def test_the_stack_separator_is_plain_ascii(extracted):
    """It survives extraction as itself, which a drawn glyph would not.

    The document's own middle dot needs an embedded font to extract correctly; this
    one is a bar, so it costs a parser nothing and needs no font of its own.
    """
    assert styles.PROJECT_STACK_SEPARATOR.strip() == "|"
    assert "|" in extracted


def test_a_project_names_no_more_technologies_than_the_heading_can_hold(generated_resume):
    """Two is the whole budget: the stack shares its line with a name and a subtitle."""
    for project in generated_resume.projects:
        assert len(project.technologies) <= MAX_PROJECT_TECHNOLOGIES


def test_projects_are_drawn_in_the_order_they_are_given(tmp_path, generated_resume):
    """Which project leads is an editorial choice, made by ordering the list.

    The editor reorders that list and re-renders; nothing else carries the intent, so
    a renderer that sorted or grouped projects would silently discard it.
    """
    reversed_projects = tuple(reversed(generated_resume.projects))
    resume = generated_resume.model_copy(update={"projects": reversed_projects})

    text = _flatten(_text_of(render_resume(resume, tmp_path / "reordered.pdf").path))

    assert [project.name for project in reversed_projects] == sorted(
        (project.name for project in reversed_projects), key=text.index
    )


def test_a_project_without_technologies_still_renders_its_heading(tmp_path, generated_resume):
    bare = generated_resume.projects[0].model_copy(update={"technologies": ()})
    resume = generated_resume.model_copy(update={"projects": (bare,)})

    text = _flatten(_text_of(render_resume(resume, tmp_path / "no_tech.pdf").path))

    assert bare.name in text
    assert bare.subtitle in text


# --- what is shown, and where ---------------------------------------------


def test_the_default_options_draw_the_same_document_as_no_options(tmp_path, generated_resume):
    """The defaults are the pipeline's layout; passing them explicitly changes nothing."""
    implicit = render_resume(generated_resume, tmp_path / "implicit.pdf")
    explicit = render_resume(generated_resume, tmp_path / "explicit.pdf", options=RenderOptions())

    assert _text_of(implicit.path) == _text_of(explicit.path)


def test_sections_are_drawn_in_the_order_asked_for(tmp_path, generated_resume):
    order = (
        ResumeSection.SKILLS,
        ResumeSection.PROJECTS,
        ResumeSection.EXPERIENCE,
        ResumeSection.EDUCATION,
    )

    text = _text_of(
        render_resume(
            generated_resume, tmp_path / "reordered.pdf", options=RenderOptions(section_order=order)
        ).path
    )

    assert _heading_order(text) == ["SKILLS", "PROJECTS", "EXPERIENCE", "EDUCATION"]


def test_the_default_order_puts_education_first(extracted):
    assert _heading_order(extracted) == ["EDUCATION", "EXPERIENCE", "PROJECTS", "SKILLS"]


def test_reordering_moves_the_content_with_its_heading(tmp_path, generated_resume):
    """A heading that moved without its body would be the worst possible outcome here."""
    order = (
        ResumeSection.SKILLS,
        ResumeSection.EDUCATION,
        ResumeSection.EXPERIENCE,
        ResumeSection.PROJECTS,
    )

    text = _text_of(
        render_resume(
            generated_resume,
            tmp_path / "skills_first.pdf",
            options=RenderOptions(section_order=order),
        ).path
    )

    assert text.index(generated_resume.skills[0]) < text.index("EDUCATION")


def test_withholding_the_summary_removes_its_heading_too(tmp_path, generated_resume):
    text = _text_of(
        render_resume(
            generated_resume,
            tmp_path / "no_summary.pdf",
            options=RenderOptions(include_summary=False),
        ).path
    )

    assert generated_resume.summary not in text
    assert "SUMMARY" not in text
    assert generated_resume.personal.name in text


def test_withholding_the_gpa_keeps_the_qualification(tmp_path, generated_resume):
    entry = generated_resume.education[0]

    text = _text_of(
        render_resume(
            generated_resume,
            tmp_path / "no_gpa_shown.pdf",
            options=RenderOptions(include_gpa=False),
        ).path
    )

    assert "GPA" not in text
    assert entry.field in text
    assert entry.institution in text


def _heading_order(text: str) -> list[str]:
    """The body section headings, in the order they appear on the page."""
    headings = ("EDUCATION", "EXPERIENCE", "PROJECTS", "SKILLS")
    return sorted(headings, key=text.index)
