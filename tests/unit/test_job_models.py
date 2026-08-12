"""Tests for the job description schema and its text normalization."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from resumelab.models.job import (
    MAX_JOB_DESCRIPTION_CHARACTERS,
    MIN_JOB_DESCRIPTION_CHARACTERS,
    JobDescription,
    JobDescriptionSource,
)
from resumelab.utils.text import normalize_text

JD_TEXT = (
    "Senior Backend Engineer at Example Corp. You will build distributed "
    "services in Go and Python, and own reliability for the ingestion path."
)


def make_jd(text=JD_TEXT, source=JobDescriptionSource.TEXT, source_path=None):
    return JobDescription(text=text, source=source, source_path=source_path)


# --- normalization --------------------------------------------------------


def test_windows_and_classic_mac_line_endings_are_unified():
    text = JD_TEXT.replace(" ", "\r\n", 1) + "\rTrailing line"

    assert "\r" not in make_jd(text).text


def test_control_characters_are_removed_but_newlines_and_tabs_survive():
    text = f"{JD_TEXT}\n\tIndented requirement\x00\x07"

    result = make_jd(text).text

    assert "\x00" not in result
    assert "\x07" not in result
    assert "\n\tIndented requirement" in result


def test_surrounding_whitespace_is_stripped():
    assert make_jd(f"\n\n  {JD_TEXT}  \n\n").text == JD_TEXT


def test_text_is_nfc_normalized_so_equivalent_strings_compare_equal():
    decomposed = f"{JD_TEXT} Café"
    composed = f"{JD_TEXT} Café"

    assert make_jd(decomposed).text == make_jd(composed).text


def test_normalize_text_is_idempotent():
    once = normalize_text(f"  {JD_TEXT}\r\n\x00 ")

    assert normalize_text(once) == once


# --- length bounds --------------------------------------------------------


def test_text_shorter_than_the_floor_is_rejected():
    with pytest.raises(ValidationError):
        make_jd("Backend engineer wanted.")


def test_text_at_the_floor_is_accepted():
    assert len(make_jd("x" * MIN_JOB_DESCRIPTION_CHARACTERS).text) == (
        MIN_JOB_DESCRIPTION_CHARACTERS
    )


def test_text_beyond_the_ceiling_is_rejected():
    with pytest.raises(ValidationError):
        make_jd("x" * (MAX_JOB_DESCRIPTION_CHARACTERS + 1))


def test_whitespace_only_text_is_rejected():
    with pytest.raises(ValidationError):
        make_jd("   \n\t  ")


# --- provenance -----------------------------------------------------------


def test_a_file_source_requires_a_path():
    with pytest.raises(ValidationError, match="source_path is required"):
        make_jd(source=JobDescriptionSource.FILE)


def test_an_inline_source_must_not_carry_a_path():
    with pytest.raises(ValidationError, match="must be omitted when the source is inline text"):
        make_jd(source=JobDescriptionSource.TEXT, source_path=Path("jd.txt"))


def test_a_file_sourced_description_records_its_path():
    jd = make_jd(source=JobDescriptionSource.FILE, source_path=Path("jd.txt"))

    assert jd.source is JobDescriptionSource.FILE
    assert jd.source_path == Path("jd.txt")


# --- immutability and helpers ---------------------------------------------


def test_the_description_is_frozen():
    jd = make_jd()

    with pytest.raises(ValidationError):
        jd.text = "something else"


def test_character_count_reflects_the_normalized_text():
    jd = make_jd(f"  {JD_TEXT}  ")

    assert jd.character_count == len(JD_TEXT)


# --- provenance for fetched postings ---------------------------------------

POSTING_URL = "https://job-boards.greenhouse.io/northlake/jobs/8077887"


def test_a_url_source_carries_its_url():
    jd = JobDescription(text=JD_TEXT, source=JobDescriptionSource.URL, source_url=POSTING_URL)

    assert jd.source_url == POSTING_URL
    assert jd.source_path is None


def test_a_url_source_requires_a_url():
    with pytest.raises(ValidationError, match="source_url is required"):
        JobDescription(text=JD_TEXT, source=JobDescriptionSource.URL)


def test_a_url_source_must_not_carry_a_path():
    with pytest.raises(ValidationError, match="source_path must be omitted"):
        JobDescription(
            text=JD_TEXT,
            source=JobDescriptionSource.URL,
            source_url=POSTING_URL,
            source_path=Path("jd.txt"),
        )


def test_a_file_source_must_not_carry_a_url():
    with pytest.raises(ValidationError, match="source_url must be omitted"):
        JobDescription(
            text=JD_TEXT,
            source=JobDescriptionSource.FILE,
            source_path=Path("jd.txt"),
            source_url=POSTING_URL,
        )


def test_an_inline_source_must_not_carry_a_url():
    with pytest.raises(ValidationError, match="must be omitted when the source is inline text"):
        JobDescription(text=JD_TEXT, source=JobDescriptionSource.TEXT, source_url=POSTING_URL)
