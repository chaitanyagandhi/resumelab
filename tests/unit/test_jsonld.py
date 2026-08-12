"""Tests for reading a schema.org JobPosting block out of a page.

This is the strongest generic strategy: the block was published for search engines,
so it holds the posting already separated from the site's navigation and footer.
"""

import json

from resumelab.fetching.jsonld import find_job_posting

POSTING = {
    "@context": "https://schema.org",
    "@type": "JobPosting",
    "title": "Software Engineer, Cloud Storage",
    "hiringOrganization": {"@type": "Organization", "name": "Northlake Systems"},
    "jobLocation": {
        "@type": "Place",
        "address": {
            "@type": "PostalAddress",
            "addressLocality": "Sunnyvale",
            "addressRegion": "CA",
            "addressCountry": "US",
        },
    },
    "description": "<p>Build distributed storage services.</p>",
}


def page(*blocks) -> str:
    scripts = "".join(
        f'<script type="application/ld+json">{json.dumps(block)}</script>' for block in blocks
    )
    return f"<html><head>{scripts}</head><body><p>rendered</p></body></html>"


# --- finding the posting --------------------------------------------------


def test_a_job_posting_block_is_read():
    posting = find_job_posting(page(POSTING))

    assert posting is not None
    assert posting.title == "Software Engineer, Cloud Storage"
    assert posting.company == "Northlake Systems"
    assert posting.location == "Sunnyvale, CA, US"
    assert posting.description_html == "<p>Build distributed storage services.</p>"


def test_a_page_without_any_block_yields_nothing():
    assert find_job_posting("<html><body><p>Careers</p></body></html>") is None


def test_a_page_whose_only_block_is_not_a_posting_yields_nothing():
    breadcrumbs = {"@type": "BreadcrumbList", "itemListElement": []}

    assert find_job_posting(page(breadcrumbs)) is None


def test_the_posting_is_found_among_other_blocks():
    """Pages routinely carry an Organization and a BreadcrumbList alongside it."""
    other = {"@type": "Organization", "name": "Northlake Systems"}

    posting = find_job_posting(page(other, POSTING))

    assert posting is not None
    assert posting.title == "Software Engineer, Cloud Storage"


def test_a_posting_inside_a_graph_wrapper_is_found():
    graph = {"@context": "https://schema.org", "@graph": [{"@type": "WebPage"}, POSTING]}

    posting = find_job_posting(page(graph))

    assert posting is not None
    assert posting.company == "Northlake Systems"


def test_a_posting_in_a_top_level_list_is_found():
    posting = find_job_posting(page([{"@type": "WebSite"}, POSTING]))

    assert posting is not None


def test_a_type_declared_as_a_list_is_recognized():
    posting = find_job_posting(page(POSTING | {"@type": ["JobPosting", "Thing"]}))

    assert posting is not None


def test_a_malformed_block_does_not_prevent_reading_a_later_one():
    """Pages often carry several blocks; one broken block is not a reason to stop."""
    html = (
        '<script type="application/ld+json">{not json}</script>'
        f'<script type="application/ld+json">{json.dumps(POSTING)}</script>'
    )

    posting = find_job_posting(html)

    assert posting is not None


def test_scripts_that_are_not_ld_json_are_ignored():
    html = '<script type="text/javascript">var a = {"@type": "JobPosting"};</script>'

    assert find_job_posting(html) is None


def test_an_untyped_script_is_ignored():
    html = f"<script>{json.dumps(POSTING)}</script>"

    assert find_job_posting(html) is None


# --- fields that vary across sites ----------------------------------------


def test_a_named_organization_string_is_accepted():
    posting = find_job_posting(page(POSTING | {"hiringOrganization": "Northlake Systems"}))

    assert posting is not None
    assert posting.company == "Northlake Systems"


def test_the_first_of_several_locations_is_taken():
    remote = {"@type": "Place", "address": {"addressLocality": "Remote"}}
    posting = find_job_posting(page(POSTING | {"jobLocation": [remote]}))

    assert posting is not None
    assert posting.location == "Remote"


def test_a_location_list_that_names_nowhere_is_absent():
    posting = find_job_posting(page(POSTING | {"jobLocation": [{"@type": "Place"}, {}]}))

    assert posting is not None
    assert posting.location is None


def test_a_plain_string_location_is_accepted():
    posting = find_job_posting(page(POSTING | {"jobLocation": "Remote, US"}))

    assert posting is not None
    assert posting.location == "Remote, US"


def test_a_place_without_an_address_falls_back_to_its_name():
    place = {"@type": "Place", "name": "Remote, US"}
    posting = find_job_posting(page(POSTING | {"jobLocation": place}))

    assert posting is not None
    assert posting.location == "Remote, US"


def test_missing_optional_fields_are_absent_rather_than_empty():
    posting = find_job_posting(page({"@type": "JobPosting", "description": "<p>Build.</p>"}))

    assert posting is not None
    assert posting.title is None
    assert posting.company is None
    assert posting.location is None


def test_a_posting_with_no_description_yields_empty_text():
    posting = find_job_posting(page({"@type": "JobPosting", "title": "Engineer"}))

    assert posting is not None
    assert posting.description_html == ""


def test_non_string_field_values_are_ignored_rather_than_stringified():
    """A `title` of 42 must not become the string "42" in the posting."""
    posting = find_job_posting(page(POSTING | {"title": 42, "jobLocation": 7}))

    assert posting is not None
    assert posting.title is None
    assert posting.location is None
