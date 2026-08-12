"""Tests for reducing posting HTML to the text the pipeline analyzes."""

from resumelab.fetching.html_text import html_to_text, tidy

# --- structure that carries meaning ---------------------------------------


def test_list_items_survive_as_separate_lines():
    """A requirements list read as one paragraph stops being a list of requirements."""
    text = html_to_text("<ul><li>Go and Java</li><li>Linux internals</li></ul>")

    assert text == "- Go and Java\n- Linux internals"


def test_paragraphs_are_separated():
    text = html_to_text("<p>About the role</p><p>What you will do</p>")

    assert text == "About the role\n\nWhat you will do"


def test_line_breaks_are_honored():
    assert html_to_text("Sunnyvale, CA<br>Hybrid") == "Sunnyvale, CA\nHybrid"


def test_inline_elements_do_not_break_a_sentence():
    """Only block elements end a line; a bolded word mid-sentence must not split it."""
    text = html_to_text("<p>Experience with <strong>NVMe</strong> devices</p>")

    assert text == "Experience with NVMe devices"


def test_headings_are_kept_as_their_own_lines():
    text = html_to_text("<h2>Requirements</h2><p>Proficiency in Go</p>")

    assert text == "Requirements\n\nProficiency in Go"


# --- content that is not the posting --------------------------------------


def test_scripts_and_styles_are_dropped():
    html = "<p>Posting</p><script>var jobs = 1;</script><style>.a { color: red }</style>"

    assert html_to_text(html) == "Posting"


def test_navigation_and_footers_are_dropped():
    """Site chrome is not part of the posting and would be analyzed as if it were."""
    html = "<nav>Home Careers</nav><p>Storage engineer</p><footer>(c) 2026</footer>"

    assert html_to_text(html) == "Storage engineer"


def test_nested_dropped_elements_do_not_end_the_skip_early():
    """A closing inner tag must not resume collection inside the outer dropped one."""
    html = "<nav>outer <nav>inner</nav> still nav</nav><p>Posting</p>"

    assert html_to_text(html) == "Posting"


# --- entity and escaping handling -----------------------------------------


def test_entities_are_decoded():
    assert html_to_text("<p>Research &amp; development</p>") == "Research & development"


def test_double_escaped_markup_is_decoded_first():
    """Greenhouse escapes the HTML in its `content` field a second time.

    Left alone, the posting would arrive with literal angle brackets in the text and
    every tag would be analyzed as if it were prose.
    """
    text = html_to_text("&lt;p&gt;Build storage systems&lt;/p&gt;&lt;p&gt;In Go&lt;/p&gt;")

    assert text == "Build storage systems\n\nIn Go"


def test_real_markup_mentioning_an_escaped_bracket_is_left_alone():
    """The double-escape fix must not fire on a posting that merely says `&lt;`."""
    text = html_to_text("<p>Latency &lt; 200ms</p>")

    assert text == "Latency < 200ms"


# --- whitespace -----------------------------------------------------------


def test_whitespace_is_collapsed():
    assert html_to_text("<p>Go   and\n\n  Java</p>") == "Go and Java"


def test_deeply_nested_divs_do_not_become_a_page_of_newlines():
    html = "<div><div><div><p>Posting</p></div></div></div>"

    assert html_to_text(html) == "Posting"


def test_tidy_leaves_at_most_one_blank_line():
    assert tidy("a\n\n\n\n\nb") == "a\n\nb"


# --- preferred regions ----------------------------------------------------


def test_a_main_region_is_taken_over_the_whole_page():
    """A page that marks its content region is telling us where the posting is."""
    html = "<body><header>Careers at Northlake</header><main><p>Storage engineer</p></main></body>"

    assert html_to_text(html, prefer=frozenset({"main"})) == "Storage engineer"


def test_the_whole_page_is_used_when_there_is_no_preferred_region():
    html = "<body><p>Storage engineer</p></body>"

    assert html_to_text(html, prefer=frozenset({"main"})) == "Storage engineer"


def test_an_empty_preferred_region_falls_back_to_the_page():
    """An empty `<main>` is a rendering shell, not an instruction to return nothing."""
    html = "<body><main></main><p>Storage engineer</p></body>"

    assert html_to_text(html, prefer=frozenset({"main"})) == "Storage engineer"


def test_preferred_regions_are_ignored_unless_requested():
    html = "<body><header>Careers</header><main><p>Storage engineer</p></main></body>"

    assert html_to_text(html) == "Careers\n\nStorage engineer"
