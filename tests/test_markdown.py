"""Tests for the markdown module, which handles the splitting of markdown files into sections."""

import pytest
from kenku.markdown import parse_sections


@pytest.fixture
def short_markdown_string():
    """A short markdown string for testing."""
    return """# Title

## Section 1

This is the first section.

This is another paragraph in the first section.

## Section 2

This is the second section.

### Subsection 2.1

This is the first subsection of the second section.

### Subsection 2.2

This is the second subsection of the second section.

## Section 3

This is the third section.
"""


def test_markdown_parse_sections(short_markdown_string):
    """Test that the parse_sections function returns the correct sections.

    The parsed sections should be a dictionary with the section titles as keys and the section contents as values (lists of strings, split by paragraph).
    Subsections should be handled correctly by nesting their dictionaries within their parent section's dictionary.
    """
    sections = parse_sections(short_markdown_string)
    assert type(sections) == dict, "The sections should be a dictionary."
    assert len(sections) == 3, "There should be three sections."
    assert "Section 1" in sections, "The first section should be in the sections dictionary."
    assert "Section 2" in sections, "The second section should be in the sections dictionary."
    assert "Section 3" in sections, "The third section should be in the sections dictionary."

    assert type(sections["Section 1"]) == list, "The first section should be a list."
    assert len(sections["Section 1"]) == 2, "The first section should have two paragraphs."
    assert sections["Section 1"][0] == "This is the first section.", "The first paragraph of the first section should be correct."
    assert sections["Section 1"][1] == "This is another paragraph in the first section.", "The second paragraph of the first section should be correct."

    assert type(sections["Section 2"]) == list, "The second section should be a list."
    assert len(sections["Section 2"]) == 3, "The second section should have three 'paragraphs' (one paragraphs and two subsections)."
    assert sections["Section 2"][0] == "This is the second section.", "The first paragraph of the second section should be correct."
    assert type(sections["Section 2"][1]) == dict, "The second paragraph of the second section should be a dictionary."
    assert len(sections["Section 2"][1]) == 2, "The second paragraph of the second section should have two subsections."
    assert "Subsection 2.1" in sections["Section 2"][1], "The first subsection of the second section should be in the subsections dictionary."
    assert "Subsection 2.2" in sections["Section 2"][1], "The second subsection of the second section should be in the subsections dictionary."
    assert type(sections["Section 2"][1]["Subsection 2.1"]) == list, "The first subsection of the second section should be a list."
    assert len(sections["Section 2"][1]["Subsection 2.1"]) == 1, "The first subsection of the second section should have one paragraph."
    assert sections["Section 2"][1]["Subsection 2.1"][0] == "This is the first subsection of the second section.", "The first paragraph of the first subsection of the second section should be correct."
    assert type(sections["Section 2"][1]["Subsection 2.2"]) == list, "The second subsection of the second section should be a list."
    assert len(sections["Section 2"][1]["Subsection 2.2"]) == 1, "The second subsection of the second section should have one paragraph."
    assert sections["Section 2"][1]["Subsection 2.2"][0] == "This is the second subsection of the second section.", "The first paragraph of the second subsection of the second section should be correct."

    assert type(sections["Section 3"]) == list, "The third section should be a list."
    assert len(sections["Section 3"]) == 1, "The third section should have one paragraph."
    assert sections["Section 3"][0] == "This is the third section.", "The first paragraph of the third section should be correct."    

