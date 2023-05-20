import pytest
from kenku.core.markdown import Section, parse_sections


def test_section_ancestors():
    root = Section("Root", [])
    child1 = Section("Child 1", [], parent=root)
    child2 = Section("Child 2", [], parent=root)
    grandchild = Section("Grandchild", [], parent=child1)
    assert grandchild.get_ancestors() == [child1, root]


def test_section_siblings():
    root = Section("Root", [])
    child1 = Section("Child 1", [], parent=root)
    child2 = Section("Child 2", [], parent=root)
    assert child1.get_siblings() == [child2]


def test_section_level():
    root = Section("Root", [])
    child1 = Section("Child 1", [], parent=root)
    child2 = Section("Child 2", [], parent=root)
    grandchild = Section("Grandchild", [], parent=child1)
    assert grandchild.get_level() == 2


def test_section_to_dict():
    root = Section("Root", [])
    child1 = Section("Child 1", [], parent=root)
    child2 = Section("Child 2", [], parent=root)
    grandchild = Section("Grandchild", [], parent=child1)
    expected_dict = {
        "title": "Root",
        "content": [],
        "children": [
            {
                "title": "Child 1",
                "content": [],
                "children": [
                    {
                        "title": "Grandchild",
                        "content": [],
                        "children": []
                    }
                ]
            },
            {
                "title": "Child 2",
                "content": [],
                "children": []
            }
        ]
    }
    assert root.to_dict() == expected_dict


def test_section_copy():
    root = Section("Root", [])
    child1 = Section("Child 1", [], parent=root)
    child2 = Section("Child 2", [], parent=root)
    grandchild = Section("Grandchild", [], parent=child1)
    copy = root.copy()
    assert copy.to_dict() == root.to_dict()
    assert copy is not root


def test_section_from_dict():
    data = {
        "title": "Root",
        "content": [],
        "children": [
            {
                "title": "Child 1",
                "content": [],
                "children": [
                    {
                        "title": "Grandchild",
                        "content": [],
                        "children": []
                    }
                ]
            },
            {
                "title": "Child 2",
                "content": [],
                "children": []
            }
        ]
    }
    root = Section.from_dict(data)
    assert root.to_dict() == data


def test_section_from_markdown():
    markdown = "# Root\n\n## Child 1\n\n### Grandchild\n\n## Child 2"
    root = parse_sections(markdown)
    expected_dict = {
        "title": "Root",
        "content": [],
        "children": [
            {
                "title": "Child 1",
                "content": [],
                "children": [
                    {
                        "title": "Grandchild",
                        "content": [],
                        "children": []
                    }
                ]
            },
            {
                "title": "Child 2",
                "content": [],
                "children": []
            }
        ]
    }
    assert root.to_dict() == expected_dict


def test_section_to_markdown():
    root = Section("Root", [])
    child1 = Section("Child 1", [], parent=root)
    child2 = Section("Child 2", [], parent=root)
    grandchild = Section("Grandchild", [], parent=child1)
    expected_markdown = "# Root\n\n## Child 1\n\n### Grandchild\n\n## Child 2"
    assert root.to_markdown() == expected_markdown

import pytest

from kenku.core.markdown import parse_sections, Section


# region: Section tests
@pytest.fixture
def section():
    """A Section object for testing."""
    return Section("Title", ["Content"])


@pytest.fixture
def child_section():
    """A child Section object for testing."""
    return Section("Child Title", "Child Content")


@pytest.fixture
def grandchild_section():
    """A grandchild Section object for testing."""
    return Section("Grandchild Title", "Grandchild Content")


# GIVEN: a Section object
# WHEN: the add_child function is called with a child section
# THEN: the child section is added to the children list
def test_section_add_child_adds_child(section, child_section):
    """Test that the add_child function adds a child section."""
    section.add_child(child_section)
    assert child_section in section.children


# GIVEN: a Section object
# WHEN: the add_children function is called with a list of child sections
# THEN: the child sections are added to the children list
def test_section_add_children_adds_children(section, child_section, grandchild_section):
    """Test that the add_children function adds child sections."""
    section.add_children([child_section, grandchild_section])
    assert child_section in section.children
    assert grandchild_section in section.children


# GIVEN: a Section object
# WHEN: the get_descendants function is called
# THEN: the function returns a list of all descendant sections
def test_section_get_descendants_returns_descendants(
    section, child_section, grandchild_section
):
    """Test that the get_descendants function returns a list of all descendant sections."""
    section.add_child(child_section)
    child_section.add_child(grandchild_section)
    descendants = section.get_descendants()
    assert child_section in descendants
    assert grandchild_section in descendants


# GIVEN: a Section object
# WHEN: the get_ancestors function is called
# THEN: the function returns a list of all ancestor sections
def test_section_get_ancestors_returns_ancestors(
    section, child_section, grandchild_section
):
    """Test that the get_ancestors function returns a list of all ancestor sections."""
    section.add_child(child_section)
    child_section.add_child(grandchild_section)
    ancestors = grandchild_section.get_ancestors()
    assert child_section in ancestors
    assert section in ancestors


# GIVEN: a Section object
# WHEN: the get_siblings function is called
# THEN: the function returns a list of all sibling sections
def test_section_get_siblings_returns_siblings(
    section, child_section, grandchild_section
):
    """Test that the get_siblings function returns a list of all sibling sections."""
    section.add_child(child_section)
    section.add_child(grandchild_section)
    siblings = child_section.get_siblings()
    assert grandchild_section in siblings
    assert section not in siblings


# GIVEN: a Section object
# WHEN: the dict function is called
# THEN: the function returns a dictionary representation of the section
def test_section_dict_returns_dict(section):
    """Test that the dict function returns a dictionary representation of the section."""
    section_dict = section.to_dict()
    assert section_dict["title"] == "Title"
    assert section_dict["content"] == ["Content"]
    assert section_dict["children"] == []


# GIVEN: a Section object with children and grandchildren
# WHEN: the dict function is called
# THEN: the function returns a dictionary representation of the section
# AND: the function returns a dictionary representation of the children, grandchildren, etc.
def test_section_dict_returns_dict_with_children(
    section, child_section, grandchild_section
):
    """Test that the dict function returns a dictionary representation of the section."""
    section.add_child(child_section)
    child_section.add_child(grandchild_section)
    section_dict = section.to_dict()
    assert section_dict["title"] == "Title"
    assert section_dict["content"] == ["Content"]
    assert section_dict["children"][0]["title"] == "Child Title"
    assert section_dict["children"][0]["content"] == ["Child Content"]
    assert section_dict["children"][0]["children"][0]["title"] == "Grandchild Title"
    assert section_dict["children"][0]["children"][0]["content"] == [
        "Grandchild Content"
    ]


# GIVEN: a Section object
# WHEN: the from_dict function is called
# THEN: the function returns a Section object from a dictionary
def test_section_from_dict_returns_section(section):
    """Test that the from_dict function returns a Section object from a dictionary."""
    section_dict = section.to_dict()
    section_from_dict = Section.from_dict(section_dict)
    assert isinstance(section_from_dict, Section)
    assert section_from_dict == section


# GIVEN: a Section object with children and grandchildren
# WHEN: the from_dict function is called
# THEN: the function returns a Section object from a dictionary
# AND: the function returns a Section object with children, grandchildren, etc.
def test_section_from_dict_returns_section_with_children(
    section, child_section, grandchild_section
):
    """Test that the from_dict function returns a Section object from a dictionary."""
    section.add_child(child_section)
    child_section.add_child(grandchild_section)
    section_dict = section.to_dict()
    section_from_dict = Section.from_dict(section_dict)
    assert isinstance(section_from_dict, Section)
    assert section_from_dict == section


# GIVEN: a Section object
# WHEN: the __eq__ function is called with an identical Section object
# THEN: the function returns True
def test_section_eq_returns_true(section):
    """Test that the __eq__ function returns True when the sections are identical."""
    section_2 = Section("Title", "Content")
    assert section == section_2


# GIVEN: a Section object
# WHEN: the __eq__ function is called with a different Section object
# THEN: the function returns False
def test_section_eq_returns_false(section):
    """Test that the __eq__ function returns False when the sections are different."""
    section_2 = Section("Title", "Different Content")
    assert section != section_2


# test creating a session object with a parent in the constructor
def test_section_constructor_with_parent(section):
    """Test that the Section constructor creates a Section object with a parent."""
    assert section.parent is None
    child_section = Section("Child Title", "Child Content", parent=section)
    assert child_section.parent == section


# test wrong content type, not a list or string
def test_section_constructor_wrong_content_type():
    """Test that the Section constructor raises a TypeError if content is not a list or string."""
    with pytest.raises(TypeError):
        Section("Title", 1)  # type: ignore


def test_section_str(section):
    """Test that the __str__ function returns the title of the section."""
    assert str(section) == "Section: Title"


def test_section_repr(section):
    """Test that the __repr__ function returns a string representation of the section."""
    assert repr(section) == "Section(title='Title', content=['Content'], children=[])"

def test_section_repr_with_parent(section):
    """Test that the __repr__ function returns a string representation of the section."""
    child_section = Section("Child Title", "Child Content", parent=section)
    assert repr(child_section) == "Section(title='Child Title', content=['Child Content'], parent='Title', children=[])"
    

def test_section_hash(section):
    """Test that the __hash__ function returns a hash of the section."""
    assert isinstance(hash(section), int)

@pytest.fixture
def simple_markdown_string():
    """A short markdown string for testing."""
    return """# Title

## Section 1

This is the first section.

## Section 2

This is the second section.
"""


# GIVEN: a markdown string
# WHEN: the from_markdown function is called
# THEN: the function returns a Section object from a markdown string
def test_section_from_markdown_returns_section(simple_markdown_string):
    """Test that the from_markdown function returns a Section object from a markdown string."""
    section = Section.from_markdown(simple_markdown_string)
    assert isinstance(section, Section)
    assert section.title == "Title"
    assert section.content == []
    assert section.children[0].title == "Section 1"
    assert section.children[0].content == ["This is the first section."]
    assert section.children[1].title == "Section 2"
    assert section.children[1].content == ["This is the second section."]
    assert section.children[0].parent == section
    assert section.children[1].parent == section


# conversion between markdown and section objects
# GIVEN: a Section object
# WHEN: the to_markdown function is called
# THEN: the function returns a markdown string from a Section object
def test_section_to_markdown_returns_markdown(section):
    """Test that the to_markdown function returns a markdown string from a Section object."""
    markdown = section.to_markdown()
    assert markdown == "# Title\n\nContent"


# GIVEN: a Section object with children and grandchildren
# WHEN: the to_markdown function is called
# THEN: the function returns a markdown string from a Section object
# AND: the function returns a markdown string with children, grandchildren, etc.
def test_section_to_markdown_returns_markdown_with_children(
    section, child_section, grandchild_section
):
    """Test that the to_markdown function returns a markdown string from a Section object."""
    section.add_child(child_section)
    child_section.add_child(grandchild_section)
    markdown = section.to_markdown()
    assert (
        markdown
        == "# Title\n\nContent\n\n## Child Title\n\nChild Content\n\n### Grandchild Title\n\nGrandchild Content"
    )


# GIVEN: a Section object created from a markdown string
# WHEN: the to_markdown function is called
# THEN: the function returns a markdown string from a Section object
# AND: the markdown string is identical to the original markdown string used to create the Section object (except for whitespace)
def test_section_to_markdown_returns_markdown_from_markdown_string(
    simple_markdown_string,
):
    """Test that the to_markdown function returns a markdown string from a Section object."""
    section = Section.from_markdown(simple_markdown_string)
    markdown = section.to_markdown()
    assert markdown.strip() == simple_markdown_string.strip()


# endregion


# region: parse_sections tests
# GIVEN: a markdown string
# WHEN: the parse_sections function is called
# THEN: the function returns a Section object
def test_markdown_parse_sections_returns_section_object(simple_markdown_string):
    """Test that the parse_sections function returns the correct sections."""
    sections = parse_sections(simple_markdown_string)
    assert isinstance(sections, Section)


# GIVEN: a markdown string
# WHEN: the parse_sections function is called
# THEN: the function returns a Section object with the correct title
def test_markdown_parse_sections_returns_section_with_correct_title(
    simple_markdown_string,
):
    """Test that the parse_sections function returns a Section object with the correct title."""
    sections = parse_sections(simple_markdown_string)
    assert sections.title == "Title"


# GIVEN: a markdown string
# WHEN: the parse_sections function is called
# THEN: the function returns a Section object with the correct content
def test_markdown_parse_sections_returns_section_with_correct_content(
    simple_markdown_string,
):
    """Test that the parse_sections function returns a Section object with the correct content."""
    sections = parse_sections(simple_markdown_string)
    assert sections.content == []


# GIVEN: a markdown string
# WHEN: the parse_sections function is called
# THEN: the function returns a Section object with the correct children
def test_markdown_parse_sections_returns_section_with_correct_children(
    simple_markdown_string,
):
    """Test that the parse_sections function returns a Section object with the correct children."""
    sections = parse_sections(simple_markdown_string)
    assert len(sections.children) == 2
    assert sections.children[0].title == "Section 1"
    assert sections.children[1].title == "Section 2"


@pytest.fixture
def complex_markdown_string():
    """A complex markdown string for testing."""
    return """# Title

## Section 1

This is the first section.

This is the first section's second paragraph.

### Subsection 1

This is the first subsection.

This is the first subsection's second paragraph.

### Subsection 2

This is the second subsection.

This is the second subsection's second paragraph.

## Section 2

This is the second section.

This is the second section's second paragraph.

### Subsection 1

This is the first subsection.

This is the first subsection's second paragraph.

#### Subsubsection 1

This is the first subsubsection.

## Section 3

This is the third section.
"""


# GIVEN: a markdown string
# WHEN: the parse_sections function is called
# THEN: the function returns a Section object with the correct children
# AND: the function returns a Section object with the correct grandchildren
# AND: the function returns a Section object with the correct great-grandchildren
def test_markdown_parse_sections_returns_section_with_correct_children_and_grandchildren(
    complex_markdown_string,
):
    """Test that the parse_sections function returns a Section object with the correct children."""
    sections = parse_sections(complex_markdown_string)
    assert len(sections.children) == 3
    assert sections.children[0].title == "Section 1"
    assert sections.children[1].title == "Section 2"
    assert sections.children[2].title == "Section 3"

    assert len(sections.children[0].children) == 2
    assert sections.children[0].children[0].title == "Subsection 1"
    assert sections.children[0].children[1].title == "Subsection 2"

    assert len(sections.children[1].children) == 1
    assert sections.children[1].children[0].title == "Subsection 1"

    assert len(sections.children[1].children[0].children) == 1
    assert sections.children[1].children[0].children[0].title == "Subsubsection 1"


def test_parse_sections_to_dict_returns_dict(complex_markdown_string):
    """Test that the parse_sections_to_dict function returns a dictionary."""
    sections = parse_sections(complex_markdown_string)
    sections_dict = sections.to_dict()
    assert isinstance(sections_dict, dict)


@pytest.fixture
def complex_markdown_skip_header_level_string():
    """A complex markdown string for testing."""
    return """# Title

## Section 1

This is the first section.

#### Subsection 1

This is a subsection but it's at the wrong level.

## Section 2

This is the second section.
"""


def test_parse_sections_to_dict_skips_header_level(
    complex_markdown_skip_header_level_string,
):
    """Test that the parse_sections_to_dict function skips header levels."""
    sections = parse_sections(complex_markdown_skip_header_level_string)
    assert len(sections.children) == 2
    assert sections.children[0].title == "Section 1"
    assert sections.children[1].title == "Section 2"

    assert len(sections.children[0].children) == 1
    assert sections.children[0].children[0].title == "Subsection 1"
    assert sections.children[0].children[0].content == [
        "This is a subsection but it's at the wrong level."
    ]
    assert sections.children[0].children[0].get_level() == 2


# endregion
