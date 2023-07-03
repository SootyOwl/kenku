"""Module for parsing markdown files into a tree of Section objects."""

import copy
import re
from typing import List, Optional

HEADINGS_REGEX = re.compile(r"(?P<level>#+)\s*(?P<heading>.+)")


class Section:
    """A class representing a section of a markdown document."""

    def __init__(
        self,
        title: str,
        content: List[str] | str,
        parent: Optional["Section"] = None,
        children: Optional[List["Section"]] = None,
    ):
        """Initialize a Section object.

        Args:
            title (str): The title of the section.
            content (List[str]): The content of the section, split by paragraph.
            parent (Section, optional): The parent section of this section. Defaults to None.
            children (List[Section], optional): The child sections of this section. Defaults to None.
        """
        self.title = title
        if isinstance(content, str):
            self.content = [content]
        elif isinstance(content, list):
            self.content = content
        else:
            raise TypeError(f"content must be of type str or list, not {type(content)}")
        self.parent = parent
        if self.parent:
            self.parent.add_child(self)
        if children:
            self.children = children
            # add self as parent to children
            for child in self.children:
                child.parent = self
        else:
            self.children = []

    def __str__(self):
        return f"Section: {self.title}"

    def __repr__(self):
        if self.parent:
            return f"Section(title='{self.title}', content={self.content}, parent='{self.parent.title}', children={[child.title for child in self.children]})"
        else:
            return f"Section(title='{self.title}', content={self.content}, children={[child.title for child in self.children]})"

    def __eq__(self, other):
        return self.to_dict() == other.to_dict() if isinstance(other, Section) else False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(
            (self.title, hash("".join(self.content)))
        )

    def add_child(self, child: "Section"):
        """Add a child section to this section.

        Args:
            child (Section): The child section to add.
        """
        self.children.append(child)
        child.parent = self

    def add_children(self, children: List["Section"]):
        """Add multiple child sections to this section.

        Args:
            children (List[Section]): The child sections to add.
        """
        for child in children:
            self.add_child(child)

    def get_descendants(self) -> List["Section"]:
        """Get all descendants of this section.

        Returns:
            List[Section]: A list of all descendant sections.
        """
        descendants = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_descendants())
        return descendants

    def get_ancestors(self) -> List["Section"]:
        """Get all ancestors of this section.

        Returns:
            List[Section]: A list of all ancestor sections.
        """
        ancestors = []
        if self.parent:
            ancestors.append(self.parent)
            ancestors.extend(self.parent.get_ancestors())
        return ancestors

    def get_siblings(self) -> List["Section"]:
        """Get all siblings of this section.

        Returns:
            List[Section]: A list of all sibling sections.
        """
        siblings = []
        if self.parent:
            siblings.extend(self.parent.children)
            siblings.remove(self)
        return siblings

    def get_level(self) -> int:
        """Get the level of this section in the tree.

        The level is defined as the number of ancestors of this section.
        The root / top-level section has level 0.

        Returns:
            int: The level of this section in the tree.
        """
        return self.parent.get_level() + 1 if self.parent else 0

    def to_dict(self) -> dict:
        """Convert this section to a dictionary, for use in JSON serialization."""
        return {
            "title": self.title,
            "content": self.content,
            "children": [child.to_dict() for child in self.children],
        }

    def copy(self) -> "Section":
        """Create a copy of this section."""
        return copy.deepcopy(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Section":
        """Create a section from a dictionary, for use in JSON deserialization."""
        return cls(
            title=data["title"],
            content=data["content"],
            children=[cls.from_dict(child) for child in data["children"]],
        )

    @classmethod
    def from_markdown(cls, markdown: str) -> "Section":
        """Create a section from a markdown string."""
        builder = SectionBuilder()
        for line in markdown.splitlines():
            builder.add_line(line)
        return builder.tree()

    def to_markdown(self, include_children=False) -> str:
        """Convert this section to a markdown string."""
        return "\n\n".join(self._md_lines(include_children=include_children))

    def _md_lines(self, include_children: bool = False) -> List[str]:
        """Get the markdown lines for this section."""
        # get the level of this section
        level = self.get_level() + 1
        # create the heading
        heading = "#" * level + " " + self.title
        md_lines = [heading, *self.content]
        # create the children
        if include_children:
            for child in self.children:
                md_lines.extend(child._md_lines(True))
        return md_lines


class SectionBuilder:
    """A class for building a tree of Section objects from a markdown string."""

    def __init__(self, sections: Optional[List[Section]] = None):
        """Initialize a SectionBuilder object."""
        self.builder = Section("", [])
        if sections is not None:
            self.builder.add_children(sections)
        self.current_section = self.builder
        self.current_level = 0

    def add_line(self, line: str):
        """Add a line of markdown to the tree of sections.

        Args:
            line (str): The line of markdown to add.
        """
        line = line.strip()
        if not line:
            return
        if match := HEADINGS_REGEX.match(line):
            level = len(match.group("level"))
            heading = match.group("heading")
            if level > self.current_level:
                self._add_child(level, heading)
            elif level == self.current_level:
                self._add_sibling(heading)
            elif level < self.current_level:
                self._add_ancestor(level, heading)
        else:
            self.current_section.content.append(line)

    def _add_ancestor(self, level, heading):
        """Add an ancestor section at the given level."""
        section = Section(heading, [])
        parent = self.current_section.parent
        while parent and parent.get_level() >= level:
            parent = parent.parent
        parent.add_child(section) if parent else self.builder.add_child(section)
        self.current_section = section
        self.current_level = level

    def _add_child(self, level, heading):
        """Add a child section to the current section."""
        section = Section(heading, [])
        self.current_section.add_child(section)
        self.current_section = section
        self.current_level = level

    def _add_sibling(self, heading):
        """Add a sibling section to the current section."""
        section = Section(heading, [])
        self.current_section.parent.add_child(section) if self.current_section.parent else self.builder.add_child(section)
        self.current_section = section

    def tree(self) -> Section:
        """Build the tree of sections.

        Returns:
            Section: The root section of the tree.
        """
        # make a deep copy of the tree, remove the root section, and return the copy
        try:
            tree = self.builder.children[0].copy()
        except IndexError:
            tree = Section("", [])
        tree.parent = None
        return tree


def parse_sections(markdown: str) -> Section:
    """Parse a markdown string into a tree of Section objects.

    Args:
        markdown (str): The markdown string to parse.

    Returns:
        Section: The root section of the parsed markdown.
    """
    lines = markdown.splitlines()
    builder = SectionBuilder()
    for line in lines:
        builder.add_line(line)
    return builder.tree()
