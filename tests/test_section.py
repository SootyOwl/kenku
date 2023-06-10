from kenku.markdown import Section


def test_section_init():
    section = Section("Title", "Content")
    assert section.title == "Title"
    assert section.content == ["Content"]
    assert section.parent is None
    assert section.children == []

    section = Section("Title", ["Content 1", "Content 2"])
    assert section.title == "Title"
    assert section.content == ["Content 1", "Content 2"]
    assert section.parent is None
    assert section.children == []


def test_section_add_child():
    parent = Section("Parent", "Parent content")
    child = Section("Child", "Child content")
    parent.add_child(child)
    assert child.parent == parent
    assert parent.children == [child]

    grandchild = Section("Grandchild", "Grandchild content")
    child.add_child(grandchild)
    assert grandchild.parent == child
    assert child.children == [grandchild]
    assert parent.children == [child]


def test_section_get_descendants():
    parent = Section("Parent", "Parent content")
    child1 = Section("Child 1", "Child 1 content")
    child2 = Section("Child 2", "Child 2 content")
    grandchild1 = Section("Grandchild 1", "Grandchild 1 content")
    grandchild2 = Section("Grandchild 2", "Grandchild 2 content")
    parent.add_children([child1, child2])
    child1.add_child(grandchild1)
    child2.add_child(grandchild2)

    descendants = parent.get_descendants()
    assert len(descendants) == 4
    assert child1 in descendants
    assert child2 in descendants
    assert grandchild1 in descendants
    assert grandchild2 in descendants


def test_section_get_ancestors():
    parent = Section("Parent", "Parent content")
    child = Section("Child", "Child content")
    grandchild = Section("Grandchild", "Grandchild content")
    parent.add_child(child)
    child.add_child(grandchild)

    ancestors = grandchild.get_ancestors()
    assert len(ancestors) == 2
    assert child in ancestors
    assert parent in ancestors


def test_section_get_siblings():
    parent = Section("Parent", "Parent content")
    child1 = Section("Child 1", "Child 1 content")
    child2 = Section("Child 2", "Child 2 content")
    child3 = Section("Child 3", "Child 3 content")
    parent.add_children([child1, child2, child3])

    siblings = child2.get_siblings()
    assert len(siblings) == 2
    assert child1 in siblings
    assert child3 in siblings


def test_section_get_level():
    parent = Section("Parent", "Parent content")
    child1 = Section("Child 1", "Child 1 content")
    child2 = Section("Child 2", "Child 2 content")
    grandchild = Section("Grandchild", "Grandchild content")
    parent.add_children([child1, child2])
    child1.add_child(grandchild)

    assert parent.get_level() == 0
    assert child1.get_level() == 1
    assert child2.get_level() == 1
    assert grandchild.get_level() == 2


def test_section_to_dict():
    parent = Section("Parent", "Parent content")
    child1 = Section("Child 1", "Child 1 content")
    child2 = Section("Child 2", "Child 2 content")
    grandchild = Section("Grandchild", "Grandchild content")
    parent.add_children([child1, child2])
    child1.add_child(grandchild)

    section_dict = parent.to_dict()
    assert section_dict["title"] == "Parent"
    assert section_dict["content"] == ["Parent content"]
    assert len(section_dict["children"]) == 2
    assert section_dict["children"][0]["title"] == "Child 1"
    assert section_dict["children"][0]["content"] == ["Child 1 content"]
    assert len(section_dict["children"][0]["children"]) == 1
    assert section_dict["children"][0]["children"][0]["title"] == "Grandchild"
    assert section_dict["children"][0]["children"][0]["content"] == ["Grandchild content"]
    assert section_dict["children"][1]["title"] == "Child 2"
    assert section_dict["children"][1]["content"] == ["Child 2 content"]
    assert len(section_dict["children"][1]["children"]) == 0


def test_section_from_dict():
    section_dict = {
        "title": "Parent",
        "content": ["Parent content"],
        "children": [
            {
                "title": "Child 1",
                "content": ["Child 1 content"],
                "children": [
                    {
                        "title": "Grandchild",
                        "content": ["Grandchild content"],
                        "children": [],
                    }
                ],
            },
            {"title": "Child 2", "content": ["Child 2 content"], "children": []},
        ],
    }

    parent = Section.from_dict(section_dict)
    assert parent.title == "Parent"
    assert parent.content == ["Parent content"]
    assert len(parent.children) == 2
    assert parent.children[0].title == "Child 1"
    assert parent.children[0].content == ["Child 1 content"]
    assert len(parent.children[0].children) == 1
    assert parent.children[0].children[0].title == "Grandchild"
    assert parent.children[0].children[0].content == ["Grandchild content"]
    assert len(parent.children[0].children[0].children) == 0
    assert parent.children[1].title == "Child 2"
    assert parent.children[1].content == ["Child 2 content"]
    assert len(parent.children[1].children) == 0


def test_section_from_markdown():
    markdown = "# Parent\n\nParent content\n\n## Child 1\n\nChild 1 content\n\n### Grandchild\n\nGrandchild content\n\n## Child 2\n\nChild 2 content"
    parent = Section.from_markdown(markdown)

    assert parent.title == "Parent"
    assert parent.content == ["Parent content"]
    assert len(parent.children) == 2
    assert parent.children[0].title == "Child 1"
    assert parent.children[0].content == ["Child 1 content"]
    assert len(parent.children[0].children) == 1
    assert parent.children[0].children[0].title == "Grandchild"
    assert parent.children[0].children[0].content == ["Grandchild content"]
    assert len(parent.children[0].children[0].children) == 0
    assert parent.children[1].title == "Child 2"
    assert parent.children[1].content == ["Child 2 content"]


def test_section_to_markdown():
    parent = Section("Parent", "Parent content")
    child1 = Section("Child 1", "Child 1 content")
    child2 = Section("Child 2", "Child 2 content")
    grandchild = Section("Grandchild", "Grandchild content")
    parent.add_children([child1, child2])
    child1.add_child(grandchild)

    markdown = parent.to_markdown()
    assert markdown == "# Parent\n\nParent content\n\n## Child 1\n\nChild 1 content\n\n### Grandchild\n\nGrandchild content\n\n## Child 2\n\nChild 2 content"