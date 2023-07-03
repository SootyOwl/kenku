# BEGIN: 8f6c5d4b9f3a
import os

import pytest

from kenku.core.embeddings import Embeddings
from kenku.core.markdown import Section

from .conftest import skip_openai


@pytest.fixture(scope="session")
def sections():
    return [
        Section(title="Introduction", parent=None, content=["Hello, world!"]),
        Section(
            title="Section 1",
            parent=None,
            content=["This is the first section of the document."],
        ),
        Section(
            title="Section 2",
            parent=None,
            content=["This is the second section of the document."],
        ),
    ]


@pytest.fixture(scope="session")
def embeddings(sections):
    # if we have a test embeddings file, load it
    if os.path.exists("tests/test_embeddings.pickle"):
        return Embeddings.load("tests/test_embeddings.pickle")
    # otherwise, generate the embeddings, save them, and return them
    e = Embeddings(sections)
    e.generate_embeddings()
    e.save("tests/test_embeddings.pickle")
    return e


@skip_openai
def test_generate_embeddings(embeddings):
    assert len(embeddings.embeddings) == 3
    assert all(
        isinstance(embedding, list) for embedding in embeddings.embeddings.values()
    )


@skip_openai
def test_get_embedding(embeddings, sections):
    embedding = embeddings.get_embedding(sections[0])
    assert isinstance(embedding, list)
    assert len(embedding) == 12288


@skip_openai
def test_get_similar_sections(embeddings, sections):
    similar_sections = embeddings.get_similar_sections(sections[1], n=2)
    assert len(similar_sections) == 2
    assert similar_sections[0][0] == sections[2]
    assert similar_sections[1][0] == sections[0]


@skip_openai
def test_save_embeddings(embeddings, tmp_path):
    embeddings.save(tmp_path / "embeddings.pickle")
    assert os.path.exists(tmp_path / "embeddings.pickle")
    # load the embeddings
    embeddings_loaded = Embeddings.load(tmp_path / "embeddings.pickle")
    # check that the embeddings exist and are the same
    assert embeddings_loaded.embeddings is not None
    assert embeddings_loaded.embeddings == embeddings.embeddings
