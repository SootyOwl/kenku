# BEGIN: 8f6c5d4b9f3a
from kenku.core.embeddings import Embeddings
from kenku.core.markdown import Section
import pytest


@pytest.fixture
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

import openai

skip_openai = pytest.mark.skipif(
    not openai.api_key,
    reason="OpenAI API key not set. Set the OPENAI_API_KEY environment variable.",
)

@skip_openai
def test_generate_embeddings(sections):
    embeddings = Embeddings(sections)
    assert len(embeddings.embeddings) == 3
    assert all(isinstance(embedding, list) for embedding in embeddings.embeddings.values())

@skip_openai
def test_get_embedding(sections):
    embeddings = Embeddings(sections)
    embedding = embeddings.get_embedding(sections[0])
    assert isinstance(embedding, list)
    assert len(embedding) == 12288  

@skip_openai
def test_get_similar_sections(sections):
    embeddings = Embeddings(sections)
    similar_sections = embeddings.get_similar_sections(sections[1], n=2)
    assert len(similar_sections) == 2
    assert similar_sections[0][0] == sections[2]
    assert similar_sections[1][0] == sections[0]
