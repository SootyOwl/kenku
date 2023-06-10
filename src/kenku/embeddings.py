"""Embeddings for Kenku.

This module contains the classes and functions for the embeddings used in Kenku.

Classes:
    Section(Protocol): A protocol for a section of a document, from markdown.py.
    """
from typing import List, Optional, Tuple, Protocol
import openai
from openai.embeddings_utils import cosine_similarity, get_embedding
import tiktoken

EMBEDDINGS_MODEL = "text-embedding-ada-002"
EMBEDDINGS_ENCODING = "cl100k_base"  # encoding for ada-002
EMBEDDINGS_MAX_TOKENS = 8000  # max tokens for ada-002

encoding = tiktoken.get_encoding(EMBEDDINGS_ENCODING)


class Section(Protocol):
    title: str
    parent: Optional["Section"]
    content: List[str]
    children: Optional[List["Section"]]

    def to_markdown(self) -> str:
        ...


class Embeddings:
    """A class for generating embeddings for sections of ."""

    def __init__(
        self,
        sections: List[Section],
        engine: str = EMBEDDINGS_MODEL,
        parallel: bool = True,
    ):
        """Initialize an Embeddings object."""
        self.sections = sections
        """The sections of the document."""
        self.engine = engine
        """The engine to use for generating embeddings."""
        self.parallel = parallel
        """Whether to generate embeddings in parallel."""
        self.embeddings = self._generate_embeddings()
        """The embeddings for each section of the document."""

    def _generate_embeddings(self) -> dict[Section, List[float]]:
        """Generate embeddings for each section of the document."""
        # check if api key is set
        if not openai.api_key:
            raise ValueError(
                "OpenAI API key not set. Set the OPENAI_API_KEY environment variable."
            )
        if self.parallel:
            # import the multiprocessing module
            import multiprocessing

            # create a pool of workers
            with multiprocessing.Pool() as pool:
                # generate the embeddings
                embeddings = pool.map(
                    get_embedding,
                    (section.to_markdown() for section in self.sections),
                    chunksize=1,
                )
        else:
            # generate the embeddings
            embeddings = [
                get_embedding(section.to_markdown())
                for section in self.sections
            ]

        # return the embeddings as a dictionary of section: embedding pairs
        return dict(zip(self.sections, embeddings))
    
    
    def get_embedding(self, section: Section) -> List[float]:
        """Get the embedding for a section."""
        return self.embeddings[section]

    def get_similar_sections(
        self, section: Section, n: int = 5
    ) -> List[Tuple[Section, float]]:
        """Get the n most similar sections to a given section."""
        if section not in self.sections:
            raise ValueError("Section not in document.")
        # get the embedding for the given section
        embedding = self.get_embedding(section)
        # get the cosine similarity between the given section and all other sections
        similarities = [
            (
                other_section,
                cosine_similarity(embedding, self.get_embedding(other_section)),
            )
            for other_section in self.sections if other_section != section
        ]
        # sort the similarities by the cosine similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        # return the n most similar sections
        return similarities[:n]
