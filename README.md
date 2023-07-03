# Kenku: Markdown Information Retrieval and Generation

Kenku is a tool for generating and retrieving information from Markdown files. It is built on top of ChatGPT and OpenAI's API. Using OpenAI Embeddings, Kenku can intelligently search for relevant context in Markdown files. Using ChatGPT, Kenku can generate Markdown files from a prompt[^1].

[^1]: This is not yet implemented.

Kenku is named after the [Kenku](https://en.wikipedia.org/wiki/Kenku), a birdlike creature from Dungeons and Dragons:

> Kenku "...are most recognizable for their lack of a voice; instead of speaking themselves, they use their gift of mimicry to communicate. [...] they can cleverly piece together voices and sounds they've heard to communicate."

## Installation

First, checkout the repository:

```bash
git clone https://github.com/SootyOwl/kenku.git
```

Then, install the dependencies:

```bash
cd kenku
poetry install
```

## Usage

Current functionality is limited to searching for context in Markdown files. To use Kenku, you must first obtain an API key from OpenAI, and put it in a file named `.apikey` or set the `OPENAI_API_KEY` environment variable. Then, you can run the following command and follow the prompts:

```bash
poetry run kenku 
```

## OpenAI Cookbook(s)

The following are links to the OpenAI Cookbook examples that are relevant to Kenku:

- [Obtain dataset](https://github.com/openai/openai-cookbook/blob/main/examples/Obtain_dataset.ipynb)
- [Semantic text search using embeddings](https://github.com/openai/openai-cookbook/blob/main/examples/Semantic_text_search_using_embeddings.ipynb)
- [Question answering using embeddings (for context)](https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb)
- [How to call functions with chat models](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_with_chat_models.ipynb)
