"""Main module for Kenku.

Provided a directory of markdown files, Kenku will first parse the files into
sections, then generate embeddings for each section, and finally use the
KenkuGPTEngine to generate responses to user input about the markdown files.

The KenkuGPTEngine has the ability to ask for context from the markdown files
using the function call `get_context`. This function will use a cosine
similarity metric to find the most similar section to the context query and
return the section's text."""
from kenku.core.markdown import Section
from kenku.core.embeddings import Embeddings
from kenku.core.chatgpt import KenkuGPTEngine
from kenku.core.utils import get_markdown_files, get_file_text
import logging

import openai

openai.api_key_path = ".apikey"

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)


# define get_context function
def get_context(query: str, n: int = 5) -> str:
    """Provide context about the query topic from the markdown files.

    This function will use a cosine similarity metric to find the most similar
    sections to the query topic and return their contents.

    Args:
        query: The query topic to get context about.
    """
    if not hasattr(get_context, "embeddings"):
        raise AttributeError("Embeddings not set.")

    similar_to_query = get_context.embeddings.get_similar_sections_by_query(query, n)
    context = "\n\n\n***\n\n\n".join(
        section.to_markdown(True) for section, _ in similar_to_query
    )
    LOGGER.debug(f"Context: {context}")
    return context


def load_markdown_files(path: str):
    # Get the files from the markdown directory
    LOGGER.info("Getting markdown files...")
    files = get_markdown_files(path)
    LOGGER.debug(f"Found {len(files)} markdown files.")
    # Get the text from each file
    LOGGER.info("Getting text from markdown files...")
    return [get_file_text(file) for file in files]


def parse_markdown_files(texts):
    # Parse the text into sections
    LOGGER.info("Parsing markdown files into sections...")
    sections = [Section.from_markdown(text) for text in texts]
    LOGGER.debug(f"Found {len(sections)} sections.")

    return sections


def generate_embeddings(sections):
    # Generate embeddings for each section
    LOGGER.info("Generating embeddings for each section...")
    embeddings = Embeddings(sections, parallel=True)
    embeddings.generate_embeddings()
    LOGGER.debug(f"Generated embeddings for {len(embeddings.sections)} sections.")

    return embeddings


def save_embeddings(embeddings):
    # save the embeddings
    LOGGER.info("Saving embeddings...")
    path = input("Enter path to save embeddings: ") or "embeddings.pickle"
    embeddings.save(path)
    LOGGER.debug("Embeddings saved.")


def main():
    if input("Load embeddings? (y/n): ").lower() == "y":
        # load the embeddings
        LOGGER.info("Loading embeddings...")
        path = input("Enter path to embeddings: ") or "embeddings.pickle"
        embeddings = Embeddings.load(path)
        LOGGER.debug("Embeddings loaded.")
    else:
        # load the markdown files
        texts = load_markdown_files(
            input("Enter path to markdown files: ") or "markdown"
        )
        # parse the markdown files
        sections = parse_markdown_files(texts)
        # generate embeddings for each section
        embeddings = generate_embeddings(sections)
        # save the embeddings
        save_embeddings(embeddings)
    # run the engine
    LOGGER.info("Running engine...")
    run_engine(embeddings)


def run_engine(embeddings: Embeddings):
    # allow the get_context function to access the embeddings
    get_context.embeddings = embeddings

    # Create the KenkuGPTEngine
    engine = KenkuGPTEngine(
        functions=[get_context],
        function_call="auto",
    )

    # Create the ConversationPrinter
    printer = ConversationPrinter()

    # Start the conversation
    # print the system messages
    printer.print_messages(engine.system_messages)
    # print the messages
    printer.print_messages(engine.messages)
    while True:
        _msg_list = engine.messages.copy()
        # get the user input
        user_input = input("user: ")
        # remove the previous line from the terminal using ANSI escape sequence
        print("\033[A                             \033[A")
        response = engine.get_response(user_input)

        # print the messages that were added
        printer.print_messages(engine.messages[len(_msg_list) :])


from termcolor import colored


class ConversationPrinter:
    def __init__(self):
        self.role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "magenta",
        }

    def print_message(self, message):
        if message["role"] == "system":
            print(
                colored(
                    f"system: {message['content']}\n",
                    self.role_to_color[message["role"]],
                )
            )
        elif message["role"] == "user":
            print(
                colored(
                    f"user: {message['content']}\n", self.role_to_color[message["role"]]
                )
            )
        elif message["role"] == "assistant" and message.get("function_call"):
            print(
                colored(
                    f"assistant: {message['function_call']}\n",
                    self.role_to_color[message["role"]],
                )
            )
        elif message["role"] == "assistant" and not message.get("function_call"):
            print(
                colored(
                    f"assistant: {message['content']}\n",
                    self.role_to_color[message["role"]],
                )
            )
        elif message["role"] == "function":
            print(
                colored(
                    f"function ({message['name']}): {message['content']}\n",
                    self.role_to_color[message["role"]],
                )
            )

    def print_messages(self, messages):
        for message in messages:
            self.print_message(message)


if __name__ == "__main__":
    main()
