"""ChatGPT module for Kenku."""

import inspect
import json
import logging
from typing import Any, Callable, Dict, List, Literal, Protocol, Tuple

import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

from kenku.core.utils import FunctionTemplateGenerator

LOGGER = logging.getLogger(__name__)


# section protocol
class Section(Protocol):
    title: str
    parent: "Section"
    content: list
    children: list


# embeddings protocol
class Embeddings(Protocol):
    sections: list
    engine: str
    parallel: bool
    embeddings: dict

    def _generate_embeddings(self) -> dict:
        ...

    def generate_embeddings(self) -> None:
        ...

    def get_embedding(self, section: Section) -> list:
        ...

    def get_similar_sections(self, section: Section, n: int = 5) -> list:
        ...

    def save(self, path: str) -> None:
        ...

    @classmethod
    def load(cls, path: str) -> "Embeddings":
        ...


class GPTEngine:
    """GPT Engine.

    Attributes:
        model (str): The model to use. Defaults to "gpt-3.5-turbo".
        messages (List[Dict[str, str]]): The messages in the conversation. Defaults to [].
        system_messages (List[Dict[str, str]]): The system messages in the conversation. Defaults to [].
        functions (List[Callable]): The functions that ChatGPT can call. Defaults to []. Functions must have a docstring. See FunctionTemplateGenerator for more information.
        function_call ("auto"|"none"|dict["name", str]): How to call functions. Defaults to "auto". If "auto", ChatGPT will automatically call functions. If "none", ChatGPT will not call functions. If a dict, ChatGPT will be forced to call functions with the given name.
        function_call_limit (int): The maximum number of function calls allowed in a single response. Defaults to 5. If you're getting a "Too many function calls" error, try increasing this value or tweak your system prompts to reduce the number of function calls that ChatGPT makes.

    Methods:
        get_response(message: str) -> str: Get a response from the engine.

    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        messages: List[Dict[str, str]] = None,
        system_messages: List[Dict[str, str]] = None,
        functions: List[Callable] = None,
        function_call: Literal["auto", "none"] | Dict[Literal["name"], str] = "auto",
        function_call_limit: int = 5,
    ):
        """Initialize the engine."""
        self.model = model

        self.system_messages = system_messages or []

        self.messages = messages or []

        self.functions = functions or []
        self.function_call = function_call
        self.function_call_limit = function_call_limit
        self._templates = FunctionTemplateGenerator(self.functions).generate_templates()

        self._functions: dict[str, Callable] = {
            function.__name__: function for function in self.functions
        }

    def get_response(self, message: str) -> str:
        """Main method for getting a response from the engine. Handles all logic for generating a response,
        including parsing function calls from the ChatGPT response, and providing function responses
        to the ChatGPT engine for further processing.

        Args:
            message (str): The message to get a response for.

        Returns:
            str: The final response to the message, after parsing function calls and calling the appropriate function.
        """
        if not message:
            return ""

        self._add_user_message(message)
        response = self._generate_response()
        self.messages.append(response)
        response = self._handle_function_calls(response)
        return response

    def _handle_function_calls(self, response) -> str:
        for _ in range(self.function_call_limit):
            function_calls = self._parse_function_calls(response)
            if function_calls[0] is None:
                break
            function_responses = self._get_function_response(function_calls)
            self.messages.append(function_responses)
            response = self._generate_response()
            self.messages.append(response)
        else:
            raise RuntimeError("Too many function calls")
        return response

    def _add_user_message(self, message: str) -> None:
        """Add a user message to the conversation."""
        self.messages.append({"content": message, "role": "user"})

    def _add_system_message(self, message: str) -> None:
        """Add a system message to the conversation."""
        self.system_messages.append({"content": message, "role": "system"})

    @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(5))
    def _generate_response(self) -> Dict[str, str]:
        """Generate a response from the model."""
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.system_messages.copy() + self.messages.copy(),
            functions=self._templates,
            function_call=self.function_call,
        )

        return response.choices[0]["message"]

    def _parse_function_calls(self, response: dict) -> Tuple[str, List[Dict[str, Any]]]:
        """Check a response to see if GPT wanted to call a function."""
        if response.get("function_call"):
            # function call arguments are returned as a json string, so we need to parse them
            response_args = json.loads(response["function_call"]["arguments"])
            return (
                response["function_call"]["name"],
                response_args,
            )
        return None, None

    def _get_function_response(
        self, function_calls: Tuple[str, List[Dict[str, Any]]]
    ) -> Dict[str, str]:
        """Get the response from the functions."""
        function_name, arguments = function_calls
        function = self._functions[function_name]
        response = function(**arguments)

        return dict(
            role="function",
            name=function_name,
            content=response,
        )

    def save_conversation(self, path: str) -> None:
        """Save the conversation to a file."""
        with open(path, "w") as f:
            json.dump(self.messages, f, indent=4)

    def load_conversation(self, path: str) -> None:
        """Load a conversation from a file."""
        with open(path, "r") as f:
            self.messages = json.load(f)
