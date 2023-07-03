"""ChatGPT module for Kenku."""

import inspect
import json
from typing import Any, Callable, Dict, List, Literal, Protocol, Tuple

import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential


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


class FunctionTemplateGenerator:
    """Takes a set of callables and tries to build a ChatGPT template for each one using introspection.

    e.g.:
    Takes the function `get_current_weather(location: string, unit: 'celsius' | 'fahrenheit')` and generates a template like:
    [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
    ]

    Note that the parameters object is in JSON Schema format, strictly.
    Each property in the parameters object is a parameter to the function.
    Docstrings are used to populate the description field for each parameter.

    The callable must define a short docstring for each variable and a type annotation for each variable.
    """

    def __init__(self, functions: List[Callable]):
        """Initialize the FunctionTemplateGenerator."""
        self.functions = functions
        """The functions to generate templates for."""

    def generate_templates(self) -> List[Dict[str, Any]]:
        """Generate templates for each function."""
        return [self._generate_template(function) for function in self.functions]

    def _generate_template(self, function: Callable) -> Dict[str, Any]:
        """Generate a template for a function."""
        return {
            "name": function.__name__,
            "description": function.__doc__.splitlines()[0],
            "parameters": self._generate_parameters(function),
        }

    def _generate_parameters(self, function: Callable) -> Dict[str, Any]:
        """Generate a parameters object for a function."""
        parameters = {
            "type": "object",
            "properties": {},
            "required": [],
        }
        for name, parameter in inspect.signature(function).parameters.items():
            if parameter.default == inspect._empty:
                parameters["required"].append(name)
            parameters["properties"][name] = {
                "type": self._get_type(parameter.annotation),
                "description": self._get_description(function, parameter),
            }
            # add default value if it exists
            if parameter.default != inspect._empty:
                parameters["properties"][name]["default"] = parameter.default

            # add enum if it exists
            if parameter.annotation != inspect._empty and hasattr(
                parameter.annotation, "__args__"
            ):
                parameters["properties"][name]["enum"] = parameter.annotation.__args__

        return parameters

    def _get_type(self, annotation: Any) -> str:
        """Get the type for a parameter."""
        if annotation == inspect._empty:
            return "string"
        if annotation == int:
            return "integer"
        if annotation == float:
            return "number"
        if annotation == bool:
            return "boolean"
        if annotation == list:
            return "array"
        if annotation == dict:
            return "object"
        return "string"

    def _get_description(self, function: Callable, parameter: inspect.Parameter) -> str:
        """Get the description for a parameter from the docstring of the function.

        The description is the first line of the docstring that starts with the parameter name.

        e.g.:
        ```
        def get_current_weather(location: str, unit: str = "celsius") -> str:
            \"""Get the current weather in a given location.

            location: The city and state, e.g. San Francisco, CA
            unit: The unit to return the temperature in, e.g. celsius or fahrenheit
            \"""
            ...
        ```
        """
        if docstring := function.__doc__:
            for line in docstring.splitlines():
                if line.strip().startswith(parameter.name):
                    return line.strip().split(":")[1].strip()


class KenkuGPTEngine:
    """Kenku GPT Engine.

    Kenku is a tool for answering questions and generating new information for a D&D campaign.

    Attributes:
        model (str): The model to use for generating responses. Defaults to "gpt-3.5-turbo".
        messages (List[Dict[str, str]]): The initial messages to use for generating responses. Defaults to None.
        system_messages (List[Dict[str, str]]): The system messages to use for generating responses.
        functions (List[Callable]): The functions to use for generating responses. Defaults to None. Must return a string.
        function_call ("auto"|"none"|dict["name", str]): The function call to use for generating responses. Defaults to "auto" for automatic detection.

    Methods:
        get_response: Get a response from the engine.
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        messages: List[Dict[str, str]] = None,
        system_messages: List[Dict[str, str]] = None,
        functions: List[Callable] = None,
        function_call: Literal["auto", "none"] | Dict[Literal["name"], str] = "auto",
    ):
        """Initialize the engine."""
        self.model = model
        """The model to use for generating responses."""

        self.system_messages = system_messages or []
        """The system messages are used to instruct the ChatGPT engine in what it should do."""

        self.messages = messages or []
        """The initial messages to use for generating responses."""

        self.functions = functions or []
        """The functions to use for generating responses."""

        self.function_call = function_call
        """The function call to use for generating responses. 
        Defaults to "auto" for automatic detection."""

        self._templates = FunctionTemplateGenerator(self.functions).generate_templates()
        """The templates to use for generating responses."""

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

        # build the prompt of all messages to be sent to the model
        self._add_user_message(message)

        # generate the response from the model
        response = self._generate_response()

        # add the response to the conversation
        self.messages.append(response)

        # parse the response for function calls
        i = 0
        while (function_calls := self._parse_function_calls(response))[0] is not None:
            # get the function responses
            function_responses = self._get_function_response(function_calls)
            # add the function responses to the conversation
            self.messages.append(function_responses)
            # generate a new response from the model
            response = self._generate_response()
            # add the response to the conversation
            self.messages.append(response)
        # return the final response
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
            response_args = json.loads(
                response["function_call"]["arguments"]
            )
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
