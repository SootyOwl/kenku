import glob
import inspect
import logging
import re
from typing import Any, Callable, Dict, List

LOGGER = logging.getLogger(__name__)


def get_markdown_files(dir, exclude_dirs_patterns: list = None):
    """Recursively get the paths to all markdown files in a directory, excluding any directories in exclude_dirs.
    Directories in exclude_dirs are matched using glob syntax and may appear anywhere in the path.
    """
    if exclude_dirs_patterns is None:
        exclude_dirs_patterns = []
    paths = []
    for path in glob.glob(f"{dir}/**", recursive=True):
        if not path.endswith(".md"):
            continue
        # Skip any paths that match any of the exclude_dirs patterns anywhere in the path
        if any(re.search(pattern, path) for pattern in exclude_dirs_patterns):
            continue
        paths.append(path)
    return paths


def get_file_text(file: str) -> str:
    """Return the text of the given file."""
    with open(file, "r") as f:
        return f.read()


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
                "type": self._get_type(parameter),
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

    def _get_type(self, parameter: inspect.Parameter) -> str:
        # sourcery skip: assign-if-exp, reintroduce-else
        """Get the type for a parameter."""
        annotation = parameter.annotation
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
        LOGGER.warning(
            f"Unknown type {annotation} for parameter {parameter.name} in function. Defaulting to string."
        )
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
            LOGGER.warning(
                f"No description found for parameter {parameter.name} in function {function.__name__}. Add this parameter to the docstring to generate a template for this function."
            )
        return ""
