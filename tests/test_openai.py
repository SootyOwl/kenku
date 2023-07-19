"""Tests for the chatGPT component of the app.

Ensures the interface to the OpenAI API is working as expected.

Uses the openai Python package to interface with the OpenAI ChatCompletions API.
Also tests that chatGPT can call the functions to glean more context from the 
embeddings of the document.
"""

import os

import openai
import pytest

from kenku.core.chatgpt import GPTEngine
from kenku.core.utils import FunctionTemplateGenerator

from .conftest import skip_openai


# test the function template generator
def test_function_template_generator_simple():
    # define a test function with a docstring and a function body
    def test_function():
        """This is a test function."""
        print("Hello, world!")

    # create a function template generator
    ftg = FunctionTemplateGenerator([test_function])

    out = ftg.generate_templates()
    assert len(out) == 1
    assert out[0]["name"] == "test_function"
    assert out[0]["description"] == "This is a test function."
    assert out[0]["parameters"]["type"] == "object"
    assert out[0]["parameters"]["properties"] == {}
    assert out[0]["parameters"]["required"] == []


def test_function_template_generator_with_parameters():
    # define a test function with a docstring and a function body
    def test_function(a: int, b: str = "hello"):
        """This is a test function.

        Args:
            a: This is the first parameter.
            b: This is the second parameter.
        """
        print("Hello, world!")

    # create a function template generator
    ftg = FunctionTemplateGenerator([test_function])

    out = ftg.generate_templates()
    assert len(out) == 1
    assert out[0]["name"] == "test_function"
    assert out[0]["description"] == "This is a test function."
    assert out[0]["parameters"]["type"] == "object"
    assert out[0]["parameters"]["properties"] == {
        "a": {"type": "integer", "description": "This is the first parameter."},
        "b": {
            "type": "string",
            "description": "This is the second parameter.",
            "default": "hello",
        },
    }
    assert out[0]["parameters"]["required"] == ["a"]


class Response:
    """Mock response object for the OpenAI API.

    Has a choices attribute. This is a list of dictionaries. Each dictionary
    has a "message" key, which contains a dictionary with a "role" key and
    a "content" key.

    choices = [
        {
            "message": {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": None,
                    "arguments": {},
                },
            },
        }
    ]
    """

    def __init__(self, choices):
        self.choices = choices


# test the chatGPT engine
class TestKenkuGPTEngine:
    # add a fixture to create the engine
    @pytest.fixture
    def engine(self):
        return GPTEngine()

    @pytest.fixture
    def response(self):
        return Response(
            [
                {
                    "message": {"role": "assistant", "content": "Hello, world!"},
                }
            ]
        )

    @pytest.fixture
    def response_with_function_call(self):
        return Response(
            [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": "get_current_weather",
                            "arguments": '{\n  "location": "Glasgow, Scotland",\n  "format": "celsius"\n}',
                        },
                    },
                }
            ]
        )

    def test_init(self, engine):
        # check the attributes
        assert engine.function_call == "auto"
        assert engine.system_messages == []
        assert engine._templates == []
        assert engine.messages == []
        assert engine.model == "gpt-3.5-turbo"

    # test the private methods
    def test__add_user_message(self, engine):
        # add a user message
        engine._add_user_message("Hello, world!")

        # check the messages
        assert engine.messages == [
            {
                "role": "user",
                "content": "Hello, world!",
            }
        ]

    def test__add_system_message(self, engine):
        # add a system message
        engine._add_system_message("Hello, world!")

        # check the messages
        assert engine.system_messages == [
            {
                "role": "system",
                "content": "Hello, world!",
            }
        ]

    def test__parse_function_calls(self, engine, response_with_function_call):
        # parse the function calls
        f_name, f_args = engine._parse_function_calls(
            response_with_function_call.choices[0]["message"]
        )

        # check the function name and arguments
        assert f_name == "get_current_weather"
        assert f_args == {
            "location": "Glasgow, Scotland",
            "format": "celsius",
        }

    def test__parse_function_calls_no_function_call(self, engine):
        # parse the function calls
        f_name, f_args = engine._parse_function_calls(
            {
                "role": "assistant",
                "content": "Hello, world!",
            }
        )

        # check the function name and arguments
        assert f_name is None
        assert f_args is None

    @pytest.fixture
    def mock_openai(self, mocker):
        # mock the openai.ChatCompletion.create method
        return mocker.patch("kenku.core.chatgpt.openai.ChatCompletion.create")

    def test__generate_response(self, engine, mock_openai, response):
        # mock the response from the OpenAI API
        mock_openai.return_value = response
        # get a response
        response = engine._generate_response()

        # check the response
        assert response["content"] == "Hello, world!"
        assert response["role"] == "assistant"
        assert mock_openai.call_count == 1

    def test__generate_response_with_function_call(
        self, engine, mock_openai, response_with_function_call
    ):
        # mock the response from the OpenAI API
        mock_openai.return_value = response_with_function_call
        # get a response
        response = engine._generate_response()

        # check the response
        assert response["content"] is None
        assert response["role"] == "assistant"
        assert mock_openai.call_count == 1

    # test the public methods
    def test_get_response_no_function_call(self, engine, mock_openai, response, mocker):
        # mock the response from the OpenAI API
        mock_openai.return_value = response
        # mock the KenkuGPTEngine methods so we can check they are called
        mocker.patch.object(engine, "_add_user_message", return_value=None)
        mocker.patch.object(engine, "_parse_function_calls", return_value=(None, None))

        # get a response
        response = engine.get_response("Hello, world!")

        # check the response
        assert response["content"] == "Hello, world!"
        assert response["role"] == "assistant"
        assert mock_openai.call_count == 1
        assert engine._add_user_message.call_count == 1
        assert engine._parse_function_calls.call_count == 1

    def test_get_response_with_function_call(
        self, engine, mock_openai, response_with_function_call, mocker
    ):
        # mock the response from the OpenAI API
        mock_openai.return_value = response_with_function_call
        # mock the KenkuGPTEngine methods so we can check they are called
        mocker.patch.object(engine, "_add_user_message", return_value=None)
        # mock the _parse_function_calls method to return a function call first, then (None, None)
        parse_function_calls_mock = mocker.patch.object(engine, "_parse_function_calls")
        parse_function_calls_mock.side_effect = [
            (
                "get_current_weather",
                {
                    "location": "Glasgow, Scotland",
                    "format": "celsius",
                },
            ),
            (None, None),
        ]
        mocker.patch.object(
            engine,
            "_get_function_response",
            return_value=dict(
                role="function",
                name="get_current_weather",
                content="It's raining",
            ),
        )

        # get a response
        response = engine.get_response("Hello, world!")

        # check the response
        assert response["content"] is None
        assert response["role"] == "assistant"
        assert mock_openai.call_count == 2
        assert engine._add_user_message.call_count == 1
        assert parse_function_calls_mock.call_count == 2
        assert engine._get_function_response.call_count == 1


@skip_openai
class TestKenkuGPTEngineConversation:
    """Test having a conversation with the chatGPT engine."""

    @pytest.fixture
    def engine(self, mocker):
        """Set up the engine."""

        # define a function to get context about a topic
        def get_context(query: str) -> dict:
            """Get more context about the topic specified in the query.

            Args:
                query (str): The topic to get more information about.
            """
            return "This is the context about the topic"

        # create the engine
        engine = GPTEngine(
            function_call="auto",
            model="gpt-3.5-turbo",
            functions=[get_context],
        )

        # return the engine
        return engine

    def test_chatgpt_calls_function_when_asked(self, engine, mocker):
        """Test that the chatGPT engine calls the function when asked."""
        mocker.patch.object(
            engine,
            "_get_function_response",
            return_value={
                "role": "function",
                "name": "get_context",
                "content": "Port Damali is a city on the Menagerie Coast.",
            },
        )
        # get a response
        response = engine.get_response("Tell me about Port Damali")
        assert response
        assert engine._get_function_response.call_count == 1
