import pytest
import openai
import os

SKIP_OPENAI_ENVVAR = os.environ.get("SKIP_OPENAI_TESTS", True)

skip_openai = pytest.mark.skipif(
    not openai.api_key or SKIP_OPENAI_ENVVAR,
    reason="OpenAI API key not set. Set the OPENAI_API_KEY environment variable.",
)