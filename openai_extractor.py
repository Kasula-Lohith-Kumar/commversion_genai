import re
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('API_KEY'))

def extract_with_openai(conversation, model, pf):
    """
    Extract structured information from a conversation using an OpenAI chat completion model.

    Sends the conversation text to the specified model with a system prompt loaded from file,
    retrieves the model's response, parses it as JSON, and returns both the parsed prediction
    and token usage statistics.

    Args:
        conversation (str):
            The full conversation text (usually formatted as a single string with roles,
            timestamps, or other markers depending on the prompt expectations).

        model (str):
            The OpenAI model identifier to use (e.g. "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo").

        pf (str | Path):
            Path to the file containing the system prompt/instructions.
            The entire content of this file will be used as the system message.

    Returns:
        dict:
            Dictionary with two keys:
            - "prediction": dict
                The parsed JSON object returned by the model (or {} if parsing failed)
            - "usage": dict
                Token usage statistics containing:
                - prompt_tokens (int)
                - completion_tokens (int)
                - total_tokens (int)

    Raises:
        FileNotFoundError: If the prompt file (pf) cannot be opened
        openai.OpenAIError: If the API call fails (rate limits, authentication, etc.)
        AttributeError/KeyError: If the response structure is unexpected

    Notes:
        - Assumes `client` is a properly initialized `openai.OpenAI` (or compatible) client
        - Relies on `parse_llm_json()` to handle common LLM JSON output variations
        - Prints a warning to stdout if JSON parsing fails
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": open(pf).read()},
            {"role": "user", "content": conversation}
        ]
    )

    prediction_text = response.choices[0].message.content

    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens
    }

    return {
        "prediction": parse_llm_json(prediction_text),
        "usage": usage
    }

def parse_llm_json(content: str):
    """
    Robustly extract a JSON object from LLM-generated text.

    Handles common output patterns produced by language models:
    - Markdown fenced code blocks: ```json {...} ```
    - Raw JSON objects that may be embedded in longer text
    - Leading/trailing whitespace or extra newlines

    Args:
        content (str):
            Raw text output from an LLM, expected to contain a JSON object.

    Returns:
        dict:
            Parsed JSON object if successful, or empty dict {} if:
            - input is empty
            - no JSON-like structure is found
            - JSON is malformed / cannot be parsed

    Behavior:
        1. Strips leading/trailing whitespace
        2. Attempts to extract content inside ```json ... ``` fences (most common pattern)
        3. If no fenced block found, tries to extract the first {...} substring
        4. Attempts to parse the extracted string with json.loads()
        5. Prints a warning to stdout and returns {} on any JSON decode error

    Example:
        >>> parse_llm_json('Some text\n```json\n{"name": "Alice", "age": 28}\n```')
        {'name': 'Alice', 'age': 28}

        >>> parse_llm_json('{"items": [1,2,3]} and some extra text')
        {'items': [1, 2, 3]}

        >>> parse_llm_json('No json here')
        {}
    """
    if not content:
        return {}

    # Remove ```json and ``` fences
    content = content.strip()

    # Case 1: markdown fenced JSON
    fence_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
    if fence_match:
        content = fence_match.group(1)

    # Case 2: fallback – first JSON object found
    else:
        brace_match = re.search(r"(\{.*\})", content, re.DOTALL)
        if brace_match:
            content = brace_match.group(1)

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print("JSON parse failed:", e)
        return {}