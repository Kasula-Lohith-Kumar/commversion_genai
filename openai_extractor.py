import re
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('API_KEY'))

def extract_with_openai(conversation, model, pf):
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