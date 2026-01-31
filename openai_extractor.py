import re
import json
from openai import OpenAI
from openai_keys import key

client = OpenAI(api_key=key)

def extract_with_openai(conversation, model="gpt-4.1-mini"):
    with open("prompt.txt", "r") as f:
        prompt_template = f.read()

    prompt = prompt_template.format(conversation=conversation)

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    content = response.choices[0].message.content
    # print(content)
    

    parsed = parse_llm_json(content)
    print(parsed)
    return parsed

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