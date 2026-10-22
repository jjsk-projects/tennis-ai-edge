#!/usr/bin/env python3
"""Drive a local oMLX model for the TDD multi-model workflow.

Usage: omllab.py <model_id> <prompt_file> [max_tokens]
Sends the prompt, prints the response text, exits non-zero on failure.
"""
import json
import sys
import urllib.request

API = "http://127.0.0.1:8000/v1/chat/completions"
KEY = json.load(open("/Users/jjsk/.omlx/settings.json"))["auth"]["api_key"]


def call(model: str, prompt: str, max_tokens: int = 8192) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        API, data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        data = json.load(r)
    return data["choices"][0]["message"]["content"]


if __name__ == "__main__":
    model = sys.argv[1]
    prompt = open(sys.argv[2]).read()
    max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
    try:
        print(call(model, prompt, max_tokens))
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
